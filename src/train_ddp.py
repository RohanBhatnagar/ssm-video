import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import modal

app = modal.App("video-frame-prediction-ddp")

# Build the Docker image: include model.py and the datasets package
image = (
    modal.Image
         .from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .pip_install([
             "torch==2.7.1",
             "torchvision==0.22.1",
             "tqdm==4.67.1",
             "numpy==2.2.6",
             "matplotlib==3.10.3",
             "pillow==11.3.0",
             "ptflops==0.7.4",
             "mamba-ssm",
         ])
         .run_commands("mkdir -p /app/datasets")
         .add_local_file("model.py", remote_path="/app/model.py")
         .add_local_file(
             "datasets/dataloader.py",
             remote_path="/app/datasets/dataloader.py"
         )
)

volume = modal.Volume.from_name("ssm-video", create_if_missing=True)

# Top-level worker function must be picklable for spawn
def ddp_worker(
    rank: int,
    world_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_every: int,
    data_paths: list,
):
    import sys
    sys.path.append("/app")
    from model import VQVAEVideo
    from datasets.dataloader import MovingMNIST
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = DDP(VQVAEVideo().to(device), device_ids=[rank])
    
    dataset = MovingMNIST(data_paths=data_paths)
    
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Batch size per GPU: {batch_size} (total {batch_size * world_size})")
        print(f"Epochs: {epochs}, LR: {lr}, WD: {weight_decay}")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for frames in loader:
            frames = frames.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                reconstruction, _, _, total_loss = model.module(frames)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()

        scheduler.step()

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} | avg loss: {avg_loss:.4f}")
            if (epoch + 1) % save_every == 0:
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                }
                path = f"/data/checkpoint_epoch_{epoch+1}.pt"
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

    # final save on rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), "/data/final_model.pt")
        print("Final model saved to /data/final_model.pt")

    dist.destroy_process_group()


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100:4",
    memory=32 * 1024,
    timeout=8 * 3600,
)
def train_worker(
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_every: int,
    data_paths: list,
):
    """
    Launch DDP across all GPUs in this container by spawning the top-level worker.
    """
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        ddp_worker,
        args=(
            world_size,
            batch_size,
            epochs,
            lr,
            weight_decay,
            save_every,
            data_paths,
        ),
        nprocs=world_size,
        join=True,
    )


@app.local_entrypoint()
def main(
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    save_every: int = 10,
    train: bool = False,
    ptflops: bool = False,
):
    """
    Kick off training in a single container using all GPUs.
    """
    if train:
        print("Fetching data paths from Modal volume...")
        data_paths = get_data_paths.remote()
        print(f"Found {len(data_paths)} data files.")
        print("Launching distributed training on Modal (single-container, multi-GPU)...")
        call = train_worker.spawn(
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            save_every=save_every,
            data_paths=data_paths,
        )
        print("Training started—waiting for it to finish...")
        result = call.get()
        print("Training completed! Result:", result)
    elif ptflops: 
        examine_model.remote()
        get_data_paths.remote()
        
        
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=7200,
)
def examine_model():   
    import sys
    sys.path.append("/app")
    from model import VQVAEVideo, SpatialEncoder, SpatialDecoder, VectorQuantizerEMA
    from ptflops import get_model_complexity_info
    
    with torch.cuda.device(0):
        encoder = SpatialEncoder().cuda()
        encoder_macs, encoder_params = get_model_complexity_info(
            encoder,
            (3, 64, 64), 
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Encoder FLOPS: {encoder_macs}")
        print(f"Encoder Params: {encoder_params}")
        
        decoder = SpatialDecoder(
            in_ch=256, 
            base_ch=64, 
            num_blocks=4, 
            out_ch=3
        ).cuda()
        decoder_macs, decoder_params = get_model_complexity_info(
            decoder,
            (256, 4, 4), # (D_emb, H', W')
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Decoder FLOPS: {decoder_macs}")
        print(f"Decoder Params: {decoder_params}")
        
        quantizer = VectorQuantizerEMA(K=512, D=256).cuda()
        quantizer_macs, quantizer_params = get_model_complexity_info(
            quantizer,
            (1, 256), # input: posterior, (N, D) flattened (B*T*H'*W', D), just use 1 for a compelxity of a single vector
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Quantizer FLOPS: {quantizer_macs}")
        print(f"Quantizer Params: {quantizer_params}")
        
        # total model 
        model = VQVAEVideo().cuda()
        total_macs, total_params = get_model_complexity_info(
            model,
            (8, 3, 64, 64),  # B, T, C, H, W
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Total FLOPS: {total_macs}")
        print(f"Total Params: {total_params}")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def get_data_paths():
    import glob
    data_paths = glob.glob("/data/batch_*.pt")
    return data_paths