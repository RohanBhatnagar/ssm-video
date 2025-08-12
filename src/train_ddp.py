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
            "h5py==3.11.0",
            "ptflops==0.7.4",
        ])
        .run_commands("mkdir -p /tmp/data")
        .add_local_file("models/originals.py", remote_path="/app/models/originals.py")
        .add_local_file("models/vqvae3d.py", remote_path="/app/models/vqvae3d.py")
        .add_local_file(
            "datasets/dataloader.py",
            remote_path="/app/datasets/dataloader.py"
        )
)

volume = modal.Volume.from_name("ssm-video", create_if_missing=True)

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
    from models.originals import VQVAEVideo, VQVAEConfig
    from datasets.dataloader import MovingMNIST
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = DDP(VQVAEVideo(VQVAEConfig()).to(device), device_ids=[rank])
    
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Batch size per GPU: {batch_size} (total {batch_size * world_size})")
        print(f"Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for frames in loader:
            frames = frames.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=False):
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
    epochs: int = 3,
    lr: float = 2e-4,
    weight_decay: float = 0, # 1e-4, no need to have wd for 3 epochs
    save_every: int = 1,
    train: bool = False,
):
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
        call.get()
        print("Training completed!")
        
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def get_data_paths():
    import glob
    h5_paths = glob.glob("/data/batch_*.h5")
    return sorted(h5_paths)