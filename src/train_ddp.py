import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import modal

# Modal App setup
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
             "ptflops==0.7.4",
             "mamba-ssm",
             "ptflops",
         ])
         .run_commands("mkdir -p /tmp/data")
         .add_local_file("model.py", remote_path="/app/model.py")
)

volume = modal.Volume.from_name("moving-mnist-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=7200,
)
def examine_model():  
    # in cuda env   
    import sys
    sys.path.append("/app")
    from model import VideoFramePredictor, Encoder, TemporalMamba, Decoder
    from ptflops import get_model_complexity_info
    
    with torch.cuda.device(0):
        encoder = Encoder().cuda()
        encoder_macs, encoder_params = get_model_complexity_info(
            encoder,
            (7, 3, 64, 64),  # 7 frames 3 channels 64x64
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Encoder FLOPS: {encoder_macs}")
        print(f"Encoder Params: {encoder_params}")
        
        temporal = TemporalMamba(input_dim=16384, hidden_dim=1024, depth=4).cuda()
        mamba_macs, mamba_params = get_model_complexity_info(
            temporal,
            (7, 16384),  # 7 timesteps 16384 latent dimension
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Mamba FLOPS: {mamba_macs}")
        print(f"Mamba Params: {mamba_params}")
        
        decoder = Decoder(latent_dim=16384, output_channels=3).cuda()
        decoder_macs, decoder_params = get_model_complexity_info(
            decoder,
            (16384,),  # single flattened latent vector
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Decoder FLOPS: {decoder_macs}")
        print(f"Decoder Params: {decoder_params}")
        
        model = VideoFramePredictor().cuda()
        total_macs, total_params = get_model_complexity_info(
            model,
            (7, 3, 64, 64),  # 7 frames, 3 channels, 64x64
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"Total FLOPS: {total_macs}")
        print(f"Total Params: {total_params}")
        
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100",
    timeout=7200,
    # cpu=8.0,
    # memory=32768,  # 32GB RAM
)
def train_worker(rank: int, world_size: int, master_addr: str, master_port: str, args: dict):
    """Worker function for distributed training on Modal"""
    import sys
    sys.path.append("/app")
    
    from src.model import VideoFramePredictor
    from src.datasets.modal_dataset_loader import MovingMNIST, ModalMovingMNIST
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    
    model = VideoFramePredictor().to(device)
    model = DDP(model, device_ids=[device])
    
    # use modal dataset
    try:
        dataset = ModalMovingMNIST(data_dir=args.get('modal_data_dir', '/data'))
        if rank == 0:
            print(f"Using Modal-generated dataset with {len(dataset)} sequences")
    except FileNotFoundError as e:
        print("Modal dataset not found!")
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args['batch_size'], 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args.get('weight_decay', 1e-4))
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.L1Loss()
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    
    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size per GPU: {args['batch_size']}")
        print(f"Total batch size: {args['batch_size'] * world_size}")
    
    for epoch in range(args['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        # only show progress bar only on rank 0
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args['epochs']}")
        else:
            pbar = dataloader
        
        for batch_idx, (frames, target) in enumerate(pbar):
            frames = frames.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                pred = model(frames)
                loss = loss_fn(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        scheduler.step()
        
        # Calculate average loss across all processes
        avg_loss = epoch_loss / num_batches
        dist.all_reduce(torch.tensor(avg_loss, device=device), op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / world_size
        
        if rank == 0:
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            if (epoch + 1) % args.get('save_every', 5) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, f"/data/checkpoint_epoch_{epoch+1}.pt")
                print(f"Checkpoint saved for epoch {epoch+1}")
    
    if rank == 0:
        final_model_path = "/data/final_model.pt"
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")
    
    dist.destroy_process_group()
    return {"final_loss": avg_loss, "epochs_completed": args['epochs']}

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=600,
)
def launch_distributed_training(world_size: int = 4, **training_args):
    """Launch distributed training across multiple Modal containers"""
    import time
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        master_port = str(s.getsockname()[1])
    
    master_addr = "127.0.0.1"
    
    print(f"Launching distributed training with {world_size} workers")
    print(f"Master address: {master_addr}:{master_port}")
    
    worker_calls = []
    for rank in range(world_size):
        call = train_worker.spawn(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            args=training_args
        )
        worker_calls.append(call)
    
    # Wait for all workers to complete
    results = []
    for i, call in enumerate(worker_calls):
        try:
            result = call.get()
            results.append(result)
            print(f"Worker {i} completed successfully")
        except Exception as e:
            print(f"Worker {i} failed with error: {e}")
            results.append({"error": str(e)})
    
    return {
        "success": True,
        "worker_results": results,
        "world_size": world_size
    }

@app.local_entrypoint()
def main():
    examine_model.remote()
    
# def main(
#     world_size: int = 4,
#     batch_size: int = 32,
#     epochs: int = 10,
#     lr: float = 2e-4,
#     weight_decay: float = 1e-4,
#     use_modal: bool = True,
#     num_sequences: int = 50000,
#     save_every: int = 5,
# ):
#     """Local entrypoint to launch distributed training on Modal"""
    
#     training_args = {
#         "batch_size": batch_size,
#         "lr": lr,
#         "epochs": epochs,
#         "weight_decay": weight_decay,
#         "use_modal": use_modal,
#         "modal_data_dir": "/data",
#         "num_sequences": num_sequences,
#         "save_every": save_every,
#     }
    
#     print("Starting distributed training on Modal...")
#     print(f"Configuration: {training_args}")
    
#     result = launch_distributed_training.remote(
#         world_size=world_size,
#         **training_args
#     )
    
#     print("Training completed!")
#     print(f"Results: {result}")

# if __name__ == "__main__":
#     print("This script is designed to run on Modal.")
#     print("To run locally, use: modal run src/train_ddp.py")
#     print("Or with custom parameters: modal run src/train_ddp.py --world-size 2 --epochs 5")
