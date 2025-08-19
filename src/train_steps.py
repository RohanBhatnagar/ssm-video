import modal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
from typing import List, Optional

app = modal.App("train_steps_vqvae")

image = (
    modal.Image
        .from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
        .pip_install([
            "torch==2.7.1",
            "torchvision==0.22.1",
            "tqdm==4.67.1",
            "numpy==2.2.6",
            "h5py==3.11.0",
        ])
        .run_commands("mkdir -p /tmp/data")
        .add_local_file("models/videogpt.py", remote_path="/app/models/videogpt.py")
        .add_local_file(
            "datasets/dataloader.py",
            remote_path="/app/datasets/dataloader.py"
        )
)

volume = modal.Volume.from_name("ssm-video", create_if_missing=True)

def train_single_gpu(
    batch_size: int,
    max_steps: int,
    lr: float,
    weight_decay: float,
    save_every: int,
    data_paths: List[str],
):
    """
    Train for steps, on a single GPU. 
    """
    import sys
    sys.path.append("/app")
    from models.videogpt import VQVAE3D, VQVAEConfig
    from datasets.dataloader import MovingMNIST
    
    os.makedirs("/data/checkpoints", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VQVAE3D(VQVAEConfig()).to(device)
    
    dataset = MovingMNIST(data_paths=data_paths)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps
    )
        
    scaler = torch.amp.GradScaler("cuda")

    print(f"Starting training on single GPU")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}, LR: {lr}")
    print(f"Dataset size: {len(dataset)} sequences")

    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = infinite_dataloader(loader)
    
    total_loss = 0.0
    recon_loss = 0.0 
    commit_loss = 0.0
    log_every = 100
    
    pbar = tqdm(range(max_steps), desc="Training", total=max_steps)
    
    for step in pbar:
        step_start_time = time.time()
        
        x = next(data_iter).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        warmup_steps = 500

        encoder_output, vq_info, decoder_output = model(x)
        
        r_loss = F.mse_loss(x, decoder_output) / 0.06
        vq_loss = vq_info["vq_loss"]
        c_loss = vq_info["commit_loss"]
        perplexity = vq_info["perplexity"]
        
        if step < warmup_steps:
            t_loss = r_loss
        else:
            t_loss = r_loss + vq_loss
        
        scaler.scale(t_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += t_loss.item()
        recon_loss += r_loss.item()
        commit_loss += c_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            "total": f"{t_loss.item():.4f}",
            "recon": f"{r_loss.item():.4f}",
            "commit": f"{c_loss.item():.4f}",
            "perplexity": f"{perplexity:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })
        
        if (step + 1) % log_every == 0:
            avg_total = total_loss / log_every
            avg_recon = recon_loss / log_every
            avg_commit = commit_loss / log_every
            
            print(f"\nStep {step+1}/{max_steps} | "
                  f"Avg Total: {avg_total:.4f} | Avg Recon: {avg_recon:.4f} | Avg Commit: {avg_commit:.4f}")
            
            total_loss = 0.0
            recon_loss = 0.0
            commit_loss = 0.0
        
        if (step + 1) % save_every == 0:
            checkpoint = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "total_loss": t_loss.item(),
            }
            
            checkpoint_path = f"/data/checkpoints/checkpoint_step_{step+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")

    os.makedirs("/data/trained_models", exist_ok=True)
    final_model_path = f"/data/trained_models/final_model_{max_steps}_steps.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining completed! Final model saved to: {final_model_path}")


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    memory=16 * 1024,
    timeout=8 * 3600,
)
def train_vqvae_single(
    batch_size: int,
    max_steps: int,
    lr: float,
    weight_decay: float,
    save_every: int,
    data_paths: List[str],
):
    train_single_gpu(
        batch_size=batch_size,
        max_steps=max_steps,
        lr=lr,
        weight_decay=weight_decay,
        save_every=save_every,
        data_paths=data_paths,
    )


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def get_data_paths():
    import glob
    h5_paths = glob.glob("/data/batch_*.h5")
    return sorted(h5_paths)


@app.local_entrypoint()
def main(
    batch_size: int = 32,
    max_steps: int = 500,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    save_every: int = 100,
    train: bool = False,
):
    """Step-based single GPU training with terminal progress"""
    if train:
        print("Fetching data paths...")
        data_paths = get_data_paths.remote()
        print(f"Found {len(data_paths)} data files.")
        
        print(f"Starting single GPU training for {max_steps} steps...")
        call = train_vqvae_single.spawn(
            batch_size=batch_size,
            max_steps=max_steps,
            lr=lr,
            weight_decay=weight_decay,
            save_every=save_every,
            data_paths=data_paths,
        )
        
        call.get()
        print("Training completed!")