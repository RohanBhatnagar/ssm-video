import torch
import numpy as np
from tqdm import tqdm
from src.dataset import MovingMNIST
import os

def precompute_dataset(num_sequences=50000, seq_len=8, save_dir="data/precomputed"):
    """Pre-generate and save MovingMNIST sequences"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = MovingMNIST(seq_len=seq_len, num_sequences=num_sequences)
    
    all_frames = []
    all_targets = []
    
    print(f"Generating {num_sequences} sequences...")
    for i in tqdm(range(num_sequences)):
        frames, target = dataset[i]
        all_frames.append(frames)
        all_targets.append(target)
    
    # Stack and save
    frames_tensor = torch.stack(all_frames)  # [N, T-1, 3, H, W]
    targets_tensor = torch.stack(all_targets)  # [N, 3, H, W]
    
    torch.save(frames_tensor, f"{save_dir}/frames.pt")
    torch.save(targets_tensor, f"{save_dir}/targets.pt")
    
    print(f"Saved to {save_dir}/")
    print(f"Frames shape: {frames_tensor.shape}")
    print(f"Targets shape: {targets_tensor.shape}")

class PrecomputedMovingMNIST(torch.utils.data.Dataset):
    """Fast dataset that loads pre-computed sequences"""
    def __init__(self, data_dir="data/precomputed"):
        self.frames = torch.load(f"{data_dir}/frames.pt")
        self.targets = torch.load(f"{data_dir}/targets.pt")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx], self.targets[idx]

if __name__ == "__main__":
    precompute_dataset() 