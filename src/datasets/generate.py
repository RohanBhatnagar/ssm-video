import modal
import torch
import numpy as np
from typing import Tuple

"""
This script generates a bunch of files containing batches for moving mnist. these files are used to make a dataset in dataloader.py for VQVAE training.
"""

app = modal.App("moving-mnist-generator")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install([
        "torch==2.7.1",
        "torchvision==0.22.1",
        "numpy==2.2.6",
        "h5py==3.11.0",
    ])
)

volume = modal.Volume.from_name("ssm-video", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600, 
)
def generate_batch(batch_start: int, batch_size: int, seq_len: int = 16, 
                   mnist_digits: int = 2, image_size: int = 64) -> str:
    """Generate a batch of MovingMNIST sequences for VQVAE training"""
    import random
    import torch
    import torch.nn.functional as F
    from torchvision.datasets import MNIST
    from torchvision import transforms
    import h5py
    
    mnist = MNIST(root="/tmp/mnist", train=True, download=True, transform=transforms.ToTensor())
    
    digit_cache = []
    for i in range(len(mnist)):
        digit_img, _ = mnist[i]
        digit_img = transforms.Resize(28)(digit_img).squeeze() 
        digit_cache.append(digit_img)
    
    all_frames = []
    
    for seq_idx in range(batch_size):
        canvas = torch.zeros((seq_len, 1, image_size, image_size))
        
        for _ in range(mnist_digits):
            # Random digit
            digit_idx = random.randint(0, len(digit_cache)-1)
            digit_img = digit_cache[digit_idx]
            
            # Random starting position and velocity
            pos = [random.randint(0, image_size - 28) for _ in range(2)]
            vel = [random.choice([-1, 1]) * random.randint(1, 3) for _ in range(2)]
            
            for t in range(seq_len):
                top, left = pos
                canvas[t, 0, top:top+28, left:left+28] += digit_img
                
                # Update position with bouncing
                for i in range(2):
                    pos[i] += vel[i]
                    if pos[i] <= 0 or pos[i] + 28 >= image_size:
                        vel[i] = -vel[i]
                        pos[i] += vel[i]
        
        canvas = torch.clamp(canvas, 0, 1)
        canvas = canvas.repeat(1, 3, 1, 1)  # [T, 3, H, W]
        all_frames.append(canvas)
    
    frames_tensor = torch.stack(all_frames)  # [batch_size, T, 3, H, W]
    
    frames_uint8 = (frames_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    batch_filename = f"/data/batch_{batch_start:06d}_{batch_start + batch_size - 1:06d}.h5"
    with h5py.File(batch_filename, "w") as f:
        f.create_dataset(
            "frames",
            data=frames_uint8,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )
        f.attrs["batch_start"] = batch_start
        f.attrs["batch_size"] = batch_size
    
    return f"Generated batch {batch_start}-{batch_start + batch_size - 1} -> {batch_filename}"

@app.local_entrypoint()
def main(
    total_sequences: int = 25000,
    batch_size: int = 1000,
    seq_len: int = 16,
    mnist_digits: int = 2,
    image_size: int = 64,
    max_workers: int = 5
):
    """Generate MovingMNIST dataset using Modal workers for VQVAE training"""
    
    print(f"Generating {total_sequences} sequences with {max_workers} workers...")
    print(f"Each sequence: {seq_len} frames, {mnist_digits} digits, {image_size}x{image_size} resolution")
    
    num_batches = (total_sequences + batch_size - 1) // batch_size
    
    batch_jobs = []
    for i in range(num_batches):
        batch_start = i * batch_size
        current_batch_size = min(batch_size, total_sequences - batch_start)
        
        job = generate_batch.spawn(
            batch_start=batch_start,
            batch_size=current_batch_size,
            seq_len=seq_len,
            mnist_digits=mnist_digits,
            image_size=image_size
        )
        batch_jobs.append(job)
        
        # Limit concurrent workers
        if len(batch_jobs) >= max_workers:
            # Wait for first batch to complete
            result = batch_jobs.pop(0).get()
            print(result)
    
    # Wait for remaining jobs
    for job in batch_jobs:
        result = job.get()
        print(result)
    
    print("All batches generated!")