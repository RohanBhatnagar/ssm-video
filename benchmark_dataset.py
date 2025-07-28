import time
import torch
from torch.utils.data import DataLoader
from src.dataset import MovingMNIST

def benchmark_dataset():
    dataset = MovingMNIST(seq_len=8, num_sequences=1000)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    
    # Warm up
    for i, (frames, target) in enumerate(dataloader):
        if i >= 10:
            break
    
    # Benchmark
    start_time = time.time()
    total_samples = 0
    
    for i, (frames, target) in enumerate(dataloader):
        total_samples += frames.shape[0]
        if i >= 100:  # Test 100 batches
            break
    
    end_time = time.time()
    
    samples_per_second = total_samples / (end_time - start_time)
    print(f"Dataset throughput: {samples_per_second:.1f} samples/second")
    print(f"Time per batch (32 samples): {(end_time - start_time) / (i+1):.3f} seconds")

if __name__ == "__main__":
    benchmark_dataset() 