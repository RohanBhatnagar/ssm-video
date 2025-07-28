import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import transforms
from tqdm import tqdm
from model import VideoFramePredictor
from dataset import MovingMNIST

def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = VideoFramePredictor().to(rank)
    model = DDP(model, device_ids=[rank])

    # Choose dataset type
    if args['use_precomputed']:
        try:
            from precompute_dataset import PrecomputedMovingMNIST
            dataset = PrecomputedMovingMNIST()
            if rank == 0:
                print("Using precomputed dataset")
        except:
            if rank == 0:
                print("Precomputed dataset not found, falling back to on-the-fly generation")
            dataset = MovingMNIST()
    else:
        dataset = MovingMNIST()
        if rank == 0:
            print("Using on-the-fly dataset generation")
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args['lr'])
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.L1Loss()

    for epoch in range(args['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch}")

        for frames, target in pbar:
            frames = frames.to(rank)
            target = target.to(rank)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(frames)
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                pbar.set_postfix(loss=loss.item())

    dist.destroy_process_group()

# ---- Entry point ----
def main():
    world_size = torch.cuda.device_count()
    args = {
        "batch_size": 32,
        "lr": 2e-4,
        "epochs": 10,
        "use_precomputed": False,  # Set to True for faster training
    }
    mp.spawn(train, args=(world_size, args), nprocs=world_size)

if __name__ == "__main__":
    main()
