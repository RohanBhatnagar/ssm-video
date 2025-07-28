import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import random

class MovingMNIST(Dataset):
    def __init__(self, mnist_digits=2, seq_len=20, image_size=64, num_sequences=10000, train=True):
        self.mnist = MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())
        self.mnist_digits = mnist_digits
        self.seq_len = seq_len
        self.image_size = image_size
        self.num_sequences = num_sequences

    def __len__(self):
        return self.num_sequences

    # randomly select 2 digits from the mnist dataset and simulate their movement
    def __getitem__(self, idx):
        canvas = torch.zeros((self.seq_len, 1, self.image_size, self.image_size))
        for _ in range(self.mnist_digits):
            digit_idx = random.randint(0, len(self.mnist)-1)
            digit_img, _ = self.mnist[digit_idx]
            digit_img = transforms.Resize(28)(digit_img)
            pos = [random.randint(0, self.image_size - 28) for _ in range(2)]
            vel = [random.choice([-1, 1]) * random.randint(1, 3) for _ in range(2)]
            for t in range(self.seq_len):
                top, left = pos
                canvas[t, 0, top:top+28, left:left+28] += digit_img.squeeze()
                # bounce
                for i in range(2):
                    pos[i] += vel[i]
                    if pos[i] <= 0 or pos[i] + 28 >= self.image_size:
                        vel[i] = -vel[i]
                        pos[i] += vel[i]
        canvas = torch.clamp(canvas, 0, 1)
        
        # RGB for 3 input channels
        canvas = canvas.repeat(1, 3, 1, 1)  # [T, 3, H, W]
        frames = canvas[:-1]  # [T-1, 3, H, W]
        target = canvas[-1]  # [3, H, W]
        
        return frames, target
