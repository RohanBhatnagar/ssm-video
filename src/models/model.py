from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

@dataclass
class VQVAEConfig:
    n_hiddens: int = 240
    n_res_layers: int = 2
    n_embeddings: int = 512
    embedding_dim: int = 64
    n_quantizers: int = 1
    n_codes: int = 256
    n_classes: int = 10
    downsample: List[int] = (2, 4, 4)
    upsample: List[int] = (2, 4, 4)
    beta: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5

class VQVAE(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, n_embeddings, n_quantizers, n_codes, n_classes, n_frames, n_channels, n_height, n_width, beta, decay, epsilon, downsample, upsample):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.n_res_layers = n_res_layers
        self.n_embeddings = n_embeddings
        self.n_quantizers = n_quantizers
        
        self.encoder = Encoder(n_channels, n_hiddens, downsample)
        self.decoder = Decoder(n_hiddens, n_res_layers, upsample)
        self.vq = VectorQuantizer(n_embeddings, n_quantizers, n_codes, n_classes, n_frames, beta, decay, epsilon)
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_info, idx = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_info, idx

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, stride=3, kernel_size=3, padding=1)
        
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, stride=3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 2 units of conv, bn, relu 
        return x + self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
    
class Encoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: List[int] = (2, 4, 4)):
        super().__init__()
        self.convs = nn.ModuleList()
        num_downsample = np.array([int(math.log2(d)) for d in downsample])
        max_ds = num_downsample.max()
        for i in range(max_ds):
            in_channels = 1 if i == 0 else out_ch
            stride = tuple([2 if d > 0 else 1 for d in num_downsample])
            conv = nn.Conv3d(in_channels, out_ch, kernel_size=3, stride=stride)
            self.convs.append(conv)
        self.final_conv = nn.Conv3d(out_ch, out_ch, kernel_size=3)
        self.res_stack = nn.Sequential(
            *[ResidualBlock(out_ch, out_ch) for _ in range(n_res_layers)],
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.final_conv(x)
        x = self.res_stack(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample: List[int] = (2, 4, 4)):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_hiddens, n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = nn.Conv3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h
    
    
# currently implements ema. add random restarts if needed. 
class VectorQuantizer(nn.Module):
    def __init__(self, K, D, beta, decay, epsilon): 
        super().__init__()
        self.K, self.D = K, D
        self.beta, self.decay, self.epsilon = beta, decay, epsilon
        self.embedding = nn.Embedding(K, D)
        nn.init.uniform_(self.embedding.weight, -1/K, 1/K)
        self.register_buffer('ema_count', torch.zeros(K))
        self.register_buffer('ema_avg', self.embedding.weight.data.clone())
        
    @torch.no_grad()
    def _update_ema(self, z_e, idx):
        one_hot = F.one_hot(idx, self.K).type(z_e.dtype)
        count = one_hot.sum(0)
        avg = one_hot.t() @ z_e
        self.ema_count.mul_(self.decay).add_(count, alpha=1-self.decay)
        self.ema_avg.mul_(self.decay).add_(avg, alpha=1-self.decay)
        
    def forward(self, z_e):
        dist = (z_e**2).sum(-1, keepdim=True) - 2*z_e @ self.embedding.weight.t() + (self.embedding.weight**2).sum(-1)
        idx = dist.argmin(-1)
        z_q = self.embedding(idx)
        if self.training: # maintain ema in training 
            self._update_ema(z_e, idx)
        commitment = F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()
        vq_info = {
            'quantized_latent': z_q,
            'commitment_loss': self.beta*commitment,
        }
        # quantized latent, info, idx 
        return z_q, vq_info, idx
    
    
def sanity_check(device: str = "cpu"):
    from ptflops import get_model_complexity_info
    model = VQVAE(VQVAEConfig())
    model_macs, model_params = get_model_complexity_info(model, (1, 16, 64, 64), as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"Model FLOPS: {model_macs}")
    print(f"Model Params: {model_params}")
    print(model)
    
if __name__ == "__main__":
    sanity_check()