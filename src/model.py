import torch.nn as nn 
import torch.nn.functional as F
from mamba_ssm import Mamba
from ptflops import get_model_complexity_info

class VideoFramePredictor(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 base_channels=64, 
                 encoder_blocks=[2, 2, 2, 2], 
                 latent_dim=16384,
                 ssm_hidden_dim=1024,
                 ssm_depth=4,
                 out_channels=3):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, 
                               base_channels=base_channels, 
                               num_blocks=encoder_blocks)
        
        self.temporal = TemporalMamba(input_dim=latent_dim, 
                                      hidden_dim=ssm_hidden_dim, 
                                      depth=ssm_depth)
        
        
        self.decoder = Decoder(latent_dim=latent_dim, 
                               output_channels=out_channels)

    def forward(self, frames):  # [B, T, 3, 64, 64]
        x = self.encoder(frames)         # [B, T, 16384]
        x = self.temporal(x)             # [B, 16384]  ← latent of next frame
        out = self.decoder(x)            # [B, 3, 64, 64]
        return out

# resnet block for encoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        if downsample or in_channels != out_channels: # no residual connection if downsample or in_channels != out_channels
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

# upres block for decoder 
class UpResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.res = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch)
        )

    def forward(self, x):
        r = self.res(x)
        x = self.upsample(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + r)
    
# resnet style encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=[2,2,2,2]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        layers = []
        channels = base_channels
        total_stages = len(num_blocks)
        for i, n_blocks in enumerate(num_blocks):
            if i == 0:
                stride = 1
            elif i == total_stages - 1:
                stride = 1        
            else:
                stride = 2

            blocks = [ResidualBlock(channels, channels*2, stride=stride, downsample=True)]
            channels *= 2
            for _ in range(n_blocks - 1):
                blocks.append(ResidualBlock(channels, channels))
            layers.append(nn.Sequential(*blocks))

        self.backbone = nn.Sequential(*layers)
        self.final_dim = channels * 4 * 4

    def forward(self, frames):  # [B, T, 3, 64, 64]
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.stem(x)        # [B*T, base_channels, 16, 16]
        x = self.backbone(x)    # [B*T, channels, 4, 4]
        x = x.view(B, T, -1)    # [B, T, final_dim]
        return x
    

class MambaBlock(nn.Module):
    def __init__(self, d_model, dt_rank=1):
        super().__init__()
        self.d_model = d_model
        self.dt_rank = dt_rank

        # Input projections
        self.in_proj = nn.Linear(d_model, 2 * d_model)  # u, Δ

        # State-space params
        self.A = nn.Parameter(torch.randn(d_model))  # state decay
        self.B = nn.Parameter(torch.randn(d_model))  # input influence

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.size()

        u, delta = self.in_proj(x).chunk(2, dim=-1)  # [B, T, D] each
        delta = torch.sigmoid(delta)  # gating input

        h = torch.zeros(B, D, device=x.device)  # initial state
        y = []

        for t in range(T):
            h = torch.exp(-self.A * delta[:, t]) * h + self.B * u[:, t]
            y.append(h)

        y = torch.stack(y, dim=1)  # [B, T, D]
        return self.out_proj(y)
    
class TemporalMamba(nn.Module):
    def __init__(self, input_dim=16384, hidden_dim=512, depth=4):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba_layers = nn.Sequential(*[
            Mamba(
                d_model=hidden_dim, 
                d_state=16, 
                d_conv=4, 
                expand=2
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):  # x: [B, T, input_dim]
        x = self.in_proj(x)        # [B, T, hidden_dim]
        x = self.mamba_layers(x)   # [B, T, hidden_dim]
        x = self.norm(x[:, -1])    # take last time step
        return self.out_proj(x)    # [B, input_dim]
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=16384, output_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.reshape_channels = 1024
        self.reshape_size = 4  # Because 1024 × 4 × 4 = 16384

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 4→8
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 8→16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 16→32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.Tanh()  # or Sigmoid, depending on frame normalization
        )

    def forward(self, x):  # x: [B, 16384]
        x = x.view(-1, self.reshape_channels, self.reshape_size, self.reshape_size)  # [B, 1024, 4, 4]
        return self.up(x)  # [B, 3, 64, 64]
    
