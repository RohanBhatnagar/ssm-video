import torch.nn as nn 
import torch 
import torch.nn.functional as F
from mamba_ssm import Mamba

class VQVAEVideo(nn.Module):
    def __init__(self,
                 in_ch=3, 
                 base_ch=64,
                 latent_dim=512, 
                 hidden_dim=256, 
                 ssm_depth=4,
                 K=512, 
                 emb_dim=256,
                 num_blocks=4,
                 img_size=64
    ):
        super().__init__()
        self.enc = SpatialEncoder(in_ch, base_ch, num_blocks)
        
        # after enc, feature-map shape = (B, C_e, H', W')
        self.C_e = base_ch*2**(num_blocks-1)
        self.H_p = self.W_p = img_size//(2**(num_blocks))
        self.emb_dim = emb_dim
        
        self.to_latent = nn.Linear(self.C_e*self.H_p*self.W_p, latent_dim)
        self.temporal  = TemporalSSM(latent_dim, hidden_dim, ssm_depth)
        self.from_latent = nn.Linear(latent_dim, self.C_e*self.H_p*self.W_p)
        self.quant_conv = nn.Conv2d(self.C_e, emb_dim, 1)
        self.vq = VectorQuantizerEMA(K, emb_dim)
        self.dec = SpatialDecoder(
            in_ch = emb_dim,
            base_ch = base_ch,
            num_blocks= num_blocks,
            out_ch = in_ch
        )

    def forward(self, x):  
        B, T, C, H, W = x.shape
        # sptial temporal encoding 
        x_ = x.view(B*T, C, H, W)
        h = self.enc(x_) # (B*T, C_e, H', W') we should collapse b, t to process images independelty in apralle 
        h = h.view(B, T, -1) # -1 indicates C_e*H'*W'. add temporal dim back in;            
        # pass into ssm
        z = self.to_latent(h) # prepare latent vec for ssm (B, T, latent_dim)  
        z = self.temporal(z) # pass into ssm 
        # back to spatial feature map 
        h2 = self.from_latent(z) #(B, T, C_e*H'*W')
        h2 = h2.view(B*T, self.C_e, self.H_p, self.W_p) # (B*T, C_e, H', W')
        # quantize, match to codebook 
        e = self.quant_conv(h2) # conv for quantization
        N, D = e.shape[0]*e.shape[2]*e.shape[3], e.shape[1] # N = B*T*H'*W', D = emb_dim
        e_flat = e.permute(0,2,3,1).reshape(N, D) # flatten e 
        e_q, loss_c = self.vq(e_flat) # (N, emb_dim)
        e_q = e_q.view(B*T, self.emb_dim, self.H_p, self.W_p) # (B*T, emb_dim, H', W')
        # decode 
        x_rec = self.dec(e_q) # (B*T,3,H,W)
        x_rec = x_rec.view(B, T, 3, H, W)
        return x_rec, loss_c


class SpatialEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_blocks=4):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_blocks):
            layers += [
                nn.Conv2d(ch, base_ch*(2**i), 4, 2, 1),
                nn.BatchNorm2d(base_ch*(2**i)),
                nn.LeakyReLU(0.2, inplace=True) # avoid dying grad 
            ]
            ch = base_ch*(2**i)
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B,3,H,W)
        return self.net(x)

class TemporalSSM(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_state: int, 
                 depth: int = 4,
                 d_conv: int = 4,
                 expand: int = 2):
        """
        d_model: latent dim
        d_state: hidden dim
        depth: num blocks  
        d_conv: conv kernel size
        expand: expansion factor
        """
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                Mamba(
                    d_model = d_model,
                    d_state = d_state,
                    d_conv  = d_conv,
                    expand  = expand,
                )
            )
        self.ssm = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.ssm(x)

class VectorQuantizerEMA(nn.Module):
    """
    K = codebook size 
    D = embedding dim 
    """
    def __init__(self, K, D, beta=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.K, self.D = K, D
        self.embedding = nn.Embedding(K, D) # randomly init codebook. 
        nn.init.uniform_(self.embedding.weight, -1/K, 1/K) # uniform categorical dist for trianing.  
        self.register_buffer('ema_count', torch.zeros(K)) # save persistent buffers with exponential moving avg. 
        self.register_buffer('ema_avg', self.embedding.weight.data.clone())
        self.beta, self.decay, self.epsilon = beta, decay, epsilon

    def forward(self, z_e):  
        # input: posterior, (N, D) flattened (B*T*H'*W', D)
        
        # ompute nearest code indices
        dist = (z_e**2).sum(-1, keepdim=True) \
             - 2*z_e @ self.embedding.weight.t() \
             + (self.embedding.weight**2).sum(-1)
        idx = dist.argmin(-1)  # (N,)
        z_q = self.embedding(idx)  # (N, D)
        
        # ema update 
        if self.training:
            one_hot = F.one_hot(idx, self.K).type(z_e.dtype)  # (N,K)
            count = one_hot.sum(0)  # (K,)
            avg = one_hot.t() @ z_e  # (K,D)
            self.ema_count.mul_(self.decay).add_(count, alpha=1-self.decay)
            self.ema_avg.  mul_(self.decay).add_(avg, alpha=1-self.decay)
            n = self.ema_count.sum()
            # laplace smoothing
            cluster_size = (self.ema_count + self.epsilon) / (n + self.K*self.epsilon) * n
            # normalize
            embed = self.ema_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed)
        # losses 
        commitment = F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()
        return z_q, commitment

class SpatialDecoder(nn.Module):
    def __init__(self,
                 in_ch: int,         # now the embedding dim you actually have
                 base_ch: int,
                 num_blocks: int,
                 out_ch: int = 3):   # e.g. RGB
        super().__init__()
        layers = []
        ch = in_ch
        # upsample num_blocks times, halving the multiplier each time
        for i in reversed(range(num_blocks)):
            layers += [
                nn.ConvTranspose2d(ch, base_ch*(2**i), 4, 2, 1),
                nn.BatchNorm2d(base_ch*(2**i)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = base_ch*(2**i)
        layers += [
            nn.Conv2d(ch, out_ch, 3, 1, 1),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B·T, in_ch, H′, W′)
        return self.net(x)


# class VideoFramePredictor(nn.Module):
#     def __init__(self, 
#                  in_channels=3, 
#                  base_channels=64, 
#                  encoder_blocks=[2, 2, 2, 2], 
#                  latent_dim=16384,
#                  ssm_hidden_dim=1024,
#                  ssm_depth=4,
#                  out_channels=3):
#         super().__init__()

#         self.encoder = Encoder(in_channels=in_channels, 
#                                base_channels=base_channels, 
#                                num_blocks=encoder_blocks)
        
#         self.temporal = TemporalMamba(input_dim=latent_dim, 
#                                       hidden_dim=ssm_hidden_dim, 
#                                       depth=ssm_depth)
        
        
#         self.decoder = Decoder(latent_dim=latent_dim, 
#                                output_channels=out_channels)

#     def forward(self, frames): # [B, T, 3, 64, 64]
#         x = self.encoder(frames) # [B, T, 16384]
#         x = self.temporal(x) # [B, 16384] 
#         out = self.decoder(x) # [B, 3, 64, 64]
#         return out

# # resnet block for encoder
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=False):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.downsample = downsample
#         if downsample or in_channels != out_channels: # no residual connection if downsample or in_channels != out_channels
#             self.residual = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.residual = nn.Identity()

#     def forward(self, x):
#         identity = self.residual(x)
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity
#         out = F.relu(out)
#         return out
    
# # resnet style encoder
# class Encoder(nn.Module):
#     def __init__(self, in_channels=3, base_channels=64, num_blocks=[2,2,2,2]):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(3, stride=2, padding=1),
#         )
        
#         layers = []
#         channels = base_channels
#         total_stages = len(num_blocks)
#         for i, n_blocks in enumerate(num_blocks):
#             if i == 0:
#                 stride = 1
#             elif i == total_stages - 1:
#                 stride = 1        
#             else:
#                 stride = 2

#             blocks = [ResidualBlock(channels, channels*2, stride=stride, downsample=True)]
#             channels *= 2
#             for _ in range(n_blocks - 1):
#                 blocks.append(ResidualBlock(channels, channels))
#             layers.append(nn.Sequential(*blocks))

#         self.backbone = nn.Sequential(*layers)
#         self.final_dim = channels * 4 * 4

#     def forward(self, frames):  # [B, T, 3, 64, 64]
#         B, T, C, H, W = frames.shape
#         x = frames.view(B * T, C, H, W)
#         x = self.stem(x) # [B*T, base_channels, 16, 16]
#         x = self.backbone(x) # [B*T, channels, 4, 4]
#         x = x.view(B, T, -1) # [B, T, final_dim]
#         return x
    
# # https://github.com/state-spaces/mamba
# class TemporalMamba(nn.Module):
#     def __init__(self, input_dim=16384, hidden_dim=512, depth=4):
#         super().__init__()
#         self.in_proj = nn.Linear(input_dim, hidden_dim)
#         self.mamba_layers = nn.Sequential(*[
#             Mamba(
#                 d_model=hidden_dim, 
#                 d_state=16, 
#                 d_conv=4, 
#                 expand=2
#             )
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.out_proj = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x): # x: [B, T, input_dim]
#         x = self.in_proj(x) # [B, T, hidden_dim]
#         x = self.mamba_layers(x)# [B, T, hidden_dim]
#         x = self.norm(x[:, -1]) # take last time step
#         return self.out_proj(x) # [B, input_dim]
    
    
# class Decoder(nn.Module):
#     def __init__(self, latent_dim=16384, output_channels=3):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.reshape_channels = 1024
#         self.reshape_size = 4  # Because 1024 × 4 × 4 = 16384

#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 4→8
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 8→16
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 16→32
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),  # 32→64
#             nn.Tanh()  # or Sigmoid, depending on frame normalization
#         )

#     def forward(self, x):  # x: [B, 16384]
#         x = x.view(-1, self.reshape_channels, self.reshape_size, self.reshape_size)  # [B, 1024, 4, 4]
#         return self.up(x)  # [B, 3, 64, 64]
    

