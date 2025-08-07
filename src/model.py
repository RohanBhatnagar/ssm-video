import torch.nn as nn 
import torch 
import torch.nn.functional as F

class VQVAEVideo(nn.Module):
    def __init__(self,
                 in_ch=3, 
                 base_ch=64, 
                 quantization_dim=512, 
                 K=512, 
                 num_blocks=4,
                 img_size=64
    ):
        super().__init__()
        self.H_p = self.W_p = img_size//(2**(num_blocks))
        self.quantization_dim = quantization_dim
        self.C_e = base_ch * (2**(num_blocks - 1)) if num_blocks > 0 else base_ch 

        self.enc = SpatialEncoder(in_ch, base_ch, num_blocks)
        
        self.vq = VectorQuantizerEMA(K, quantization_dim)
        
        self.dec = SpatialDecoder(
            in_ch = quantization_dim,
            base_ch = base_ch,
            num_blocks= num_blocks,
            out_ch = in_ch
        )

    def forward(self, x):  
        B, T, C, H, W = x.shape 
        x_ = x.view(B*T, C, H, W)
        latent = self.enc(x_)
        flattened_latent = latent.permute(0, 2, 3, 1).contiguous().view(-1, self.C_e)
        quantized_latent, commitment_loss, codebook_indices = self.vq(flattened_latent)
        quantized_latent_reshaped = quantized_latent.view(B*T, self.H_p, self.W_p, self.C_e).permute(0, 3, 1, 2).contiguous()
        reconstruction_ = self.dec(quantized_latent_reshaped)
        reconstruction = reconstruction_.view(B, T, C, H, W)
        recon_loss = F.l1_loss(reconstruction, x)
        total_loss = recon_loss + commitment_loss
        return reconstruction, recon_loss, commitment_loss, total_loss

class SpatialEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_blocks=4):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_blocks):
            layers += [
                nn.Conv2d(ch, base_ch*(2**i), 4, 2, 1),
                nn.BatchNorm2d(base_ch*(2**i)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            ch = base_ch*(2**i)
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B,3,H,W)
        return self.net(x)

class VectorQuantizerEMA(nn.Module):
    """
    K = codebook size 
    D = embedding dim 
    """
    def __init__(self, K, D, beta=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.K, self.D = K, D
        self.embedding = nn.Embedding(K, D) # initialize codebook 
        nn.init.uniform_(self.embedding.weight, -1/K, 1/K) # uniform categorical dist for trianing.  
        self.register_buffer('ema_count', torch.zeros(K)) # save persistent buffers with ema. 
        self.register_buffer('ema_avg', self.embedding.weight.data.clone())
        self.beta, self.decay, self.epsilon = beta, decay, epsilon

    def forward(self, z_e):  
        # input: posterior (B*T*H'*W', D)
        dist = (z_e**2).sum(-1, keepdim=True) \
             - 2*z_e @ self.embedding.weight.t() \
             + (self.embedding.weight**2).sum(-1)
        idx = dist.argmin(-1)  # (N,)
        z_q = self.embedding(idx)  # (N, D)
        
        if self.training:
            one_hot = F.one_hot(idx, self.K).type(z_e.dtype)  # (N,K)
            count = one_hot.sum(0)  # (K,)
            avg = one_hot.t() @ z_e  # (K,D)
            self.ema_count.mul_(self.decay).add_(count, alpha=1-self.decay)
            self.ema_avg.  mul_(self.decay).add_(avg, alpha=1-self.decay)
            n = self.ema_count.sum()
            cluster_size = (self.ema_count + self.epsilon) / (n + self.K*self.epsilon) * n
            embed = self.ema_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed)  
            
        commitment = F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach() # straight estimate of codebook 
        return z_q, self.beta*commitment, idx

class SpatialDecoder(nn.Module):
    def __init__(self,
                 in_ch: int,         
                 base_ch: int,
                 num_blocks: int,
                 out_ch: int): 
        super().__init__()
        layers = []
        ch = in_ch
        
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
        return self.net(x)

# class TemporalSSM(nn.Module):
#     from mamba_ssm import Mamba

#     def __init__(self, 
#                  d_model: int, 
#                  d_state: int, 
#                  depth: int = 4,
#                  d_conv: int = 4,
#                  expand: int = 2):
#         """
#         d_model: latent dim
#         d_state: hidden dim
#         depth: num blocks  
#         d_conv: conv kernel size
#         expand: expansion factor
#         """
#         super().__init__()
#         layers = []
#         for _ in range(depth):
#             layers.append(
#                 Mamba(
#                     d_model = d_model,
#                     d_state = d_state,
#                     d_conv  = d_conv,
#                     expand  = expand,
#                 )
#             )
#         self.ssm = nn.Sequential(*layers)

#     def forward(self, x):
#         # x: (B, T, d_model)
#         return self.ssm(x)