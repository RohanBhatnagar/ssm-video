# Next-frame prediction with VQ-VAE and SSMs

The goal of this project is to generate future frames given a prior. To do this, we first train a Vector quantized variational autoencoder (VQ-VAE), then a state space model (SSM) to predict the next frame (autoregressively) given an ancestral prior. 

## Stage 1 · VQ-VAE

The first stage is a **Vector-Quantized Variational Auto-Encoder** that acts purely on
spatial information (per-frame).  It compresses each RGB frame into a compact grid of
discrete tokens that the temporal model will later use.

```
Input frame  (3 × 64 × 64)
    │
    ▼
Encoder (CNN) ──► Continuous latents ──► Vector-Quantiser ──► Discrete codes ──► Decoder (CNN) ──► Reconstruction
```

### Components

1. **Spatial Encoder (3 D-ResNet-style)**
   • Processes each frame independently.  
   • Downsamples via strided convolutions: `(3, 64, 64) → Cₑ × H′ × W′`.

2. **EMA Vector-Quantiser**
   • Maps each latent vector to the nearest entry in a learnable *codebook* of size K.  
   • Uses exponential-moving-average updates for stable training.  
   • Outputs discrete indices and a *commitment* loss.

3. **Spatial Decoder (ConvTranspose)**
   • Upsamples the quantised feature map back to `(3, 64, 64)`.  
   • Mirror of the encoder with transposed convs + batch-norm + LeakyReLU.

> **Note** The *Temporal Mamba SSM* that models dynamics is trained in **Stage 2** and is
> therefore documented in the *Training Pipeline* section below, not here.

## Model Specs

Architecture is changing. Encoder/decoder are very simple right now, doing downsampling and reconstruction with only 2d convolutions. The current model is about 25M params.

## Dataset

Initial training on Moving MNIST. Essentially, MNIST digits randomly selected and placed on a 64x64 black canvas, and set in motion given `initial_pos` and `velocity`

See more here: https://www.cs.toronto.edu/~nitish/unsupervised_video/

## Training Pipeline (2-Stage)

```
raw frames ──► Stage 1: VQ-VAE ──► discrete tokens ──► Stage 2: SSM prior ──► next-frame tokens ──► VQ-VAE decoder ──► next frame
```

All training is done on 4 A100 GPUs using PyTorch's Distributed Data Parallel (DDP).

### Stage 1 · Vector-Quantised VAE

1. **Goal** – Learn a spatial compressor that maps each 64 × 64 RGB frame to a compact grid of *discrete* latent codes.
2. **Loss** – L1 reconstruction + β·commitment (β = 0.25).
3. **Code** – `src/model.py` implements an encoder → EMA vector-quantiser → decoder.  Training is launched via `src/train_ddp.py` which runs **Distributed Data Parallel** jobs on Modal GPUs.

> ℹ️ Why VQ-VAE?  Discrete latents drastically reduce the sequence length for the temporal model and remove colour/texture ambiguity, letting the second stage focus purely on dynamics.

### Stage 2 · Autoregressive State-Space Prior (Mamba)

1. **Preparation** – Freeze the trained VQ-VAE.  Encode every frame in the dataset to obtain sequences of codebook indices.
2. **Model** – A *Mamba* State-Space Model is trained to predict the next token given a context window.  Because Mamba mixes states in **linear time**, it scales well to long video sequences.
3. **Sampling** – At inference we sample tokens autoregressively (ancestral sampling) and decode them back to RGB frames with the VQ-VAE decoder.

> ℹ️ Compared with Transformers this SSM prior has ϴ(L) memory/computation and excels at very long horizons. While the current prior will not be very long, I hope to achieve similar performance to architectures that use transformers. 

### Modal Training Configuration

```python
train_cfg = dict(
    batch_size = 32,    # per-GPU
    lr         = 2e-4,
    epochs     = 10,
    weight_decay = 1e-4,
    save_every = 5,
)
```

Additional flags (`--train`, `--ptflops`) can be passed to `src/train_ddp.py` for launching distributed runs or measuring FLOPs/params.

## Project Structure

```
ssm-video/
├── src/
│   ├── model.py              # VideoFramePredictor architecture
│   ├── train_ddp.py          # Distributed training on Modal
│   └── datasets/
│       └── modal_dataset_loader.py  # MovingMNIST data loaders
├── requirements.txt          # Python dependencies
└── README.md                # This file
```