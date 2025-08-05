# Video prediction with state-space-models

Next-2-frame prediction model using **State Space Models (Mamba)** for efficient temporal modeling with **linear complexity** in sequence length.


## 🏗️ Architecture

The model follows an **Encoder-Temporal-Decoder** architecture:

```
Input: [B, T, 3, 64, 64] → Output: [B, num_next_frames, 3, 64, 64]
```

### Components:

1. **Encoder (ResNet-style)**
   - Processes each frame independently
   - Extracts spatial features: `(T, 3, 64, 64) → (T, 16384)`
   - ResNet blocks with downsampling

2. **TemporalMamba (SSM)**
   - Models temporal dependencies across context window
   - Selective state space model: `(T, 16384) → (num_next_frames,16384)`
   - 4-layers with `hidden_dim=1024`

3. **Decoder (ConvTranspose)**
   - Reconstructs next frame from latent representation
   - Upsampling with skip connections: `(16384) → (3, 64, 64)`
   - Nearest-neighbor upsampling for sharp details

## Model Specs

Architecture is prone to change but between 200 and 700 M params, and 4.6 GMacs.

## Dataset

Initial training on Moving MNIST. Essentially MNIST digits randomly selected and placed on a 64x64 black canvas, and set in motion given `initial_pos` and `velocity`

https://www.cs.toronto.edu/~nitish/unsupervised_video/


## Training

### Distributed Data Parallel (DDP) on Modal
- **Multi-GPU**: H100/A100 distributed training with DDP
- **Mixed Precision**: FP16 + GradScaler for efficiency  
- **Learning Rate**: Cosine annealing schedule
- **Optimizer**: AdamW with weight decay
- **Loss**: L1 loss for sharp reconstruction

### Training Configuration
```python
training_args = {
    "batch_size": 16,          # Per-GPU batch size
    "lr": 2e-4,               # Learning rate
    "epochs": 10,             # Training epochs
    "weight_decay": 1e-4,     # AdamW weight decay
    "save_every": 5,          # Checkpoint frequency
}
```

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