import modal
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


app = modal.App("video-reconstruct-app")

image = (
    modal.Image
        .from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
        .pip_install([
            "torch==2.7.1",
            "torchvision==0.22.1",
            "numpy==2.2.6",
            "matplotlib==3.10.3",
        ])
        .run_commands("mkdir -p /app /data")
        .add_local_file("models/originals.py", remote_path="/app/models/originals.py")
)


def _generate_moving_mnist_batch(
    batch_size: int,
    seq_len: int,
    mnist_digits: int,
    image_size: int = 64,
) -> torch.Tensor:
    """Generate a batch of MovingMNIST sequences in-memory.

    Returns a float tensor in [0, 1] with shape (B, T, 3, H, W) for RGB models.
    """
    import random
    import torch
    from torchvision.datasets import MNIST
    from torchvision import transforms

    mnist = MNIST(root="/tmp/mnist", train=True, download=True, transform=transforms.ToTensor())

    digit_cache = []
    for i in range(len(mnist)):
        digit_img, _ = mnist[i]
        digit_img = transforms.Resize(28)(digit_img).squeeze()  # [28, 28]
        digit_cache.append(digit_img)

    all_frames = []
    for _ in range(batch_size):
        canvas = torch.zeros((seq_len, 1, image_size, image_size))
        for _ in range(mnist_digits):
            digit_idx = random.randint(0, len(digit_cache) - 1)
            digit_img = digit_cache[digit_idx]

            pos = [random.randint(0, image_size - 28) for _ in range(2)]
            vel = [random.choice([-1, 1]) * random.randint(1, 3) for _ in range(2)]

            for t in range(seq_len):
                top, left = pos
                canvas[t, 0, top:top + 28, left:left + 28] += digit_img

                for k in range(2):
                    pos[k] += vel[k]
                    if pos[k] <= 0 or pos[k] + 28 >= image_size:
                        vel[k] = -vel[k]
                        pos[k] += vel[k]

        canvas = torch.clamp(canvas, 0, 1)
        
        canvas_rgb = canvas.repeat(1, 3, 1, 1)  # [T, 3, H, W]
        all_frames.append(canvas_rgb)

    frames_tensor = torch.stack(all_frames)  # [B, T, 3, H, W]
    return frames_tensor


def _save_side_by_side_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    save_path: str,
) -> None:
    """Save a 2xT grid comparing originals (top row) and reconstructions (bottom row).

    originals, reconstructions: tensors in [0, 1] with shape (T, 3, H, W).
    """
    # Convert RGB to grayscale for visualization: take first channel or convert properly
    if originals.shape[1] == 3:  # (T, 3, H, W)
        originals = originals[:, 0]  # Take red channel for visualization (T, H, W)
    else:
        originals = originals.squeeze(1)  # (T, H, W)
        
    if reconstructions.shape[1] == 3:  # (T, 3, H, W)
        reconstructions = reconstructions[:, 0]  # Take red channel for visualization (T, H, W)
    else:
        reconstructions = reconstructions.squeeze(1)  # (T, H, W)

    originals = originals.detach().cpu().numpy()  # (T, H, W)
    reconstructions = reconstructions.detach().cpu().numpy()  # (T, H, W)

    T = originals.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
    for t in range(T):
        ax_top = axes[0, t] if T > 1 else axes[0]
        ax_bot = axes[1, t] if T > 1 else axes[1]

        # For grayscale images, use the single channel
        img_o = originals[t]  # (H, W)
        img_r = reconstructions[t]  # (H, W)

        ax_top.imshow(img_o, cmap='gray', vmin=0, vmax=1)
        ax_top.axis("off")
        if t == 0:
            ax_top.set_title("Original")

        ax_bot.imshow(img_r, cmap='gray', vmin=0, vmax=1)
        ax_bot.axis("off")
        if t == 0:
            ax_bot.set_title("Reconstruction")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": modal.Volume.from_name("ssm-video", create_if_missing=True)},
    timeout=1800,
)
def reconstruct(
    checkpoint_path: str = "/data/final_model.pt",
    batch_size: int = 4,
    seq_len: int = 16,
    mnist_digits: int = 2,
) -> list:
    import sys
    sys.path.append("/app")
    import torch
    from models.originals import VQVAEVideo, VQVAEConfig

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare data - now returns (B, 1, T, H, W)
    frames = _generate_moving_mnist_batch(batch_size, seq_len, mnist_digits)  # (B, 1, T, 64, 64)
    frames = frames.to(device)

    # Load model
    model = VQVAEVideo(VQVAEConfig()).to(device)
    model.eval()

    # Load weights (support both raw state_dict and full checkpoint dict)
    try:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}. Using randomly initialised weights.")

    with torch.no_grad():
        recon, _, _, _ = model(frames)

    saved_paths = []
    for i in range(batch_size):
        save_path = f"/data/reconstruction_{i}.png"
        _save_side_by_side_grid(frames[i], recon[i], save_path)
        saved_paths.append(save_path)

    print("Saved reconstructions:", saved_paths)
    return saved_paths


@app.local_entrypoint()
def main(
    checkpoint_path: str = "/data/final_model.pt",
    batch_size: int = 1,
    seq_len: int = 16,
    mnist_digits: int = 2,
):
    print("Launching reconstruction job on Modal…")
    result = reconstruct.remote(
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        seq_len=seq_len,
        mnist_digits=mnist_digits,
    )
    print("Reconstruction complete. Files:")
    for p in result:
        print(p)