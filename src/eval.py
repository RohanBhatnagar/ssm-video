import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import modal

app = modal.App("vqvae-video-evaluation")

image = (
    modal.Image
    .from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .pip_install([
        "torch==2.7.1",
        "torchvision==0.22.1",
        "matplotlib==3.10.3",
        "numpy==2.2.6",
        "pillow==11.3.0",
        "mamba-ssm",
    ])
    .run_commands("mkdir -p /app/datasets")
    .add_local_file("model.py", remote_path="/app/model.py")
    .add_local_file(
        "datasets/modal_dataset_loader.py",
        remote_path="/app/datasets/modal_dataset_loader.py"
    )
)

volume = modal.Volume.from_name("moving-mnist-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=1800,
)
def evaluate_multiple_sequences(num_sequences: int = 5):
    """Evaluate model on multiple sequences and compute average metrics"""
    import sys
    sys.path.append("/app")
    
    from model import VQVAEVideo
    from datasets.modal_dataset_loader import MovingMNIST
    
    print(f"📊 Evaluating model on {num_sequences} sequences...")
    
    # Load model
    device = torch.device("cuda")
    model = VQVAEVideo().to(device)
    
    checkpoint_path = "/data/checkpoint_epoch_10.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate test sequences
    dataset = MovingMNIST(num_sequences=num_sequences, seq_len=32)
    
    all_mse = []
    all_psnr = []
    
    with torch.no_grad():
        for seq_idx in range(num_sequences):
            print(f"🔮 Evaluating sequence {seq_idx+1}/{num_sequences}")
            
            input_frames, _ = dataset[seq_idx]
            input_sequence = input_frames[:16].unsqueeze(0).to(device)
            gt_sequence = input_frames[16:32].unsqueeze(0).to(device)
            
            # Generate frames autoregressively
            generated_frames = []
            current_input = input_sequence
            
            for i in range(16):
                model_input = current_input[:, -8:] if current_input.size(1) > 8 else current_input
                reconstructed, _ = model(model_input)
                next_frame = reconstructed[:, -1:]
                generated_frames.append(next_frame)
                current_input = torch.cat([current_input, next_frame], dim=1)
            
            generated_sequence = torch.cat(generated_frames, dim=1)
            
            # Calculate metrics
            mse = torch.mean((generated_sequence - gt_sequence) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            all_mse.append(mse.item())
            all_psnr.append(psnr.item())
    
    # Compute averages
    avg_mse = np.mean(all_mse)
    avg_psnr = np.mean(all_psnr)
    std_mse = np.std(all_mse)
    std_psnr = np.std(all_psnr)
    
    print(f"\n📊 Evaluation Results ({num_sequences} sequences):")
    print(f"  Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    
    return {
        'avg_mse': avg_mse,
        'avg_psnr': avg_psnr,
        'std_mse': std_mse,
        'std_psnr': std_psnr,
        'all_mse': all_mse,
        'all_psnr': all_psnr
    }

@app.local_entrypoint()
def main(
    mode: str = "generate",  # "generate" or "evaluate"
    num_sequences: int = 5
):
    """
    Run video generation evaluation
    
    Args:
        mode: "generate" for single sequence with visualization, "evaluate" for multiple sequences
        num_sequences: Number of sequences to evaluate (only for evaluate mode)
    """
    if mode == "generate":
        print("🎬 Running single sequence generation with visualization...")
        result = generate_video_sequences.remote()
        print(f"✅ Generation completed: {result}")
        
    elif mode == "evaluate":
        print(f"📊 Running evaluation on {num_sequences} sequences...")
        result = evaluate_multiple_sequences.remote(num_sequences)
        print(f"✅ Evaluation completed: {result}")
        
    else:
        print("❌ Invalid mode. Use 'generate' or 'evaluate'")

if __name__ == "__main__":
    print("🎯 VQVAE Video Evaluation")
    print("Usage:")
    print("  modal run src/eval.py --mode generate")
    print("  modal run src/eval.py --mode evaluate --num-sequences 10")