"""
Optical Flow Computation Utility
Generates optical flow maps using pre-trained RAFT model for 4D-GS supervision.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import sys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm


def load_image(image_path: str, device: str = "cuda") -> torch.Tensor:
    """
    Load image and convert to tensor format [3, H, W] normalized to [0, 1].
    
    Args:
        image_path: Path to image file
        device: Device to move tensor to
    
    Returns:
        Image tensor [3, H, W] in range [0, 1]
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    return img_tensor.to(device)


def load_raft_model(model_path: str = "models/raft-small.pth", device: str = "cuda"):
    """
    Load pre-trained RAFT model.
    
    Args:
        model_path: Path to RAFT model weights
        device: Device to load model to
    
    Returns:
        RAFT model in eval mode
    
    Note:
        If model is not found locally, this will download the torchvision version.
    """
    try:
        # Try to import RAFT from torchvision
        from torchvision.models.optical_flow import raft_large, raft_small
        model = raft_small(pretrained=True, progress=True)
        print("Loaded RAFT-small from torchvision")
    except ImportError:
        print("torchvision optical flow not available, attempting local load...")
        # Fallback: manual RAFT implementation expected
        raise ImportError("Please install a recent version of torchvision with optical flow support")
    
    model = model.to(device)
    model.eval()
    return model


def compute_flow(model, frame1: torch.Tensor, frame2: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Compute optical flow from frame1 to frame2 using RAFT.
    
    Args:
        model: RAFT model
        frame1: Image tensor [3, H, W] at time t
        frame2: Image tensor [3, H, W] at time t+1
        device: Device to compute on
    
    Returns:
        Optical flow tensor [2, H, W] where flow[0] = x-displacement, flow[1] = y-displacement
    """
    # Ensure tensors are on the correct device and properly shaped
    frame1 = frame1.unsqueeze(0).to(device)  # [1, 3, H, W]
    frame2 = frame2.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    with torch.no_grad():
        # RAFT returns a list of flow predictions; we use the last one
        flow_list = model(frame1, frame2)
        flow = flow_list[-1]  # [1, 2, H, W]
    
    return flow.squeeze(0)  # [2, H, W]


def flow_to_rgb(flow: torch.Tensor, scale: float = 50.0) -> np.ndarray:
    """
    Convert optical flow to RGB image for visualization.
    
    Args:
        flow: Optical flow tensor [2, H, W]
        scale: Scaling factor for visualization (higher = brighter colors)
    
    Returns:
        RGB image [H, W, 3] in range [0, 1]
    """
    flow_np = flow.detach().cpu().numpy()  # [2, H, W]
    fx, fy = flow_np[0], flow_np[1]
    
    # Compute magnitude and angle
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)
    
    # Normalize magnitude
    mag_normalized = mag / (mag.max() + 1e-6)
    
    # Create HSV image
    h = (ang + np.pi) / (2 * np.pi)  # Hue: direction
    s = np.ones_like(h)  # Saturation: full
    v = mag_normalized  # Value: magnitude
    
    # HSV to RGB conversion
    from matplotlib.colors import hsv_to_rgb
    hsv = np.stack([h, s, v], axis=-1)
    rgb = hsv_to_rgb(hsv)
    
    return rgb


def generate_optical_flow_dataset(
    image_dir: str,
    output_dir: str,
    model_checkpoint: Optional[str] = None,
    device: str = "cuda",
    save_visualization: bool = True,
    max_frames: Optional[int] = None
) -> None:
    """
    Generate optical flow maps for all consecutive frame pairs in a directory.
    
    Args:
        image_dir: Directory containing sorted training images
        output_dir: Directory to save .npy flow files
        model_checkpoint: Optional path to RAFT model checkpoint
        device: Device to compute on
        save_visualization: Whether to save RGB visualizations of flow
        max_frames: Maximum number of frames to process (for testing)
    
    Process:
        - Scans image_dir for image files
        - Sorts by filename (assumes temporal ordering in filename)
        - Computes forward flow between consecutive frames
        - Saves flows as .npy files maintaining directory structure
        - Optionally saves visualizations
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    viz_path = output_path / "visualization"
    if save_visualization:
        viz_path.mkdir(exist_ok=True)
    
    # Load model
    print("Loading RAFT model...")
    model = load_raft_model(model_checkpoint, device)
    
    # Find and sort images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    if max_frames:
        image_files = image_files[:max_frames]
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) < 2:
        print("Need at least 2 images to compute optical flow")
        return
    
    # Process consecutive frame pairs
    for i in range(len(image_files) - 1):
        frame_idx = i
        
        # Load frames
        frame1_path = os.path.join(image_dir, image_files[i])
        frame2_path = os.path.join(image_dir, image_files[i + 1])
        
        print(f"Processing frames {i} -> {i+1}: {image_files[i]} -> {image_files[i+1]}")
        
        try:
            frame1 = load_image(frame1_path, device)
            frame2 = load_image(frame2_path, device)
            
            # Compute flow
            flow = compute_flow(model, frame1, frame2, device)
            
            # Save flow as .npy
            flow_save_path = output_path / f"flow_{frame_idx:06d}.npy"
            np.save(str(flow_save_path), flow.cpu().numpy())
            print(f"  Saved flow to {flow_save_path}")
            
            # Save visualization
            if save_visualization:
                rgb = flow_to_rgb(flow)
                viz_save_path = viz_path / f"flow_{frame_idx:06d}.png"
                Image.fromarray((rgb * 255).astype(np.uint8)).save(str(viz_save_path))
                print(f"  Saved visualization to {viz_save_path}")
            
            # Free memory
            del frame1, frame2, flow
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"  Error processing frames {i} -> {i+1}: {e}")
            continue
    
    # Handle last frame (no flow to t+1)
    last_flow_path = output_path / f"flow_{len(image_files)-1:06d}.npy"
    last_flow = np.zeros_like(np.load(str(output_path / f"flow_{len(image_files)-2:06d}.npy")))
    np.save(str(last_flow_path), last_flow)
    print(f"Saved zero flow for last frame to {last_flow_path}")
    
    print(f"\nOptical flow generation complete! Files saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate optical flow maps using RAFT")
    parser.add_argument("image_dir", help="Directory containing training images (sorted by timestamp)")
    parser.add_argument("--output_dir", default="flow", help="Output directory for flow maps")
    parser.add_argument("--model_checkpoint", default=None, help="Path to RAFT model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--no_viz", action="store_true", help="Skip saving visualizations")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    generate_optical_flow_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        save_visualization=not args.no_viz,
        max_frames=args.max_frames
    )
