"""
Coarse-to-Fine Training Strategy with Optical Flow Supervision
Implements resolution progression and loss integration
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy


class CoarseToFineScheduler:
    """
    Manages resolution and loss weight scheduling for coarse-to-fine training.
    """
    
    def __init__(self,
                 warmup_iters: int = 3000,
                 downsample_factor: int = 4,
                 lambda_flow_start: float = 0.1,
                 lambda_flow_end: float = 0.5,
                 lambda_rigidity: float = 0.01):
        """
        Args:
            warmup_iters: Number of iterations for coarse stage
            downsample_factor: Downsampling factor during warmup (e.g., 4x)
            lambda_flow_start: Initial optical flow loss weight
            lambda_flow_end: Final optical flow loss weight
            lambda_rigidity: Rigidity loss weight
        """
        self.warmup_iters = warmup_iters
        self.downsample_factor = downsample_factor
        self.lambda_flow_start = lambda_flow_start
        self.lambda_flow_end = lambda_flow_end
        self.lambda_rigidity = lambda_rigidity
    
    def is_warmup(self, iteration: int) -> bool:
        """Check if we're in warmup (coarse) phase."""
        return iteration < self.warmup_iters
    
    def get_resolution_scale(self, iteration: int) -> float:
        """
        Get resolution scale factor (0 to 1, where 1 = full resolution).
        During warmup: returns 1/downsample_factor
        After warmup: linearly increases from 1/downsample_factor to 1
        """
        if iteration < self.warmup_iters:
            return 1.0 / self.downsample_factor
        else:
            # Smooth transition from warmup to full resolution
            # We use a short transition window (e.g., next 1000 iters)
            transition_iters = 1000
            progress = min(1.0, (iteration - self.warmup_iters) / transition_iters)
            return (1.0 / self.downsample_factor) + progress * (1.0 - 1.0 / self.downsample_factor)
    
    def get_loss_weights(self, iteration: int) -> Dict[str, float]:
        """
        Get loss weights for current iteration.
        
        Returns:
            Dict with keys: 'flow', 'rigidity', 'flow_enabled'
        """
        # Start using flow loss after warmup
        flow_enabled = iteration >= self.warmup_iters
        
        if flow_enabled:
            # Ramp up flow weight over time
            elapsed_since_warmup = iteration - self.warmup_iters
            warmup_flow_iters = 2000
            flow_progress = min(1.0, elapsed_since_warmup / warmup_flow_iters)
            lambda_flow = self.lambda_flow_start + flow_progress * (self.lambda_flow_end - self.lambda_flow_start)
        else:
            lambda_flow = 0.0
        
        return {
            'flow': lambda_flow,
            'rigidity': self.lambda_rigidity,
            'flow_enabled': flow_enabled
        }


def downsample_image(image: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsample image by average pooling.
    
    Args:
        image: Image tensor [C, H, W]
        factor: Downsampling factor (e.g., 4)
    
    Returns:
        Downsampled image [C, H//factor, W//factor]
    """
    if factor == 1:
        return image
    
    # Average pooling
    image = image.unsqueeze(0)  # [1, C, H, W]
    image = F.avg_pool2d(image, kernel_size=factor, stride=factor)
    return image.squeeze(0)  # [C, H//factor, W//factor]


def downsample_flow(flow: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsample optical flow.
    Flow values need to be scaled, not just averaged.
    
    Args:
        flow: Flow tensor [2, H, W]
        factor: Downsampling factor
    
    Returns:
        Downsampled flow [2, H//factor, W//factor]
    """
    if factor == 1:
        return flow
    
    # Downsample and scale flow values
    flow = flow.unsqueeze(0)  # [1, 2, H, W]
    downsampled = F.avg_pool2d(flow, kernel_size=factor, stride=factor)
    downsampled = downsampled / factor  # Scale flow values
    return downsampled.squeeze(0)  # [2, H//factor, W//factor]


def adjust_camera_intrinsics(camera, resolution_scale: float):
    """
    Adjust camera intrinsics for downsampled resolution.
    
    For downsampled images, focal lengths should scale proportionally.
    
    Args:
        camera: Camera object
        resolution_scale: Scale factor (0 to 1)
    
    Returns:
        Modified camera with adjusted intrinsics
    """
    if resolution_scale == 1.0:
        return camera
    
    # Create a copy to avoid modifying original
    cam_copy = copy.deepcopy(camera)
    
    # Scale field of view (focal length scales inversely with image resolution)
    # When resolution is smaller, we need larger field of view to capture same content
    fov_scale = 1.0 / resolution_scale
    cam_copy.FoVx = camera.FoVx * fov_scale
    cam_copy.FoVy = camera.FoVy * fov_scale
    
    # Adjust image dimensions
    cam_copy.image_width = int(camera.image_width * resolution_scale)
    cam_copy.image_height = int(camera.image_height * resolution_scale)
    
    # Recompute projection matrix with new FoV
    from utils.graphics_utils import getProjectionMatrix
    cam_copy.projection_matrix = getProjectionMatrix(
        znear=cam_copy.znear,
        zfar=cam_copy.zfar,
        fovX=cam_copy.FoVx,
        fovY=cam_copy.FoVy
    ).transpose(0, 1)
    
    # Recompute full projection transform
    cam_copy.full_proj_transform = (
        cam_copy.world_view_transform.unsqueeze(0).bmm(
            cam_copy.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)
    
    return cam_copy


def apply_coarse_to_fine_preprocessing(
    viewpoint_cams: list,
    gt_images: torch.Tensor,
    resolution_scale: float,
    scheduler: CoarseToFineScheduler
) -> Tuple[list, torch.Tensor]:
    """
    Apply coarse-to-fine preprocessing to cameras and images.
    
    Args:
        viewpoint_cams: List of camera objects
        gt_images: Ground truth images [B, 3, H, W]
        resolution_scale: Resolution scale factor
        scheduler: Scheduler object
    
    Returns:
        Tuple of (adjusted_cameras, downsampled_images)
    """
    if resolution_scale >= 1.0:
        return viewpoint_cams, gt_images
    
    # Adjust cameras
    adjusted_cams = [adjust_camera_intrinsics(cam, resolution_scale) for cam in viewpoint_cams]
    
    # Downsample images
    downsampled_images = downsample_image(gt_images, int(1.0 / resolution_scale))
    
    return adjusted_cams, downsampled_images


def compute_combined_loss(
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    l1_loss_fn,
    flow_loss_value: Optional[torch.Tensor] = None,
    rigidity_loss_value: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined loss with optional flow and rigidity terms.
    
    Args:
        rendered_image: Rendered image [3, H, W] or [B, 3, H, W]
        gt_image: Ground truth image [3, H, W] or [B, 3, H, W]
        l1_loss_fn: L1 loss function
        flow_loss_value: Pre-computed optical flow loss or None
        rigidity_loss_value: Pre-computed rigidity loss or None
        loss_weights: Dict with keys 'flow', 'rigidity'
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    if loss_weights is None:
        loss_weights = {
            'flow': 0.0,
            'rigidity': 0.0,
            'flow_enabled': False
        }
    
    # Base L1 loss
    l1_loss = l1_loss_fn(rendered_image, gt_image)
    
    # Initialize loss dict
    loss_dict = {'l1': l1_loss.item()}
    
    # Add flow loss if enabled and provided
    total_loss = l1_loss
    if loss_weights.get('flow_enabled', False) and flow_loss_value is not None:
        lambda_flow = loss_weights.get('flow', 0.0)
        total_loss = total_loss + lambda_flow * flow_loss_value
        loss_dict['flow'] = flow_loss_value.item()
    
    # Add rigidity loss if provided
    if rigidity_loss_value is not None:
        lambda_rigidity = loss_weights.get('rigidity', 0.0)
        total_loss = total_loss + lambda_rigidity * rigidity_loss_value
        loss_dict['rigidity'] = rigidity_loss_value.item()
    
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


# ============================================================================
# Integration helper: sample deformation points for rigidity loss
# ============================================================================

def sample_gaussian_neighborhood_points(
    gaussian_positions: torch.Tensor,
    num_samples_per_gaussian: int = 5,
    neighborhood_radius: float = 0.5
) -> torch.Tensor:
    """
    Sample points in neighborhoods of Gaussians for rigidity loss.
    
    Args:
        gaussian_positions: Gaussian centers [N_gaussians, 3]
        num_samples_per_gaussian: How many points to sample per Gaussian
        neighborhood_radius: Radius of neighborhood
    
    Returns:
        Sampled points [N_gaussians * num_samples_per_gaussian, 3]
    """
    device = gaussian_positions.device
    num_gaussians = gaussian_positions.shape[0]
    
    # Sample random perturbations
    perturbations = torch.randn(
        num_gaussians * num_samples_per_gaussian,
        3,
        device=device
    )
    
    # Normalize and scale
    perturbations = perturbations / (perturbations.norm(dim=1, keepdim=True) + 1e-6)
    perturbations = perturbations * neighborhood_radius
    
    # Create samples by adding perturbations to Gaussian positions
    gaussian_positions_expanded = gaussian_positions.repeat_interleave(num_samples_per_gaussian, dim=0)
    sampled_points = gaussian_positions_expanded + perturbations
    
    return sampled_points


if __name__ == "__main__":
    print("Coarse-to-fine training utilities loaded successfully")
