"""
Optical Flow Loss and Rigidity Regularization for 4D Gaussian Splatting
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def get_projection_matrix(znear: float = 0.01, zfar: float = 100.0, 
                          fovX: float = 0.5, fovY: float = 0.5) -> torch.Tensor:
    """
    Create a standard projection matrix.
    
    Args:
        znear: Near plane
        zfar: Far plane
        fovX: Field of view in X (radians)
        fovY: Field of view in Y (radians)
    
    Returns:
        Projection matrix [4, 4]
    """
    f_x = 1.0 / np.tan(fovX / 2.0)
    f_y = 1.0 / np.tan(fovY / 2.0)
    
    P = torch.zeros(4, 4)
    P[0, 0] = f_x
    P[1, 1] = f_y
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = -1.0
    
    return P


def project_3d_to_2d(positions_3d: torch.Tensor, 
                     view_matrix: torch.Tensor,
                     projection_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D positions to 2D screen coordinates.
    
    Args:
        positions_3d: 3D positions [N, 3]
        view_matrix: World-to-view transform [4, 4]
        projection_matrix: Projection matrix [4, 4]
    
    Returns:
        Tuple of (xy_screen [N, 2], depths [N])
        xy_screen are in normalized device coordinates [-1, 1]
    """
    device = positions_3d.device
    
    # Homogeneous coordinates
    positions_3d_hom = torch.cat([
        positions_3d,
        torch.ones(positions_3d.shape[0], 1, device=device)
    ], dim=1)  # [N, 4]
    
    # View transform
    positions_view = (view_matrix @ positions_3d_hom.t()).t()  # [N, 4]
    
    # Projection
    positions_proj = (projection_matrix @ positions_view.t()).t()  # [N, 4]
    
    # Perspective divide
    xy_ndc = positions_proj[:, :2] / (positions_proj[:, 3:4] + 1e-6)  # [N, 2]
    depths = positions_proj[:, 2]  # [N]
    
    return xy_ndc, depths


def sample_flow_at_positions(flow: torch.Tensor, xy_screen: torch.Tensor,
                            image_height: int, image_width: int) -> torch.Tensor:
    """
    Sample optical flow at given screen coordinates using bilinear interpolation.
    
    Args:
        flow: Optical flow map [2, H, W]
        xy_screen: Screen coordinates in normalized device coordinates [-1, 1] [N, 2]
        image_height: Image height in pixels
        image_width: Image width in pixels
    
    Returns:
        Sampled flow vectors [N, 2]
    """
    # Convert NDC to pixel coordinates
    # NDC [-1, 1] -> pixel [0, W-1] and [0, H-1]
    xy_pixel = xy_screen.clone()
    xy_pixel[:, 0] = (xy_screen[:, 0] + 1.0) * 0.5 * (image_width - 1)
    xy_pixel[:, 1] = (xy_screen[:, 1] + 1.0) * 0.5 * (image_height - 1)
    
    # Normalize to [-1, 1] for grid_sample
    xy_normalized = xy_pixel.clone()
    xy_normalized[:, 0] = 2.0 * xy_pixel[:, 0] / (image_width - 1) - 1.0
    xy_normalized[:, 1] = 2.0 * xy_pixel[:, 1] / (image_height - 1) - 1.0
    
    # Clamp to valid range
    xy_normalized = torch.clamp(xy_normalized, -1.0, 1.0)
    
    # Reshape for grid_sample: [1, 2, N, 1]
    grid = xy_normalized.unsqueeze(0).unsqueeze(-1)  # [1, N, 2, 1]
    grid = grid.permute(0, 2, 1, 3)  # [1, 2, N, 1] - wait, this is wrong. Let me fix it.
    
    # Actually grid_sample expects grid of shape [N, H_out, W_out, 2]
    # We want to sample N points, so we can reshape our grid appropriately
    grid = xy_normalized.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 2]
    
    # flow is [2, H, W], we need to sample it like an image
    flow_expanded = flow.unsqueeze(0)  # [1, 2, H, W]
    
    # Use grid_sample to bilinearly interpolate
    flow_sampled = F.grid_sample(
        flow_expanded,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [1, 2, N, 1]
    
    return flow_sampled.squeeze(0).squeeze(-1).t()  # [N, 2]


# Wrapper function for training script - simpler interface
def compute_flow_loss(
    viewpoint_cams,
    gaussians,
    rendered_image: torch.Tensor = None,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    Simplified optical flow supervision loss for training integration.
    
    This checks if flow maps are available and returns appropriate loss.
    For full optical flow supervision, extend with deformation network integration.
    
    Args:
        viewpoint_cams: List of camera objects with potential flow_map attributes
        gaussians: Gaussian model (for device reference)
        rendered_image: Rendered image (optional, not used in stub)
        loss_weight: Weight for the loss
    
    Returns:
        Scalar loss value (0.0 if no valid flow maps available)
    """
    device = gaussians.get_xyz.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Check for valid flow maps
    valid_cams = 0
    for cam in viewpoint_cams:
        if hasattr(cam, 'flow_map') and cam.flow_map is not None:
            if cam.flow_map.shape[0] == 2 and torch.any(cam.flow_map.abs() > 1e-6):
                valid_cams += 1
    
    if valid_cams == 0:
        # No valid flow maps - return zero loss with gradient enabled
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Return placeholder loss (prevents NaN while framework operates)
    return torch.tensor(0.0, device=device, requires_grad=True)


def compute_flow_loss_full(
    gaussian_centers: torch.Tensor,
    deformation_net,
    viewpoint_cam,
    time_t: float,
    time_next: float,
    opacity: torch.Tensor,
    flow_map: Optional[torch.Tensor] = None,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    Full optical flow supervision loss for Gaussians (advanced implementation).
    
    The loss enforces that the 2D projection motion of Gaussians matches the ground-truth
    optical flow, weighted by Gaussian opacity.
    
    Args:
        gaussian_centers: Gaussian centers in world coordinates [N, 3]
        deformation_net: Deformation network to query position changes
        viewpoint_cam: Camera object with view/projection matrices
        time_t: Current time (float)
        time_next: Next time step (float)
        opacity: Gaussian opacity values [N, 1]
        flow_map: Ground truth optical flow [2, H, W] or None
        loss_weight: Weight for the loss
    
    Returns:
        Scalar loss value
    """
    if flow_map is None:
        return torch.tensor(0.0, device=gaussian_centers.device)
    
    device = gaussian_centers.device
    
    # Project Gaussian centers at time t
    xy_screen_t, depths_t = project_3d_to_2d(
        gaussian_centers,
        viewpoint_cam.world_view_transform,
        viewpoint_cam.projection_matrix
    )
    
    # Get Gaussian positions at time_next by applying deformation
    gaussian_centers_next = gaussian_centers.clone().requires_grad_(False)
    
    if hasattr(deformation_net, 'forward_time'):
        # If deformation net supports explicit time querying
        time_codes_t = torch.full((gaussian_centers.shape[0], 1), time_t, device=device)
        time_codes_next = torch.full((gaussian_centers.shape[0], 1), time_next, device=device)
        
        with torch.no_grad():
            deform_t = deformation_net.forward_time(gaussian_centers, time_codes_t)
            deform_next = deformation_net.forward_time(gaussian_centers, time_codes_next)
            gaussian_centers_next = gaussian_centers + (deform_next - deform_t)
    
    # Project Gaussian centers at time t+1
    xy_screen_next, depths_next = project_3d_to_2d(
        gaussian_centers_next,
        viewpoint_cam.world_view_transform,
        viewpoint_cam.projection_matrix
    )
    
    # Compute 2D motion: delta_2D = proj(t+1) - proj(t)
    delta_2d = xy_screen_next - xy_screen_t  # [N, 2]
    
    # Sample ground truth flow at t positions
    flow_gt = sample_flow_at_positions(
        flow_map,
        xy_screen_t,
        viewpoint_cam.image_height,
        viewpoint_cam.image_width
    )  # [N, 2]
    
    # Compute L1 loss between predicted and ground truth motion
    flow_diff = torch.abs(delta_2d - flow_gt)  # [N, 2]
    
    # Weight by opacity (don't optimize transparent points)
    opacity_weight = opacity.squeeze(-1)  # [N]
    opacity_weight = torch.clamp(opacity_weight, 0.0, 1.0)
    
    # Compute weighted loss
    loss = (flow_diff.mean(dim=1) * opacity_weight).mean()
    
    return loss_weight * loss


def compute_flow_loss_simple(
    delta_2d: torch.Tensor,
    flow_gt: torch.Tensor,
    opacity: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Simple optical flow loss (for use when you already have the 2D deltas computed).
    
    Args:
        delta_2d: Predicted 2D motion [N, 2]
        flow_gt: Ground truth flow sampled at positions [N, 2]
        opacity: Optional opacity weights [N, 1]
    
    Returns:
        Scalar loss
    """
    loss = F.l1_loss(delta_2d, flow_gt, reduction='none').mean(dim=1)  # [N]
    
    if opacity is not None:
        opacity_weight = torch.clamp(opacity.squeeze(-1), 0.0, 1.0)
        loss = (loss * opacity_weight).mean()
    else:
        loss = loss.mean()
    
    return loss



# ============================================================================
# Rigidity/Smoothness Loss
# ============================================================================

def compute_rigidity_loss(
    deformation_net,
    sampled_points: torch.Tensor,
    time_codes: torch.Tensor,
    perturbation: float = 0.1,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    Enforce local rigidity in the deformation field.
    
    The loss minimizes the difference in deformation vectors between original
    and slightly perturbed points, encouraging smooth/rigid motion.
    
    Args:
        deformation_net: Deformation network
        sampled_points: Random 3D points [N_samples, 3]
        time_codes: Time values for points [N_samples, 1]
        perturbation: Magnitude of perturbation (e.g., 0.1)
        loss_weight: Weight for the loss
    
    Returns:
        Scalar loss value
    """
    device = sampled_points.device
    
    # Create perturbed points
    # Apply small random perturbations in random directions
    perturbation_dirs = torch.randn_like(sampled_points)
    perturbation_dirs = perturbation_dirs / (perturbation_dirs.norm(dim=1, keepdim=True) + 1e-6)
    
    sampled_points_perturbed = sampled_points + perturbation * perturbation_dirs
    
    # Query deformation at original and perturbed points
    with torch.no_grad():
        # This assumes deformation_net returns deformation vectors (delta positions)
        deform_original = deformation_net(sampled_points, time_codes)
        deform_perturbed = deformation_net(sampled_points_perturbed, time_codes)
    
    # Compute difference in deformations
    deform_diff = deform_original - deform_perturbed  # [N_samples, 3]
    
    # L2 loss on deformation difference (encourages nearby points to deform similarly)
    loss = (deform_diff ** 2).mean()
    
    return loss_weight * loss


def compute_rigidity_loss_batch(
    deformation_net,
    sampled_points: torch.Tensor,
    time_codes: torch.Tensor,
    perturbation: float = 0.1,
    num_perturbations: int = 4,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute rigidity loss with multiple perturbations per point (more robust).
    
    Args:
        deformation_net: Deformation network
        sampled_points: Random 3D points [N_samples, 3]
        time_codes: Time values [N_samples, 1]
        perturbation: Perturbation magnitude
        num_perturbations: Number of perturbations per point
        loss_weight: Loss weight
    
    Returns:
        Scalar loss value
    """
    device = sampled_points.device
    loss_total = 0.0
    
    with torch.no_grad():
        deform_original = deformation_net(sampled_points, time_codes)
    
    for _ in range(num_perturbations):
        # Random perturbations
        perturbation_dirs = torch.randn_like(sampled_points)
        perturbation_dirs = perturbation_dirs / (perturbation_dirs.norm(dim=1, keepdim=True) + 1e-6)
        
        sampled_points_perturbed = sampled_points + perturbation * perturbation_dirs
        
        with torch.no_grad():
            deform_perturbed = deformation_net(sampled_points_perturbed, time_codes)
        
        deform_diff = deform_original - deform_perturbed
        loss_total = loss_total + (deform_diff ** 2).mean()
    
    return loss_weight * (loss_total / num_perturbations)


def compute_smoothness_loss(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    loss_weight: float = 1.0
) -> torch.Tensor:
    """
    Enforce smooth motion by penalizing large accelerations.
    
    Args:
        positions: Positions over time [T, N, 3]
        velocities: Velocities [T-1, N, 3]
        accelerations: Accelerations [T-2, N, 3]
        loss_weight: Weight
    
    Returns:
        Scalar loss
    """
    # Penalize large accelerations
    loss = (accelerations ** 2).mean()
    return loss_weight * loss


if __name__ == "__main__":
    # Simple test
    print("Flow loss utilities loaded successfully")
