"""
Validation and Verification Tests for Optical Flow Supervision
Tests the flow loss, Gaussian tracking, and coarse-to-fine resolution switching.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class FlowValidation:
    """
    Validation utilities for optical flow supervision in 4D-GS.
    """
    
    def __init__(self, output_dir: str = "flow_validation"):
        """
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}
    
    # ========================================================================
    # TEST 1: Flow Loss Progression Tracking
    # ========================================================================
    
    def test_flow_loss_decreasing(
        self,
        flow_losses: List[float],
        window_size: int = 100,
        expected_trend: str = "decreasing"
    ) -> Dict:
        """
        Test 1: Verify that optical flow loss is decreasing over training.
        
        Args:
            flow_losses: List of flow loss values per iteration
            window_size: Smoothing window for loss curve
            expected_trend: "decreasing" or "increasing"
        
        Returns:
            Dict with test results including visualization
        """
        if len(flow_losses) < window_size:
            return {
                'status': 'SKIP',
                'reason': f'Insufficient samples ({len(flow_losses)} < {window_size})'
            }
        
        # Compute moving average
        losses_array = np.array(flow_losses)
        moving_avg = np.convolve(losses_array, np.ones(window_size)/window_size, mode='valid')
        
        # Check if trend matches expectation
        start_avg = moving_avg[:10].mean()
        end_avg = moving_avg[-10:].mean()
        
        if expected_trend == "decreasing":
            improvement = (start_avg - end_avg) / (start_avg + 1e-6)
            trend_satisfied = improvement > 0.1  # At least 10% improvement
        else:
            improvement = (end_avg - start_avg) / (start_avg + 1e-6)
            trend_satisfied = improvement > 0.1
        
        # Save plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(flow_losses, alpha=0.3, label='Raw loss')
        ax.plot(range(window_size-1, len(flow_losses)), moving_avg, linewidth=2, label=f'Moving avg (window={window_size})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Flow Loss')
        ax.set_title(f'Flow Loss Over Training (Trend: {expected_trend})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'test1_flow_loss_progression.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'status': 'PASS' if trend_satisfied else 'FAIL',
            'improvement_percent': improvement * 100,
            'start_loss': float(start_avg),
            'end_loss': float(end_avg),
            'threshold': 10.0,
            'visualization': 'test1_flow_loss_progression.png'
        }
    
    # ========================================================================
    # TEST 2: Gaussian Centers Alignment with Optical Flow
    # ========================================================================
    
    def test_gaussian_tracking_alignment(
        self,
        gaussian_positions_t0: torch.Tensor,
        gaussian_positions_t1: torch.Tensor,
        flow_gt: torch.Tensor,
        camera,
        opacity: torch.Tensor,
        threshold_pixels: float = 2.0
    ) -> Dict:
        """
        Test 2: Verify that Gaussian center displacements match optical flow.
        
        Visualizes:
        - Gaussian projected positions on image plane
        - Optical flow field overlaid
        - Discrepancy between predicted and ground-truth motion
        
        Args:
            gaussian_positions_t0: 3D Gaussian centers at time t [N, 3]
            gaussian_positions_t1: 3D Gaussian centers at time t+1 [N, 3]
            flow_gt: Ground truth optical flow [2, H, W]
            camera: Camera object with projection matrices
            opacity: Gaussian opacity [N, 1]
            threshold_pixels: Max acceptable error in pixels
        
        Returns:
            Dict with test results and visualization
        """
        device = gaussian_positions_t0.device
        
        # Project Gaussians to 2D
        from utils.flow_loss_utils import project_3d_to_2d
        
        xy_2d_t0, _ = project_3d_to_2d(
            gaussian_positions_t0,
            camera.world_view_transform.to(device),
            camera.projection_matrix.to(device)
        )
        
        xy_2d_t1, _ = project_3d_to_2d(
            gaussian_positions_t1,
            camera.world_view_transform.to(device),
            camera.projection_matrix.to(device)
        )
        
        # Predicted 2D motion
        predicted_motion = xy_2d_t1 - xy_2d_t0  # [N, 2] in NDC
        
        # Convert to pixel coordinates
        predicted_motion_pixels = predicted_motion.clone()
        predicted_motion_pixels[:, 0] = predicted_motion[:, 0] * camera.image_width / 2.0
        predicted_motion_pixels[:, 1] = predicted_motion[:, 1] * camera.image_height / 2.0
        
        # Sample ground truth flow at projected positions
        from utils.flow_loss_utils import sample_flow_at_positions
        
        flow_gt_sampled = sample_flow_at_positions(
            flow_gt,
            xy_2d_t0,
            camera.image_height,
            camera.image_width
        )  # [N, 2] in pixels
        
        # Compute error (weighted by opacity)
        error = torch.abs(predicted_motion_pixels - flow_gt_sampled)  # [N, 2]
        opacity_weight = torch.clamp(opacity.squeeze(-1), 0.0, 1.0)
        weighted_error = (error.mean(dim=1) * opacity_weight).mean().item()
        
        # Count points within threshold
        error_magnitude = error.norm(dim=1)
        within_threshold = (error_magnitude < threshold_pixels).float() * opacity_weight
        accuracy_percent = (within_threshold.sum() / opacity_weight.sum().clamp(min=1e-6)).item() * 100
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert flow to RGB
        flow_np = flow_gt.cpu().numpy()
        fx, fy = flow_np[0], flow_np[1]
        mag = np.sqrt(fx**2 + fy**2)
        ang = np.arctan2(fy, fx)
        h = (ang + np.pi) / (2 * np.pi)
        s = np.ones_like(h)
        v = mag / (mag.max() + 1e-6)
        from matplotlib.colors import hsv_to_rgb
        hsv = np.stack([h, s, v], axis=-1)
        flow_rgb = hsv_to_rgb(hsv)
        
        # Plot 1: Flow field
        axes[0].imshow(flow_rgb)
        axes[0].set_title('Ground Truth Optical Flow')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Plot 2: Gaussian projections with predicted motion
        xy_2d_t0_pixels = xy_2d_t0.clone().cpu()
        xy_2d_t0_pixels[:, 0] = (xy_2d_t0[:, 0].cpu() + 1.0) * 0.5 * (camera.image_width - 1)
        xy_2d_t0_pixels[:, 1] = (xy_2d_t0[:, 1].cpu() + 1.0) * 0.5 * (camera.image_height - 1)
        
        # Only plot well-lit Gaussians
        opacity_np = opacity_weight.cpu().numpy()
        visible_idx = opacity_np > 0.3
        
        axes[1].imshow(flow_rgb)
        
        # Draw vectors for visible Gaussians
        pred_motion_np = predicted_motion_pixels.cpu().numpy()
        scale = 1.0  # Scale for visualization
        axes[1].quiver(
            xy_2d_t0_pixels[visible_idx, 0],
            xy_2d_t0_pixels[visible_idx, 1],
            pred_motion_np[visible_idx, 0] * scale,
            pred_motion_np[visible_idx, 1] * scale,
            angles='xy', scale_units='xy', scale=1, color='cyan', width=0.5, alpha=0.7
        )
        axes[1].set_title(f'Predicted Gaussian Motion (Opacity > 0.3)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        
        # Plot 3: Error map
        error_np = error_magnitude.cpu().numpy()
        error_map = np.full((camera.image_height, camera.image_width), np.nan)
        
        # Sample error at grid points
        y_grid, x_grid = np.mgrid[0:camera.image_height:10, 0:camera.image_width:10]
        
        axes[2].imshow(flow_rgb, alpha=0.3)
        scatter = axes[2].scatter(
            xy_2d_t0_pixels[visible_idx, 0],
            xy_2d_t0_pixels[visible_idx, 1],
            c=error_np[visible_idx],
            cmap='RdYlGn_r',
            vmin=0,
            vmax=threshold_pixels * 2,
            s=50,
            alpha=0.8
        )
        axes[2].set_title(f'Motion Error (pixels)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        cbar = plt.colorbar(scatter, ax=axes[2])
        cbar.set_label('Error (pixels)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test2_gaussian_tracking.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'status': 'PASS' if accuracy_percent > 70 else 'WARN',
            'mean_error_pixels': float(weighted_error),
            'accuracy_percent': float(accuracy_percent),
            'threshold_pixels': threshold_pixels,
            'num_gaussians_visible': int(visible_idx.sum()),
            'visualization': 'test2_gaussian_tracking.png'
        }
    
    # ========================================================================
    # TEST 3: Coarse-to-Fine Resolution Transition
    # ========================================================================
    
    def test_coarse_to_fine_transition(
        self,
        images_coarse: torch.Tensor,
        images_fine: torch.Tensor,
        downsample_factor: int = 4,
        max_smoothness_error: float = 0.05
    ) -> Dict:
        """
        Test 3: Verify that coarse-to-fine transition doesn't cause artifacts.
        
        Checks:
        - Downsampled fine images match coarse images
        - No discontinuities at resolution transition
        - Brightness/contrast consistency
        
        Args:
            images_coarse: Rendered images at coarse resolution [B, 3, H_coarse, W_coarse]
            images_fine: Rendered images at fine resolution [B, 3, H_fine, W_fine]
            downsample_factor: Downsampling factor (e.g., 4)
            max_smoothness_error: Max acceptable reconstruction error
        
        Returns:
            Dict with test results
        """
        device = images_coarse.device
        
        # Downsample fine images to match coarse
        fine_downsampled = F.avg_pool2d(
            images_fine,
            kernel_size=downsample_factor,
            stride=downsample_factor
        )
        
        # Compute L1 error
        l1_error = F.l1_loss(fine_downsampled, images_coarse).item()
        
        # Compute perceptual metrics (simple: mean intensity, contrast)
        coarse_mean = images_coarse.mean().item()
        coarse_std = images_coarse.std().item()
        
        fine_mean = images_fine.mean().item()
        fine_std = images_fine.std().item()
        
        mean_diff = abs(coarse_mean - fine_mean) / (coarse_mean + 1e-6)
        std_diff = abs(coarse_std - fine_std) / (coarse_std + 1e-6)
        
        # Check consistency
        mean_consistent = mean_diff < 0.05
        std_consistent = std_diff < 0.1
        l1_acceptable = l1_error < max_smoothness_error
        
        status = 'PASS' if (mean_consistent and std_consistent and l1_acceptable) else 'WARN'
        
        # Visualize transition
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show a sample image from batch
        if images_coarse.shape[0] > 0:
            sample_idx = 0
            
            # Coarse original
            axes[0, 0].imshow(images_coarse[sample_idx].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            axes[0, 0].set_title(f'Coarse Resolution ({images_coarse.shape[-2]}x{images_coarse.shape[-1]})')
            axes[0, 0].axis('off')
            
            # Fine original
            axes[0, 1].imshow(images_fine[sample_idx].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            axes[0, 1].set_title(f'Fine Resolution ({images_fine.shape[-2]}x{images_fine.shape[-1]})')
            axes[0, 1].axis('off')
            
            # Fine downsampled
            axes[0, 2].imshow(fine_downsampled[sample_idx].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            axes[0, 2].set_title(f'Fine Downsampled ({fine_downsampled.shape[-2]}x{fine_downsampled.shape[-1]})')
            axes[0, 2].axis('off')
            
            # Difference maps
            coarse_rgb = images_coarse[sample_idx].permute(1, 2, 0).cpu().numpy()
            downsampled_rgb = fine_downsampled[sample_idx].permute(1, 2, 0).cpu().numpy()
            diff = np.abs(coarse_rgb - downsampled_rgb).mean(axis=2)
            
            im = axes[1, 0].imshow(diff, cmap='hot')
            axes[1, 0].set_title('L1 Difference Map')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Metrics text
            axes[1, 1].text(0.1, 0.9, f'Coarse Mean: {coarse_mean:.4f}\nFine Mean: {fine_mean:.4f}\nDiff: {mean_diff*100:.2f}%',
                           transform=axes[1, 1].transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].text(0.1, 0.5, f'Coarse Std: {coarse_std:.4f}\nFine Std: {fine_std:.4f}\nDiff: {std_diff*100:.2f}%',
                           transform=axes[1, 1].transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            axes[1, 1].axis('off')
            
            # Status
            status_color = 'lightgreen' if status == 'PASS' else 'lightyellow'
            axes[1, 2].text(0.1, 0.7, f'Status: {status}\n\nL1 Error: {l1_error:.6f}\nThreshold: {max_smoothness_error:.6f}\n\nMean Consistent: {mean_consistent}\nStd Consistent: {std_consistent}',
                           transform=axes[1, 2].transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.7))
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test3_coarse_to_fine_transition.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'status': status,
            'l1_error': float(l1_error),
            'l1_threshold': max_smoothness_error,
            'mean_intensity_diff_percent': float(mean_diff * 100),
            'std_diff_percent': float(std_diff * 100),
            'mean_consistent': mean_consistent,
            'std_consistent': std_consistent,
            'visualization': 'test3_coarse_to_fine_transition.png'
        }
    
    # ========================================================================
    # Generate Report
    # ========================================================================
    
    def generate_report(self, test_results: Dict) -> None:
        """
        Generate a comprehensive report of all validation tests.
        
        Args:
            test_results: Dict with results from all three tests
        """
        report_path = self.output_dir / 'validation_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("FLOW SUPERVISION VALIDATION REPORT")
        print(f"{'='*60}\n")
        
        for test_name, result in test_results.items():
            print(f"{test_name}:")
            print(f"  Status: {result.get('status', 'N/A')}")
            for key, value in result.items():
                if key not in ['status', 'visualization']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
            if 'visualization' in result:
                print(f"  Visualization: {result['visualization']}")
            print()
        
        print(f"{'='*60}")
        print(f"Report saved to: {report_path}")
        print(f"Visualizations saved to: {self.output_dir}")
        print(f"{'='*60}\n")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("Optical Flow Validation Utilities")
    print("=" * 60)
    print("\nUsage examples:\n")
    
    print("1. Flow Loss Progression Test:")
    print("   validator = FlowValidation()")
    print("   result = validator.test_flow_loss_decreasing(flow_losses_list)")
    print()
    
    print("2. Gaussian Tracking Test:")
    print("   result = validator.test_gaussian_tracking_alignment(")
    print("       gaussian_positions_t0, gaussian_positions_t1,")
    print("       flow_gt, camera, opacity)")
    print()
    
    print("3. Coarse-to-Fine Transition Test:")
    print("   result = validator.test_coarse_to_fine_transition(")
    print("       images_coarse, images_fine, downsample_factor)")
    print()
    
    print("4. Generate Report:")
    print("   validator.generate_report(test_results)")
    print()
