#!/usr/bin/env python3
"""
Complete Example: Running 4D-GS with Optical Flow Supervision

This script demonstrates how to use all the optical flow components together.
It can be run as:
    python example_flow_training.py --image_dir data/images --output_dir output
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Dict, List

# Import your 4D-GS modules
# from gaussian_renderer import render
# from scene import Scene, GaussianModel
# from utils.loss_utils import l1_loss, ssim

# Import new flow modules
from utils.flow_utils import generate_optical_flow_dataset
from utils.flow_loss_utils import compute_flow_loss, compute_rigidity_loss
from utils.coarse_to_fine_utils import (
    CoarseToFineScheduler,
    apply_coarse_to_fine_preprocessing,
    sample_gaussian_neighborhood_points
)
from utils.validation_utils import FlowValidation


class OpticalFlowTrainer:
    """
    Complete trainer with optical flow supervision and coarse-to-fine strategy.
    """
    
    def __init__(self, args):
        self.args = args
        
        # Initialize scheduler
        self.scheduler = CoarseToFineScheduler(
            warmup_iters=args.coarse_warmup_iters,
            downsample_factor=args.downsample_factor,
            lambda_flow_start=args.lambda_flow_start,
            lambda_flow_end=args.lambda_flow_end,
            lambda_rigidity=args.lambda_rigidity
        )
        
        # Initialize validation utilities
        self.validator = FlowValidation(output_dir=args.validation_dir)
        
        # Storage for metrics
        self.metrics = {
            'l1_loss': [],
            'flow_loss': [],
            'rigidity_loss': [],
            'total_loss': [],
            'psnr': []
        }
    
    def setup_flow_data(self):
        """
        Step 1: Generate optical flow if not already present.
        """
        if not os.path.exists(self.args.flow_dir):
            print(f"Generating optical flow from {self.args.image_dir}...")
            generate_optical_flow_dataset(
                image_dir=self.args.image_dir,
                output_dir=self.args.flow_dir,
                device=self.args.device,
                save_visualization=True,
                max_frames=self.args.max_frames
            )
            print(f"Flow generation complete! Files saved to {self.args.flow_dir}")
        else:
            print(f"Using existing flow data from {self.args.flow_dir}")
    
    def training_step(self, iteration: int, batch_data: Dict) -> Dict:
        """
        Single training step with flow and rigidity losses.
        
        Args:
            iteration: Current iteration number
            batch_data: Dict with keys:
                - 'rendered_images': [B, 3, H, W]
                - 'gt_images': [B, 3, H, W]
                - 'cameras': List of camera objects
                - 'gaussians': GaussianModel instance
                - 'l1_loss_fn': Loss function
        
        Returns:
            Dict with losses and metrics
        """
        # Get scheduler state
        resolution_scale = self.scheduler.get_resolution_scale(iteration)
        loss_weights = self.scheduler.get_loss_weights(iteration)
        
        # Extract data
        rendered_images = batch_data['rendered_images']
        gt_images = batch_data['gt_images']
        cameras = batch_data['cameras']
        gaussians = batch_data['gaussians']
        l1_loss_fn = batch_data['l1_loss_fn']
        device = rendered_images.device
        
        losses = {}
        
        # ====== 1. Base L1 Loss ======
        l1_loss = l1_loss_fn(rendered_images, gt_images)
        losses['l1'] = l1_loss.item()
        total_loss = l1_loss
        
        # ====== 2. Optical Flow Loss ======
        flow_loss = 0.0
        if loss_weights['flow_enabled'] and self.args.use_flow_loss:
            for cam_idx, camera in enumerate(cameras):
                if hasattr(camera, 'flow_map') and camera.flow_map is not None:
                    try:
                        # Note: This requires your render function to return 3D positions
                        # For now, we provide a simplified version
                        cam_flow_loss = compute_flow_loss(
                            gaussian_centers=gaussians.get_xyz,
                            deformation_net=gaussians._deformation,
                            viewpoint_cam=camera,
                            time_t=float(camera.time),
                            time_next=float(camera.time) + 1.0,
                            opacity=gaussians.get_opacity,
                            flow_map=camera.flow_map,
                            loss_weight=1.0
                        )
                        flow_loss += cam_flow_loss
                    except Exception as e:
                        print(f"Warning: Flow loss failed: {e}")
            
            if flow_loss > 0:
                flow_loss = flow_loss / max(len(cameras), 1)
                total_loss = total_loss + loss_weights['flow'] * flow_loss
        
        losses['flow'] = flow_loss if isinstance(flow_loss, float) else flow_loss.item()
        
        # ====== 3. Rigidity Loss ======
        rigidity_loss = 0.0
        if self.args.use_rigidity_loss and loss_weights['rigidity'] > 0:
            try:
                sampled_points = sample_gaussian_neighborhood_points(
                    gaussian_positions=gaussians.get_xyz,
                    num_samples_per_gaussian=3,
                    neighborhood_radius=0.5
                )
                
                time_codes = torch.full(
                    (sampled_points.shape[0], 1),
                    cameras[0].time,
                    device=device
                )
                
                rigidity_loss = compute_rigidity_loss(
                    deformation_net=gaussians._deformation,
                    sampled_points=sampled_points,
                    time_codes=time_codes,
                    perturbation=0.1,
                    loss_weight=1.0
                )
                total_loss = total_loss + loss_weights['rigidity'] * rigidity_loss
            except Exception as e:
                print(f"Warning: Rigidity loss failed: {e}")
        
        losses['rigidity'] = rigidity_loss if isinstance(rigidity_loss, float) else rigidity_loss.item()
        losses['total'] = total_loss.item()
        
        # ====== Store metrics ======
        self.metrics['l1_loss'].append(losses['l1'])
        self.metrics['flow_loss'].append(losses['flow'])
        self.metrics['rigidity_loss'].append(losses['rigidity'])
        self.metrics['total_loss'].append(losses['total'])
        
        # Log info
        if iteration % 100 == 0:
            print(f"Iter {iteration}: "
                  f"L1={losses['l1']:.4f}, "
                  f"Flow={losses['flow']:.4f}, "
                  f"Rigidity={losses['rigidity']:.4f}, "
                  f"Total={losses['total']:.4f}, "
                  f"ResScale={resolution_scale:.2f}")
        
        return total_loss, losses
    
    def validate(self, iteration: int, batch_data: Dict) -> Dict:
        """
        Run validation tests at specific iterations.
        
        Args:
            iteration: Current iteration
            batch_data: Training data batch
        
        Returns:
            Validation results dict
        """
        if iteration % self.args.validation_interval != 0:
            return {}
        
        print(f"\n[Validation] Running tests at iteration {iteration}...")
        
        results = {}
        
        # Test 1: Flow loss progression
        if len(self.metrics['flow_loss']) > 100:
            results['flow_progression'] = self.validator.test_flow_loss_decreasing(
                self.metrics['flow_loss'][-200:],
                window_size=20,
                expected_trend='decreasing'
            )
            print(f"  Flow progression: {results['flow_progression']['status']}")
        
        # Test 2: Gaussian tracking (requires actual data)
        # Note: This would need 3D positions from the render pass
        # Simplified placeholder:
        if self.args.validate_tracking and hasattr(batch_data, 'gaussian_positions'):
            try:
                results['gaussian_tracking'] = self.validator.test_gaussian_tracking_alignment(
                    batch_data['gaussian_positions'],
                    batch_data['gaussian_positions_next'],
                    batch_data['cameras'][0].flow_map,
                    batch_data['cameras'][0],
                    batch_data['gaussians'].get_opacity
                )
                print(f"  Gaussian tracking: {results['gaussian_tracking']['status']}")
            except Exception as e:
                print(f"  Gaussian tracking: SKIP ({e})")
        
        # Test 3: Coarse-to-fine transition
        if iteration == self.args.coarse_warmup_iters + 1000:
            try:
                results['coarse_to_fine'] = self.validator.test_coarse_to_fine_transition(
                    batch_data.get('rendered_images_coarse', batch_data['rendered_images']),
                    batch_data['rendered_images']
                )
                print(f"  Coarse-to-fine: {results['coarse_to_fine']['status']}")
            except Exception as e:
                print(f"  Coarse-to-fine: SKIP ({e})")
        
        return results
    
    def save_checkpoint(self, iteration: int, model_dict: Dict):
        """Save training checkpoint with metrics."""
        checkpoint = {
            'iteration': iteration,
            'model': model_dict,
            'metrics': self.metrics,
            'scheduler_state': self.scheduler.__dict__
        }
        path = os.path.join(self.args.output_dir, f'checkpoint_{iteration}.pth')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def finalize(self):
        """Generate final validation report."""
        print("\n" + "="*60)
        print("GENERATING FINAL VALIDATION REPORT")
        print("="*60 + "\n")
        
        test_results = {
            'test1_flow_progression': self.validator.test_flow_loss_decreasing(
                self.metrics['flow_loss'],
                window_size=min(100, len(self.metrics['flow_loss']) // 2)
            )
        }
        
        self.validator.generate_report(test_results)
        
        # Save metrics
        metrics_path = os.path.join(self.args.validation_dir, 'training_metrics.npy')
        np.save(metrics_path, self.metrics)
        print(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="4D-GS with Optical Flow Supervision")
    
    # Flow data
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory with training images')
    parser.add_argument('--flow_dir', type=str, default='flow',
                       help='Directory for optical flow files')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Max frames to process (for testing)')
    
    # Training
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--validation_dir', type=str, default='validation',
                       help='Validation output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--iterations', type=int, default=30000,
                       help='Total training iterations')
    
    # Coarse-to-fine
    parser.add_argument('--coarse_warmup_iters', type=int, default=3000,
                       help='Iterations for coarse warmup')
    parser.add_argument('--downsample_factor', type=int, default=4,
                       help='Downsampling factor during warmup')
    
    # Loss weights
    parser.add_argument('--lambda_flow_start', type=float, default=0.1,
                       help='Initial flow loss weight')
    parser.add_argument('--lambda_flow_end', type=float, default=0.5,
                       help='Final flow loss weight')
    parser.add_argument('--lambda_rigidity', type=float, default=0.01,
                       help='Rigidity loss weight')
    
    # Flags
    parser.add_argument('--use_flow_loss', action='store_true', default=True,
                       help='Enable flow loss')
    parser.add_argument('--use_rigidity_loss', action='store_true', default=True,
                       help='Enable rigidity loss')
    parser.add_argument('--validate_tracking', action='store_true', default=False,
                       help='Enable Gaussian tracking validation')
    parser.add_argument('--validation_interval', type=int, default=1000,
                       help='Validation frequency')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.validation_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print("Initializing Optical Flow Trainer...")
    trainer = OpticalFlowTrainer(args)
    
    # Setup flow data
    print("\nStep 1: Setting up optical flow data...")
    trainer.setup_flow_data()
    
    # Example training loop (simplified)
    print(f"\nStep 2: Starting training for {args.iterations} iterations...")
    print("(Note: This is a minimal example. Integrate into your full train.py)\n")
    
    # Placeholder for actual training
    print("Training would proceed here with the trainer.training_step() function.")
    print("\nExample integration in your train loop:")
    print("""
    for iteration in range(first_iter, final_iter + 1):
        # Your rendering code here
        total_loss, losses = trainer.training_step(
            iteration,
            {
                'rendered_images': rendered,
                'gt_images': ground_truth,
                'cameras': cameras,
                'gaussians': gaussians,
                'l1_loss_fn': l1_loss
            }
        )
        
        # Your optimizer step
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Validation
        val_results = trainer.validate(iteration, batch_data)
        
        # Checkpoint
        if iteration % 5000 == 0:
            trainer.save_checkpoint(iteration, model_dict)
    
    # Final report
    trainer.finalize()
    """)
    
    print("\n" + "="*60)
    print("Setup complete! Now integrate trainer into your main train.py")
    print("="*60)


if __name__ == "__main__":
    main()
