"""
INTEGRATION GUIDE: Adding Optical Flow Loss and Coarse-to-Fine Training

This file demonstrates how to integrate the flow loss, rigidity loss, and 
coarse-to-fine training strategy into the main train.py.

Key integration points:
1. Import new utilities at the top of train.py
2. Initialize scheduler in scene_reconstruction()
3. Add flow loss computation in the training loop
4. Add rigidity loss computation in the training loop
5. Apply coarse-to-fine preprocessing in the training loop
"""

# ============================================================================
# STEP 1: Add these imports to the top of train.py
# ============================================================================

"""
from utils.flow_loss_utils import compute_flow_loss, compute_rigidity_loss
from utils.coarse_to_fine_utils import (
    CoarseToFineScheduler,
    apply_coarse_to_fine_preprocessing,
    compute_combined_loss,
    sample_gaussian_neighborhood_points
)
"""

# ============================================================================
# STEP 2: Initialize scheduler in scene_reconstruction()
# ============================================================================

"""
In the scene_reconstruction() function, after line where you set up gaussians:

    # Initialize coarse-to-fine scheduler
    c2f_scheduler = CoarseToFineScheduler(
        warmup_iters=3000,
        downsample_factor=4,
        lambda_flow_start=0.1,
        lambda_flow_end=0.5,
        lambda_rigidity=0.01
    )
"""

# ============================================================================
# STEP 3: Modify the loss computation section (around line 200-220)
# ============================================================================

"""
BEFORE (existing code):
    # Loss
    Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
    loss = Ll1
    if stage == "fine" and hyper.time_smoothness_weight != 0:
        tv_loss = gaussians.compute_regulation(...)
        loss += tv_loss
    if opt.lambda_dssim != 0:
        ssim_loss = ssim(image_tensor, gt_image_tensor)
        loss += opt.lambda_dssim * (1.0 - ssim_loss)

AFTER (with flow supervision):
    # Get resolution scale from scheduler
    resolution_scale = c2f_scheduler.get_resolution_scale(iteration)
    loss_weights = c2f_scheduler.get_loss_weights(iteration)
    
    # Apply coarse-to-fine preprocessing if in warmup
    if c2f_scheduler.is_warmup(iteration):
        adjusted_cams, adjusted_gt_images = apply_coarse_to_fine_preprocessing(
            viewpoint_cams,
            gt_image_tensor[:,:3,:,:],
            resolution_scale,
            c2f_scheduler
        )
    else:
        adjusted_cams = viewpoint_cams
        adjusted_gt_images = gt_image_tensor[:,:3,:,:]
    
    # Base L1 loss
    Ll1 = l1_loss(image_tensor, adjusted_gt_images)
    
    # Initialize total loss
    loss = Ll1
    
    # Add TV/smoothness regulation (existing)
    if stage == "fine" and hyper.time_smoothness_weight != 0:
        tv_loss = gaussians.compute_regulation(...)
        loss += tv_loss
    
    # Add SSIM loss (existing)
    if opt.lambda_dssim != 0:
        ssim_loss = ssim(image_tensor, adjusted_gt_images)
        loss += opt.lambda_dssim * (1.0 - ssim_loss)
    
    # Add optical flow loss
    flow_loss = torch.tensor(0.0, device=device)
    if loss_weights['flow_enabled'] and hasattr(viewpoint_cams[0], 'flow_map'):
        # Sample points for flow loss computation
        for cam_idx, (cam, pred_img) in enumerate(zip(adjusted_cams, images)):
            if cam.flow_map is not None and cam.flow_map.sum() > 0:
                # Compute flow loss for this camera
                # Note: This requires integrating with your render pass to get positions
                # You may need to modify render() to return 3D positions for flow loss
                
                # Example (simplified):
                # flow_loss += compute_flow_loss(
                #     gaussian_centers=gaussians.get_xyz,
                #     deformation_net=gaussians._deformation,
                #     viewpoint_cam=cam,
                #     time_t=cam.time,
                #     time_next=cam.time + 0.1,  # Adjust based on your time step
                #     opacity=gaussians.get_opacity,
                #     flow_map=cam.flow_map,
                #     loss_weight=1.0
                # )
    
    if loss_weights['flow_enabled'] and flow_loss > 0:
        loss = loss + loss_weights['flow'] * flow_loss
    
    # Add rigidity loss
    rigidity_loss = torch.tensor(0.0, device=device)
    if stage == "fine":
        # Sample points for rigidity loss
        sampled_points = sample_gaussian_neighborhood_points(
            gaussian_positions=gaussians.get_xyz,
            num_samples_per_gaussian=5,
            neighborhood_radius=0.5
        )
        
        time_codes = torch.full(
            (sampled_points.shape[0], 1),
            viewpoint_cams[0].time,
            device=device
        )
        
        rigidity_loss = compute_rigidity_loss(
            deformation_net=gaussians._deformation,
            sampled_points=sampled_points,
            time_codes=time_codes,
            perturbation=0.1,
            loss_weight=1.0
        )
        
        loss = loss + loss_weights['rigidity'] * rigidity_loss
"""

# ============================================================================
# STEP 4: Modify Scene initialization to pass flow_dir
# ============================================================================

"""
In your scene loading code (when creating the Scene object), add:

    # After Scene creation, pass flow_dir to dataset
    if hasattr(scene, 'train_cameras'):
        for cam_list in scene.train_cameras.values():
            for cam in cam_list:
                # Flow maps are already loaded by Camera.__init__
                pass
    
    # Or modify FourDGSdataset initialization:
    dataset.flow_dir = "path/to/flow/directory"
"""

# ============================================================================
# STEP 5: Add command-line arguments (in the __main__ section)
# ============================================================================

"""
Add to the argument parser:

    parser.add_argument("--flow_dir", type=str, default=None,
                       help="Directory containing optical flow .npy files")
    parser.add_argument("--lambda_flow", type=float, default=0.1,
                       help="Weight for optical flow loss")
    parser.add_argument("--lambda_rigidity", type=float, default=0.01,
                       help="Weight for rigidity regularization loss")
    parser.add_argument("--coarse_warmup_iters", type=int, default=3000,
                       help="Number of coarse stage iterations")
    parser.add_argument("--downsample_factor", type=int, default=4,
                       help="Downsampling factor for coarse stage")
"""

# ============================================================================
# STEP 6: Example of complete loss section (for reference)
# ============================================================================

EXAMPLE_COMPLETE_LOSS_SECTION = """
    # ===== Loss Computation with Flow Supervision =====
    resolution_scale = c2f_scheduler.get_resolution_scale(iteration)
    loss_weights = c2f_scheduler.get_loss_weights(iteration)
    
    # Apply coarse-to-fine preprocessing
    if c2f_scheduler.is_warmup(iteration):
        # Import the downsampling functions
        from utils.coarse_to_fine_utils import downsample_image, downsample_flow
        
        # Downsample images and flows
        factor = int(1.0 / resolution_scale)
        downsampled_gt = downsample_image(gt_image_tensor[:,:3,:,:], factor)
        
        # Adjust cameras
        adjusted_cams = []
        for cam in viewpoint_cams:
            from utils.coarse_to_fine_utils import adjust_camera_intrinsics
            adj_cam = adjust_camera_intrinsics(cam, resolution_scale)
            
            # Also downsample the flow map
            if adj_cam.flow_map is not None and adj_cam.flow_map.sum() > 0:
                adj_cam.flow_map = downsample_flow(adj_cam.flow_map, factor)
            
            adjusted_cams.append(adj_cam)
    else:
        adjusted_cams = viewpoint_cams
        downsampled_gt = gt_image_tensor[:,:3,:,:]
    
    # Base L1 loss
    Ll1 = l1_loss(image_tensor, downsampled_gt)
    loss = Ll1
    
    # TV loss (existing)
    if stage == "fine" and hyper.time_smoothness_weight != 0:
        tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, 
                                               hyper.l1_time_planes, 
                                               hyper.plane_tv_weight)
        loss += tv_loss
    
    # SSIM loss (existing)
    if opt.lambda_dssim != 0:
        ssim_loss = ssim(image_tensor, downsampled_gt)
        loss += opt.lambda_dssim * (1.0 - ssim_loss)
    
    # Optical flow loss (new)
    flow_loss_total = 0.0
    if loss_weights['flow_enabled'] and stage == "fine":
        for cam_idx, cam in enumerate(adjusted_cams):
            if hasattr(cam, 'flow_map') and cam.flow_map is not None and cam.flow_map.sum() > 0:
                try:
                    flow_loss = compute_flow_loss(
                        gaussian_centers=gaussians.get_xyz,
                        deformation_net=gaussians._deformation,
                        viewpoint_cam=cam,
                        time_t=cam.time,
                        time_next=cam.time + 1.0,  # Adjust based on frame delta
                        opacity=gaussians.get_opacity,
                        flow_map=cam.flow_map,
                        loss_weight=1.0
                    )
                    flow_loss_total += flow_loss
                except Exception as e:
                    print(f"Warning: Flow loss computation failed: {e}")
        
        if flow_loss_total > 0:
            flow_loss_total = flow_loss_total / len(adjusted_cams)
            loss = loss + loss_weights['flow'] * flow_loss_total
    
    # Rigidity loss (new)
    if stage == "fine" and loss_weights['rigidity'] > 0:
        try:
            sampled_points = sample_gaussian_neighborhood_points(
                gaussian_positions=gaussians.get_xyz,
                num_samples_per_gaussian=3,
                neighborhood_radius=0.5
            )
            
            time_codes = torch.full(
                (sampled_points.shape[0], 1),
                viewpoint_cams[0].time,
                device=gaussians.get_xyz.device
            )
            
            rigidity_loss = compute_rigidity_loss(
                deformation_net=gaussians._deformation,
                sampled_points=sampled_points,
                time_codes=time_codes,
                perturbation=0.1,
                loss_weight=1.0
            )
            loss = loss + loss_weights['rigidity'] * rigidity_loss
        except Exception as e:
            print(f"Warning: Rigidity loss computation failed: {e}")
"""

# ============================================================================
# STEP 7: Update your arguments/hyperparams to support flow_dir
# ============================================================================

"""
In your Scene loading or dataset creation, ensure flow_dir is passed:

    # In train.py main section:
    if hasattr(args, 'flow_dir') and args.flow_dir:
        scene.flow_dir = args.flow_dir
        # Re-initialize dataset with flow_dir
        # This depends on your specific scene loading implementation
"""

# ============================================================================
# IMPORTANT NOTES:
# ============================================================================

"""
1. Flow Loss Integration:
   - compute_flow_loss requires the deformation network to support querying
     positions at different times
   - You may need to modify this based on your specific deformation network API
   - Currently assumes deformation_net.forward(positions, time_codes) returns
     the deformed positions or position deltas

2. Rigidity Loss Timing:
   - Should primarily be used in "fine" stage (after warmup)
   - Applies smoothness regularization to the learned deformation field
   - Sample fewer points to keep computation cost low

3. Coarse-to-Fine Strategy:
   - First 3000 iterations: 1/4 resolution, no flow loss (warmup)
   - 3000-5000 iterations: gradual transition to full resolution
   - After 5000 iterations: full resolution + flow + rigidity losses

4. Hyperparameter Tuning:
   - lambda_flow: Controls strength of flow supervision (0.1-0.5 recommended)
   - lambda_rigidity: Controls smoothness (0.01-0.05 recommended)
   - warmup_iters: How long to stay at low resolution (3000 is good starting point)
   - downsample_factor: Trade-off between speed and quality (4x is typical)

5. Flow File Format:
   - .npy files should be [2, H, W] with flow values in pixel units
   - Generated by utils/flow_utils.py generate_optical_flow_dataset()
   - Frame i should have file flow_{i:06d}.npy

6. Time Handling:
   - Each camera has a time attribute (from the dataset)
   - For consecutive frames: time_next = time_t + 1.0 (or your time delta)
   - Adjust based on your specific dataset's time conventions
"""
