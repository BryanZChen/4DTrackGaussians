"""
QUICK START GUIDE: Optical Flow Supervision for 4D Gaussian Splatting

This file provides a step-by-step guide to implement optical flow supervision
for handling fast-moving objects in your 4D-GS project.
"""

# ============================================================================
# PHASE 1: Generate Optical Flow Data
# ============================================================================

"""
Step 1.1: Prepare your training images
- Organize them in a single directory
- Ensure they're sorted by timestamp (e.g., frame_0000.png, frame_0001.png)
- Images should be in RGB format (JPG, PNG, etc.)

Step 1.2: Generate optical flow maps using RAFT

    python -m utils.flow_utils <image_directory> \
        --output_dir flow \
        --device cuda \
        --max_frames 100  # optional, for testing

This will generate:
- flow/flow_000000.npy, flow_000001.npy, ... (optical flow files)
- flow/visualization/flow_000000.png, ... (visualizations for verification)
- Last frame gets zero flow (since there's no next frame)

Expected output structure:
    flow/
    ├── flow_000000.npy
    ├── flow_000001.npy
    ├── ...
    ├── flow_000099.npy
    └── visualization/
        ├── flow_000000.png
        ├── flow_000001.png
        └── ...

Tip: Review the visualization images to ensure optical flow is computed correctly.
"""

# ============================================================================
# PHASE 2: Update Your Training Configuration
# ============================================================================

"""
Step 2.1: Add command-line arguments to your train script:

    python train.py \
        --source_path <path_to_dataset> \
        --model_path <output_path> \
        --flow_dir flow \
        --lambda_flow 0.2 \
        --lambda_rigidity 0.01 \
        --coarse_warmup_iters 3000 \
        --downsample_factor 4 \
        --iterations 30000

Step 2.2: Arguments explanation:
    --flow_dir: Path to directory with optical flow .npy files
    --lambda_flow: Weight for flow loss (typical: 0.1-0.5)
    --lambda_rigidity: Weight for rigidity/smoothness loss (typical: 0.01-0.05)
    --coarse_warmup_iters: Iterations to stay at low resolution (typical: 3000)
    --downsample_factor: Resolution reduction during warmup (typical: 4)

Step 2.3: (Optional) Add to your hyperparameter config file:
    lambda_flow: 0.2
    lambda_rigidity: 0.01
    coarse_warmup_iters: 3000
    downsample_factor: 4
"""

# ============================================================================
# PHASE 3: Integrate into train.py
# ============================================================================

"""
Follow the FLOW_INTEGRATION_GUIDE.md for detailed integration steps.

Summary of changes needed:
1. Import new utility modules (flow_loss_utils, coarse_to_fine_utils, validation_utils)
2. Initialize CoarseToFineScheduler in scene_reconstruction()
3. Modify loss computation to include:
   - Resolution scaling for coarse stage
   - Optical flow loss
   - Rigidity regularization loss
4. Add command-line argument parsing for new hyperparameters
5. Pass flow_dir to dataset loader

Expected training progression:
- Iterations 0-3000 (WARMUP):
  * 1/4 resolution images
  * L1 loss only
  * Quick convergence on coarse structure
  
- Iterations 3000-5000 (TRANSITION):
  * Gradual resolution increase
  * Flow loss ramps up
  
- Iterations 5000+ (FINE):
  * Full resolution
  * L1 + Flow + Rigidity losses
  * Capture fine motion details
"""

# ============================================================================
# PHASE 4: Monitor Training
# ============================================================================

"""
During training, you can monitor:

1. Flow Loss Curve:
   - Should decrease from ~0.2-0.5 to ~0.05-0.1
   - If stuck at high values: try increasing lambda_flow or checking flow data
   - If zero: ensure flow files exist and camera has valid flow_map

2. Rigidity Loss Curve:
   - Should stabilize around 0.001-0.01
   - Prevents Gaussians from tearing apart
   
3. Total Loss:
   - Should show steady decrease
   - May have jumps at resolution transition (normal)

4. PSNR:
   - Should improve after warmup phase
   - Better matching with ground truth images
"""

# ============================================================================
# PHASE 5: Validation and Testing
# ============================================================================

"""
After training completes, validate the results:

Step 5.1: Use validation_utils.py to test:

    from utils.validation_utils import FlowValidation
    
    validator = FlowValidation(output_dir="validation_results")
    
    # Test 1: Flow loss progression
    result1 = validator.test_flow_loss_decreasing(flow_losses_from_training)
    
    # Test 2: Gaussian tracking alignment
    result2 = validator.test_gaussian_tracking_alignment(
        gaussian_positions_t0,
        gaussian_positions_t1,
        flow_gt,
        camera,
        opacity
    )
    
    # Test 3: Coarse-to-fine transition
    result3 = validator.test_coarse_to_fine_transition(
        images_coarse_resolution,
        images_fine_resolution
    )
    
    # Generate report
    results = {
        'Test1_FlowProgression': result1,
        'Test2_GaussianTracking': result2,
        'Test3_C2FTransition': result3
    }
    validator.generate_report(results)

This generates:
- validation_report.json (numeric results)
- test1_flow_loss_progression.png (loss curve)
- test2_gaussian_tracking.png (Gaussian alignment visualization)
- test3_coarse_to_fine_transition.png (resolution transition check)

Step 5.2: Interpret results:
- All tests should show "PASS" or "WARN"
- Test 1: Flow loss should decrease by >10%
- Test 2: Gaussian accuracy should be >70%
- Test 3: L1 error should be < 0.05
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: Flow loss stays at zero
Solution:
- Check that flow files exist in flow_dir
- Verify flow file names match: flow_000000.npy, flow_000001.npy, etc.
- Ensure Camera.flow_map is not None
- Try: print(camera.flow_map.shape, camera.flow_map.sum())

Problem: Flow loss NaN or exploding
Solution:
- Reduce lambda_flow (start with 0.05 instead of 0.2)
- Check optical flow values are reasonable (magnitude ~1-10 pixels)
- Verify projection matrix is correct
- Try: print(flow_map.min(), flow_map.max(), flow_map.mean())

Problem: Total loss increases at iteration 3000
Solution:
- This is normal during resolution transition
- Should decrease again within a few hundred iterations
- If it increases for >500 iterations, may need slower transition
- Adjust coarse_warmup_iters or downsample_factor

Problem: No improvement after adding flow loss
Solution:
- Check that loss_weights['flow_enabled'] is True after iteration 3000
- Verify camera.time values are correct and time_next is properly set
- Try increasing lambda_flow gradually (0.05 → 0.1 → 0.2)
- Check deformation network can handle time codes

Problem: Out of memory with flow loss
Solution:
- Reduce batch_size
- Reduce num_samples_per_gaussian in rigidity loss
- Skip rigidity loss (set lambda_rigidity=0)
- Use coarser initial resolution

Problem: Visible artifacts at resolution transition
Solution:
- Increase transition window from 1000 to 2000 iterations
- Modify adjust_camera_intrinsics to smooth FoV transition
- Manually verify focal length calculations are correct
"""

# ============================================================================
# PARAMETER TUNING GUIDE
# ============================================================================

"""
For Fast-Moving Objects:
- lambda_flow: 0.3-0.5 (stronger flow supervision)
- lambda_rigidity: 0.02-0.05 (more smoothness)
- coarse_warmup_iters: 5000 (longer warmup)
- downsample_factor: 4 (4x downsampling)

For Subtle Motion:
- lambda_flow: 0.1-0.2 (lighter flow supervision)
- lambda_rigidity: 0.005-0.01 (less regularization)
- coarse_warmup_iters: 2000 (shorter warmup)
- downsample_factor: 2 (2x downsampling)

For Limited VRAM:
- lambda_flow: 0.1 (smaller computation)
- lambda_rigidity: 0.0 (skip rigidity)
- coarse_warmup_iters: 3000 (standard)
- batch_size: 1 (smaller batches)

For High Quality:
- lambda_flow: 0.2-0.4 (balanced supervision)
- lambda_rigidity: 0.01-0.02 (fine smoothness)
- coarse_warmup_iters: 3000-5000
- downsample_factor: 4
- iterations: 40000-60000 (more total steps)
"""

# ============================================================================
# EXPECTED IMPROVEMENTS
# ============================================================================

"""
What you should observe with optical flow supervision:

Before Flow Supervision:
- Fast-moving objects become semi-transparent ("teleportation")
- Ghosting artifacts
- Inconsistent structure across frames
- PSNR plateaus early (~24-26 dB)

After Flow Supervision (with coarse-to-fine):
- Sharp, coherent object motion
- No teleportation artifacts
- Consistent Gaussian structure
- PSNR improvement of 1-3 dB
- More stable training curves
- Better generalization to unseen views

Typical PSNR progression:
- Iterations 0-3000 (coarse): 18-22 dB (rapid increase)
- Iterations 3000-5000 (transition): 22-24 dB (continuing increase)
- Iterations 5000+ (fine): 25-28 dB (slow refinement)
- Final: 26-30 dB (depends on dataset complexity)
"""

# ============================================================================
# TECHNICAL DETAILS
# ============================================================================

"""
How Optical Flow Loss Works:

1. Project Gaussian centers from time t to 2D image coordinates
2. Query deformation network for Gaussian positions at time t+1
3. Project these new positions to 2D
4. Compute 2D displacement: Δ₂D = proj(t+1) - proj(t)
5. Sample ground-truth flow at projected positions: F_gt
6. Compute L1 loss: L_flow = |Δ₂D - F_gt|
7. Weight by opacity: avoid optimizing transparent regions

How Rigidity Loss Works:

1. Sample random points in space
2. Add small perturbations (ε ≈ 0.1 units)
3. Query deformation network for both points
4. Compute deformation difference: Δdef = def(x) - def(x+ε)
5. Enforce smoothness: L_rigid = |Δdef|²
6. Encourages neighboring points to deform similarly

How Coarse-to-Fine Works:

1. Warmup phase (0-3000 iter): 1/4 resolution
   - Faster convergence on structure
   - Less memory, compute time
   
2. Transition phase (3000-5000 iter): Gradual resolution increase
   - Smooth handoff to full resolution
   - Avoids optimization jumps
   
3. Fine phase (5000+ iter): Full resolution
   - Refine details with flow supervision
   - Include rigidity regularization

Benefits:
- Faster initial training
- More stable overall convergence
- Better handling of motion
- Reduced artifacts
"""

# ============================================================================
# FILES CREATED/MODIFIED
# ============================================================================

"""
New files created:
1. utils/flow_utils.py - RAFT-based optical flow generation
2. utils/flow_loss_utils.py - Flow and rigidity loss functions
3. utils/coarse_to_fine_utils.py - Coarse-to-fine training utilities
4. utils/validation_utils.py - Validation and testing framework
5. FLOW_INTEGRATION_GUIDE.md - Detailed integration instructions

Modified files:
1. scene/cameras.py - Added flow_map loading
2. scene/dataset.py - Added flow_dir parameter

These changes are backward compatible. Existing code will work with flow_dir=None.
"""

# ============================================================================
# REFERENCES AND FURTHER READING
# ============================================================================

"""
Papers cited:
- RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (Teed & Deng, 2020)
- 4D Gaussian Splatting (Luiten et al., 2023)
- Dynamic Gaussian Avatars (Chen et al., 2023)
- Neural Radiance Fields (Mildenhall et al., 2020)

Optical Flow Concepts:
- Temporal consistency enforcement
- 2D projection of 3D motion
- Flow as weak supervision for motion
- Handling occlusions and disocclusions

4D Gaussian Splatting:
- Deformation networks for temporal modeling
- Position, scale, rotation updates over time
- Real-time rendering with temporal coherence
"""
