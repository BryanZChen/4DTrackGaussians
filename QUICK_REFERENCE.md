"""
QUICK REFERENCE CARD: Optical Flow Supervision for 4D-GS
═══════════════════════════════════════════════════════════════════════════
Print this out or keep it open while integrating!
"""

# ============================================================================
# 60-SECOND OVERVIEW
# ============================================================================

OVERVIEW = """
Problem: Fast-moving objects cause "teleportation" (semi-transparency) in 4D-GS
Solution: Add optical flow supervision to guide motion learning

6 Phases:
1. Generate flow with RAFT
2. Load flow into Camera.flow_map
3. Add compute_flow_loss() to training
4. Add compute_rigidity_loss() to training
5. Use CoarseToFineScheduler for multi-resolution training
6. Validate with 3 tests

Result: +1-3 dB PSNR, coherent fast motion, stable training
"""

# ============================================================================
# FILE REFERENCE CARD
# ============================================================================

FILES = """
NEW FILES:
──────────────────────────────────────────────────────────────────────────
utils/flow_utils.py                  Generate optical flow from RAFT
utils/flow_loss_utils.py             Flow & rigidity loss functions
utils/coarse_to_fine_utils.py        Scheduling & resolution management
utils/validation_utils.py            Validation tests & visualization

MODIFIED FILES:
──────────────────────────────────────────────────────────────────────────
scene/cameras.py                     + flow_map attribute
scene/dataset.py                     + flow_dir parameter

DOCUMENTATION:
──────────────────────────────────────────────────────────────────────────
OPTICAL_FLOW_README.md               Complete reference
OPTICAL_FLOW_QUICKSTART.md           Quick start + troubleshooting
FLOW_INTEGRATION_GUIDE.md            Detailed integration steps
example_flow_training.py             Complete example trainer
IMPLEMENTATION_SUMMARY.txt           This summary
"""

# ============================================================================
# COMMAND CHEAT SHEET
# ============================================================================

COMMANDS = """
STEP 1: Generate Flow
──────────────────────────────────────────────────────────────────────────
python -m utils.flow_utils ./data/images --output_dir flow

Options:
  --output_dir flow              Where to save .npy files
  --device cuda                  Device (cuda/cpu)
  --no_viz                       Skip PNG visualizations
  --max_frames 100               Only process first 100 frames (for testing)


STEP 2: Start Training
──────────────────────────────────────────────────────────────────────────
python train.py \\
    --source_path data \\
    --model_path output \\
    --flow_dir flow \\
    --lambda_flow 0.2 \\
    --lambda_rigidity 0.01 \\
    --coarse_warmup_iters 3000 \\
    --downsample_factor 4 \\
    --iterations 30000

Key parameters:
  --flow_dir flow                Directory with flow_*.npy files
  --lambda_flow 0.2              Flow loss weight (0.1-0.5)
  --lambda_rigidity 0.01         Rigidity loss weight (0.01-0.05)
  --coarse_warmup_iters 3000     Iterations at 1/4 resolution
  --downsample_factor 4          Resolution reduction (2-8)


STEP 3: Validate Results
──────────────────────────────────────────────────────────────────────────
python -c "
from utils.validation_utils import FlowValidation
validator = FlowValidation()

# Collect metrics during training and pass here
result1 = validator.test_flow_loss_decreasing(flow_losses)
result2 = validator.test_gaussian_tracking_alignment(...)
result3 = validator.test_coarse_to_fine_transition(...)

validator.generate_report({'Test1': result1, 'Test2': result2, 'Test3': result3})
"

Output: validation/validation_report.json + PNG visualizations
"""

# ============================================================================
# CODE SNIPPETS
# ============================================================================

SNIPPETS = """
SNIPPET 1: Import All Utilities
──────────────────────────────────────────────────────────────────────────
from utils.flow_loss_utils import compute_flow_loss, compute_rigidity_loss
from utils.coarse_to_fine_utils import (
    CoarseToFineScheduler,
    apply_coarse_to_fine_preprocessing,
    sample_gaussian_neighborhood_points
)


SNIPPET 2: Initialize Scheduler
──────────────────────────────────────────────────────────────────────────
scheduler = CoarseToFineScheduler(
    warmup_iters=3000,
    downsample_factor=4,
    lambda_flow_start=0.1,
    lambda_flow_end=0.5,
    lambda_rigidity=0.01
)


SNIPPET 3: Per-Iteration Loss Computation
──────────────────────────────────────────────────────────────────────────
# In your training loop:
for iteration in range(num_iters):
    # Get scheduling state
    resolution_scale = scheduler.get_resolution_scale(iteration)
    loss_weights = scheduler.get_loss_weights(iteration)
    
    # Base loss
    l1_loss = l1_loss_fn(rendered, gt_images)
    total_loss = l1_loss
    
    # Flow loss (after warmup)
    if loss_weights['flow_enabled'] and camera.flow_map is not None:
        flow_loss = compute_flow_loss(
            gaussian_centers=gaussians.get_xyz,
            deformation_net=gaussians._deformation,
            viewpoint_cam=camera,
            time_t=camera.time,
            time_next=camera.time + 1.0,
            opacity=gaussians.get_opacity,
            flow_map=camera.flow_map
        )
        total_loss = total_loss + loss_weights['flow'] * flow_loss
    
    # Rigidity loss (fine stage)
    if stage == "fine":
        sampled_points = sample_gaussian_neighborhood_points(
            gaussians.get_xyz,
            num_samples_per_gaussian=3,
            neighborhood_radius=0.5
        )
        time_codes = torch.full(
            (sampled_points.shape[0], 1),
            camera.time,
            device=device
        )
        rigidity_loss = compute_rigidity_loss(
            gaussians._deformation,
            sampled_points,
            time_codes
        )
        total_loss = total_loss + loss_weights['rigidity'] * rigidity_loss
    
    # Update
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()


SNIPPET 4: Verify Flow Loading
──────────────────────────────────────────────────────────────────────────
# Test that flow is loaded correctly
for camera in cameras:
    assert hasattr(camera, 'flow_map'), "flow_map not found!"
    assert camera.flow_map.shape == (2, camera.image_height, camera.image_width)
    print(f"Flow shape: {camera.flow_map.shape}, mean: {camera.flow_map.mean():.4f}")
"""

# ============================================================================
# DEBUGGING CHECKLIST
# ============================================================================

DEBUGGING = """
Flow Loss = 0.0?
□ Check flow files exist:    ls -la flow/flow_*.npy
□ Check dataset passes flow: print(camera.flow_map.shape)
□ Check loss_weights:         print(loss_weights['flow_enabled'])
□ Print actual loss values:   print(f"Flow loss: {flow_loss}")


Flow Loss = NaN?
□ Reduce lambda_flow to 0.05
□ Check flow value range:     print(flow_map.min(), flow_map.max())
□ Verify projection correct:  print(projection_matrix)
□ Check for divide by zero:   add eps to denominators


Training Crashes at Iter 3000?
□ This is normal (resolution transition)
□ Check if loss recovers after 500 iters
□ If still increasing: check focal length scaling


Visible Artifacts at Resolution Switch?
□ Extend transition window from 1000 to 2000 iters
□ Smooth FoV ramp manually
□ Verify adjust_camera_intrinsics() is correct


Memory Issues?
□ Reduce batch_size to 1
□ Set lambda_rigidity = 0
□ Reduce num_samples_per_gaussian to 1
□ Increase downsample_factor to 8
"""

# ============================================================================
# PERFORMANCE TARGETS
# ============================================================================

TARGETS = """
GOOD SIGNS (Training is Working)
─────────────────────────────────────────────────────────────────────────
✓ Flow loss starts at 0.2-0.5, decreases to 0.05-0.1
✓ No NaN values in any loss term
✓ Rigidity loss stabilizes around 0.001-0.01
✓ Total loss shows smooth decrease (minor spike at iter 3000)
✓ PSNR increases from ~18 dB (coarse) to ~26-30 dB (fine)
✓ Gaussian structure looks coherent in renders


BAD SIGNS (Needs Tuning)
─────────────────────────────────────────────────────────────────────────
✗ Flow loss stays at 0.0 → check flow files
✗ Flow loss = NaN → reduce lambda_flow
✗ Total loss diverges → reduce learning rate or lambda_flow
✗ PSNR doesn't improve after warmup → check camera.time values
✗ Memory errors → reduce batch_size or downsample_factor
✗ Ghosting artifacts → increase lambda_flow or lambda_rigidity


VALIDATION TARGETS (After Training)
─────────────────────────────────────────────────────────────────────────
Test 1: Flow Progression
  Expected: PASS (>10% loss decrease)
  Status:   Loss curve should be smooth and monotonically decreasing

Test 2: Gaussian Tracking
  Expected: >70% accuracy (Gaussians within 2 pixels of ground truth)
  Visualizes: Gaussian motion vectors overlaid on optical flow

Test 3: Coarse-to-Fine Transition
  Expected: PASS (L1 error <0.05, intensity diff <5%)
  Checks: Consistency between coarse and fine resolutions
"""

# ============================================================================
# HYPERPARAMETER TEMPLATES
# ============================================================================

TEMPLATES = """
FOR FAST-MOVING OBJECTS (e.g., dancing, sports)
──────────────────────────────────────────────────────────────────────────
--lambda_flow 0.4              Strong flow supervision
--lambda_rigidity 0.03         Tight smoothness
--coarse_warmup_iters 5000     Longer warmup
--downsample_factor 4          Standard 4x


FOR SUBTLE MOTION (e.g., slow rotations, breathing)
──────────────────────────────────────────────────────────────────────────
--lambda_flow 0.1              Light flow supervision
--lambda_rigidity 0.005        Minimal smoothness
--coarse_warmup_iters 2000     Shorter warmup
--downsample_factor 2          Less downsampling


FOR MEMORY-CONSTRAINED SETUP
──────────────────────────────────────────────────────────────────────────
--lambda_flow 0.1              Conservative
--lambda_rigidity 0.0          Skip rigidity (save memory)
--coarse_warmup_iters 3000     Standard
--downsample_factor 8          Max downsampling
--batch_size 1                 Single sample


FOR HIGH-QUALITY RESULTS
──────────────────────────────────────────────────────────────────────────
--lambda_flow 0.3              Balanced
--lambda_rigidity 0.015        Moderate smoothness
--coarse_warmup_iters 5000     Longer coarse phase
--downsample_factor 4          Standard
--iterations 50000             More training iterations
"""

# ============================================================================
# KEY FUNCTIONS QUICK REF
# ============================================================================

FUNCTIONS = """
From utils/flow_utils.py
──────────────────────────────────────────────────────────────────────────
generate_optical_flow_dataset(image_dir, output_dir, device='cuda')
→ Generates flow_*.npy files from image sequence


From utils/flow_loss_utils.py
──────────────────────────────────────────────────────────────────────────
compute_flow_loss(gaussian_centers, deformation_net, viewpoint_cam, ...)
→ Returns scalar flow loss term

compute_rigidity_loss(deformation_net, sampled_points, time_codes, ...)
→ Returns scalar rigidity loss term


From utils/coarse_to_fine_utils.py
──────────────────────────────────────────────────────────────────────────
CoarseToFineScheduler(warmup_iters=3000, downsample_factor=4, ...)
  .is_warmup(iteration) → bool
  .get_resolution_scale(iteration) → float [0, 1]
  .get_loss_weights(iteration) → dict

downsample_image(image, factor)
→ Downsample tensor by average pooling

adjust_camera_intrinsics(camera, resolution_scale)
→ Return adjusted camera with scaled FoV


From utils/validation_utils.py
──────────────────────────────────────────────────────────────────────────
FlowValidation(output_dir='validation')
  .test_flow_loss_decreasing(flow_losses, ...) → dict
  .test_gaussian_tracking_alignment(pos_t0, pos_t1, flow, ...) → dict
  .test_coarse_to_fine_transition(images_coarse, images_fine, ...) → dict
  .generate_report(test_results) → Generates JSON + PNGs
"""

# ============================================================================
# DOCUMENTATION LINKS
# ============================================================================

DOCS = """
Main Guides:
  OPTICAL_FLOW_README.md         ← Start here (comprehensive reference)
  OPTICAL_FLOW_QUICKSTART.md     ← Practical guide + troubleshooting
  FLOW_INTEGRATION_GUIDE.md      ← Detailed train.py integration

Examples:
  example_flow_training.py       ← Complete trainer class
  
Source Code (inline docs):
  utils/flow_utils.py            ← Flow generation with docstrings
  utils/flow_loss_utils.py       ← Loss functions with formulas
  utils/coarse_to_fine_utils.py  ← Scheduling with examples
  utils/validation_utils.py      ← Validation tests with output specs

This File:
  QUICK_REFERENCE.md             ← You are here
"""

print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║         QUICK REFERENCE: Optical Flow Supervision for 4D-GS            ║
╚════════════════════════════════════════════════════════════════════════╝

📋 60-SECOND OVERVIEW
{OVERVIEW}

📁 FILE STRUCTURE
{FILES}

💻 COMMANDS
{COMMANDS}

🔧 CODE SNIPPETS
{SNIPPETS}

🐛 DEBUGGING
{DEBUGGING}

🎯 PERFORMANCE TARGETS
{TARGETS}

⚙️  HYPERPARAMETER TEMPLATES
{TEMPLATES}

🔍 FUNCTION REFERENCE
{FUNCTIONS}

📖 DOCUMENTATION
{DOCS}

═══════════════════════════════════════════════════════════════════════════
Print this page and keep it handy while implementing!
For detailed information, see OPTICAL_FLOW_README.md
═══════════════════════════════════════════════════════════════════════════
""")
