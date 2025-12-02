# HW4 Code - Vision-Based Grasping with Visual Servoing

## Overview
This folder contains code for homework 4, which implements **vision-based robotic grasping** using a UR5e manipulator with a wrist-mounted camera. The robot must locate, grasp, and place two colored blocks (red and blue) into a bin using **visual servoing** and **computer vision**.

## Task Description
- **Objective**: Pick up red block first, then blue block, and place both in a bin
- **Challenge**: Block locations vary between tests - must use vision to locate them
- **Robot**: UR5e manipulator (6-DOF) with wrist-mounted camera
- **Vision**: Camera provides BGR images; use color thresholding to detect blocks
- **Method**: Visual servoing to align gripper over blocks, then grasp
- **Constraints**:
  - Block orientations remain constant
  - Only modify `your_code_here.py`
  - Must implement `getRobotCommand(self, tt, current_q, current_image_bgr)` method signature

## Files

### `your_code_here.py`
The main implementation file containing a comprehensive vision-based grasping system:

#### **Core Structure: Finite State Machine (FSM)**
Implements an 8-phase FSM for each block (16 states total + completion):

**Red Block Pipeline (States 0-7)**:
1. `STATE_SEARCH_RED` - Move to overhead search pose, detect red block
2. `STATE_COARSE_ALIGN_RED` - Sequential X-Y alignment to center block in camera
3. `STATE_FINE_ALIGN_RED` - Descend to approach height, precise lateral alignment
4. `STATE_DESCEND_RED` - Controlled descent to grasp height
5. `STATE_GRASP_RED` - Close gripper while holding position
6. `STATE_LIFT_RED` - Lift block to safe carry height
7. `STATE_CARRY_RED` - Transport to bin using Cartesian motion
8. `STATE_RELEASE_RED` - Open gripper to drop block

**Blue Block Pipeline (States 8-15)**: Identical sequence for blue block

**Completion**: `STATE_DONE` - Task complete, hold position

#### **Class: `RobotPlacerWithVision`**

**`__init__()`**: Initializes FSM state and tracking variables
- State management (current state, timers, counters)
- Alignment tracking (coarse/fine phases, attempt counters)
- Motion tracking (descent, grasp, lift, carry timers)

**`getRobotCommand(tt, current_q, current_image_bgr)`**: Main control loop
- **Parameters**:
  - `tt`: Timestep counter (int)
  - `current_q`: Current joint angles [q1...q6] (radians)
  - `current_image_bgr`: Camera image (480×640×3) in BGR format
- **Returns**: `[q1, q2, q3, q4, q5, q6, gripper_bool]`
  - 6 joint angles (radians) + gripper state (True=closed, False=open)
- **Logic**: State machine dispatcher that executes appropriate behavior for current state

**`transition_to_state(new_state, tt)`**: State transition handler
- Logs state changes for debugging
- Resets phase-specific counters and timers

#### **Vision Functions**

**`process_image(bgr_image, color)`**: Detects colored blocks using HSV thresholding
- **Input**: BGR image, color ('red' or 'blue')
- **Output**: (cx, cy, detected) - centroid pixel coordinates and detection flag
- **Method**:
  - Convert BGR → HSV color space
  - Color thresholding with saturation filtering (filters brown table)
  - Red: Two ranges (0-10° and 170-180° hue) to handle wraparound
  - Blue: Single range (100-130° hue)
  - Morphological operations (closing/opening) to clean noise
  - Find largest contour, compute centroid using moments
- **Visualization**: Shows detected block, target position, and image center (debug mode)

**`compute_pixel_error(cx, cy)`**: Calculates pixel error from target
- Returns (err_x, err_y) in image frame

**`pixel_to_robot_motion(err_x, err_y, scale)`**: Converts pixel errors to robot motion
- **Camera rotation mapping** (90° rotation between camera and gripper):
  - Image X (horizontal) → Robot Y (forward/backward)
  - Image Y (vertical) → Robot X (left/right)
- Applies calibrated pixel-to-meter scaling factor

#### **Kinematics Functions**

**`get_full_fk(q)`**: Forward kinematics (base to end-effector)
- Chains transforms: T01 @ T12 @ T23 @ T34 @ T45 @ T56
- Returns 4×4 homogeneous transformation matrix

**`get_jacobian(q)`**: Computes 6×6 geometric Jacobian
- Top 3 rows: Linear velocity (J_v = z_i × (p_e - p_i))
- Bottom 3 rows: Angular velocity (J_ω = z_i)
- Used for IK-based Cartesian motion control

**`move_to_position(current_q, target_pos, max_step, dt)`**: Velocity-based IK controller
- Moves end-effector toward target position using Jacobian pseudoinverse
- Maintains gripper orientation (pointing down)
- Implements orientation correction using rotation vector
- Uses damped least squares for stability
- Clips velocities to prevent aggressive motion

#### **Hardcoded Reference Poses**

**`POSE_SEARCH_BLOCKS`**: Overhead search position
- Joint angles: `[-0.2, -1.4, 1.4, -1.55, -1.57, -0.2]`
- Gripper: Open
- Height: ~0.5m above table for wide field of view

**`POSE_OVER_BIN`**: Position above bin for dropping
- Joint angles: `[1.0, -1.05, 1.0, -1.2, -0.57, 0.0]`
- Gripper: Closed (until release)

**`HEIGHTS`**: Critical Z-coordinates (meters)
- `search_height`: 0.5m - Initial search altitude
- `approach_height`: 0.15m - Height for fine alignment
- `grasp_height`: 0.165m - Height to close gripper
- `lift_height`: 0.6m - Safe carrying altitude
- `bin_drop_height`: 0.5m - Height above bin

#### **Tunable Parameters**

**Vision (HSV thresholding)**:
- Red: Hue [0-10, 170-180], Saturation [160-255], Value [100-255]
- Blue: Hue [100-130], Saturation [160-255], Value [100-255]
- Min contour area: 100 pixels

**Coarse Alignment**:
- Step size: 20mm per move
- Tolerance: ±30 pixels (X and Y)
- Required stability: 5 consecutive frames

**Fine Alignment**:
- Approach height: 0.25m
- Step size: 3mm per move
- Tolerance: ±3 pixels (X), ±3 pixels (Y)
- Max attempts: 150

**Motion Control**:
- Descent step: 4mm per step
- Wait time: 15 timesteps between steps
- Grasp duration: 92 timesteps (~1.5 seconds)
- Lift wait: 3 timesteps between steps
- Carry max step: 10mm per timestep

**Visual Servoing**:
- Target pixel: (380, 240) for coarse alignment
- Fine target: (440, 240) - tuned to compensate for camera offset
- Pixel-to-meter scale: 0.0002 m/pixel (approximate at 0.5m height)

### `ur5e_controller.py`
Webots robot controller that interfaces with the UR5e:
- Initializes 6-DOF arm motors with position sensors
- Controls Robotiq 2F-85 gripper (left/right finger motors)
- Manages wrist-mounted camera (640×480 resolution)
- Converts BGRA camera output to BGR for OpenCV
- Main loop:
  - Captures camera image
  - Calls `getRobotCommand()` from `RobotPlacerWithVision`
  - Commands joint motors and gripper actuators
- Provides live camera preview using OpenCV (optional)

### `kinematic_helpers.py`
UR5e DH-parameter transformation matrices:
- `T01(q)` through `T56(q)`: Individual joint transforms
- Based on UR5e specifications (link lengths, joint offsets)
- Used for forward kinematics and Jacobian calculation

## Key Concepts

### Visual Servoing
Closed-loop control using visual feedback to align the gripper:
1. **Detect**: Locate block centroid in image coordinates
2. **Error**: Compute pixel offset from target position
3. **Motion**: Convert pixel error to robot motion (accounting for camera rotation)
4. **Move**: Command robot to reduce error
5. **Iterate**: Repeat until aligned within tolerance

### Two-Phase Alignment Strategy
1. **Coarse Alignment** (at search height ~0.5m):
   - Sequential X then Y alignment
   - Large steps (20mm), loose tolerance (±30 pixels)
   - Fast convergence to rough alignment

2. **Fine Alignment** (at approach height ~0.15m):
   - Descend to lower altitude first (better pixel resolution)
   - Simultaneous X-Y alignment
   - Small steps (3mm), tight tolerance (±3 pixels)
   - Precise positioning for reliable grasp

### HSV Color Detection
Uses HSV color space for robust color segmentation:
- **Hue**: Primary color identifier (red vs blue)
- **Saturation**: Filters out low-saturation colors (brown table)
- **Value**: Brightness threshold
- **Red wraparound**: Hue wraps at 180°, requires two ranges
- **Morphological ops**: Remove noise, fill gaps in detected regions

### Inverse Kinematics for Cartesian Control
Velocity-based IK using Jacobian:
- Computes joint velocities from desired Cartesian velocity
- Uses damped pseudoinverse for singularity avoidance
- Maintains orientation constraint (gripper pointing down)
- Enables smooth Cartesian trajectories (carrying, descent)

## State Machine Flow

```
Red Block:  SEARCH → COARSE_ALIGN → FINE_ALIGN → DESCEND → GRASP → LIFT → CARRY → RELEASE
Blue Block: SEARCH → COARSE_ALIGN → FINE_ALIGN → DESCEND → GRASP → LIFT → CARRY → RELEASE
Final:      DONE
```

Each state implements specific behavior and transitions to the next state when its completion criteria are met.

## Implementation Notes
- **All angles in radians**, positions in meters
- **Camera frame vs robot frame**: 90° rotation must be handled in pixel-to-motion conversion
- **Rate limiting**: Descent/lift motions use counters to slow down movements
- **Stability requirements**: Alignment states require multiple consecutive frames within tolerance
- **Debug output**: Extensive logging every 30-50 timesteps for troubleshooting
- **Gripper state**: Carried through state transitions (closed during carry, open otherwise)
- **Robust to block placement**: Visual servoing adapts to different block locations

## Success Criteria
The robot should successfully:
1. Locate red block using vision
2. Align gripper precisely over red block
3. Grasp and transport red block to bin
4. Return and locate blue block
5. Grasp and transport blue block to bin
6. Complete task with both blocks in bin

Works reliably across different block placements on the gray mat.
