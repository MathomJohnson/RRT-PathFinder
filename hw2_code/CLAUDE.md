# HW2 Code - Inverse Kinematics with Newton-Raphson Method

## Overview
This folder contains code for homework 2, which implements **inverse kinematics (IK)** for a UR5e manipulator using the **Newton-Raphson method**. The robot's task is to trace a "W" pattern (for UW-Madison) by following a predefined Cartesian path.

## Task Description
- **Objective**: Implement inverse kinematics to make the UR5e end-effector track a red marker that traces a "W" shape
- **Method**: Newton-Raphson method (Damped Least Squares variant)
- **Robot**: UR5e manipulator (6-DOF arm)
- **Input**: Desired 6D pose (position + orientation as axis-angle)
- **Output**: Joint angles (6 values) to achieve the desired pose

## Files

### `your_code_here.py`
The main implementation file containing the IK solver:

#### **`getDesiredRobotCommand(tt, desired_pose, current_q)`**
The core IK function called at each timestep:
- **Parameters**:
  - `tt`: Current simulation timestep (0.016s intervals, 62.5 Hz)
  - `desired_pose`: List of 6 values - [x, y, z, rx, ry, rz]
    - Position (3): Desired Cartesian coordinates in meters
    - Orientation (3): Desired rotation as axis-angle (exponential coordinates)
  - `current_q`: Current/last commanded joint angles (6 values in radians)
- **Returns**: List of 6 joint angles in radians

#### **Newton-Raphson Implementation**
Uses **Damped Least Squares (DLS)** method:
1. Compute forward kinematics to get current pose
2. Calculate 6D error (position + orientation)
3. Compute geometric Jacobian at current configuration
4. Solve: `Δq = J^T × (J×J^T + λ²I)^(-1) × error`
5. Update: `q_new = q_old + Δq`
6. Iterate (default: 10 iterations per timestep)

**Tunable Parameters**:
- `num_iterations = 10` - Number of IK iterations per timestep
- `damping = 0.1` - Damping factor (λ) for numerical stability

#### **Helper Functions**:
- **`get_full_fk(q)`**: Computes full forward kinematics (base to end-effector transform)
  - Returns 4×4 homogeneous transformation matrix T_06
  - Chains all joint transforms: T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56

- **`get_jacobian(q)`**: Calculates the 6×6 geometric Jacobian
  - Top 3 rows: Linear velocity (J_v = z_i × (p_e - p_i))
  - Bottom 3 rows: Angular velocity (J_ω = z_i)
  - Important: Includes base frame (T_00 = I) for first joint rotation

### `kinematic_helpers.py`
Contains DH-parameter-based transformation matrices for each UR5e joint:
- `T01(q)` through `T56(q)`: Individual joint transformations
- Each function takes a single joint angle and returns a 4×4 homogeneous transform
- Based on UR5e specifications with link lengths and offsets

### `ur5e_controller.py`
Webots robot controller that orchestrates the simulation:
- **`getWPose(tt)`**: Defines the "W" trajectory path
  - 5 segments over 800 timesteps
  - Linear interpolation between waypoints
  - Returns desired pose as [x, y, z, 0, 0, 0] (zero rotation)
- **`visualizeCommand(pose)`**: Updates visual marker position to show the target
- Main loop:
  - Gets desired pose from trajectory
  - Calls `getDesiredRobotCommand()` for IK solution
  - Commands joint motors with computed angles

## Key Concepts

### Newton-Raphson for IK
Iteratively solves: **J(q) × Δq = error**
- **Jacobian J(q)**: Maps joint velocities to end-effector velocities
- **Error**: 6D difference between desired and current pose
- **Damped Least Squares**: Adds λ²I term for numerical stability and singularity avoidance

### Orientation Representation
- **Input**: Axis-angle representation (rotation vector)
- **Conversion**: Uses `scipy.spatial.transform.Rotation` to convert between rotation matrices and axis-angle
- **Error calculation**: Difference between desired and current rotation vectors

## Implementation Notes
- All joint angles are in **radians**
- Position values are in **meters**
- The Jacobian must include the base frame (identity matrix) as the first frame
- Damped Least Squares prevents instability near singularities
- Multiple iterations per timestep improve tracking accuracy
- The robot does not need to use the timestep value `tt` directly (path is predefined)

## Success Criteria
The robot should smoothly trace the "W" pattern by tracking the red marker through inverse kinematics, demonstrating successful implementation of the Newton-Raphson method.
