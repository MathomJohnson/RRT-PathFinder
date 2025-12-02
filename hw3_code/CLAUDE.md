# HW3 Code - Trajectory Generation with Cubic Polynomials

## Overview
This folder contains code for homework 3, which focuses on smooth trajectory generation for robot manipulation. The task is to pick up two blocks and place them in a bin using **third-order polynomial motions** instead of direct joint angle commands.

## Task Description
- **Objective**: Pick up two blocks and place them in a bin on the table
- **Key Constraint**: Joint velocities must not exceed 0.5 rad/s
- **Method**: Use third-order polynomial trajectory generation between keyframe joint configurations
- **Simulation**: Discrete timestep simulation (0.016s, 62.5 Hz)

## Key Difference from HW1
Unlike HW1 which directly commanded joint angles (causing fast, high-acceleration motions), this implementation uses **smooth cubic trajectory interpolation** to ensure controlled, continuous motion with zero velocity at start and end of each segment.

## Files

### `webots_your_code_here.py`
The main implementation file containing:
- **`get_cubic_s(t_current, T_total)`**: Computes the cubic polynomial scaling factor s(t) = 3(t/T)² - 2(t/T)³
- **Keyframe definitions**: Predefined joint angle configurations for the complete pick-and-place sequence
  - Block 1: home → pre-grasp → grasp → lift → bin → release
  - Block 2: pre-grasp → grasp → lift → bin → release
- **Segment duration computation**: Pre-calculates timing for each motion segment based on max joint velocity constraint
- **`getDesiredRobotCommand(tt)`**: State machine that returns smooth joint angle commands at each timestep

### `panda_controller.py`
Webots robot controller that:
- Interfaces with the Panda robot (7-DOF arm + gripper)
- Calls `getDesiredRobotCommand()` at each timestep
- Commands joint motors and gripper actuators

### `pygame_your_code_here.py`
Contains an RRT (Rapidly-exploring Random Tree) path planning implementation. This appears to be a separate exercise or testing code.

## Trajectory Generation Details

**Cubic polynomial**: q(t) = q_start + s(t) × (q_end - q_start)
- **s(t)**: Scaling factor from 0 to 1
- **Velocity**: Zero at t=0 and t=T (smooth starts/stops)
- **Duration T**: Automatically calculated to satisfy the 0.5 rad/s velocity limit

**Duration calculation**:
```
T = 3.0 × max_Δθ / max_velocity
N_timesteps = ceil(T / timestep) + 1
```

## Usage Notes
- All joint angles are in radians
- Gripper actions involve pausing at a keyframe (50 timesteps ≈ 0.8 seconds)
- The system pre-computes all segment durations before execution begins
- Motions are executed sequentially through a state machine indexed by segment
