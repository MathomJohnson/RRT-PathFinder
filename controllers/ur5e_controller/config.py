"""Configuration for RRT path planner.

Defines coordinate frames, obstacles, and planning parameters.
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# AABB (Axis-Aligned Bounding Box) Class
# =============================================================================

class AABB:
    """Axis-Aligned Bounding Box for obstacle representation."""

    def __init__(self, min_corner: np.ndarray, max_corner: np.ndarray):
        """Initialize AABB with min and max corners.

        Args:
            min_corner: [x_min, y_min, z_min] in meters
            max_corner: [x_max, y_max, z_max] in meters
        """
        self.min = np.array(min_corner, dtype=float)
        self.max = np.array(max_corner, dtype=float)

    def __repr__(self):
        return f"AABB(min={self.min}, max={self.max})"


# =============================================================================
# Robot Pose in World Frame (from hw2_ur5e.wbt)
# =============================================================================

# UR5e base position in world coordinates
ROBOT_BASE_WORLD = np.array([0.0, 2.2, 0.7])

# UR5e rotation around Z-axis (radians)
ROBOT_ROTATION_Z = 1.570846325  # ~90 degrees


# =============================================================================
# Coordinate Frame Transformation
# =============================================================================

def world_to_robot_frame(point_world: np.ndarray) -> np.ndarray:
    """Transform a point from world frame to robot's local frame.

    The robot is:
    - Translated to (0, 2.2, 0.7) in world frame
    - Rotated 90° around Z-axis

    Transformation steps:
    1. Subtract robot base position (translation)
    2. Apply inverse rotation (-90° around Z)

    Args:
        point_world: [x, y, z] position in world coordinates (meters)

    Returns:
        [x, y, z] position in robot's local frame (meters)
    """
    # Step 1: Translate to robot's origin
    point_relative = point_world - ROBOT_BASE_WORLD

    # Step 2: Rotate by -90° around Z (inverse of robot's rotation)
    angle = -ROBOT_ROTATION_Z
    x_rel, y_rel, z_rel = point_relative

    x_robot = x_rel * np.cos(angle) - y_rel * np.sin(angle)
    y_robot = x_rel * np.sin(angle) + y_rel * np.cos(angle)
    z_robot = z_rel

    return np.array([x_robot, y_robot, z_robot])


# =============================================================================
# Obstacle Definitions
# =============================================================================

# --- Obstacle1 (from hw2_ur5e.wbt) ---
# Webots definition:
#   translation: -0.5 2.51 0.96 (center)
#   size: 0.1 0.6 0.5 (width, length, height)

# Calculate AABB corners in WORLD frame
OBSTACLE1_CENTER_WORLD = np.array([-0.5, 2.51, 0.96])
OBSTACLE1_SIZE = np.array([0.1, 0.6, 0.5])
OBSTACLE1_HALF_SIZE = OBSTACLE1_SIZE / 2.0

OBSTACLE1_MIN_WORLD = OBSTACLE1_CENTER_WORLD - OBSTACLE1_HALF_SIZE
OBSTACLE1_MAX_WORLD = OBSTACLE1_CENTER_WORLD + OBSTACLE1_HALF_SIZE

# Create AABB in WORLD frame (for reference)
OBSTACLE1_WORLD = AABB(OBSTACLE1_MIN_WORLD, OBSTACLE1_MAX_WORLD)

# Transform all 8 corners to ROBOT frame
obstacle1_corners_world = [
    OBSTACLE1_MIN_WORLD,
    OBSTACLE1_MAX_WORLD,
    [OBSTACLE1_MIN_WORLD[0], OBSTACLE1_MIN_WORLD[1], OBSTACLE1_MAX_WORLD[2]],
    [OBSTACLE1_MIN_WORLD[0], OBSTACLE1_MAX_WORLD[1], OBSTACLE1_MIN_WORLD[2]],
    [OBSTACLE1_MIN_WORLD[0], OBSTACLE1_MAX_WORLD[1], OBSTACLE1_MAX_WORLD[2]],
    [OBSTACLE1_MAX_WORLD[0], OBSTACLE1_MIN_WORLD[1], OBSTACLE1_MIN_WORLD[2]],
    [OBSTACLE1_MAX_WORLD[0], OBSTACLE1_MIN_WORLD[1], OBSTACLE1_MAX_WORLD[2]],
    [OBSTACLE1_MAX_WORLD[0], OBSTACLE1_MAX_WORLD[1], OBSTACLE1_MIN_WORLD[2]],
]

obstacle1_corners_robot = [world_to_robot_frame(np.array(corner)) for corner in obstacle1_corners_world]

# Create AABB in ROBOT frame (for collision detection)
OBSTACLE1_MIN_ROBOT = np.min(obstacle1_corners_robot, axis=0)
OBSTACLE1_MAX_ROBOT = np.max(obstacle1_corners_robot, axis=0)
OBSTACLE1_ROBOT = AABB(OBSTACLE1_MIN_ROBOT, OBSTACLE1_MAX_ROBOT)

# List of all obstacles in ROBOT frame (used for collision detection)
OBSTACLES = [OBSTACLE1_ROBOT]


# =============================================================================
# UR5e Joint Limits (radians)
# =============================================================================

JOINT_LIMITS = [
    (-2*np.pi, 2*np.pi),  # shoulder_pan_joint
    (-2*np.pi, 2*np.pi),  # shoulder_lift_joint
    (-2*np.pi, 2*np.pi),  # elbow_joint
    (-2*np.pi, 2*np.pi),  # wrist_1_joint
    (-2*np.pi, 2*np.pi),  # wrist_2_joint
    (-2*np.pi, 2*np.pi),  # wrist_3_joint
]


# =============================================================================
# RRT Planning Parameters (will be tuned in Sprint 4)
# =============================================================================

RRT_STEP_SIZE = 0.1          # Radians to step toward sampled config
RRT_MAX_ITERATIONS = 5000    # Max iterations before giving up
RRT_GOAL_BIAS = 0.1          # 10% probability of sampling goal
RRT_GOAL_THRESHOLD = 0.2     # Distance to goal to declare success (radians)


# =============================================================================
# Target Locations (will be defined in Sprint 3)
# =============================================================================

# Bin location (world coordinates) - to be determined
BIN_POSITION_WORLD = None  # Will be filled in Sprint 3
BIN_POSITION_ROBOT = None  # Will be filled in Sprint 3
