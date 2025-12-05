"""Collision detection for RRT path planning.

This module provides simplified collision detection using end-effector position only.
The robot arm joints can pass through obstacles; only the end-effector must avoid collisions.
"""

import numpy as np
from typing import List
from config import AABB
from kinematic_helpers import T01, T12, T23, T34, T45, T56


# =============================================================================
# Basic Point-in-AABB Collision Detection
# =============================================================================

def is_point_in_aabb(point: np.ndarray, aabb: AABB) -> bool:
    """Check if a 3D point is inside an axis-aligned bounding box.

    Args:
        point: [x, y, z] position in meters (robot frame)
        aabb: Axis-aligned bounding box with min/max corners

    Returns:
        True if point is inside AABB (including boundaries), False otherwise

    Example:
        >>> aabb = AABB(min=[0, 0, 0], max=[1, 1, 1])
        >>> is_point_in_aabb(np.array([0.5, 0.5, 0.5]), aabb)
        True
        >>> is_point_in_aabb(np.array([1.5, 0.5, 0.5]), aabb)
        False
    """
    return np.all(point >= aabb.min) and np.all(point <= aabb.max)


def is_point_in_collision(point: np.ndarray, obstacles: List[AABB]) -> bool:
    """Check if a 3D point collides with any obstacle in the list.

    Args:
        point: [x, y, z] position in meters (robot frame)
        obstacles: List of AABB obstacles to check against

    Returns:
        True if point is inside any obstacle, False otherwise

    Example:
        >>> from config import OBSTACLES
        >>> end_effector_pos = np.array([0.3, 0.5, 0.2])
        >>> is_point_in_collision(end_effector_pos, OBSTACLES)
        True  # If inside obstacle1
    """
    for obstacle in obstacles:
        if is_point_in_aabb(point, obstacle):
            return True
    return False


# =============================================================================
# Forward Kinematics
# =============================================================================

def get_end_effector_position(q: np.ndarray) -> np.ndarray:
    """Compute end-effector 3D position using forward kinematics.

    Chains the DH parameter transformations from base to end-effector
    and extracts the position component.

    Args:
        q: Joint configuration [q1, q2, q3, q4, q5, q6] in radians
           - q1: shoulder_pan_joint
           - q2: shoulder_lift_joint
           - q3: elbow_joint
           - q4: wrist_1_joint
           - q5: wrist_2_joint
           - q6: wrist_3_joint

    Returns:
        [x, y, z] position of end-effector in robot frame (meters)
        Origin is at robot base.

    Example:
        >>> q_home = np.array([0, 0, 0, 0, 0, 0])
        >>> pos = get_end_effector_position(q_home)
        >>> print(pos)  # Should be within UR5e workspace (~0.85m reach)

    Note:
        This returns position in ROBOT frame (origin at robot base),
        NOT world frame. Use config.world_to_robot_frame() for world coordinates.
    """
    # Ensure q is a numpy array
    q = np.asarray(q)

    # Chain transformation matrices from base to end-effector
    # T_base_to_ee = T01(q1) @ T12(q2) @ T23(q3) @ T34(q4) @ T45(q5) @ T56(q6)
    T_base_to_ee = T01(q[0]) @ T12(q[1]) @ T23(q[2]) @ T34(q[3]) @ T45(q[4]) @ T56(q[5])

    # Extract position from transformation matrix
    # Position is the first 3 elements of the 4th column (translation vector)
    position = T_base_to_ee[:3, 3]

    return position


# =============================================================================
# Configuration Collision Detection
# =============================================================================

def is_in_collision(q: np.ndarray, obstacles: List[AABB]) -> bool:
    """Check if a robot configuration causes end-effector to collide with obstacles.

    This is the main collision detection function used by RRT planner.
    Uses simplified collision model: only end-effector position is checked.
    Robot arm joints can pass through obstacles.

    Args:
        q: Joint configuration [q1, q2, q3, q4, q5, q6] in radians
        obstacles: List of AABB obstacles to check against (in robot frame)

    Returns:
        True if end-effector position is inside any obstacle, False otherwise

    Example:
        >>> from config import OBSTACLES
        >>> q_test = np.array([0, -np.pi/2, 0, 0, 0, 0])
        >>> if is_in_collision(q_test, OBSTACLES):
        ...     print("Configuration collides!")
        ... else:
        ...     print("Configuration is safe")
    """
    # Get end-effector position for this configuration
    ee_position = get_end_effector_position(q)

    # Check if position collides with any obstacle
    return is_point_in_collision(ee_position, obstacles)
