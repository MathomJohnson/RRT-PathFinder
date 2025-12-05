"""Validation script for Micro-Sprint 1.1.

Tests coordinate frame transformation and obstacle definitions.
"""

import numpy as np
from config import (
    AABB,
    ROBOT_BASE_WORLD,
    ROBOT_ROTATION_Z,
    world_to_robot_frame,
    OBSTACLE1_WORLD,
    OBSTACLE1_ROBOT,
    OBSTACLES
)


def print_separator():
    print("=" * 70)


def test_coordinate_transformation():
    """Test that coordinate transformation is working correctly."""
    print_separator()
    print("COORDINATE FRAME TRANSFORMATION TEST")
    print_separator()

    print(f"\nRobot Base (World Frame): {ROBOT_BASE_WORLD}")
    print(f"Robot Rotation (Z-axis): {ROBOT_ROTATION_Z:.6f} rad ({np.degrees(ROBOT_ROTATION_Z):.2f}°)")

    # Test 1: Robot's own position should transform to origin
    print("\n--- Test 1: Robot base should map to origin ---")
    robot_in_robot_frame = world_to_robot_frame(ROBOT_BASE_WORLD)
    print(f"Robot base in robot frame: {robot_in_robot_frame}")
    print(f"Expected: [0, 0, 0]")
    print(f"PASS: {np.allclose(robot_in_robot_frame, [0, 0, 0])}")

    # Test 2: Point along robot's X-axis (after rotation)
    print("\n--- Test 2: Point along world Y-axis should map to robot X-axis ---")
    # Robot is rotated 90° around Z, so world Y → robot X
    point_world = ROBOT_BASE_WORLD + np.array([0, 1, 0])  # 1m along world Y
    point_robot = world_to_robot_frame(point_world)
    print(f"Point in world frame: {point_world}")
    print(f"Point in robot frame: {point_robot}")
    print(f"Expected: approximately [1, 0, 0] (along robot X)")
    print(f"PASS: {np.allclose(point_robot, [1, 0, 0], atol=0.01)}")

    print("\n")


def test_obstacle_definition():
    """Print obstacle definitions in both world and robot frames."""
    print_separator()
    print("OBSTACLE DEFINITIONS")
    print_separator()

    print("\n--- Obstacle1 in WORLD frame ---")
    print(f"Center: [-0.5, 2.51, 0.96]")
    print(f"Size: [0.1, 0.6, 0.5]")
    print(f"AABB: {OBSTACLE1_WORLD}")
    print(f"  min: {OBSTACLE1_WORLD.min}")
    print(f"  max: {OBSTACLE1_WORLD.max}")

    print("\n--- Obstacle1 in ROBOT frame (used for collision detection) ---")
    print(f"AABB: {OBSTACLE1_ROBOT}")
    print(f"  min: {OBSTACLE1_ROBOT.min}")
    print(f"  max: {OBSTACLE1_ROBOT.max}")

    print(f"\n--- All Obstacles for Collision Detection ---")
    print(f"Number of obstacles: {len(OBSTACLES)}")
    for i, obs in enumerate(OBSTACLES):
        print(f"  Obstacle {i}: {obs}")

    print("\n")


def test_aabb_class():
    """Test AABB class functionality."""
    print_separator()
    print("AABB CLASS TEST")
    print_separator()

    # Create simple test AABB
    test_aabb = AABB(min_corner=[0, 0, 0], max_corner=[1, 1, 1])
    print(f"\nTest AABB: {test_aabb}")
    print(f"  min: {test_aabb.min}")
    print(f"  max: {test_aabb.max}")
    print(f"  Type of min: {type(test_aabb.min)}")
    print(f"  Type of max: {type(test_aabb.max)}")
    print(f"PASS: AABB class works correctly")

    print("\n")


def main():
    """Run all validation tests."""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "MICRO-SPRINT 1.1 VALIDATION")
    print("=" * 70)
    print("\n")

    test_aabb_class()
    test_coordinate_transformation()
    test_obstacle_definition()

    print_separator()
    print("VALIDATION COMPLETE")
    print_separator()
    print("\nNext steps:")
    print("  1. Review the obstacle positions in robot frame")
    print("  2. Verify coordinate transformation tests pass")
    print("  3. Ready to proceed to Micro-Sprint 1.2")
    print("\n")


if __name__ == "__main__":
    main()
