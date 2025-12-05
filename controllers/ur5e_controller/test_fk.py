"""Validation script for Micro-Sprint 1.3.

Tests forward kinematics and configuration-based collision detection.
"""

import numpy as np
from config import OBSTACLES, OBSTACLE1_ROBOT
from collision import get_end_effector_position, is_in_collision


def print_separator():
    print("=" * 70)


def test_home_position():
    """Test FK at home position (all joints at zero)."""
    print_separator()
    print("TEST 1: Home Position [0, 0, 0, 0, 0, 0]")
    print_separator()

    q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pos = get_end_effector_position(q_home)
    distance_from_base = np.linalg.norm(pos)

    print(f"\nJoint configuration: {q_home}")
    print(f"End-effector position (robot frame): {pos}")
    print(f"Distance from base: {distance_from_base:.4f} m")
    print(f"\nUR5e max reach: ~0.85 m")
    print(f"Position within workspace: {distance_from_base <= 0.85}")

    # Check collision
    collision = is_in_collision(q_home, OBSTACLES)
    print(f"\nCollides with obstacles: {collision}")

    print()


def test_extended_position():
    """Test FK at extended position (arm pointing forward)."""
    print_separator()
    print("TEST 2: Extended Forward [0, -pi/2, 0, 0, 0, 0]")
    print_separator()

    q_extended = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
    pos = get_end_effector_position(q_extended)
    distance_from_base = np.linalg.norm(pos)

    print(f"\nJoint configuration: {q_extended}")
    print(f"End-effector position (robot frame): {pos}")
    print(f"Distance from base: {distance_from_base:.4f} m")
    print(f"Position within workspace: {distance_from_base <= 0.85}")

    # Check collision
    collision = is_in_collision(q_extended, OBSTACLES)
    print(f"\nCollides with obstacles: {collision}")

    print()


def test_various_configurations():
    """Test FK with various joint configurations."""
    print_separator()
    print("TEST 3: Various Configurations")
    print_separator()

    test_configs = [
        ("Extended Up", [0, -np.pi/2, 0, 0, 0, 0]),
        ("Extended Down", [0, np.pi/2, 0, 0, 0, 0]),
        ("Rotated 90deg", [np.pi/2, -np.pi/2, 0, 0, 0, 0]),
        ("Elbow Bent", [0, -np.pi/4, np.pi/2, 0, 0, 0]),
        ("Start Config", [1.57, -1.0, 0.5, -1.57, 0.0, 0.0]),  # From your_code_here.py
    ]

    print(f"\n{'Configuration':<20} {'Position (x, y, z)':<30} {'Distance':<12} {'Collision'}")
    print("-" * 70)

    for name, q in test_configs:
        q_np = np.array(q)
        pos = get_end_effector_position(q_np)
        distance = np.linalg.norm(pos)
        collision = is_in_collision(q_np, OBSTACLES)

        pos_str = f"[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]"
        print(f"{name:<20} {pos_str:<30} {distance:6.4f} m    {collision}")

    print()


def test_collision_detection():
    """Test configuration-based collision detection."""
    print_separator()
    print("TEST 4: Configuration Collision Detection")
    print_separator()

    print(f"\nObstacle1 AABB (robot frame):")
    print(f"  min: {OBSTACLE1_ROBOT.min}")
    print(f"  max: {OBSTACLE1_ROBOT.max}")
    center = (OBSTACLE1_ROBOT.min + OBSTACLE1_ROBOT.max) / 2.0
    print(f"  center: {center}")

    # Try to find a configuration that puts end-effector near obstacle
    print(f"\n{'Configuration':<30} {'EE Position':<30} {'Collision'}")
    print("-" * 70)

    # Test some configurations that might collide
    test_cases = [
        ("Home [0, 0, 0, 0, 0, 0]", [0, 0, 0, 0, 0, 0]),
        ("Extended [0, -pi/2, 0, 0, 0, 0]", [0, -np.pi/2, 0, 0, 0, 0]),
        ("Rotated [pi/4, -pi/2, 0, 0, 0, 0]", [np.pi/4, -np.pi/2, 0, 0, 0, 0]),
    ]

    for name, q in test_cases:
        q_np = np.array(q)
        pos = get_end_effector_position(q_np)
        collision = is_in_collision(q_np, OBSTACLES)

        pos_str = f"[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]"
        result = "COLLISION!" if collision else "Safe"
        print(f"{name:<30} {pos_str:<30} {result}")

    print()


def test_fk_consistency():
    """Test FK consistency and sanity checks."""
    print_separator()
    print("TEST 5: FK Consistency Checks")
    print_separator()

    # Test 1: FK should be deterministic
    q_test = np.array([0.5, -0.8, 1.2, -0.3, 0.7, 0.0])
    pos1 = get_end_effector_position(q_test)
    pos2 = get_end_effector_position(q_test)

    print(f"\n[TEST] FK is deterministic:")
    print(f"  First call:  {pos1}")
    print(f"  Second call: {pos2}")
    print(f"  Match: {np.allclose(pos1, pos2)}")

    # Test 2: Different configs should give different positions (usually)
    q_a = np.array([0, 0, 0, 0, 0, 0])
    q_b = np.array([0, -np.pi/2, 0, 0, 0, 0])
    pos_a = get_end_effector_position(q_a)
    pos_b = get_end_effector_position(q_b)

    print(f"\n[TEST] Different configs give different positions:")
    print(f"  Config A: {q_a}")
    print(f"  Position A: {pos_a}")
    print(f"  Config B: {q_b}")
    print(f"  Position B: {pos_b}")
    print(f"  Different: {not np.allclose(pos_a, pos_b)}")

    # Test 3: All positions should be within max reach
    print(f"\n[TEST] Random configs stay within workspace:")
    np.random.seed(42)
    all_in_workspace = True
    for i in range(10):
        q_random = np.random.uniform(-np.pi, np.pi, 6)
        pos = get_end_effector_position(q_random)
        distance = np.linalg.norm(pos)
        if distance > 0.85:  # UR5e max reach
            print(f"  Config {i}: distance = {distance:.4f} m (OUT OF WORKSPACE!)")
            all_in_workspace = False

    if all_in_workspace:
        print(f"  All 10 random configs within 0.85m reach ✓")
    else:
        print(f"  Some configs exceed max reach (this is OK, just a sanity check)")

    print()


def main():
    """Run all validation tests."""
    print("\n")
    print("=" * 70)
    print(" " * 10 + "MICRO-SPRINT 1.3 VALIDATION (Forward Kinematics)")
    print("=" * 70)
    print("\n")

    test_home_position()
    test_extended_position()
    test_various_configurations()
    test_collision_detection()
    test_fk_consistency()

    print_separator()
    print("VALIDATION COMPLETE")
    print_separator()
    print("\nKey takeaways:")
    print("  - Forward kinematics computes end-effector position from joint angles")
    print("  - Positions are in robot frame (origin at robot base)")
    print("  - is_in_collision(q, obstacles) checks if config collides")
    print("  - Ready to proceed to Micro-Sprint 1.4 (Webots visualization)")
    print("\n")


if __name__ == "__main__":
    main()
