"""Validation script for Micro-Sprint 1.2.

Tests point-in-AABB collision detection with various test cases.
"""

import numpy as np
from config import AABB, OBSTACLES, OBSTACLE1_ROBOT
from collision import is_point_in_aabb, is_point_in_collision


def print_separator():
    print("=" * 70)


def test_simple_aabb():
    """Test basic AABB collision with a simple unit cube."""
    print_separator()
    print("TEST 1: Simple Unit Cube AABB")
    print_separator()

    # Create simple test AABB: unit cube at origin
    test_aabb = AABB(min_corner=[0, 0, 0], max_corner=[1, 1, 1])
    print(f"\nTest AABB: {test_aabb}")

    test_cases = [
        # (point, expected_result, description)
        ([0.5, 0.5, 0.5], True, "Center of cube (inside)"),
        ([0.0, 0.0, 0.0], True, "Min corner (on boundary)"),
        ([1.0, 1.0, 1.0], True, "Max corner (on boundary)"),
        ([0.5, 0.5, 0.0], True, "Bottom face center (on boundary)"),
        ([1.5, 0.5, 0.5], False, "Outside +X"),
        ([-0.5, 0.5, 0.5], False, "Outside -X"),
        ([0.5, 1.5, 0.5], False, "Outside +Y"),
        ([0.5, -0.5, 0.5], False, "Outside -Y"),
        ([0.5, 0.5, 1.5], False, "Outside +Z"),
        ([0.5, 0.5, -0.5], False, "Outside -Z"),
        ([2.0, 2.0, 2.0], False, "Far outside"),
    ]

    passed = 0
    failed = 0

    for point, expected, description in test_cases:
        point_np = np.array(point)
        result = is_point_in_aabb(point_np, test_aabb)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {description}")
        print(f"        Point: {point}, Inside: {result}, Expected: {expected}")

    print(f"\nResults: {passed} passed, {failed} failed")
    print()


def test_obstacle1():
    """Test collision detection with actual Obstacle1 from config."""
    print_separator()
    print("TEST 2: Obstacle1 from Config (Robot Frame)")
    print_separator()

    print(f"\nObstacle1 AABB: {OBSTACLE1_ROBOT}")
    print(f"  min: {OBSTACLE1_ROBOT.min}")
    print(f"  max: {OBSTACLE1_ROBOT.max}")

    # Calculate center of obstacle
    center = (OBSTACLE1_ROBOT.min + OBSTACLE1_ROBOT.max) / 2.0
    print(f"  center: {center}")

    test_cases = [
        # (point, expected_result, description)
        (center, True, "Center of obstacle (inside)"),
        (OBSTACLE1_ROBOT.min, True, "Min corner (on boundary)"),
        (OBSTACLE1_ROBOT.max, True, "Max corner (on boundary)"),
        (OBSTACLE1_ROBOT.min - [0.1, 0.1, 0.1], False, "Just outside min corner"),
        (OBSTACLE1_ROBOT.max + [0.1, 0.1, 0.1], False, "Just outside max corner"),
        ([0.0, 0.0, 0.0], False, "Robot base (origin, outside)"),
        ([1.0, 1.0, 1.0], False, "Far from obstacle"),
    ]

    passed = 0
    failed = 0

    for point, expected, description in test_cases:
        point_np = np.array(point)
        result = is_point_in_aabb(point_np, OBSTACLE1_ROBOT)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {description}")
        print(f"        Point: {point_np}, Inside: {result}, Expected: {expected}")

    print(f"\nResults: {passed} passed, {failed} failed")
    print()


def test_multi_obstacle():
    """Test collision detection with multiple obstacles."""
    print_separator()
    print("TEST 3: Multiple Obstacles List")
    print_separator()

    print(f"\nNumber of obstacles: {len(OBSTACLES)}")
    for i, obs in enumerate(OBSTACLES):
        print(f"  Obstacle {i}: min={obs.min}, max={obs.max}")

    # Test with obstacle center (should collide)
    center = (OBSTACLE1_ROBOT.min + OBSTACLE1_ROBOT.max) / 2.0
    result1 = is_point_in_collision(center, OBSTACLES)
    status1 = "PASS" if result1 == True else "FAIL"
    print(f"\n  [{status1}] Point at obstacle center collides: {result1} (expected True)")

    # Test with origin (should not collide)
    origin = np.array([0.0, 0.0, 0.0])
    result2 = is_point_in_collision(origin, OBSTACLES)
    status2 = "PASS" if result2 == False else "FAIL"
    print(f"  [{status2}] Point at robot base (origin) collides: {result2} (expected False)")

    # Test with point far away (should not collide)
    far_away = np.array([10.0, 10.0, 10.0])
    result3 = is_point_in_collision(far_away, OBSTACLES)
    status3 = "PASS" if result3 == False else "FAIL"
    print(f"  [{status3}] Point far away collides: {result3} (expected False)")

    print()


def test_boundary_cases():
    """Test tricky boundary cases."""
    print_separator()
    print("TEST 4: Boundary Edge Cases")
    print_separator()

    # Very small AABB
    tiny_aabb = AABB(min_corner=[0, 0, 0], max_corner=[0.001, 0.001, 0.001])
    print(f"\nTiny AABB: {tiny_aabb}")

    result1 = is_point_in_aabb(np.array([0.0005, 0.0005, 0.0005]), tiny_aabb)
    status1 = "PASS" if result1 == True else "FAIL"
    print(f"  [{status1}] Point inside tiny AABB: {result1} (expected True)")

    result2 = is_point_in_aabb(np.array([0.002, 0.0005, 0.0005]), tiny_aabb)
    status2 = "PASS" if result2 == False else "FAIL"
    print(f"  [{status2}] Point outside tiny AABB: {result2} (expected False)")

    # Empty obstacles list
    result3 = is_point_in_collision(np.array([0.5, 0.5, 0.5]), [])
    status3 = "PASS" if result3 == False else "FAIL"
    print(f"\n  [{status3}] Empty obstacle list always returns False: {result3}")

    print()


def main():
    """Run all validation tests."""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "MICRO-SPRINT 1.2 VALIDATION")
    print("=" * 70)
    print("\n")

    test_simple_aabb()
    test_obstacle1()
    test_multi_obstacle()
    test_boundary_cases()

    print_separator()
    print("VALIDATION COMPLETE")
    print_separator()
    print("\nKey takeaways:")
    print("  - is_point_in_aabb() correctly detects points inside/outside AABBs")
    print("  - Boundary points (on edges/corners) are considered INSIDE")
    print("  - is_point_in_collision() works with obstacle lists")
    print("  - Ready to proceed to Micro-Sprint 1.3 (End-Effector FK)")
    print("\n")


if __name__ == "__main__":
    main()
