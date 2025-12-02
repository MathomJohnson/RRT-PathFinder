# you may find all of these packages helpful!
from kinematic_helpers import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

# ============================================================================
# KINEMATICS FUNCTIONS
# ============================================================================

def get_full_fk(q):
    """
    Compute the full forward kinematics (base to end-effector transform).

    Args:
        q: List/array of 6 joint angles [q1, q2, q3, q4, q5, q6]

    Returns:
        T_06: 4x4 homogeneous transformation matrix from base to end-effector
    """
    T_06 = T01(q[0]) @ T12(q[1]) @ T23(q[2]) @ T34(q[3]) @ T45(q[4]) @ T56(q[5])
    return T_06


def get_jacobian(q):
    """
    Calculate the 6x6 geometric Jacobian for the UR5e at a given joint configuration.
    This is the same algorithm from HW2.

    Args:
        q: List/array of 6 joint angles

    Returns:
        J: 6x6 Jacobian matrix
    """
    J = np.zeros((6, 6))

    # Calculate all intermediate transforms from base to each joint
    T_00 = np.eye(4)
    T_01 = T01(q[0])
    T_02 = T_01 @ T12(q[1])
    T_03 = T_02 @ T23(q[2])
    T_04 = T_03 @ T34(q[3])
    T_05 = T_04 @ T45(q[4])
    T_06 = T_05 @ T56(q[5])

    # End-effector position in base frame
    p_end = T_06[0:3, 3]

    # List of transforms from base to each joint frame
    transforms = [T_00, T_01, T_02, T_03, T_04, T_05]

    # For each joint, compute its contribution to the Jacobian
    for i in range(6):
        T_current = transforms[i]
        z_i = T_current[0:3, 2]
        p_i = T_current[0:3, 3]
        J[0:3, i] = np.cross(z_i, p_end - p_i)
        J[3:6, i] = z_i

    return J


def move_to_position(current_q, target_pos, max_step=0.02, dt=0.016):
    """
    Move toward target position using velocity-based IK with orientation constraint.
    Keeps gripper pointing straight down (perpendicular to mat) at all times.

    Args:
        current_q: Current joint configuration [q1, q2, q3, q4, q5, q6]
        target_pos: Target Cartesian position [x, y, z]
        max_step: Maximum position change per timestep (m)
        dt: Timestep (seconds)

    Returns:
        q_next: Next joint configuration
    """
    T_06 = get_full_fk(current_q)
    current_pos = T_06[0:3, 3]
    current_R = T_06[0:3, 0:3]  # Current rotation matrix

    # Position error
    pos_error = target_pos - current_pos
    distance = np.linalg.norm(pos_error)

    # Convert to Cartesian velocity with max_step limiting
    if distance > max_step:
        velocity = (pos_error / distance) * max_step / dt
    else:
        velocity = pos_error / dt

    # Desired orientation: gripper pointing straight down (same as search pose)
    # Z-axis pointing down: [0, 0, -1]
    # X-axis and Y-axis should maintain the same orientation as search pose
    T_search = get_full_fk(POSE_SEARCH_BLOCKS['q'])
    desired_R = T_search[0:3, 0:3]

    # Orientation error using rotation vector (axis-angle)
    # R_error = desired_R @ current_R.T (rotation from current to desired)
    R_error = desired_R @ current_R.T

    # Convert rotation matrix to axis-angle (rotation vector)
    # Using Rodrigues formula approximation for small angles
    rot_vec = np.array([
        R_error[2, 1] - R_error[1, 2],
        R_error[0, 2] - R_error[2, 0],
        R_error[1, 0] - R_error[0, 1]
    ]) / 2.0

    # Orientation velocity (proportional control)
    k_orient = 2.0  # Orientation correction gain
    angular_velocity = k_orient * rot_vec

    # Combined Cartesian velocity (position + orientation)
    v_cart = np.zeros(6)
    v_cart[0:3] = velocity  # Position velocity
    v_cart[3:6] = angular_velocity  # Orientation velocity

    # Get Jacobian and compute joint velocities
    J = get_jacobian(current_q)

    try:
        # Jacobian pseudoinverse with damping
        lamb = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lamb * np.eye(6))
        q_dot = J_pinv @ v_cart

        # Clip joint velocities to prevent aggressive motion
        q_dot = np.clip(q_dot, -1.0, 1.0)

        return (np.array(current_q) + q_dot * dt).tolist()
    except:
        # If IK fails, return current position
        return current_q

# ============================================================================
# HARDCODED REFERENCE POSES
# ============================================================================

# SEARCH_BLOCKS - Overhead view of block area (also used as starting position)
POSE_SEARCH_BLOCKS = {
    'q': [-0.2, -1.4, 1.4, -1.55, -1.57, -0.2],
    #'q': [-0.1, -1.5, 1.2, -1.3, -1.57, -0.1],  # TUNE THESE
    'gripper': False,
    'description': 'High overhead position to see blocks'
}

# OVER_BIN - Position above bin for dropping
POSE_OVER_BIN = {
    'q': [1.0, -1.05, 1.0, -1.2, -0.57, 0.0],  # TUNE THESE
    'gripper': True,
    'description': 'Above bin, ready to release'
}

# Critical Heights
HEIGHTS = {
    'search_height': 0.5,      # Height for searching (m)
    'approach_height': 0.15,   # Height before descent (m)
    'grasp_height': 0.165,     # Height to close gripper (m)
    'lift_height': 0.6,        # Safe carry height (m)
    'bin_drop_height': 0.5,    # Height above bin (m)
}

# ============================================================================
# VISION PARAMETERS
# ============================================================================

# HSV color ranges (tune as needed)
HSV_RED_LOWER = np.array([0, 100, 100])
HSV_RED_UPPER = np.array([10, 255, 255])

HSV_BLUE_LOWER = np.array([100, 160, 100])  # Saturation 160 to filter out brown table
HSV_BLUE_UPPER = np.array([130, 255, 255])

# Minimum contour area to filter noise (pixels)
MIN_CONTOUR_AREA = 100

# Wait time for robot to reach search pose (timesteps)
SEARCH_POSE_WAIT_TIME = 100

# ============================================================================
# ALIGNMENT PARAMETERS
# ============================================================================

# Coarse alignment - large moves, sequential X then Y
COARSE_STEP_SIZE = 0.02        # 2cm per move
COARSE_TOLERANCE_X = 30        # ±30 pixels in X
COARSE_TOLERANCE_Y = 30        # ±30 pixels in Y
REQUIRED_ALIGN_FRAMES = 5      # Require 5 consecutive frames within tolerance

# Fine alignment - two-phase approach (Sprint 7)
# Phase A: Pre-alignment descent to approach height
APPROACH_HEIGHT = 0.25         # Height to descend to before fine align (m)
PRE_DESCENT_STEP = 0.004       # 4mm per step (same as final descent)
PRE_DESCENT_WAIT = 5           # Timesteps between descent steps

# Phase B: Fine alignment at lower height
FINE_STEP_SIZE = 0.003          # 1cm per move
FINE_TOLERANCE_X = 3          # ±40 pixels in X
FINE_TOLERANCE_Y = 3         # ±120 pixels in Y (loose to avoid oscillation)
MAX_FINE_ATTEMPTS = 150         # Force descent after this many attempts

# Move timing
MOVE_SETTLE_TIME = 20          # Timesteps to wait between moves

# ============================================================================
# DESCENT PARAMETERS (Sprint 8)
# ============================================================================

# Final descent from approach_height to grasp_height
DESCENT_STEP_SIZE = 0.004      # 4mm per step
DESCENT_WAIT_TIME = 15          # Timesteps between descent steps
GRASP_HEIGHT = HEIGHTS['grasp_height']  # 0.16m - height to close gripper

# ============================================================================
# GRASP PARAMETERS (Sprint 9)
# ============================================================================

# Grasp duration - how long to hold position while closing gripper
GRASP_DURATION = 92            # Timesteps (~1.0 seconds at 62.5 Hz)

# ============================================================================
# LIFT PARAMETERS (Sprint 10)
# ============================================================================

# Lift motion - symmetric with descent for smooth operation
LIFT_STEP_SIZE = 0.004         # 4mm per step (same as descent)
LIFT_WAIT_TIME = 3             # Timesteps between lift steps (same as descent)
LIFT_HEIGHT = HEIGHTS['lift_height']  # 0.6m - safe carry height

# ============================================================================
# CARRY PARAMETERS (Sprint 11)
# ============================================================================

# Carry to bin - smooth Cartesian motion
CARRY_MAX_STEP = 0.01          # 1cm max step per timestep for smooth motion
CARRY_ARRIVAL_THRESHOLD = 0.05 # 5cm - consider arrived when within this distance
CARRY_SETTLE_TIME = 62         # Timesteps to wait after arrival (~1 second)

# ============================================================================
# RELEASE PARAMETERS (Sprint 12)
# ============================================================================

# Release duration - how long to hold position while opening gripper
RELEASE_DURATION = 62          # Timesteps (~1 second at 62.5 Hz)

# ============================================================================
# VISUAL SERVOING PARAMETERS
# ============================================================================

# Target pixel position (where we want the block to appear in the image)
# Camera is offset 0.05m ahead in +Y, so target is right of center to compensate
TARGET_CX = 380  # Horizontal target (for coarse alignment)
TARGET_CY = 240  # Vertical target (for coarse alignment)

# Fine alignment targets (tune these independently to center gripper over block)
# Image dimensions: 640 wide x 480 tall
# Center would be: (320, 240)
FINE_TARGET_CX = 440  # Horizontal target for fine alignment - TUNE THIS
FINE_TARGET_CY = 240  # Vertical target for fine alignment - TUNE THIS

# Pixel-to-meter scale factor (TUNE THIS)
# Approximate: at 0.5m height, 1 pixel ≈ 0.0002m
PIXEL_TO_METER_SCALE = 0.0002

# ============================================================================
# VISUAL SERVOING FUNCTIONS
# ============================================================================

def compute_pixel_error(cx, cy):
    """
    Compute pixel error from target position.

    Args:
        cx: Current block x-position in image (pixels)
        cy: Current block y-position in image (pixels)

    Returns:
        (err_x, err_y): Pixel errors in image frame
    """
    err_x = cx - TARGET_CX  # Horizontal error
    err_y = cy - TARGET_CY  # Vertical error
    return err_x, err_y


def pixel_to_robot_motion(err_x, err_y, scale=PIXEL_TO_METER_SCALE):
    """
    Convert pixel errors to robot motion in world frame.

    Camera is rotated 90° relative to gripper:
    - Image X (horizontal) → Robot Y (forward/backward)
    - Image Y (vertical) → Robot X (left/right)

    Args:
        err_x: Pixel error in image x (horizontal)
        err_y: Pixel error in image y (vertical)
        scale: Meters per pixel (default from constant)

    Returns:
        (delta_x, delta_y): Robot motion in world frame (meters)
    """
    # Handle 90° rotation:
    # Based on actual testing, the correct mapping is:
    # - Image UP (negative err_y) → Robot RIGHT (positive delta_x)
    # - Image DOWN (positive err_y) → Robot LEFT (negative delta_x)
    # - Image LEFT (negative err_x) → Robot FORWARD (positive delta_y)
    # - Image RIGHT (positive err_x) → Robot BACKWARD (negative delta_y)

    delta_robot_x = -err_y * scale     # Image vertical → Robot left/right (NEGATED)
    delta_robot_y = -err_x * scale     # Image horizontal → Robot forward/back (negated)

    return delta_robot_x, delta_robot_y

# ============================================================================
# VISION FUNCTIONS
# ============================================================================

def process_image(bgr_image, color='red'):
    """
    Detect colored block in image and return centroid

    Args:
        bgr_image: (H, W, 3) BGR image from camera
        color: 'red' or 'blue'

    Returns:
        (cx, cy, detected): pixel coordinates and detection flag
        If not detected, returns (None, None, False)
    """
    if bgr_image is None:
        return None, None, False

    # Convert BGR to HSV
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Select color range
    if color == 'red':
        # Red wraps around in HSV (0-10 and 170-180)
        # Use high saturation (160) to filter out brown table
        lower1 = np.array([0, 160, 100])      # Lower red range (0-10 hue)
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 160, 100])    # Upper red range (170-180 hue)
        upper2 = np.array([180, 255, 255])

        # Create two masks and combine them
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.add(mask1, mask2)  # Combine both red ranges
    elif color == 'blue':
        lower = HSV_BLUE_LOWER
        upper = HSV_BLUE_UPPER
        mask = cv2.inRange(hsv, lower, upper)
    else:
        return None, None, False

    # Optional: Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None, False

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # Check if contour is large enough
    if area < MIN_CONTOUR_AREA:
        return None, None, False

    # Calculate centroid
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return None, None, False

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # ============ VISUALIZATION FOR DEBUGGING ============
    # Create a copy of the image for drawing
    vis_image = bgr_image.copy()

    # Draw target position (green) - where we want the block to be
    cv2.circle(vis_image, (TARGET_CX, TARGET_CY), 8, (0, 255, 0), 2)  # Green circle
    cv2.drawMarker(vis_image, (TARGET_CX, TARGET_CY), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(vis_image, "TARGET", (TARGET_CX + 10, TARGET_CY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw actual center of image (cyan) - for reference
    center_x, center_y = 320, 240
    cv2.circle(vis_image, (center_x, center_y), 5, (255, 255, 0), 2)  # Cyan circle
    cv2.putText(vis_image, "CENTER", (center_x + 10, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Draw detected block position (red)
    cv2.circle(vis_image, (cx, cy), 8, (0, 0, 255), 2)  # Red circle
    cv2.drawMarker(vis_image, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(vis_image, f"{color.upper()} ({cx},{cy})", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show the visualization window
    cv2.imshow('Vision Debug', vis_image)
    cv2.waitKey(1)  # Required for window to update
    # ====================================================

    return cx, cy, True

# ============================================================================
# STATE CONSTANTS
# ============================================================================

# Red block states (0-7)
STATE_SEARCH_RED = 0
STATE_COARSE_ALIGN_RED = 1
STATE_FINE_ALIGN_RED = 2
STATE_DESCEND_RED = 3
STATE_GRASP_RED = 4
STATE_LIFT_RED = 5
STATE_CARRY_RED = 6
STATE_RELEASE_RED = 7

# Blue block states (8-15)
STATE_SEARCH_BLUE = 8
STATE_COARSE_ALIGN_BLUE = 9
STATE_FINE_ALIGN_BLUE = 10
STATE_DESCEND_BLUE = 11
STATE_GRASP_BLUE = 12
STATE_LIFT_BLUE = 13
STATE_CARRY_BLUE = 14
STATE_RELEASE_BLUE = 15

# Completion state
STATE_DONE = 99

# Test states
STATE_TEST_HARDCODED_MOVE = 100
STATE_TEST_IK_CARTESIAN = 101

# State names for debugging
STATE_NAMES = {
    0: "SEARCH_RED",
    1: "COARSE_ALIGN_RED",
    2: "FINE_ALIGN_RED",
    3: "DESCEND_RED",
    4: "GRASP_RED",
    5: "LIFT_RED",
    6: "CARRY_RED",
    7: "RELEASE_RED",
    8: "SEARCH_BLUE",
    9: "COARSE_ALIGN_BLUE",
    10: "FINE_ALIGN_BLUE",
    11: "DESCEND_BLUE",
    12: "GRASP_BLUE",
    13: "LIFT_BLUE",
    14: "CARRY_BLUE",
    15: "RELEASE_BLUE",
    99: "DONE",
    100: "TEST_HARDCODED_MOVE",
    101: "TEST_IK_CARTESIAN"
}

# ============================================================================
# MAIN CONTROLLER CLASS
# ============================================================================

class RobotPlacerWithVision():
    def __init__(self):
        """Initialize the FSM controller"""
        # State tracking
        self.state = STATE_SEARCH_RED  # Sprint 4: Search for red block
        self.state_start_tt = 0
        self.move_timer = 0  # Tracks time since last move (for settle time)

        # Coarse alignment tracking (Sprint 6)
        self.coarse_phase = 'align_x'  # Tracks coarse alignment phase: 'align_x' or 'align_y'
        self.align_counter_x = 0  # Counter for X alignment verification
        self.align_counter_y = 0  # Counter for Y alignment verification

        # Fine alignment tracking (Sprint 7)
        self.fine_phase = 'descending'  # 'descending' or 'aligning'
        self.fine_descent_counter = 0   # Rate limiting counter for descent
        self.fine_attempt_counter = 0   # Counts alignment attempts
        self.fine_align_counter = 0     # Counts consecutive frames within tolerance

        # Descent tracking (Sprint 8)
        self.descent_counter = 0        # Rate limiting counter for final descent

        # Grasp tracking (Sprint 9)
        self.grasp_start_tt = 0         # Timestep when grasp started

        # Lift tracking (Sprint 10)
        self.lift_counter = 0           # Rate limiting counter for lift

        # Carry tracking (Sprint 11)
        self.carry_arrival_tt = None    # Timestep when arrived at bin (None = not arrived yet)

        # Release tracking (Sprint 12)
        self.release_start_tt = 0       # Timestep when release started

        # Debug flag
        self.debug = True

        print("="*60)
        print("RobotPlacerWithVision initialized")
        print("Starting in state:", STATE_NAMES[self.state])
        print("="*60)

    def getRobotCommand(self, tt, current_q, current_image_bgr):
        """
        Main control loop - called every timestep

        Args:
            tt: Timestep counter (int)
            current_q: Current joint angles [q1, q2, q3, q4, q5, q6]
            current_image_bgr: Camera image (480, 640, 3) BGR format

        Returns:
            [q1, q2, q3, q4, q5, q6, gripper_bool]
        """

        # Print debug info every 50 timesteps
        if self.debug and tt % 50 == 0:
            time_elapsed = tt * 0.016  # seconds
            state_duration = (tt - self.state_start_tt) * 0.016
            print(f"[tt={tt:5d} | t={time_elapsed:6.2f}s] State: {STATE_NAMES[self.state]:20s} | Duration: {state_duration:5.2f}s")

        # ====================================================================
        # STATE MACHINE
        # ====================================================================

        if self.state == STATE_SEARCH_RED:
            # Move to search position first
            state_duration = tt - self.state_start_tt

            if state_duration < SEARCH_POSE_WAIT_TIME:
                # Still moving to search pose
                if tt % 50 == 0:
                    print(f"[SEARCH_RED] Moving to search pose... ({state_duration}/{SEARCH_POSE_WAIT_TIME})")
                return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]
            else:
                # Arrived at search pose, start detecting red block
                cx, cy, detected = process_image(current_image_bgr, 'red')

                if detected:
                    # Test Sprint 5 functions: compute errors and motions
                    err_x, err_y = compute_pixel_error(cx, cy)
                    delta_x, delta_y = pixel_to_robot_motion(err_x, err_y)

                    print(f"[SEARCH_RED] RED BLOCK DETECTED at pixel ({cx}, {cy})")
                    print(f"[SEARCH_RED] Target pixels: ({TARGET_CX}, {TARGET_CY})")
                    print(f"[SEARCH_RED] Pixel errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                    print(f"[SEARCH_RED] Robot motion: delta_x={delta_x:.4f}m, delta_y={delta_y:.4f}m")
                    print(f"[SEARCH_RED] Transitioning to coarse alignment...")
                    self.transition_to_state(STATE_COARSE_ALIGN_RED, tt)
                    return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]
                else:
                    if tt % 50 == 0:
                        print(f"[SEARCH_RED] Searching for red block... (not detected yet)")
                    return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]

        elif self.state == STATE_COARSE_ALIGN_RED:
            # Coarse alignment - sequential X then Y alignment
            # Phase 1: Align X until within tolerance
            # Phase 2: Align Y until within tolerance
            # Then transition to fine alignment

            # DEBUG: Entry check
            if tt % 50 == 0:
                print(f"[COARSE_DEBUG] Entered coarse alignment state, phase={self.coarse_phase}")

            # Detect red block
            cx, cy, detected = process_image(current_image_bgr, 'red')

            # DEBUG: Detection status
            if tt % 50 == 0:
                print(f"[COARSE_DEBUG] Detection: detected={detected}, cx={cx}, cy={cy}")

            if not detected:
                # Lost the block - stay in place and keep searching
                if tt % 50 == 0:
                    print(f"[COARSE_ALIGN_RED] Block not detected, searching...")
                return current_q + [False]

            # Compute pixel errors
            err_x, err_y = compute_pixel_error(cx, cy)

            # PHASE 1: Align X direction
            if self.coarse_phase == 'align_x':
                # DEBUG: Phase check
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] In align_x phase, err_x={err_x:.1f}, tolerance={COARSE_TOLERANCE_X}")

                # Check if X is aligned
                if abs(err_x) < COARSE_TOLERANCE_X:
                    self.align_counter_x += 1
                    print(f"[COARSE] X within tolerance ({self.align_counter_x}/{REQUIRED_ALIGN_FRAMES})")

                    if self.align_counter_x >= REQUIRED_ALIGN_FRAMES:
                        print(f"[COARSE_ALIGN_RED] X aligned! Moving to Y alignment phase...")
                        self.coarse_phase = 'align_y'
                        self.align_counter_x = 0  # Reset for next time
                        return current_q + [False]
                    else:
                        # Within tolerance but waiting for stability
                        return current_q + [False]
                else:
                    # Not within tolerance - reset counter
                    self.align_counter_x = 0

                # DEBUG: Not aligned
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] X not aligned yet (err_x={err_x:.1f} > {COARSE_TOLERANCE_X})")

                # X not aligned - check settle time
                time_since_move = tt - self.move_timer

                # DEBUG: Settle time check
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] Settle check: time_since_move={time_since_move}, threshold={MOVE_SETTLE_TIME}, move_timer={self.move_timer}")

                if time_since_move < MOVE_SETTLE_TIME:
                    if tt % 50 == 0:
                        print(f"[COARSE_DEBUG] Still settling, waiting...")
                    return current_q + [False]

                # DEBUG: Ready to move
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] Ready to move!")

                # Move in X direction only (zero out Y error)
                delta_x, delta_y = pixel_to_robot_motion(err_x, 0)

                # DEBUG: Show raw delta
                print(f"[COARSE_X_MOVE] tt={tt} | err_x={err_x:.1f} | RAW delta_x={delta_x:.6f}m")

                # Limit to coarse step size
                delta_magnitude = abs(delta_x)
                if delta_magnitude > COARSE_STEP_SIZE:
                    delta_x = COARSE_STEP_SIZE if delta_x > 0 else -COARSE_STEP_SIZE

                # DEBUG: Show limited delta
                print(f"[COARSE_X_MOVE] After limiting: delta_x={delta_x:.6f}m (limit={COARSE_STEP_SIZE})")

                # Get current position and create target
                T_current = get_full_fk(current_q)
                current_pos = T_current[0:3, 3]
                # Image X (horizontal) controls Robot Y, so move in Y direction
                target_pos = current_pos + np.array([0.0, delta_y, 0.0])

                # DEBUG: Show positions
                print(f"[COARSE_X_MOVE] Current pos: {current_pos}")
                print(f"[COARSE_X_MOVE] Target pos:  {target_pos}")

                # Move using IK
                q_next = move_to_position(current_q, target_pos, max_step=COARSE_STEP_SIZE, dt=0.016)
                self.move_timer = tt

                # DEBUG: Show joint changes
                q_diff = np.array(q_next) - np.array(current_q)
                print(f"[COARSE_X_MOVE] Joint changes: {q_diff}")

                return q_next + [False]

            # PHASE 2: Align Y direction
            elif self.coarse_phase == 'align_y':
                # Check if Y is aligned
                if abs(err_y) < COARSE_TOLERANCE_Y:
                    self.align_counter_y += 1
                    print(f"[COARSE] Y within tolerance ({self.align_counter_y}/{REQUIRED_ALIGN_FRAMES})")

                    if self.align_counter_y >= REQUIRED_ALIGN_FRAMES:
                        print(f"[COARSE_ALIGN_RED] Y aligned! Coarse alignment complete!")
                        print(f"[COARSE_ALIGN_RED] Final errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                        self.coarse_phase = 'align_x'  # Reset for next use
                        self.align_counter_y = 0  # Reset for next time
                        self.transition_to_state(STATE_FINE_ALIGN_RED, tt)
                        return current_q + [False]
                    else:
                        # Within tolerance but waiting for stability
                        return current_q + [False]
                else:
                    # Not within tolerance - reset counter
                    self.align_counter_y = 0

                # Y not aligned - check settle time
                time_since_move = tt - self.move_timer
                if time_since_move < MOVE_SETTLE_TIME:
                    return current_q + [False]

                # Move in Y direction only (zero out X error)
                delta_x, delta_y = pixel_to_robot_motion(0, err_y)

                # Limit to coarse step size
                delta_magnitude = abs(delta_y)
                if delta_magnitude > COARSE_STEP_SIZE:
                    delta_y = COARSE_STEP_SIZE if delta_y > 0 else -COARSE_STEP_SIZE

                # Get current position and create target
                T_current = get_full_fk(current_q)
                current_pos = T_current[0:3, 3]
                # Image Y (vertical) controls Robot X, so move in X direction
                target_pos = current_pos + np.array([delta_x, 0.0, 0.0])

                # Move using IK
                q_next = move_to_position(current_q, target_pos, max_step=COARSE_STEP_SIZE, dt=0.016)
                self.move_timer = tt

                if tt % 20 == 0:
                    print(f"[COARSE_Y] Block at ({cx}, {cy}) | err_y={err_y:.1f} | Moving delta_y={delta_y:.4f}m")

                return q_next + [False]

        elif self.state == STATE_FINE_ALIGN_RED:
            # Sprint 7: Two-phase fine alignment
            # Phase A: Descend to approach_height (~15cm)
            # Phase B: Fine lateral alignment at that lower height

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            current_z = current_pos[2]

            # Debug: Print phase every 50 timesteps
            if tt % 50 == 0:
                print(f"[FINE_DEBUG] tt={tt} | phase={self.fine_phase} | height={current_z:.4f}m | target={APPROACH_HEIGHT}m")

            # PHASE A: Pre-alignment descent
            if self.fine_phase == 'descending':
                # Rate-limited descent
                self.fine_descent_counter += 1

                if self.fine_descent_counter >= PRE_DESCENT_WAIT:
                    # Time to take another descent step
                    target_z = max(current_z - PRE_DESCENT_STEP, APPROACH_HEIGHT)

                    # Check if we've reached approach height (target_z stopped decreasing)
                    if abs(target_z - APPROACH_HEIGHT) < 0.001:  # Within 1mm of target
                        print(f"[FINE_ALIGN] Reached approach height ({current_z:.4f}m). Starting fine alignment phase...")
                        self.fine_phase = 'aligning'
                        self.move_timer = tt  # Reset move timer for alignment phase
                        return current_q + [False]

                    target_pos = current_pos.copy()
                    target_pos[2] = target_z

                    # Move using IK (keep X, Y constant, only change Z)
                    q_next = move_to_position(current_q, target_pos, max_step=0.008, dt=0.016)
                    self.fine_descent_counter = 0  # Reset counter

                    print(f"[FINE_DESCEND] tt={tt} | Height: {current_z:.4f}m -> {target_z:.4f}m")

                    return q_next + [False]
                else:
                    # Still waiting for next descent step
                    return current_q + [False]

            # PHASE B: Fine alignment at lower height
            elif self.fine_phase == 'aligning':
                # Detect red block
                cx, cy, detected = process_image(current_image_bgr, 'red')

                # Debug: Detection status
                if tt % 50 == 0:
                    print(f"[FINE_DEBUG] Aligning phase | detected={detected} | cx={cx}, cy={cy}")

                if not detected:
                    # Lost the block - stay in place
                    if tt % 50 == 0:
                        print(f"[FINE_ALIGN] Block not detected, searching...")
                    return current_q + [False]

                # Compute pixel errors using FINE alignment targets
                err_x = cx - FINE_TARGET_CX
                err_y = cy - FINE_TARGET_CY

                # Debug: Show errors and targets
                if tt % 50 == 0:
                    print(f"[FINE_DEBUG] Block at ({cx}, {cy}) | Target: ({FINE_TARGET_CX}, {FINE_TARGET_CY})")
                    print(f"[FINE_DEBUG] err_x={err_x:.1f}, err_y={err_y:.1f} | tol_x={FINE_TOLERANCE_X}, tol_y={FINE_TOLERANCE_Y}")

                # Check if aligned (both X and Y within tolerance)
                x_aligned = abs(err_x) < FINE_TOLERANCE_X
                y_aligned = abs(err_y) < FINE_TOLERANCE_Y

                if x_aligned and y_aligned:
                    self.fine_align_counter += 1
                    print(f"[FINE_ALIGN] Aligned! ({self.fine_align_counter}/{REQUIRED_ALIGN_FRAMES})")

                    if self.fine_align_counter >= REQUIRED_ALIGN_FRAMES:
                        print(f"[FINE_ALIGN] Fine alignment complete!")
                        print(f"[FINE_ALIGN] Final position: cx={cx}, cy={cy} | Target: ({FINE_TARGET_CX}, {FINE_TARGET_CY})")
                        print(f"[FINE_ALIGN] Final errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                        print(f"[FINE_ALIGN] Final height: {current_z:.4f}m")
                        self.transition_to_state(STATE_DESCEND_RED, tt)
                        return current_q + [False]
                    else:
                        # Within tolerance but waiting for stability
                        return current_q + [False]
                else:
                    # Not aligned - reset counter
                    self.fine_align_counter = 0

                # Check if we've exceeded max attempts
                if self.fine_attempt_counter >= MAX_FINE_ATTEMPTS:
                    print(f"[FINE_ALIGN] Max attempts ({MAX_FINE_ATTEMPTS}) reached. Forcing descent...")
                    print(f"[FINE_ALIGN] Final position: cx={cx}, cy={cy} | Target: ({FINE_TARGET_CX}, {FINE_TARGET_CY})")
                    print(f"[FINE_ALIGN] Final errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                    self.transition_to_state(STATE_DESCEND_RED, tt)
                    return current_q + [False]

                # Check settle time
                time_since_move = tt - self.move_timer
                if time_since_move < MOVE_SETTLE_TIME:
                    return current_q + [False]

                # Ready to make alignment move
                self.fine_attempt_counter += 1

                # Convert pixel errors to robot motion
                delta_x, delta_y = pixel_to_robot_motion(err_x, err_y)

                # Limit to fine step size
                delta_magnitude = np.sqrt(delta_x**2 + delta_y**2)
                if delta_magnitude > FINE_STEP_SIZE:
                    scale = FINE_STEP_SIZE / delta_magnitude
                    delta_x *= scale
                    delta_y *= scale

                # Create target position - EXPLICITLY maintain Z at approach height
                target_pos = current_pos.copy()
                target_pos[0] += delta_x  # Apply X adjustment
                target_pos[1] += delta_y  # Apply Y adjustment
                target_pos[2] = APPROACH_HEIGHT  # Lock Z at approach height

                # Move using IK
                q_next = move_to_position(current_q, target_pos, max_step=FINE_STEP_SIZE, dt=0.016)
                self.move_timer = tt

                if tt % 30 == 0:
                    print(f"[FINE_ALIGN] Attempt {self.fine_attempt_counter} | Block at ({cx}, {cy}) → Target ({FINE_TARGET_CX}, {FINE_TARGET_CY}) | err=({err_x:.1f}, {err_y:.1f}) | delta=({delta_x:.4f}, {delta_y:.4f})")

                return q_next + [False]

        elif self.state == STATE_DESCEND_RED:
            # Sprint 8: Controlled descent from approach_height to grasp_height
            # Rate-limited, small steps, keep X/Y constant

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            current_z = current_pos[2]

            # Check if we've reached grasp height (with 1mm tolerance for floating-point precision)
            if current_z < GRASP_HEIGHT + 0.001:
                print(f"[DESCEND_RED] Reached grasp height ({current_z:.4f}m). Transitioning to grasp...")
                self.transition_to_state(STATE_GRASP_RED, tt)
                return current_q + [False]

            # Rate-limited descent
            self.descent_counter += 1

            if self.descent_counter >= DESCENT_WAIT_TIME:
                # Time to take another descent step
                target_z = max(current_z - DESCENT_STEP_SIZE, GRASP_HEIGHT)
                target_pos = current_pos.copy()
                target_pos[2] = target_z

                # Move using IK (keep X, Y constant, only change Z)
                q_next = move_to_position(current_q, target_pos, max_step=0.008, dt=0.016)
                self.descent_counter = 0  # Reset counter

                if tt % 30 == 0:
                    print(f"[DESCEND_RED] tt={tt} | Height: {current_z:.4f}m → {target_z:.4f}m (target: {GRASP_HEIGHT}m)")

                return q_next + [False]
            else:
                # Still waiting for next descent step
                return current_q + [False]

        elif self.state == STATE_GRASP_RED:
            # Sprint 9: Close gripper while holding position
            # Hold for GRASP_DURATION timesteps to ensure secure grasp

            # Calculate how long we've been grasping
            grasp_duration = tt - self.grasp_start_tt

            # Check if grasp is complete
            if grasp_duration >= GRASP_DURATION:
                print(f"[GRASP_RED] Grasp complete ({grasp_duration} timesteps). Transitioning to lift...")
                self.transition_to_state(STATE_LIFT_RED, tt)
                return current_q + [True]  # Keep gripper closed

            # Hold position while gripper closes
            # Use small max_step to minimize drift
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            target_pos = current_pos.copy()  # Stay at current position

            q_next = move_to_position(current_q, target_pos, max_step=0.005, dt=0.016)

            # Debug output
            if grasp_duration == 0:
                print(f"[GRASP_RED] Starting grasp at height {current_pos[2]:.4f}m")
            if tt % 30 == 0:
                print(f"[GRASP_RED] Holding position... ({grasp_duration}/{GRASP_DURATION})")

            return q_next + [True]  # Gripper closed

        elif self.state == STATE_LIFT_RED:
            # Sprint 10: Controlled lift from grasp_height to lift_height
            # Rate-limited, small steps, keep X/Y constant, gripper closed

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            current_z = current_pos[2]

            # Check if we've reached lift height (with 1mm tolerance for floating-point precision)
            if current_z > LIFT_HEIGHT - 0.001:
                print(f"[LIFT_RED] Reached lift height ({current_z:.4f}m). Transitioning to carry...")
                self.transition_to_state(STATE_CARRY_RED, tt)
                return current_q + [True]  # Keep gripper closed

            # Rate-limited lift
            self.lift_counter += 1

            if self.lift_counter >= LIFT_WAIT_TIME:
                # Time to take another lift step
                target_z = min(current_z + LIFT_STEP_SIZE, LIFT_HEIGHT)
                target_pos = current_pos.copy()
                target_pos[2] = target_z

                # Move using IK (keep X, Y constant, only change Z)
                q_next = move_to_position(current_q, target_pos, max_step=0.008, dt=0.016)
                self.lift_counter = 0  # Reset counter

                if tt % 30 == 0:
                    print(f"[LIFT_RED] tt={tt} | Height: {current_z:.4f}m → {target_z:.4f}m (target: {LIFT_HEIGHT}m)")

                return q_next + [True]  # Keep gripper closed
            else:
                # Still waiting for next lift step
                return current_q + [True]  # Keep gripper closed

        elif self.state == STATE_CARRY_RED:
            # Sprint 11: Carry block to bin using smooth Cartesian motion
            # Move gradually to bin position, pause to settle, then transition to release

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]

            # Get target position from POSE_OVER_BIN
            T_bin = get_full_fk(POSE_OVER_BIN['q'])
            target_pos = T_bin[0:3, 3]

            # Calculate distance to target
            distance = np.linalg.norm(target_pos - current_pos)

            # Check if we've arrived at bin
            if distance < CARRY_ARRIVAL_THRESHOLD:
                # Mark arrival time if not already marked
                if self.carry_arrival_tt is None:
                    self.carry_arrival_tt = tt
                    print(f"[CARRY_RED] Arrived at bin (distance: {distance:.4f}m). Settling...")

                # Check if settle time is complete
                settle_duration = tt - self.carry_arrival_tt
                if settle_duration >= CARRY_SETTLE_TIME:
                    print(f"[CARRY_RED] Settle complete ({settle_duration} timesteps). Transitioning to release...")
                    self.transition_to_state(STATE_RELEASE_RED, tt)
                    return current_q + [True]  # Keep gripper closed
                else:
                    # Still settling - hold position
                    if tt % 30 == 0:
                        print(f"[CARRY_RED] Settling... ({settle_duration}/{CARRY_SETTLE_TIME})")
                    return current_q + [True]  # Keep gripper closed

            # First timestep - print start message
            carry_duration = tt - self.state_start_tt
            if carry_duration == 0:
                print(f"[CARRY_RED] Starting carry from {current_pos} to bin at {target_pos}")
                print(f"[CARRY_RED] Distance to bin: {distance:.4f}m")

            # Smooth motion towards bin using IK
            q_next = move_to_position(current_q, target_pos, max_step=CARRY_MAX_STEP, dt=0.016)

            # Debug output
            if tt % 50 == 0:
                print(f"[CARRY_RED] Moving to bin... distance={distance:.4f}m")

            return q_next + [True]  # Keep gripper closed

        elif self.state == STATE_RELEASE_RED:
            # Sprint 12: Open gripper and drop block while holding position
            # Hold for RELEASE_DURATION timesteps to let block fall

            # Calculate how long we've been releasing
            release_duration = tt - self.release_start_tt

            # Check if release is complete
            if release_duration >= RELEASE_DURATION:
                print(f"[RELEASE_RED] Release complete ({release_duration} timesteps). Transitioning to search blue...")
                self.transition_to_state(STATE_SEARCH_BLUE, tt)
                return current_q + [False]  # Keep gripper open

            # Hold position while gripper opens
            # Use small max_step to minimize drift
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            target_pos = current_pos.copy()  # Stay at current position

            q_next = move_to_position(current_q, target_pos, max_step=0.005, dt=0.016)

            # Debug output
            if release_duration == 0:
                print(f"[RELEASE_RED] Opening gripper at height {current_pos[2]:.4f}m")
            if tt % 30 == 0:
                print(f"[RELEASE_RED] Releasing... ({release_duration}/{RELEASE_DURATION})")

            return q_next + [False]  # Gripper open

        elif self.state == STATE_SEARCH_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_SEARCH_RED - Sprint 13
            # ============================================================
            # Move to search position first
            state_duration = tt - self.state_start_tt

            if state_duration < SEARCH_POSE_WAIT_TIME:
                # Still moving to search pose
                if tt % 50 == 0:
                    print(f"[SEARCH_BLUE] Moving to search pose... ({state_duration}/{SEARCH_POSE_WAIT_TIME})")
                return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]
            else:
                # Arrived at search pose, start detecting blue block
                cx, cy, detected = process_image(current_image_bgr, 'blue')

                if detected:
                    # Test Sprint 5 functions: compute errors and motions
                    err_x, err_y = compute_pixel_error(cx, cy)
                    delta_x, delta_y = pixel_to_robot_motion(err_x, err_y)

                    print(f"[SEARCH_BLUE] BLUE BLOCK DETECTED at pixel ({cx}, {cy})")
                    print(f"[SEARCH_BLUE] Target pixels: ({TARGET_CX}, {TARGET_CY})")
                    print(f"[SEARCH_BLUE] Pixel errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                    print(f"[SEARCH_BLUE] Robot motion: delta_x={delta_x:.4f}m, delta_y={delta_y:.4f}m")
                    print(f"[SEARCH_BLUE] Transitioning to coarse alignment...")
                    self.transition_to_state(STATE_COARSE_ALIGN_BLUE, tt)
                    return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]
                else:
                    if tt % 50 == 0:
                        print(f"[SEARCH_BLUE] Searching for blue block... (not detected yet)")
                    return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]

        elif self.state == STATE_COARSE_ALIGN_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_COARSE_ALIGN_RED - Sprint 13
            # ============================================================
            # Coarse alignment - sequential X then Y alignment
            # Phase 1: Align X until within tolerance
            # Phase 2: Align Y until within tolerance
            # Then transition to fine alignment

            # DEBUG: Entry check
            if tt % 50 == 0:
                print(f"[COARSE_DEBUG] Entered coarse alignment state, phase={self.coarse_phase}")

            # Detect blue block
            cx, cy, detected = process_image(current_image_bgr, 'blue')

            # DEBUG: Detection status
            if tt % 50 == 0:
                print(f"[COARSE_DEBUG] Detection: detected={detected}, cx={cx}, cy={cy}")

            if not detected:
                # Lost the block - stay in place and keep searching
                if tt % 50 == 0:
                    print(f"[COARSE_ALIGN_BLUE] Block not detected, searching...")
                return current_q + [False]

            # Compute pixel errors
            err_x, err_y = compute_pixel_error(cx, cy)

            # PHASE 1: Align X direction
            if self.coarse_phase == 'align_x':
                # DEBUG: Phase check
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] In align_x phase, err_x={err_x:.1f}, tolerance={COARSE_TOLERANCE_X}")

                # Check if X is aligned
                if abs(err_x) < COARSE_TOLERANCE_X:
                    self.align_counter_x += 1
                    print(f"[COARSE] X within tolerance ({self.align_counter_x}/{REQUIRED_ALIGN_FRAMES})")

                    if self.align_counter_x >= REQUIRED_ALIGN_FRAMES:
                        print(f"[COARSE_ALIGN_BLUE] X aligned! Moving to Y alignment phase...")
                        self.coarse_phase = 'align_y'
                        self.align_counter_x = 0  # Reset for next time
                        return current_q + [False]
                    else:
                        # Within tolerance but waiting for stability
                        return current_q + [False]
                else:
                    # Not within tolerance - reset counter
                    self.align_counter_x = 0

                # DEBUG: Not aligned
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] X not aligned yet (err_x={err_x:.1f} > {COARSE_TOLERANCE_X})")

                # X not aligned - check settle time
                time_since_move = tt - self.move_timer

                # DEBUG: Settle time check
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] Settle check: time_since_move={time_since_move}, threshold={MOVE_SETTLE_TIME}, move_timer={self.move_timer}")

                if time_since_move < MOVE_SETTLE_TIME:
                    if tt % 50 == 0:
                        print(f"[COARSE_DEBUG] Still settling, waiting...")
                    return current_q + [False]

                # DEBUG: Ready to move
                if tt % 50 == 0:
                    print(f"[COARSE_DEBUG] Ready to move!")

                # Move in X direction only (zero out Y error)
                delta_x, delta_y = pixel_to_robot_motion(err_x, 0)

                # DEBUG: Show raw delta
                print(f"[COARSE_X_MOVE] tt={tt} | err_x={err_x:.1f} | RAW delta_x={delta_x:.6f}m")

                # Limit to coarse step size
                delta_magnitude = abs(delta_x)
                if delta_magnitude > COARSE_STEP_SIZE:
                    delta_x = COARSE_STEP_SIZE if delta_x > 0 else -COARSE_STEP_SIZE

                # DEBUG: Show limited delta
                print(f"[COARSE_X_MOVE] After limiting: delta_x={delta_x:.6f}m (limit={COARSE_STEP_SIZE})")

                # Get current position and create target
                T_current = get_full_fk(current_q)
                current_pos = T_current[0:3, 3]
                # Image X (horizontal) controls Robot Y, so move in Y direction
                target_pos = current_pos + np.array([0.0, delta_y, 0.0])

                # DEBUG: Show positions
                print(f"[COARSE_X_MOVE] Current pos: {current_pos}")
                print(f"[COARSE_X_MOVE] Target pos:  {target_pos}")

                # Move using IK
                q_next = move_to_position(current_q, target_pos, max_step=COARSE_STEP_SIZE, dt=0.016)
                self.move_timer = tt

                # DEBUG: Show joint changes
                q_diff = np.array(q_next) - np.array(current_q)
                print(f"[COARSE_X_MOVE] Joint changes: {q_diff}")

                return q_next + [False]

            # PHASE 2: Align Y direction
            elif self.coarse_phase == 'align_y':
                # Check if Y is aligned
                if abs(err_y) < COARSE_TOLERANCE_Y:
                    self.align_counter_y += 1
                    print(f"[COARSE] Y within tolerance ({self.align_counter_y}/{REQUIRED_ALIGN_FRAMES})")

                    if self.align_counter_y >= REQUIRED_ALIGN_FRAMES:
                        print(f"[COARSE_ALIGN_BLUE] Y aligned! Coarse alignment complete!")
                        print(f"[COARSE_ALIGN_BLUE] Final errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                        self.coarse_phase = 'align_x'  # Reset for next use
                        self.align_counter_y = 0  # Reset for next time
                        self.transition_to_state(STATE_FINE_ALIGN_BLUE, tt)
                        return current_q + [False]
                    else:
                        # Within tolerance but waiting for stability
                        return current_q + [False]
                else:
                    # Not within tolerance - reset counter
                    self.align_counter_y = 0

                # Y not aligned - check settle time
                time_since_move = tt - self.move_timer
                if time_since_move < MOVE_SETTLE_TIME:
                    return current_q + [False]

                # Move in Y direction only (zero out X error)
                delta_x, delta_y = pixel_to_robot_motion(0, err_y)

                # Limit to coarse step size
                delta_magnitude = abs(delta_y)
                if delta_magnitude > COARSE_STEP_SIZE:
                    delta_y = COARSE_STEP_SIZE if delta_y > 0 else -COARSE_STEP_SIZE

                # Get current position and create target
                T_current = get_full_fk(current_q)
                current_pos = T_current[0:3, 3]
                # Image Y (vertical) controls Robot X, so move in X direction
                target_pos = current_pos + np.array([delta_x, 0.0, 0.0])

                # Move using IK
                q_next = move_to_position(current_q, target_pos, max_step=COARSE_STEP_SIZE, dt=0.016)
                self.move_timer = tt

                if tt % 20 == 0:
                    print(f"[COARSE_Y] Block at ({cx}, {cy}) | err_y={err_y:.1f} | Moving delta_y={delta_y:.4f}m")

                return q_next + [False]

        elif self.state == STATE_FINE_ALIGN_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_FINE_ALIGN_RED - Sprint 13
            # ============================================================
            # Sprint 7: Two-phase fine alignment
            # Phase A: Descend to approach_height (~15cm)
            # Phase B: Fine lateral alignment at that lower height

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            current_z = current_pos[2]

            # Debug: Print phase every 50 timesteps
            if tt % 50 == 0:
                print(f"[FINE_DEBUG] tt={tt} | phase={self.fine_phase} | height={current_z:.4f}m | target={APPROACH_HEIGHT}m")

            # PHASE A: Pre-alignment descent
            if self.fine_phase == 'descending':
                # Rate-limited descent
                self.fine_descent_counter += 1

                if self.fine_descent_counter >= PRE_DESCENT_WAIT:
                    # Time to take another descent step
                    target_z = max(current_z - PRE_DESCENT_STEP, APPROACH_HEIGHT)

                    # Check if we've reached approach height (target_z stopped decreasing)
                    if abs(target_z - APPROACH_HEIGHT) < 0.001:  # Within 1mm of target
                        print(f"[FINE_ALIGN] Reached approach height ({current_z:.4f}m). Starting fine alignment phase...")
                        self.fine_phase = 'aligning'
                        self.move_timer = tt  # Reset move timer for alignment phase
                        return current_q + [False]

                    target_pos = current_pos.copy()
                    target_pos[2] = target_z

                    # Move using IK (keep X, Y constant, only change Z)
                    q_next = move_to_position(current_q, target_pos, max_step=0.008, dt=0.016)
                    self.fine_descent_counter = 0  # Reset counter

                    print(f"[FINE_DESCEND] tt={tt} | Height: {current_z:.4f}m -> {target_z:.4f}m")

                    return q_next + [False]
                else:
                    # Still waiting for next descent step
                    return current_q + [False]

            # PHASE B: Fine alignment at lower height
            elif self.fine_phase == 'aligning':
                # Detect blue block
                cx, cy, detected = process_image(current_image_bgr, 'blue')

                # Debug: Detection status
                if tt % 50 == 0:
                    print(f"[FINE_DEBUG] Aligning phase | detected={detected} | cx={cx}, cy={cy}")

                if not detected:
                    # Lost the block - stay in place
                    if tt % 50 == 0:
                        print(f"[FINE_ALIGN] Block not detected, searching...")
                    return current_q + [False]

                # Compute pixel errors using FINE alignment targets
                err_x = cx - FINE_TARGET_CX
                err_y = cy - FINE_TARGET_CY

                # Debug: Show errors and targets
                if tt % 50 == 0:
                    print(f"[FINE_DEBUG] Block at ({cx}, {cy}) | Target: ({FINE_TARGET_CX}, {FINE_TARGET_CY})")
                    print(f"[FINE_DEBUG] err_x={err_x:.1f}, err_y={err_y:.1f} | tol_x={FINE_TOLERANCE_X}, tol_y={FINE_TOLERANCE_Y}")

                # Check if aligned (both X and Y within tolerance)
                x_aligned = abs(err_x) < FINE_TOLERANCE_X
                y_aligned = abs(err_y) < FINE_TOLERANCE_Y

                if x_aligned and y_aligned:
                    self.fine_align_counter += 1
                    print(f"[FINE_ALIGN] Aligned! ({self.fine_align_counter}/{REQUIRED_ALIGN_FRAMES})")

                    if self.fine_align_counter >= REQUIRED_ALIGN_FRAMES:
                        print(f"[FINE_ALIGN] Fine alignment complete!")
                        print(f"[FINE_ALIGN] Final position: cx={cx}, cy={cy} | Target: ({FINE_TARGET_CX}, {FINE_TARGET_CY})")
                        print(f"[FINE_ALIGN] Final errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                        print(f"[FINE_ALIGN] Final height: {current_z:.4f}m")
                        self.transition_to_state(STATE_DESCEND_BLUE, tt)
                        return current_q + [False]
                    else:
                        # Within tolerance but waiting for stability
                        return current_q + [False]
                else:
                    # Not aligned - reset counter
                    self.fine_align_counter = 0

                # Check if we've exceeded max attempts
                if self.fine_attempt_counter >= MAX_FINE_ATTEMPTS:
                    print(f"[FINE_ALIGN] Max attempts ({MAX_FINE_ATTEMPTS}) reached. Forcing descent...")
                    print(f"[FINE_ALIGN] Final position: cx={cx}, cy={cy} | Target: ({FINE_TARGET_CX}, {FINE_TARGET_CY})")
                    print(f"[FINE_ALIGN] Final errors: err_x={err_x:.1f}, err_y={err_y:.1f}")
                    self.transition_to_state(STATE_DESCEND_BLUE, tt)
                    return current_q + [False]

                # Check settle time
                time_since_move = tt - self.move_timer
                if time_since_move < MOVE_SETTLE_TIME:
                    return current_q + [False]

                # Ready to make alignment move
                self.fine_attempt_counter += 1

                # Convert pixel errors to robot motion
                delta_x, delta_y = pixel_to_robot_motion(err_x, err_y)

                # Limit to fine step size
                delta_magnitude = np.sqrt(delta_x**2 + delta_y**2)
                if delta_magnitude > FINE_STEP_SIZE:
                    scale = FINE_STEP_SIZE / delta_magnitude
                    delta_x *= scale
                    delta_y *= scale

                # Create target position - EXPLICITLY maintain Z at approach height
                target_pos = current_pos.copy()
                target_pos[0] += delta_x  # Apply X adjustment
                target_pos[1] += delta_y  # Apply Y adjustment
                target_pos[2] = APPROACH_HEIGHT  # Lock Z at approach height

                # Move using IK
                q_next = move_to_position(current_q, target_pos, max_step=FINE_STEP_SIZE, dt=0.016)
                self.move_timer = tt

                if tt % 30 == 0:
                    print(f"[FINE_ALIGN] Attempt {self.fine_attempt_counter} | Block at ({cx}, {cy}) → Target ({FINE_TARGET_CX}, {FINE_TARGET_CY}) | err=({err_x:.1f}, {err_y:.1f}) | delta=({delta_x:.4f}, {delta_y:.4f})")

                return q_next + [False]

        elif self.state == STATE_DESCEND_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_DESCEND_RED - Sprint 13
            # ============================================================
            # Sprint 8: Controlled descent from approach_height to grasp_height
            # Rate-limited, small steps, keep X/Y constant

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            current_z = current_pos[2]

            # DEBUG: Print height and counter status
            if tt % 50 == 0:
                print(f"[DESCEND_BLUE_DEBUG] current_z={current_z:.4f}m, GRASP_HEIGHT={GRASP_HEIGHT:.4f}m, counter={self.descent_counter}/{DESCENT_WAIT_TIME}")

            # Check if we've reached grasp height (with 1mm tolerance for floating-point precision)
            if current_z < GRASP_HEIGHT + 0.001:
                print(f"[DESCEND_BLUE] Reached grasp height ({current_z:.4f}m). Transitioning to grasp...")
                self.transition_to_state(STATE_GRASP_BLUE, tt)
                return current_q + [False]

            # Rate-limited descent
            self.descent_counter += 1

            if self.descent_counter >= DESCENT_WAIT_TIME:
                # Time to take another descent step
                target_z = max(current_z - DESCENT_STEP_SIZE, GRASP_HEIGHT)
                target_pos = current_pos.copy()
                target_pos[2] = target_z

                # Move using IK (keep X, Y constant, only change Z)
                q_next = move_to_position(current_q, target_pos, max_step=0.008, dt=0.016)
                self.descent_counter = 0  # Reset counter

                if tt % 30 == 0:
                    print(f"[DESCEND_BLUE] tt={tt} | Height: {current_z:.4f}m → {target_z:.4f}m (target: {GRASP_HEIGHT}m)")

                return q_next + [False]
            else:
                # Still waiting for next descent step
                if tt % 50 == 0:
                    print(f"[DESCEND_BLUE_DEBUG] Waiting... counter={self.descent_counter}/{DESCENT_WAIT_TIME}")
                return current_q + [False]

        elif self.state == STATE_GRASP_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_GRASP_RED - Sprint 13
            # ============================================================
            # Sprint 9: Close gripper while holding position
            # Hold for GRASP_DURATION timesteps to ensure secure grasp

            # Calculate how long we've been grasping
            grasp_duration = tt - self.grasp_start_tt

            # Check if grasp is complete
            if grasp_duration >= GRASP_DURATION:
                print(f"[GRASP_BLUE] Grasp complete ({grasp_duration} timesteps). Transitioning to lift...")
                self.transition_to_state(STATE_LIFT_BLUE, tt)
                return current_q + [True]  # Keep gripper closed

            # Hold position while gripper closes
            # Use small max_step to minimize drift
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            target_pos = current_pos.copy()  # Stay at current position

            q_next = move_to_position(current_q, target_pos, max_step=0.005, dt=0.016)

            # Debug output
            if grasp_duration == 0:
                print(f"[GRASP_BLUE] Starting grasp at height {current_pos[2]:.4f}m")
            if tt % 30 == 0:
                print(f"[GRASP_BLUE] Holding position... ({grasp_duration}/{GRASP_DURATION})")

            return q_next + [True]  # Gripper closed

        elif self.state == STATE_LIFT_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_LIFT_RED - Sprint 13
            # ============================================================
            # Sprint 10: Controlled lift from grasp_height to lift_height
            # Rate-limited, small steps, keep X/Y constant, gripper closed

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            current_z = current_pos[2]

            # Check if we've reached lift height (with 1mm tolerance for floating-point precision)
            if current_z > LIFT_HEIGHT - 0.001:
                print(f"[LIFT_BLUE] Reached lift height ({current_z:.4f}m). Transitioning to carry...")
                self.transition_to_state(STATE_CARRY_BLUE, tt)
                return current_q + [True]  # Keep gripper closed

            # Rate-limited lift
            self.lift_counter += 1

            if self.lift_counter >= LIFT_WAIT_TIME:
                # Time to take another lift step
                target_z = min(current_z + LIFT_STEP_SIZE, LIFT_HEIGHT)
                target_pos = current_pos.copy()
                target_pos[2] = target_z

                # Move using IK (keep X, Y constant, only change Z)
                q_next = move_to_position(current_q, target_pos, max_step=0.008, dt=0.016)
                self.lift_counter = 0  # Reset counter

                if tt % 30 == 0:
                    print(f"[LIFT_BLUE] tt={tt} | Height: {current_z:.4f}m → {target_z:.4f}m (target: {LIFT_HEIGHT}m)")

                return q_next + [True]  # Keep gripper closed
            else:
                # Still waiting for next lift step
                return current_q + [True]  # Keep gripper closed

        elif self.state == STATE_CARRY_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_CARRY_RED - Sprint 13
            # ============================================================
            # Sprint 11: Carry block to bin using smooth Cartesian motion
            # Move gradually to bin position, pause to settle, then transition to release

            # Get current position
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]

            # Get target position from POSE_OVER_BIN
            T_bin = get_full_fk(POSE_OVER_BIN['q'])
            target_pos = T_bin[0:3, 3]

            # Calculate distance to target
            distance = np.linalg.norm(target_pos - current_pos)

            # Check if we've arrived at bin
            if distance < CARRY_ARRIVAL_THRESHOLD:
                # Mark arrival time if not already marked
                if self.carry_arrival_tt is None:
                    self.carry_arrival_tt = tt
                    print(f"[CARRY_BLUE] Arrived at bin (distance: {distance:.4f}m). Settling...")

                # Check if settle time is complete
                settle_duration = tt - self.carry_arrival_tt
                if settle_duration >= CARRY_SETTLE_TIME:
                    print(f"[CARRY_BLUE] Settle complete ({settle_duration} timesteps). Transitioning to release...")
                    self.transition_to_state(STATE_RELEASE_BLUE, tt)
                    return current_q + [True]  # Keep gripper closed
                else:
                    # Still settling - hold position
                    if tt % 30 == 0:
                        print(f"[CARRY_BLUE] Settling... ({settle_duration}/{CARRY_SETTLE_TIME})")
                    return current_q + [True]  # Keep gripper closed

            # First timestep - print start message
            carry_duration = tt - self.state_start_tt
            if carry_duration == 0:
                print(f"[CARRY_BLUE] Starting carry from {current_pos} to bin at {target_pos}")
                print(f"[CARRY_BLUE] Distance to bin: {distance:.4f}m")

            # Smooth motion towards bin using IK
            q_next = move_to_position(current_q, target_pos, max_step=CARRY_MAX_STEP, dt=0.016)

            # Debug output
            if tt % 50 == 0:
                print(f"[CARRY_BLUE] Moving to bin... distance={distance:.4f}m")

            return q_next + [True]  # Keep gripper closed

        elif self.state == STATE_RELEASE_BLUE:
            # ============================================================
            # DUPLICATE OF STATE_RELEASE_RED - Sprint 13
            # ============================================================
            # Sprint 12: Open gripper and drop block while holding position
            # Hold for RELEASE_DURATION timesteps to let block fall

            # Calculate how long we've been releasing
            release_duration = tt - self.release_start_tt

            # Check if release is complete
            if release_duration >= RELEASE_DURATION:
                print(f"[RELEASE_BLUE] Release complete ({release_duration} timesteps). Transitioning to done...")
                self.transition_to_state(STATE_DONE, tt)  # <- Changed to STATE_DONE
                return current_q + [False]  # Keep gripper open

            # Hold position while gripper opens
            # Use small max_step to minimize drift
            T_current = get_full_fk(current_q)
            current_pos = T_current[0:3, 3]
            target_pos = current_pos.copy()  # Stay at current position

            q_next = move_to_position(current_q, target_pos, max_step=0.005, dt=0.016)

            # Debug output
            if release_duration == 0:
                print(f"[RELEASE_BLUE] Opening gripper at height {current_pos[2]:.4f}m")
            if tt % 30 == 0:
                print(f"[RELEASE_BLUE] Releasing... ({release_duration}/{RELEASE_DURATION})")

            return q_next + [False]  # Gripper open

        elif self.state == STATE_TEST_HARDCODED_MOVE:
            # Test: directly command SEARCH_BLOCKS joint angles
            if tt % 50 == 0:
                T_current = get_full_fk(current_q)
                T_target = get_full_fk(POSE_SEARCH_BLOCKS['q'])
                pos_current = T_current[0:3, 3]
                pos_target = T_target[0:3, 3]
                distance = np.linalg.norm(pos_target - pos_current)
                print(f"[TEST_HARDCODED_MOVE] At SEARCH pose | Distance to target: {distance:.3f}m")
            return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]

        elif self.state == STATE_TEST_IK_CARTESIAN:
            # Test: Use IK to lift end-effector 10cm straight up from SEARCH position
            state_duration = tt - self.state_start_tt

            if state_duration < 100:
                # Hold at SEARCH position first
                if tt % 50 == 0:
                    print(f"[TEST_IK_CARTESIAN] Holding at SEARCH position")
                return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]
            else:
                # Use IK to move to target position (10cm above SEARCH)
                # Calculate target: SEARCH end-effector position + [0, 0, 0.1]
                T_search = get_full_fk(POSE_SEARCH_BLOCKS['q'])
                search_pos = T_search[0:3, 3]
                target_pos = search_pos + np.array([0.0, 0.0, 0.1])  # Lift 10cm in Z

                # Use move_to_position to approach target
                q_next = move_to_position(current_q, target_pos, max_step=0.02, dt=0.016)

                # Debug: print distance to target every 50 timesteps
                if tt % 50 == 0:
                    T_current = get_full_fk(current_q)
                    current_pos = T_current[0:3, 3]
                    distance = np.linalg.norm(target_pos - current_pos)
                    print(f"[TEST_IK_CARTESIAN] Distance to target: {distance:.4f}m | Target: {target_pos}")

                return q_next + [POSE_SEARCH_BLOCKS['gripper']]

        elif self.state == STATE_DONE:
            # TODO: Sprint 14
            # Hold final position
            return current_q + [False]

        # Default fallback - return to search position
        return POSE_SEARCH_BLOCKS['q'] + [POSE_SEARCH_BLOCKS['gripper']]

    def transition_to_state(self, new_state, tt):
        """Helper function to transition between states"""
        if self.debug:
            print(f"\n[tt={tt:5d}] STATE TRANSITION: {STATE_NAMES[self.state]} -> {STATE_NAMES[new_state]}\n")
        self.state = new_state
        self.state_start_tt = tt

        # Reset coarse phase and move timer when entering coarse alignment states
        if new_state == STATE_COARSE_ALIGN_RED or new_state == STATE_COARSE_ALIGN_BLUE:
            self.coarse_phase = 'align_x'
            self.move_timer = 0  # Reset so first move happens immediately
            self.align_counter_x = 0  # Reset alignment counters
            self.align_counter_y = 0

        # Reset fine alignment phase when entering fine alignment states
        if new_state == STATE_FINE_ALIGN_RED or new_state == STATE_FINE_ALIGN_BLUE:
            self.fine_phase = 'descending'
            self.fine_descent_counter = 0
            self.fine_attempt_counter = 0
            self.fine_align_counter = 0
            self.move_timer = 0

        # Reset descent counter when entering descent states
        if new_state == STATE_DESCEND_RED or new_state == STATE_DESCEND_BLUE:
            self.descent_counter = 0

        # Reset grasp timer when entering grasp states
        if new_state == STATE_GRASP_RED or new_state == STATE_GRASP_BLUE:
            self.grasp_start_tt = tt

        # Reset lift counter when entering lift states
        if new_state == STATE_LIFT_RED or new_state == STATE_LIFT_BLUE:
            self.lift_counter = 0

        # Reset carry arrival timer when entering carry states
        if new_state == STATE_CARRY_RED or new_state == STATE_CARRY_BLUE:
            self.carry_arrival_tt = None

        # Reset release timer when entering release states
        if new_state == STATE_RELEASE_RED or new_state == STATE_RELEASE_BLUE:
            self.release_start_tt = tt
