import math

# --- 1. Constants and Helper Functions ---

# Simulation timestep in seconds
TIMESTEP = 0.016  # 62.5 Hz
# Maximum allowed joint velocity in rad/s
MAX_JOINT_VEL = 0.5
# Gripper pause duration in timesteps (0.8 seconds)
GRIPPER_PAUSE_TIMESTEPS = 50

def get_cubic_s(t_current, T_total):
    """
    Calculates the cubic polynomial scaling factor 's' and its derivative 's_dot'.
    s(t) goes from 0 to 1 as t goes from 0 to T_total.
    This is the standard "starts and ends at rest" cubic trajectory.
    
    s(t) = 3*(t/T)^2 - 2*(t/T)^3
    """
    if T_total == 0:
        return 1.0 # Instantly at the end
        
    t_norm = t_current / T_total
    
    # Clamp to [0, 1] to prevent overshooting
    if t_norm > 1.0:
        t_norm = 1.0
    elif t_norm < 0.0:
        t_norm = 0.0

    s = 3.0 * (t_norm ** 2) - 2.0 * (t_norm ** 3)
    return s

# --- 2. Keyframe Definitions ---
# These are the "goal" poses from your HW1 solution.
# Each keyframe is: ([list_of_7_joint_angles], gripper_is_closed_bool)

# We define a neutral "home" position to start from.
KF_HOME = ([0.0, 0.0, 0.0, -1.70, -1.57, 1.4, 0.785], False)

# --- Block 1 Pick-and-Place ---
KF_PRE_GRASP_1 = ([-0.2, 0.0, 0.0, -1.70, -1.57, 1.4, 0.785], False)
KF_DOWN_TO_BLOCK_1 = ([-0.2, 0.8, 0.0, -1.70, -1.57, 1.4, 0.785], False)
KF_ADJUST_GRASP_1 = ([-0.2, 0.82, 0.0, -1.70, -1.57, 1.7, 0.7], False)
KF_CLOSE_GRIPPER_1 = ([-0.2, 0.82, 0.0, -1.70, -1.57, 1.7, 0.7], True)  # Gripper action
KF_LIFT_BLOCK_1 = ([-0.2, 0.0, 0.0, -1.70, -1.57, 1.7, 0.7], True)
KF_MOVE_TO_BIN_1 = ([1.0, 0.0, 0.0, -1.70, -1.57, 1.7, 0.7], True)
KF_OVER_BIN_1 = ([1.0, 0.0, 0.0, -1.70, -1.57, 2.3, 0.7], True)
KF_OPEN_GRIPPER_1 = ([1.0, 0.0, 0.0, -1.70, -1.57, 2.3, 0.7], False) # Gripper action

# --- Block 2 Pick-and-Place ---
KF_PRE_GRASP_2 = ([-0.3, 0.0, 0.0, -1.70, -1.57, 1.4, 0.785], False)
KF_DOWN_TO_BLOCK_2 = ([-0.3, 0.75, 0.0, -1.70, -1.57, 1.4, 0.785], False)
KF_ADJUST_GRASP_2A = ([-0.3, 0.75, 0.0, -1.70, -1.57, 2.5, 0.785], False)
KF_ADJUST_GRASP_2B = ([-0.3, 0.87, 0.0, -1.70, -1.57, 2.5, 0.385], False)
KF_CLOSE_GRIPPER_2 = ([-0.3, 0.87, 0.0, -1.70, -1.57, 2.5, 0.385], True) # Gripper action
KF_LIFT_BLOCK_2 = ([-0.3, 0.0, 0.0, -1.70, -1.57, 2.5, 0.385], True)
KF_MOVE_TO_BIN_2 = ([1.0, 0.0, 0.0, -1.70, -1.57, 2.5, 0.385], True)
KF_OPEN_GRIPPER_2 = ([1.0, 0.0, 0.0, -1.70, -1.57, 2.5, 0.385], False) # Gripper action
KF_FINAL_POSE = ([1.0, 0.0, 0.0, -1.70, -1.57, 2.5, 0.385], False) # Stay here

# This list defines the entire sequence of motions
KEYFRAMES = [
    KF_HOME,
    KF_PRE_GRASP_1,
    KF_DOWN_TO_BLOCK_1,
    KF_ADJUST_GRASP_1,
    KF_CLOSE_GRIPPER_1,  # Segment 3 (Pause)
    KF_LIFT_BLOCK_1,     # Segment 4
    KF_MOVE_TO_BIN_1,
    KF_OVER_BIN_1,
    KF_OPEN_GRIPPER_1,   # Segment 7 (Pause)
    KF_PRE_GRASP_2,
    KF_DOWN_TO_BLOCK_2,
    KF_ADJUST_GRASP_2A,
    KF_ADJUST_GRASP_2B,
    KF_CLOSE_GRIPPER_2,  # Segment 12 (Pause)
    KF_LIFT_BLOCK_2,
    KF_MOVE_TO_BIN_2,
    KF_OPEN_GRIPPER_2,   # Segment 15 (Pause)
    KF_FINAL_POSE        # Stay here
]

# --- 3. Pre-computation (Runs ONCE) ---
# Calculate the duration (in timesteps) for each segment.
# A "segment" is the motion from one keyframe to the next.

SEGMENT_DURATIONS_TIMESTEPS = []
for i in range(len(KEYFRAMES) - 1):
    start_angles = KEYFRAMES[i][0]
    end_angles = KEYFRAMES[i+1][0]
    
    # Check if this is a "pause" segment (for gripper)
    if start_angles == end_angles:
        # This is a gripper action, so we just pause
        N_timesteps = GRIPPER_PAUSE_TIMESTEPS
    else:
        # This is a motion segment. Find the duration T based on the
        # single joint that has to move the farthest/fastest.
        
        # Find the largest angle change |delta_theta| for any joint
        max_delta_theta = 0.0
        for j in range(7):
            delta = abs(end_angles[j] - start_angles[j])
            if delta > max_delta_theta:
                max_delta_theta = delta
        
        # From our cubic trajectory, max_vel = (1.5 / T) * |delta_theta|
        # We solve for T: T = 1.5 * |delta_theta| / max_vel
        # We use the provided MAX_JOINT_VEL, 0.5 rad/s
        # T_seconds = 1.5 * max_delta_theta / MAX_JOINT_VEL
        
        # A simpler derivation: T = 3 * |delta_theta_max| / max_vel
        T_seconds = 3.0 * max_delta_theta / MAX_JOINT_VEL
        
        # Convert total time in seconds to total timesteps
        # Add 1 to ensure we have at least one step and avoid division by zero
        N_timesteps = int(math.ceil(T_seconds / TIMESTEP)) + 1
        
    SEGMENT_DURATIONS_TIMESTEPS.append(N_timesteps)

# --- 4. State Machine Variables ---
# These global variables track our progress through the trajectory
segment_index = 0
segment_start_tt = 0

# --- 5. Main Robot Command Function ---

def getDesiredRobotCommand(tt):
    """
    This function is called by the simulator at every timestep.
    It calculates and returns the correct joint angles for the smooth trajectory.
    """
    global segment_index, segment_start_tt

    # --- Part A: Check if we are done with all segments ---
    if segment_index >= len(SEGMENT_DURATIONS_TIMESTEPS):
        # We've completed all motions. Just stay at the last keyframe.
        final_angles = KEYFRAMES[-1][0]
        final_grip = KEYFRAMES[-1][1]
        return final_angles + [final_grip]

    # --- Part B: Get current segment info ---
    N_timesteps_for_segment = SEGMENT_DURATIONS_TIMESTEPS[segment_index]
    tt_in_segment = tt - segment_start_tt
    
    # --- Part C: Check if we need to advance to the NEXT segment ---
    if tt_in_segment >= N_timesteps_for_segment:
        # This segment is done. Move to the next one.
        segment_index += 1
        segment_start_tt = tt
        
        # Reset trackers for the new segment
        tt_in_segment = 0
        
        # Check again if we just finished the *last* segment
        if segment_index >= len(SEGMENT_DURATIONS_TIMESTEPS):
            final_angles = KEYFRAMES[-1][0]
            final_grip = KEYFRAMES[-1][1]
            return final_angles + [final_grip]
            
        # Get the duration for the new, active segment
        N_timesteps_for_segment = SEGMENT_DURATIONS_TIMESTEPS[segment_index]

    # --- Part D: Calculate the command for the CURRENT segment ---
    
    # Get the start and end keyframes for *this* segment
    start_kf = KEYFRAMES[segment_index]
    end_kf = KEYFRAMES[segment_index + 1]
    
    start_angles = start_kf[0]
    start_grip = start_kf[1]
    end_angles = end_kf[0]
    end_grip = end_kf[1]

    # Check if this is a "pause" segment (gripper action)
    if start_angles == end_angles:
        # We are in a pause. Hold the position.
        # Command the *new* gripper state so it has time to act.
        return end_angles + [end_grip]
        
    # --- Part E: It's a motion segment. Calculate the trajectory! ---
    
    # Calculate total segment time in seconds
    T_total_seconds = N_timesteps_for_segment * TIMESTEP
    
    # Calculate how far into this segment we are (in seconds)
    t_current_seconds = tt_in_segment * TIMESTEP
    
    # Get the scaling factor 's' from our cubic polynomial
    s = get_cubic_s(t_current_seconds, T_total_seconds)
    
    # Calculate the intermediate joint angles
    current_angles = []
    for i in range(7):
        # Linear interpolation: q(t) = q_start + s(t) * (q_end - q_start)
        delta = end_angles[i] - start_angles[i]
        angle = start_angles[i] + s * delta
        current_angles.append(angle)
        
    # During the motion, we keep the gripper in its *starting* state
    # (e.g., we keep the gripper closed while moving the block)
    return current_angles + [start_grip]

#eof


#def getDesiredRobotCommand(tt):
    # tt: current simulation timestep
    # a timestep is 0.016s (62.5 Hz)

    # write your code here...
    # you should return a eight-element list
    # the first seven entries are the joint angles
    # the last entry is a bool for whether the gripper
    # should be closed

    # you should use linear motion with 3rd-order polynomial scaling
    # in the configuration space to move between the different joint configurations

    # your maximum joint velocity should be 0.5 rad/s

    
    # replace the following code with your code:
    #if tt < 200:
        #return [0.0, 0.0, 0.0, -1.70, -1.57, 1.4, 0.785, True]
    #elif tt < 400:
        #return [0.5, 0.2, 0.0, -1.50, -1.57, 1.4, 0.785, False]
    #else:
        #return [-0.5, 0.2, 0.0, -1.50, -1.57, 1.4, 0.785, True]

