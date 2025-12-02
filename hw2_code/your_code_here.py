import numpy as np
from scipy.spatial.transform import Rotation as R # you might find this useful for calculating the error
from kinematic_helpers import *
    

def getDesiredRobotCommand(tt, desired_pose, current_q):
    # tt: current simulation timestep
    # desired_pose: a list of six for the desired position (3) and desired exponential coordinates (3)
    # current q is the list of six joint angles: initially the current six joint angles, and then the last commanded joint angles after that
    # a timestep is 0.016s (62.5 Hz)

    # write your code here...
    # you should return a six-element list of the joint angles in radians

    # your goal is to program the inverse kinematics so the 
    # robot tracks the red circle to trace a W using the Newton-Raphson method
    
    # you are provided with the desired_pose
    # you do not need to use tt in this assignment
    
    # you have been provided with kinematic helper functions
    # for each one of the robot transforms. You can use these to form the
    # Jacobian. Note, the first z-axis in the jacobian should be for the identity
    # matrix (the first joint rotates about the z-axis). If you start with T01, you
    # will miss the first Jacobian component!
    
    # you are welcome to break it down with as many helper functions as necessary
    # you do not need to analytically solve for each entry of the Jacobian matrix.

    # replace this with your IK solution! :)
    
    
    # --- Tunable Parameters ---
    num_iterations = 10  # Number of IK iterations per timestep
    damping = 0.1      # Damping factor (lambda) for DLS
    
    # --- Convert inputs to numpy arrays ---
    q_command = np.array(current_q)
    pos_desired = np.array(desired_pose[0:3])
    rot_desired_vec = np.array(desired_pose[3:6]) # This is an axis-angle vector
    
    I = np.identity(6) # Identity matrix
    
    # --- Newton-Raphson (Damped Least Squares) Loop ---
    for i in range(num_iterations):
        
        # 1. Get current pose (Forward Kinematics)
        T_06 = get_full_fk(q_command)
        pos_current = T_06[0:3, 3]
        rot_current_mat = T_06[0:3, 0:3]
        
        # 2. Calculate the 6D pose error
        # Position error is simple subtraction
        pos_error = pos_desired - pos_current
        
        # Orientation error
        r_current = R.from_matrix(rot_current_mat)
        rot_current_vec = r_current.as_rotvec()
        rot_error = rot_desired_vec - rot_current_vec
        
        # Combine into a 6D error vector
        error_6D = np.concatenate((pos_error, rot_error))

        # 3. Get the Jacobian at the current configuration
        J = get_jacobian(q_command)
        
        # 4. Solve for delta_q using Damped Least Squares
        # The formula is: delta_q = J_T * inv(J * J_T + damping^2 * I) * error_6D
        J_T = J.T
        
        # Solve the linear system: (J @ J_T + damping**2 * I) @ y = error_6D
        # This is more stable than computing the inverse directly
        y = np.linalg.solve(J @ J_T + (damping**2) * I, error_6D)
        
        # Get delta_q: delta_q = J_T @ y
        delta_q = J_T @ y
        
        # 5. Update the joint angle command
        q_command = q_command + delta_q
        
    # After all iterations, return the final command as a list
    return q_command.tolist()
    
    #q = [0,0,0,0,0,0]
    #return q
   
def get_full_fk(q):
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2], q[3], q[4], q[5]
    T_01 = T01(q1)
    T_12 = T12(q2)
    T_23 = T23(q3)
    T_34 = T34(q4)
    T_45 = T45(q5)
    T_56 = T56(q6)
    
    T_06 = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56
    return T_06
    
def get_jacobian(q):
    """
    Calculates the 6x6 geometric Jacobian for the UR5e at a given joint configuration 'q'.
    """
    J = np.zeros((6, 6))
    
    # Base frame (for joint 1)
    T_00 = np.identity(4)
    
    # Calculate all intermediate transforms
    T_01 = T01(q[0])
    T_02 = T_01 @ T12(q[1])
    T_03 = T_02 @ T23(q[2])
    T_04 = T_03 @ T34(q[3])
    T_05 = T_04 @ T45(q[4])
    T_06 = T_05 @ T56(q[5])
    
    # End-effector position
    p_e = T_06[0:3, 3]
    
    # List of transforms from T00 to T05
    transforms = [T_00, T_01, T_02, T_03, T_04, T_05]
    
    # Loop to fill each column of the Jacobian
    for i in range(6):
        T_current = transforms[i]
        
        # Get z-axis (rotation axis) and position of the current joint frame
        z_i = T_current[0:3, 2] # 3rd column of rotation matrix
        p_i = T_current[0:3, 3] # 4th column (position)
        
        # Linear velocity part: J_v = z_i x (p_e - p_i)
        J[0:3, i] = np.cross(z_i, p_e - p_i)
        
        # Angular velocity part: J_w = z_i
        J[3:6, i] = z_i
        
    return J
