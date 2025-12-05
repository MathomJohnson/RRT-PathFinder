# you may find all of these packages helpful!
from kinematic_helpers import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

# This is the starter class you can modify with whatever methods/members you might need!
class RobotPlacerWithVision():
    def __init__(self):
        # Starting configuration - safe position away from obstacle
        # Based on Sprint 1 tests: Extended Up [0, -π/2, 0, 0, 0, 0] is collision-free
        # You can adjust these values to find a better starting position
        self.start_config = [
            1.57,      # shoulder_pan_joint
            -1.0,   # shoulder_lift_joint (pointing up, away from obstacle)
            0.5,      # elbow_joint
            -1.57,      # wrist_1_joint
            0.0,      # wrist_2_joint
            0.0       # wrist_3_joint
        ]
        self.initialized = False

    def getRobotCommand(self, tt, current_q, current_image_bgr):
        # This is the code we will call. Please don't change the signature!

        # Keep the robot at its current joint positions and keep gripper open (0).
        return list(current_q) + [0]
