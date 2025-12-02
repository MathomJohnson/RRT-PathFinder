"""panda_controller controller."""

from controller import Robot
from controller import Supervisor
from your_code_here import getDesiredRobotCommand
import numpy as np

# create the Robot instance.
robot = Supervisor()

marker = robot.getFromDef("ROBOTCOMMAND")
if marker is None:
    print("Could not find DEF MARKER")
    exit(1)
marker_translation_field = marker.getField("translation")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Get all motors (joint actuators)
motors = []
motors.append(robot.getDevice("shoulder_pan_joint"))
motors.append(robot.getDevice("shoulder_lift_joint"))
motors.append(robot.getDevice("elbow_joint"))
motors.append(robot.getDevice("wrist_1_joint"))
motors.append(robot.getDevice("wrist_2_joint"))
motors.append(robot.getDevice("wrist_3_joint"))

# Enable joint position sensors (radians)
sensors = []
for m in motors:
    ps = m.getPositionSensor()
    ps.enable(timestep)
    sensors.append(ps)


# Main loop:
tt = 0

def visualizeCommand(pose):
    # add offsets from robot (to avoid parent/child)
    x = -pose[1]+0.0
    y = pose[0]+2.2
    z = pose[2]+0.7
    marker_translation_field.setSFVec3f([x, y, z])

def getWPose(tt):
    if tt<200:
       initial_pos = np.array([0.6,-0.3,0.4])
       final_pos = np.array([0.6, -0.2, 0.1])
       pos = initial_pos + (final_pos-initial_pos)*(tt/200.)
    elif tt<400:
       temp_tt = tt-200
       initial_pos = np.array([0.6, -0.2, 0.1])
       final_pos = np.array([0.6, -0.1, 0.25])
       pos = initial_pos + (final_pos-initial_pos)*(temp_tt/200.)
    elif tt<600:
       temp_tt = tt-400
       initial_pos = np.array([0.6, -0.1, 0.25])
       final_pos = np.array([0.6, 0.0, 0.1])
       pos = initial_pos + (final_pos-initial_pos)*(temp_tt/200.)
    elif tt<800:
       temp_tt = tt-600
       initial_pos = np.array([0.6, 0.0, 0.1])
       final_pos = np.array([0.6,0.1,0.4])
       pos = initial_pos + (final_pos-initial_pos)*(temp_tt/200.)
    else:
       pos = np.array([0.6,0.1,0.4])
       
    pos = pos.tolist()
    # return pos + [1.5708, 0.0, 0.0 ] # zero rotation
    return pos + [0, 0, 0 ]

# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    print(f"Timestep {tt}")
    
    desired_pose = getWPose(tt)
    
    if tt==0:
        last_q = []
        for sensor in sensors:
            last_q.append(sensor.getValue())
        
    desired_command = getDesiredRobotCommand(tt, desired_pose, last_q)
    
    last_q = desired_command
    
    visualizeCommand(desired_pose)
    
    # 7 Robot Joints
    for j, motor in enumerate(motors):
        motor.setPosition(desired_command[j])
        
    tt+=1
