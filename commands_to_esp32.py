from robot_communication import RobotController

robot = RobotController()
robot.connect()

# IK will automatically try up to 20 times to find valid solution
#success, solution = robot.send_target_position([0, 0, 500])
robot.send_target_position([100, -200, 300])
#robot.send_joint_angles([0, 0, 0, 0, 0, 0])
robot.home()
