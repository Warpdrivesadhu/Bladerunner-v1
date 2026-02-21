
from motion_executor import MotionExecutor
from robot_communication import RobotController

robot = RobotController()
robot.connect()

executor = MotionExecutor(use_robot=True)


robot.home()

# waypoints = [{'position': [200, 200, 400]}]
# executor.execute_custom_waypoints(waypoints)



# from draw_text import draw_hello_world

# from motion_executor import MotionExecutor

# executor = MotionExecutor(use_robot=True)
# waypoints = [{'position': [150, -400, 250]}]
# executor.execute_custom_waypoints(waypoints)
# # Draw on YZ plane (vertical wall)
# draw_hello_world(
#     executor,
#     plane='YZ',         # YZ plane (vertical)
#     x_pos=150,          # X stays at 150mm
#     scale=60,           # Letters are 30mm tall
#     spacing=20,         # 10mm between letters
#     orientation=[0, 0, 0]  # End effector pointing down
# )




# custom_waypoints = generate_text_waypoints(
#         "HELLO WORLD",
#         plane='YZ',
#         x_pos=250,
#         scale=30,
#         spacing=10,
#         orientation=[0, 0, 0],  # Tilted 45 degrees
#         points_per_segment=10
#     )
    
# executor.execute_custom_waypoints(custom_waypoints)
    

executor.disconnect()

# Mode 3 - Specific
#executor.execute_linear_motion([250, 400, 30], [250, 0, 30])
                             #  orientation=[0, 0, 0])
#robot.home()
# waypoints = [{'position': [300, 150, 300]}]
# executor.execute_custom_waypoints(waypoints)

# executor.execute_circular_motion(
#             center=[300, 0, 300],
#             radius=150,
#             num_points=100,
#             axis='x')
        
# executor.execute_spiral_motion(
#     center=[220, 0, 100],
#     radius=80,
#     height=300,
#     num_points=100,
#     turns=5
# )
executor.disconnect()



#######Two Point LINEAR MOVEMENT EXAMPLE########

# from motion_executor import MotionExecutor

# # Initialize
# executor = MotionExecutor(use_robot=True)

# # Just give it XYZ coordinates - it handles IK automatically!
# executor.execute_linear_motion(
#     start_pos=[100, 100, 300],  # Current position (or use robot's current pos)
#     end_pos=[200, 150, 350],    # Your target point
#     num_points=2                # Just 1 point = direct move
# )


########SINGLE POINT MOVEMENT EXAMPLE########
# from motion_executor import MotionExecutor

# executor = MotionExecutor(use_robot=True)

# # Move to a single XYZ point
# waypoints = [{'position': [150, 200, 320]}]
# executor.execute_custom_waypoints(waypoints)
