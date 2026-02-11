"""
Motion Executor - Execute trajectories on real robot or in simulation
Integrates trajectory planning with robot control and simulation
"""

import numpy as np
import time
from trajectory_planner import (
    TrajectoryExecutor, 
    linear_trajectory, 
    circular_trajectory,
    spiral_trajectory,
    rectangular_trajectory
)
from robot_communication import RobotController
from robot_simulator import RobotSimulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_FIRST = True  # Always simulate before sending to real robot
DELAY_BETWEEN_WAYPOINTS = 0  # seconds between waypoints (real robot)

# ============================================================================

class MotionExecutor:
    """
    Execute trajectories on real robot with simulation preview
    """
    
    def __init__(self, serial_port='/dev/tty.usbserial-0001', use_robot=True):
        """
        Initialize motion executor
        
        Parameters:
        -----------
        serial_port : str
            Serial port for robot connection
        use_robot : bool
            Whether to connect to real robot
        """
        self.use_robot = use_robot
        self.robot = None
        self.simulator = RobotSimulator()
        self.trajectory_planner = TrajectoryExecutor()
        
        # Try to connect to robot if requested
        if use_robot:
            print("\n🤖 Initializing robot connection...")
            self.robot = RobotController(serial_port)
            self.robot.connect()
        else:
            print("\n🎬 Simulation-only mode")
    
    def execute_trajectory(self, waypoints, ik_method='optimization', 
                          simulate_first=SIMULATION_FIRST,
                          animate=True, show_visualization=True):
        """
        Execute a trajectory with full workflow
        
        Parameters:
        -----------
        waypoints : list of dicts
            List of target poses
        ik_method : str
            IK solving method
        simulate_first : bool
            Show simulation before sending to real robot
        animate : bool
            Show animation
        show_visualization : bool
            Show static trajectory visualization
            
        Returns:
        --------
        success : bool
            True if trajectory executed successfully
        """
        print("\n" + "="*70)
        print("TRAJECTORY EXECUTION WORKFLOW")
        print("="*70)
        
        # Step 1: Plan trajectory (solve IK for all waypoints)
        print("\n📋 Step 1: Planning trajectory...")
        success = self.trajectory_planner.plan_trajectory(waypoints, ik_method)
        
        if not success:
            print("❌ Trajectory planning failed")
            return False
        
        joint_trajectory = self.trajectory_planner.joint_trajectory
        
        if len(joint_trajectory) == 0:
            print("❌ No valid waypoints found")
            return False
        
        # Step 2: Show static visualization
        if show_visualization:
            print("\n📊 Step 2: Showing trajectory visualization...")
            self.trajectory_planner.visualize_trajectory(show_path=True, show_frames=False)
        
        # Step 3: Animate trajectory
        if animate:
            print("\n🎬 Step 3: Animating trajectory...")
            self.trajectory_planner.animate_trajectory(duration=5.0)
        
        # Step 4: Ask for confirmation if using real robot
        if self.use_robot and self.robot and self.robot.connected:
            print("\n" + "="*70)
            response = input("Execute this trajectory on REAL ROBOT? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Trajectory execution cancelled")
                return False
            
            # Step 5: Execute on real robot
            print("\n🤖 Step 4: Executing on real robot...")
            return self._execute_on_robot(joint_trajectory)
        
        else:
            print("\n✓ Trajectory planning and simulation complete!")
            print("  (No real robot connected)")
            return True
    
    def _execute_on_robot(self, joint_trajectory):
        """
        Send trajectory to real robot waypoint by waypoint
        
        Parameters:
        -----------
        joint_trajectory : list
            List of joint angle configurations
            
        Returns:
        --------
        success : bool
            True if all waypoints executed successfully
        """
        total = len(joint_trajectory)
        print(f"\n🤖 Executing {total} waypoints on real robot...")
        print("-"*70)
        
        for i, angles in enumerate(joint_trajectory):
            # Show progress
            progress = (i + 1) / total * 100
            print(f"[{i+1}/{total}] ({progress:.1f}%) Waypoint {i+1}...", end='', flush=True)
            
            # Send command with skip_confirmation=True (no asking yes/no for each)
            success = self.robot.send_joint_angles(
                angles, 
                wait_for_response=True,
                skip_confirmation=True  # Don't ask for each waypoint
            )
            
            if not success:
                print(f" ❌ Failed!")
                return False
            
            print(" ✓")
            
            # Small delay between waypoints
            if i < total - 1:
                time.sleep(DELAY_BETWEEN_WAYPOINTS)
        
        print("\n" + "="*70)
        print("✓ Trajectory execution complete!")
        print("="*70)
        return True
    
    def execute_linear_motion(self, start_pos, end_pos, num_points=30):
        """
        Execute a linear motion from start to end
        
        Parameters:
        -----------
        start_pos : array-like
            Start position [x, y, z]
        end_pos : array-like
            End position [x, y, z]
        num_points : int
            Number of intermediate points
        """
        print(f"\n🎯 Linear Motion: {start_pos} → {end_pos}")
        waypoints = linear_trajectory(start_pos, end_pos, num_points)
        return self.execute_trajectory(waypoints)
    
    def execute_circular_motion(self, center, radius, num_points=40, axis='z'):
        """
        Execute a circular motion
        
        Parameters:
        -----------
        center : array-like
            Center of circle [x, y, z]
        radius : float
            Radius of circle
        num_points : int
            Number of points
        axis : str
            Rotation axis ('x', 'y', or 'z')
        """
        print(f"\n🔵 Circular Motion: center={center}, radius={radius}, axis={axis}")
        waypoints = circular_trajectory(center, radius, num_points, axis)
        return self.execute_trajectory(waypoints)
    
    def execute_spiral_motion(self, center, radius, height, num_points=50, turns=2):
        """
        Execute a spiral motion
        
        Parameters:
        -----------
        center : array-like
            Center point [x, y, z_start]
        radius : float
            Spiral radius
        height : float
            Total height
        num_points : int
            Number of points
        turns : float
            Number of turns
        """
        print(f"\n🌀 Spiral Motion: center={center}, radius={radius}, height={height}")
        waypoints = spiral_trajectory(center, radius, height, num_points, turns)
        return self.execute_trajectory(waypoints)
    
    def execute_rectangular_motion(self, corner1, corner2, num_points=80):
        """
        Execute a rectangular motion
        
        Parameters:
        -----------
        corner1 : array-like
            First corner [x, y, z]
        corner2 : array-like
            Opposite corner [x, y, z]
        num_points : int
            Number of points
        """
        print(f"\n▭ Rectangular Motion: {corner1} ↔ {corner2}")
        waypoints = rectangular_trajectory(corner1, corner2, num_points)
        return self.execute_trajectory(waypoints)
    
    def execute_pick_and_place(self, pick_pos, place_pos, approach_height=50):
        """
        Execute a pick and place motion
        
        Parameters:
        -----------
        pick_pos : array-like
            Pick position [x, y, z]
        place_pos : array-like
            Place position [x, y, z]
        approach_height : float
            Height above positions for approach/retract
        """
        print(f"\n📦 Pick and Place: {pick_pos} → {place_pos}")
        
        pick_pos = np.array(pick_pos)
        place_pos = np.array(place_pos)
        
        # Create waypoints
        waypoints = [
            {'position': pick_pos + [0, 0, approach_height]},  # Approach pick
            {'position': pick_pos},                             # Pick
            {'position': pick_pos + [0, 0, approach_height]},  # Retract from pick
            {'position': place_pos + [0, 0, approach_height]}, # Approach place
            {'position': place_pos},                            # Place
            {'position': place_pos + [0, 0, approach_height]}, # Retract from place
        ]
        
        return self.execute_trajectory(waypoints, ik_method='global')
    
    def execute_custom_waypoints(self, waypoints):
        """
        Execute custom waypoints
        
        Parameters:
        -----------
        waypoints : list of dicts
            Custom waypoint list
        """
        print(f"\n⚙️ Custom Motion: {len(waypoints)} waypoints")
        return self.execute_trajectory(waypoints)
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.robot:
            self.robot.disconnect()

# ============================================================================
# Example Usage and Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MOTION EXECUTOR - DEMO")
    print("="*70)
    
    # Create executor (set use_robot=False for simulation only)
    executor = MotionExecutor(
        serial_port='/dev/tty.usbserial-0001',
        use_robot=False  # Change to True to use real robot
    )
    
    try:
        # Demo 1: Linear motion
        print("\n" + "="*70)
        print("DEMO 1: Linear Motion")
        print("="*70)
        executor.execute_linear_motion(
            start_pos=[100, 100, 300],
            end_pos=[200, 200, 350],
            num_points=20
        )
        
        # Demo 2: Circular motion
        print("\n" + "="*70)
        print("DEMO 2: Circular Motion")
        print("="*70)
        executor.execute_circular_motion(
            center=[150, 150, 300],
            radius=50,
            num_points=30,
            axis='z'
        )
        
        # Demo 3: Spiral motion
        print("\n" + "="*70)
        print("DEMO 3: Spiral Motion")
        print("="*70)
        executor.execute_spiral_motion(
            center=[150, 150, 250],
            radius=40,
            height=100,
            num_points=40,
            turns=2
        )
        
        # Demo 4: Pick and place
        print("\n" + "="*70)
        print("DEMO 4: Pick and Place")
        print("="*70)
        executor.execute_pick_and_place(
            pick_pos=[120, 120, 280],
            place_pos=[180, 180, 280],
            approach_height=50
        )
        
        # Demo 5: Custom waypoints with orientation
        print("\n" + "="*70)
        print("DEMO 5: Custom Waypoints")
        print("="*70)
        custom_waypoints = [
            {'position': [100, 100, 300], 'euler': [0, 0, 0]},
            {'position': [150, 150, 350], 'euler': [0, 45, 45]},
            {'position': [200, 100, 320], 'euler': [0, 0, 90]},
            {'position': [150, 50, 300], 'euler': [0, -45, 180]},
        ]
        executor.execute_custom_waypoints(custom_waypoints)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        executor.disconnect()
        print("\n" + "="*70)
        print("Demo complete!")
        print("="*70)