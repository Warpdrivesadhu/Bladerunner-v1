"""
Robot Simulator - Animate robot movement in Python
Simulates movement before sending to real robot
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from forward_kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics
from visualize_robot import plot_robot_3d, draw_coordinate_frame

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_FPS = 30  # Frames per second
MOVEMENT_DURATION = 2.0  # seconds for each movement
SHOW_FRAMES = True  # Show coordinate frames during animation
FRAME_SCALE = 40  # Size of coordinate frame arrows

# ============================================================================

class RobotSimulator:
    """
    Simulate robot movement with animation
    """
    
    def __init__(self, initial_angles=None):
        """
        Initialize simulator
        
        Parameters:
        -----------
        initial_angles : list, optional
            Starting joint angles [j1, j2, j3, j4, j5, j6]
        """
        if initial_angles is None:
            initial_angles = [0, 0, 0, 0, 0, 0]
        
        self.current_angles = np.array(initial_angles, dtype=float)
        self.target_angles = np.array(initial_angles, dtype=float)
        self.is_moving = False
        
        # Animation data
        self.trajectory = []
        self.fig = None
        self.ax = None
        
    def interpolate_angles(self, start_angles, end_angles, num_steps):
        """
        Generate smooth trajectory between two angle sets
        
        Parameters:
        -----------
        start_angles : array
            Starting joint angles
        end_angles : array
            Ending joint angles
        num_steps : int
            Number of interpolation steps
            
        Returns:
        --------
        trajectory : ndarray
            Array of intermediate angle configurations
        """
        trajectory = []
        for i in range(num_steps):
            t = i / (num_steps - 1)  # 0 to 1
            # Smooth S-curve interpolation (ease in-out)
            t_smooth = (np.sin((t - 0.5) * np.pi) + 1) / 2
            angles = start_angles + (end_angles - start_angles) * t_smooth
            trajectory.append(angles)
        
        return np.array(trajectory)
    
    def move_to_angles(self, target_angles, duration=MOVEMENT_DURATION, animate=True):
        """
        Move robot to target joint angles
        
        Parameters:
        -----------
        target_angles : list or array
            Target joint angles [j1, j2, j3, j4, j5, j6]
        duration : float
            Duration of movement in seconds
        animate : bool
            Whether to show animation
            
        Returns:
        --------
        success : bool
            True if movement completed
        """
        target_angles = np.array(target_angles, dtype=float)
        
        if len(target_angles) != 6:
            print(f"✗ Error: Expected 6 angles, got {len(target_angles)}")
            return False
        
        # Generate trajectory
        num_steps = int(duration * SIMULATION_FPS)
        self.trajectory = self.interpolate_angles(self.current_angles, target_angles, num_steps)
        
        print(f"\n🤖 Moving to: {np.round(target_angles, 1)}°")
        print(f"   From: {np.round(self.current_angles, 1)}°")
        print(f"   Duration: {duration}s ({num_steps} steps)")
        
        if animate:
            self._animate_movement()
        
        # Update current position
        self.current_angles = target_angles.copy()
        self.target_angles = target_angles.copy()
        
        return True
    
    def move_to_position(self, target_xyz, duration=MOVEMENT_DURATION, 
                        ik_method='global', animate=True):
        """
        Move robot to target XYZ position using IK
        
        Parameters:
        -----------
        target_xyz : list or array
            Target position [x, y, z] in mm
        duration : float
            Duration of movement in seconds
        ik_method : str
            IK solving method ('optimization' or 'global')
        animate : bool
            Whether to show animation
            
        Returns:
        --------
        success : bool
            True if movement completed
        solution : dict
            IK solution details
        """
        print(f"\n🎯 Target Position: {target_xyz} mm")
        print(f"   Solving IK using '{ik_method}' method...")
        
        # Solve IK
        solution = inverse_kinematics(target_xyz, 
                                     initial_guess=self.current_angles.tolist(),
                                     method=ik_method)
        
        if not solution['success']:
            print(f"✗ IK failed: Error = {solution['error']:.2f} mm")
            return False, solution
        
        print(f"✓ IK solved: Error = {solution['error']:.6f} mm")
        print(f"   Joint Angles: {np.round(solution['joint_angles'], 1)}°")
        
        # Move to the solved angles
        success = self.move_to_angles(solution['joint_angles'], duration, animate)
        
        return success, solution
    
    def home(self, duration=MOVEMENT_DURATION, animate=True):
        """Move to home position (all joints at 0°)"""
        print("\n🏠 Homing robot...")
        return self.move_to_angles([0, 0, 0, 0, 0, 0], duration, animate)
    
    def _animate_movement(self):
        """Create animation of robot movement"""
        # Create figure
        self.fig = plt.figure(figsize=(6, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Animation function
        def update(frame):
            self.ax.clear()
            
            # Get current angles for this frame
            angles = self.trajectory[frame]
            
            # Calculate FK
            positions, transforms, ee_pose = forward_kinematics(angles)
            
            # Plot robot
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            
            # Plot links
            self.ax.plot(x, y, z, 'b-o', linewidth=3, markersize=8, alpha=0.7)
            
            # Plot joints
            self.ax.scatter(x[1:-1], y[1:-1], z[1:-1], c='red', s=100, 
                          zorder=5, edgecolors='black', linewidths=1)
            
            # Plot base
            self.ax.scatter(x[0], y[0], z[0], c='black', s=200, marker='s', 
                          zorder=5, edgecolors='white', linewidths=2)
            
            # Plot end effector
            self.ax.scatter(x[-1], y[-1], z[-1], c='green', s=250, marker='*', 
                          zorder=5, edgecolors='black', linewidths=1.5)
            
            # Plot coordinate frames
            if SHOW_FRAMES:
                base_T = np.eye(4)
                draw_coordinate_frame(self.ax, base_T, FRAME_SCALE, label='Base/J1')
                
                for i, T in enumerate(transforms):
                    if i < len(transforms)-1:
                        label = f'J{i+2}'
                    else:
                        label = 'EE'
                    draw_coordinate_frame(self.ax, T, FRAME_SCALE, label=label)
            
            # Add labels
            for i in range(len(positions)):
                if i == 0:
                    label = 'Base/J1'
                elif i == len(positions) - 1:
                    label = 'EE'
                else:
                    label = f'J{i+1}'
                self.ax.text(x[i], y[i], z[i], f'  {label}', fontsize=8, weight='bold')
            
            # Set labels and title
            self.ax.set_xlabel('X (mm)', fontsize=10, weight='bold')
            self.ax.set_ylabel('Y (mm)', fontsize=10, weight='bold')
            self.ax.set_zlabel('Z (mm)', fontsize=10, weight='bold')
            
            # Progress info
            progress = (frame + 1) / len(self.trajectory) * 100
            title = f"Robot Movement Simulation\nProgress: {progress:.1f}%\n"
            title += f"Angles: {np.round(angles, 1)}°"
            self.ax.set_title(title, fontsize=12, weight='bold', pad=20)
            
            # Equal aspect ratio
            max_range = np.array([
                x.max() - x.min(),
                y.max() - y.min(),
                z.max() - z.min()
            ]).max() / 2.0
            
            mid_x = (x.max() + x.min()) * 0.5
            mid_y = (y.max() + y.min()) * 0.5
            mid_z = (z.max() + z.min()) * 0.5
            
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            self.ax.grid(True, alpha=0.3)
            self.ax.set_box_aspect([1,1,1])
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, update, 
            frames=len(self.trajectory),
            interval=1000/SIMULATION_FPS,
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()
    
    def get_current_angles(self):
        """Get current joint angles"""
        return self.current_angles.copy()
    
    def get_current_end_effector(self):
        """Get current end effector position"""
        positions, _, ee_pose = forward_kinematics(self.current_angles)
        return ee_pose['position']
    
    def visualize_current_state(self):
        """Show current robot configuration (static)"""
        positions, transforms, ee_pose = forward_kinematics(self.current_angles)
        
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        ee_pos = ee_pose['position']
        title = f"Current Robot State\n"
        title += f"Angles: {np.round(self.current_angles, 1)}°\n"
        title += f"End Effector: [{ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f}] mm"
        
        plot_robot_3d(positions, ax, title=title, color='blue',
                     transforms=transforms, show_frames=True, frame_scale=50)
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ROBOT MOVEMENT SIMULATOR")
    print("="*70)
    
    # Create simulator
    sim = RobotSimulator()
    
    # Test sequence
    print("\n📋 Running test sequence...")
    
    # Move 1: Joint angles
    print("\n" + "-"*70)
    print("Movement 1: Specific joint angles")
    sim.move_to_angles([45, 30, 60, 20, 90, 45], duration=2.0)
    
    # Move 2: Different angles
    print("\n" + "-"*70)
    print("Movement 2: Different configuration")
    sim.move_to_angles([70, 60, 120, 0, 180, 90], duration=2.0)
    
    # Move 3: Target position (IK)
    print("\n" + "-"*70)
    print("Movement 3: Move to XYZ position")
    success, solution = sim.move_to_position([100, 150, 300], duration=2.5, ik_method='global')
    
    # Move 4: Another position
    print("\n" + "-"*70)
    print("Movement 4: Move to another XYZ position")
    sim.move_to_position([150, 100, 350], duration=2.5, ik_method='global')
    
    # Move 5: Home
    print("\n" + "-"*70)
    print("Movement 5: Return home")
    sim.home(duration=2.0)
    
    # Show final state
    print("\n" + "="*70)
    print("Final State:")
    print(f"  Joint Angles: {np.round(sim.get_current_angles(), 2)}°")
    print(f"  End Effector: {np.round(sim.get_current_end_effector(), 2)} mm")
    print("="*70)
    
    # Show static visualization of final state
    print("\nShowing final state...")
    sim.visualize_current_state()
