"""
Trajectory Planning and Visualization Module
Create and visualize complex robot trajectories with position and orientation control
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from forward_kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics
from visualize_robot import draw_coordinate_frame

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAJECTORY_FPS = 30
DEFAULT_DURATION = 3.0

# ============================================================================
# Orientation Utilities
# ============================================================================

def rotation_matrix_from_euler(roll, pitch, yaw, degrees=True):
    """
    Create rotation matrix from Euler angles (ZYX convention)
    
    Parameters:
    -----------
    roll : float
        Rotation around X axis
    pitch : float
        Rotation around Y axis
    yaw : float
        Rotation around Z axis
    degrees : bool
        If True, angles are in degrees, else radians
        
    Returns:
    --------
    R : ndarray (3x3)
        Rotation matrix
    """
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation (ZYX order)
    R = Rz @ Ry @ Rx
    return R

def euler_from_rotation_matrix(R):
    """
    Extract Euler angles from rotation matrix (ZYX convention)
    
    Parameters:
    -----------
    R : ndarray (3x3)
        Rotation matrix
        
    Returns:
    --------
    roll, pitch, yaw : float
        Euler angles in degrees
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    
    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)

def interpolate_poses(start_pose, end_pose, num_steps):
    """
    Interpolate between two poses (position + orientation)
    
    Parameters:
    -----------
    start_pose : dict
        {'position': [x,y,z], 'orientation': R (3x3) or 'euler': [r,p,y]}
    end_pose : dict
        {'position': [x,y,z], 'orientation': R (3x3) or 'euler': [r,p,y]}
    num_steps : int
        Number of interpolation steps
        
    Returns:
    --------
    trajectory : list of dicts
        List of interpolated poses
    """
    trajectory = []
    
    # Get start and end positions
    start_pos = np.array(start_pose['position'])
    end_pos = np.array(end_pose['position'])
    
    # Get start and end orientations
    if 'orientation' in start_pose:
        start_R = start_pose['orientation']
    elif 'euler' in start_pose:
        start_R = rotation_matrix_from_euler(*start_pose['euler'])
    else:
        start_R = np.eye(3)
    
    if 'orientation' in end_pose:
        end_R = end_pose['orientation']
    elif 'euler' in end_pose:
        end_R = rotation_matrix_from_euler(*end_pose['euler'])
    else:
        end_R = np.eye(3)
    
    # Extract Euler angles for interpolation
    start_euler = euler_from_rotation_matrix(start_R)
    end_euler = euler_from_rotation_matrix(end_R)
    
    for i in range(num_steps):
        t = i / (num_steps - 1)
        t_smooth = (np.sin((t - 0.5) * np.pi) + 1) / 2  # S-curve
        
        # Interpolate position
        pos = start_pos + (end_pos - start_pos) * t_smooth
        
        # Interpolate orientation (simple Euler angle interpolation)
        euler = [
            start_euler[0] + (end_euler[0] - start_euler[0]) * t_smooth,
            start_euler[1] + (end_euler[1] - start_euler[1]) * t_smooth,
            start_euler[2] + (end_euler[2] - start_euler[2]) * t_smooth
        ]
        R = rotation_matrix_from_euler(*euler)
        
        trajectory.append({
            'position': pos,
            'orientation': R,
            'euler': euler
        })
    
    return trajectory

# ============================================================================
# Trajectory Types
# ============================================================================

def linear_trajectory(start_pos, end_pos, num_points=50, orientation=None):
    """
    Create a straight line trajectory
    
    Parameters:
    -----------
    start_pos : array-like
        Start position [x, y, z]
    end_pos : array-like
        End position [x, y, z]
    num_points : int
        Number of points along the line
    orientation : dict, optional
        Constant orientation or None
        
    Returns:
    --------
    waypoints : list of dicts
        List of waypoint poses
    """
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        pos = start_pos + (end_pos - start_pos) * t
        
        waypoint = {'position': pos}
        if orientation:
            waypoint.update(orientation)
        
        waypoints.append(waypoint)
    
    return waypoints

def circular_trajectory(center, radius, num_points=50, axis='z', orientation=None):
    """
    Create a circular trajectory
    
    Parameters:
    -----------
    center : array-like
        Center of circle [x, y, z]
    radius : float
        Radius of circle
    num_points : int
        Number of points along the circle
    axis : str
        Axis of rotation ('x', 'y', or 'z')
    orientation : dict, optional
        Orientation strategy or None
        
    Returns:
    --------
    waypoints : list of dicts
        List of waypoint poses
    """
    center = np.array(center)
    waypoints = []
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        
        if axis == 'z':
            # Circle in XY plane
            pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])
        elif axis == 'y':
            # Circle in XZ plane
            pos = center + np.array([
                radius * np.cos(angle),
                0,
                radius * np.sin(angle)
            ])
        elif axis == 'x':
            # Circle in YZ plane
            pos = center + np.array([
                0,
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
        
        waypoint = {'position': pos}
        
        # Optionally make orientation follow the path
        if orientation == 'follow':
            # Point Z-axis toward center
            direction = center - pos
            direction = direction / np.linalg.norm(direction)
            # Simple orientation (could be improved)
            yaw = np.rad2deg(np.arctan2(direction[1], direction[0]))
            waypoint['euler'] = [0, 0, yaw]
        elif orientation:
            waypoint.update(orientation)
        
        waypoints.append(waypoint)
    
    return waypoints

def spiral_trajectory(center, radius, height, num_points=100, turns=2):
    """
    Create a spiral trajectory
    
    Parameters:
    -----------
    center : array-like
        Center point [x, y, z_start]
    radius : float
        Spiral radius
    height : float
        Total height of spiral
    num_points : int
        Number of points
    turns : float
        Number of complete turns
        
    Returns:
    --------
    waypoints : list of dicts
        List of waypoint poses
    """
    center = np.array(center)
    waypoints = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = 2 * np.pi * turns * t
        
        pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height * t
        ])
        
        waypoints.append({'position': pos})
    
    return waypoints

def rectangular_trajectory(corner1, corner2, num_points=100):
    """
    Create a rectangular trajectory
    
    Parameters:
    -----------
    corner1 : array-like
        First corner [x, y, z]
    corner2 : array-like
        Opposite corner [x, y, z]
    num_points : int
        Total number of points
        
    Returns:
    --------
    waypoints : list of dicts
        List of waypoint poses
    """
    c1 = np.array(corner1)
    c2 = np.array(corner2)
    
    # Define 4 corners
    corners = [
        c1,
        [c2[0], c1[1], c1[2]],
        c2,
        [c1[0], c2[1], c1[2]]
    ]
    
    waypoints = []
    points_per_side = num_points // 4
    
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        
        for j in range(points_per_side):
            t = j / points_per_side
            pos = start + (end - start) * t
            waypoints.append({'position': pos})
    
    return waypoints

# ============================================================================
# Trajectory Execution and Visualization
# ============================================================================

class TrajectoryExecutor:
    """
    Execute and visualize trajectories with IK solving
    """
    
    def __init__(self):
        self.waypoints = []
        self.joint_trajectory = []
        self.current_angles = [0, 0, 0, 0, 0, 0]
        
    def plan_trajectory(self, waypoints, ik_method='optimization'):
        """
        Plan joint angles for all waypoints using IK
        
        Parameters:
        -----------
        waypoints : list of dicts
            List of target poses
        ik_method : str
            IK solving method
            
        Returns:
        --------
        success : bool
            True if all waypoints solved successfully
        """
        self.waypoints = waypoints
        self.joint_trajectory = []
        
        print(f"\n📋 Planning trajectory for {len(waypoints)} waypoints...")
        
        failed = 0
        for i, waypoint in enumerate(waypoints):
            target_pos = waypoint['position']
            
            # Solve IK
            solution = inverse_kinematics(
                target_pos,
                initial_guess=self.current_angles,
                method=ik_method
            )
            
            if solution['success'] and solution['within_limits']:
                self.joint_trajectory.append(solution['joint_angles'])
                self.current_angles = solution['joint_angles']
            else:
                failed += 1
                if failed < 5:  # Only print first few failures
                    print(f"  ⚠ Waypoint {i+1}/{len(waypoints)} failed")
        
        success_rate = (len(waypoints) - failed) / len(waypoints) * 100
        print(f"✓ Trajectory planned: {len(self.joint_trajectory)}/{len(waypoints)} waypoints ({success_rate:.1f}%)")
        
        return failed == 0
    
    def visualize_trajectory(self, show_path=True, show_frames=False):
        """
        Visualize the entire trajectory
        
        Parameters:
        -----------
        show_path : bool
            Show the end effector path
        show_frames : bool
            Show coordinate frames
        """
        if not self.joint_trajectory:
            print("No trajectory to visualize")
            return
        
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate all end effector positions
        ee_positions = []
        for angles in self.joint_trajectory:
            positions, transforms, ee_pose = forward_kinematics(angles)
            ee_positions.append(ee_pose['position'])
        
        ee_positions = np.array(ee_positions)
        
        # Plot path
        if show_path:
            ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                   'b-', linewidth=2, alpha=0.6, label='End Effector Path')
            
            # Plot waypoints
            waypoint_positions = np.array([wp['position'] for wp in self.waypoints])
            ax.scatter(waypoint_positions[:, 0], waypoint_positions[:, 1], 
                      waypoint_positions[:, 2],
                      c='orange', s=50, marker='o', label='Waypoints', zorder=5)
            
            # Mark start and end
            ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
                      c='green', s=200, marker='o', label='Start', zorder=6)
            ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                      c='red', s=200, marker='X', label='End', zorder=6)
        
        # Plot robot at key positions
        num_snapshots = min(5, len(self.joint_trajectory))
        snapshot_indices = np.linspace(0, len(self.joint_trajectory)-1, num_snapshots, dtype=int)
        
        for idx in snapshot_indices:
            angles = self.joint_trajectory[idx]
            positions, transforms, _ = forward_kinematics(angles)
            
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            
            alpha = 0.3 if idx not in [0, len(self.joint_trajectory)-1] else 0.8
            ax.plot(x, y, z, 'gray', linewidth=1, alpha=alpha)
            
            if show_frames and idx == len(self.joint_trajectory)-1:
                # Show frames at final position
                base_T = np.eye(4)
                draw_coordinate_frame(ax, base_T, 30, label='Base/J1')
                for i, T in enumerate(transforms):
                    label = f'J{i+2}' if i < len(transforms)-1 else 'EE'
                    draw_coordinate_frame(ax, T, 30, label=label)
        
        # Labels
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_zlabel('Z (mm)', fontweight='bold')
        ax.set_title(f'Trajectory Visualization ({len(self.waypoints)} waypoints)', 
                    fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Equal aspect
        all_points = np.vstack([ee_positions, waypoint_positions])
        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0
        
        mid = all_points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        plt.show()
    
    def animate_trajectory(self, duration=DEFAULT_DURATION):
        """
        Animate the robot following the trajectory
        
        Parameters:
        -----------
        duration : float
            Total animation duration in seconds
        """
        if not self.joint_trajectory:
            print("No trajectory to animate")
            return
        
        print(f"\n🎬 Animating trajectory...")
        
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Pre-calculate all positions for the path
        all_ee_positions = []
        for angles in self.joint_trajectory:
            _, _, ee_pose = forward_kinematics(angles)
            all_ee_positions.append(ee_pose['position'])
        all_ee_positions = np.array(all_ee_positions)
        
        def update(frame):
            ax.clear()
            
            # Get current configuration
            angles = self.joint_trajectory[frame]
            positions, transforms, ee_pose = forward_kinematics(angles)
            
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            
            # Plot robot
            ax.plot(x, y, z, 'b-o', linewidth=3, markersize=8, alpha=0.7, label='Robot')
            ax.scatter(x[1:-1], y[1:-1], z[1:-1], c='red', s=100, zorder=5)
            ax.scatter(x[0], y[0], z[0], c='black', s=200, marker='s', zorder=5)
            ax.scatter(x[-1], y[-1], z[-1], c='green', s=250, marker='*', zorder=5)
            
            # Plot path traveled so far
            if frame > 0:
                ax.plot(all_ee_positions[:frame+1, 0], 
                       all_ee_positions[:frame+1, 1],
                       all_ee_positions[:frame+1, 2],
                       'orange', linewidth=2, alpha=0.8, label='Path Traveled')
            
            # Plot remaining path
            if frame < len(self.joint_trajectory) - 1:
                ax.plot(all_ee_positions[frame:, 0],
                       all_ee_positions[frame:, 1],
                       all_ee_positions[frame:, 2],
                       'gray', linewidth=1, alpha=0.3, linestyle='--', label='Remaining Path')
            
            # Labels
            for i in range(len(positions)):
                if i == 0:
                    label = 'Base/J1'
                elif i == len(positions) - 1:
                    label = 'EE'
                else:
                    label = f'J{i+1}'
                ax.text(x[i], y[i], z[i], f'  {label}', fontsize=8, weight='bold')
            
            ax.set_xlabel('X (mm)', fontweight='bold')
            ax.set_ylabel('Y (mm)', fontweight='bold')
            ax.set_zlabel('Z (mm)', fontweight='bold')
            
            progress = (frame + 1) / len(self.joint_trajectory) * 100
            ax.set_title(f'Trajectory Animation\nProgress: {progress:.1f}% ({frame+1}/{len(self.joint_trajectory)})',
                        fontweight='bold', fontsize=12)
            
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Fixed bounds
            max_range = np.array([
                all_ee_positions[:, 0].max() - all_ee_positions[:, 0].min(),
                all_ee_positions[:, 1].max() - all_ee_positions[:, 1].min(),
                all_ee_positions[:, 2].max() - all_ee_positions[:, 2].min()
            ]).max() / 2.0
            
            mid = all_ee_positions.mean(axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            ax.set_box_aspect([1,1,1])
        
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(self.joint_trajectory),
            interval=duration * 1000 / len(self.joint_trajectory),
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRAJECTORY PLANNING AND VISUALIZATION")
    print("="*70)
    
    executor = TrajectoryExecutor()
    
    # Test 1: Linear trajectory
    print("\n" + "-"*70)
    print("Test 1: Linear Trajectory")
    print("-"*70)
    waypoints = linear_trajectory([100, 100, 300], [200, 200, 350], num_points=30)
    executor.plan_trajectory(waypoints, ik_method='optimization')
    executor.visualize_trajectory(show_path=True, show_frames=False)
    executor.animate_trajectory(duration=3.0)
    
    # Test 2: Circular trajectory
    print("\n" + "-"*70)
    print("Test 2: Circular Trajectory")
    print("-"*70)
    waypoints = circular_trajectory([150, 150, 300], radius=50, num_points=40, axis='z')
    executor.plan_trajectory(waypoints, ik_method='optimization')
    executor.visualize_trajectory(show_path=True)
    executor.animate_trajectory(duration=4.0)
    
    # Test 3: Spiral trajectory
    print("\n" + "-"*70)
    print("Test 3: Spiral Trajectory")
    print("-"*70)
    waypoints = spiral_trajectory([150, 150, 250], radius=40, height=100, 
                                  num_points=50, turns=2)
    executor.plan_trajectory(waypoints, ik_method='optimization')
    executor.visualize_trajectory(show_path=True)
    executor.animate_trajectory(duration=5.0)
    
    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)