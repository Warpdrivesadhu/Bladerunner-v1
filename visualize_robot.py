"""
Robot Visualization Module
Visualizes robot configuration for target positions using FK and IK
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from forward_kinematics import forward_kinematics, print_fk_summary
from inverse_kinematics import inverse_kinematics, print_ik_summary

# ============================================================================
# USER INPUT: Change these values to visualize different target positions
# ============================================================================

# Option 1: Target end effector position (x, y, z) in mm
TARGET_POSITION = [300, 250, 300]
USE_TARGET_POSITION = True  # Set to True to use IK with target position

# Option 2: Target joint angles (degrees)
TARGET_ANGLES = [0, -45, 0, -90, -45, 0]
USE_TARGET_ANGLES = True  # Set to True to visualize target angles

# IK method: 'optimization' (fast) or 'global' (slower, more reliable)
IK_METHOD = 'global'

# Show both configurations side-by-side if both are enabled
SHOW_COMPARISON = True  # Set to False to show only one at a time

# ============================================================================

def draw_coordinate_frame(ax, T, scale=50, label='', linewidth=2):
    """
    Draw a coordinate frame (X=Red, Y=Green, Z=Blue) at the given transformation
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to draw on
    T : ndarray (4x4)
        Homogeneous transformation matrix
    scale : float
        Length of the axes arrows (mm)
    label : str
        Label for this frame
    linewidth : float
        Width of the arrow lines
    """
    # Origin of the frame
    origin = T[:3, 3]
    
    # Direction vectors (columns of rotation matrix)
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale
    
    # Draw X axis (Red)
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='red', arrow_length_ratio=0.2, linewidth=linewidth,
              alpha=0.8)
    
    # Draw Y axis (Green)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='green', arrow_length_ratio=0.2, linewidth=linewidth,
              alpha=0.8)
    
    # Draw Z axis (Blue)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='blue', arrow_length_ratio=0.2, linewidth=linewidth,
              alpha=0.8)
    
    # Add label near the origin
    if label:
        offset = scale * 0.3
        ax.text(origin[0] + offset, origin[1] + offset, origin[2] + offset,
                label, fontsize=7, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         alpha=0.6, edgecolor='black', linewidth=0.5))

def plot_robot_3d(positions, ax=None, title="Robot Configuration", 
                  target_pos=None, color='blue', alpha=0.7, transforms=None,
                  show_frames=True, frame_scale=50):
    """
    Plot the robot in 3D space with coordinate frames
    
    Parameters:
    -----------
    positions : ndarray
        Array of joint positions
    ax : matplotlib axis, optional
        3D axis to plot on
    title : str
        Plot title
    target_pos : array-like, optional
        Target position to display as a marker
    color : str
        Color for the robot links
    alpha : float
        Transparency of robot links
    transforms : list, optional
        List of transformation matrices for each joint
    show_frames : bool
        Whether to show coordinate frames at each joint
    frame_scale : float
        Scale of the coordinate frame arrows (mm)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Plot links
    ax.plot(x, y, z, color=color, linewidth=3, markersize=8, 
            marker='o', label='Robot Links', alpha=alpha)
    
    # Plot joints
    ax.scatter(x[1:-1], y[1:-1], z[1:-1], c='red', s=100, 
               label='Joints', zorder=5, edgecolors='black', linewidths=1)
    
    # Plot base
    ax.scatter(x[0], y[0], z[0], c='black', s=200, marker='s', 
               label='Base', zorder=5, edgecolors='white', linewidths=2)
    
    # Plot end effector
    ax.scatter(x[-1], y[-1], z[-1], c='green', s=250, marker='*', 
               label='End Effector', zorder=5, edgecolors='black', linewidths=1.5)
    
    # Plot coordinate frames at each joint
    if show_frames and transforms is not None:
        # Base frame (world frame) - same as J1
        base_T = np.eye(4)
        draw_coordinate_frame(ax, base_T, frame_scale, label='Base/J1')
        
        # Frames at each joint (J2, J3, J4, J5, J6, EE)
        for i, T in enumerate(transforms):
            if i < len(transforms)-1:
                label = f'J{i+2}'  # J2, J3, J4, J5, J6
            else:
                label = 'EE'
            draw_coordinate_frame(ax, T, frame_scale, label=label)
    
    # Plot target position if provided
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                  c='orange', s=250, marker='X', 
                  label='Target Position', zorder=5, edgecolors='black', linewidths=1.5)
        
        # Draw line from end effector to target
        ax.plot([x[-1], target_pos[0]], 
               [y[-1], target_pos[1]], 
               [z[-1], target_pos[2]], 
               'r--', linewidth=1, alpha=0.5, label='Error Vector')
    
    # Add joint labels (Base and J1 at same location, then J2-J6, EE)
    for i in range(len(positions)):
        if i == 0:
            label = 'Base/J1'
        elif i == len(positions) - 1:
            label = 'EE'
        else:
            label = f'J{i+1}'  # J2, J3, J4, J5, J6
        ax.text(x[i], y[i], z[i], f'  {label}', fontsize=8, 
                weight='bold')
    
    # Labels and styling
    ax.set_xlabel('X (mm)', fontsize=10, weight='bold')
    ax.set_ylabel('Y (mm)', fontsize=10, weight='bold')
    ax.set_zlabel('Z (mm)', fontsize=10, weight='bold')
    ax.set_title(title, fontsize=12, weight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=8)
    
    # Equal aspect ratio
    all_points = np.vstack([positions, [target_pos]] if target_pos is not None else positions)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect([1,1,1])
    
    return ax

def visualize_dual_mode(target_position=None, target_angles=None, 
                       use_target_pos=True, use_target_angles=True,
                       ik_method='optimization', show_comparison=True):
    """
    Visualize robot with both target position (IK) and target angles (FK)
    
    Parameters:
    -----------
    target_position : array-like
        Target XYZ position for IK
    target_angles : array-like
        Target joint angles for FK
    use_target_pos : bool
        Whether to use target position (IK mode)
    use_target_angles : bool
        Whether to use target angles (FK mode)
    ik_method : str
        IK solving method ('optimization' or 'global')
    show_comparison : bool
        Show both side-by-side if both are enabled
    """
    
    # Determine what to show
    show_ik = use_target_pos and target_position is not None
    show_fk = use_target_angles and target_angles is not None
    
    if not show_ik and not show_fk:
        print("ERROR: No target specified! Enable either USE_TARGET_POSITION or USE_TARGET_ANGLES")
        return None, None
    
    # Calculate FK solution if needed
    fk_data = None
    if show_fk:
        print("\n" + "="*70)
        print("FORWARD KINEMATICS (Target Angles)")
        print("="*70)
        positions_fk, transforms_fk, ee_pose_fk = forward_kinematics(target_angles)
        fk_data = {
            'positions': positions_fk,
            'ee_pose': ee_pose_fk,
            'angles': target_angles,
            'transforms': transforms_fk
        }
        print_fk_summary(target_angles)
    
    # Calculate IK solution if needed
    ik_data = None
    if show_ik:
        print("\n" + "="*70)
        print("INVERSE KINEMATICS (Target Position)")
        print("="*70)
        print(f"Target Position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] mm")
        print(f"IK Method: {ik_method}")
        print("-"*70)
        
        solution = inverse_kinematics(target_position, method=ik_method)
        print_ik_summary(solution)
        
        positions_ik, transforms_ik, ee_pose_ik = forward_kinematics(solution['joint_angles'])
        ik_data = {
            'positions': positions_ik,
            'solution': solution,
            'ee_pose': ee_pose_ik,
            'transforms': transforms_ik
        }
    
    # Create visualization
    if show_ik and show_fk and show_comparison:
        # Side-by-side comparison
        fig = plt.figure(figsize=(18, 8))
        
        # Plot FK (Target Angles)
        ax1 = fig.add_subplot(121, projection='3d')
        ee_pos_fk = fk_data['ee_pose']['position']
        title_fk = f"Target Angles (FK)\n"
        title_fk += f"Angles: {np.round(target_angles, 1)}\n"
        title_fk += f"End Effector: [{ee_pos_fk[0]:.1f}, {ee_pos_fk[1]:.1f}, {ee_pos_fk[2]:.1f}] mm"
        plot_robot_3d(fk_data['positions'], ax1, title=title_fk, color='blue',
                     transforms=fk_data['transforms'], show_frames=True, frame_scale=40)
        
        # Plot IK (Target Position)
        ax2 = fig.add_subplot(122, projection='3d')
        error = ik_data['solution']['error']
        title_ik = f"Target Position (IK)\n"
        title_ik += f"Target: [{target_position[0]:.1f}, {target_position[1]:.1f}, {target_position[2]:.1f}] mm\n"
        title_ik += f"Error: {error:.4f} mm"
        
        if error > 0.01:
            plot_robot_3d(ik_data['positions'], ax2, title=title_ik, 
                         target_pos=target_position, color='green',
                         transforms=ik_data['transforms'], show_frames=True, frame_scale=40)
        else:
            plot_robot_3d(ik_data['positions'], ax2, title=title_ik, color='green',
                         transforms=ik_data['transforms'], show_frames=True, frame_scale=40)
        
        # Add info boxes
        info_fk = f"Joint Angles (deg):\n"
        for i, angle in enumerate(target_angles):
            info_fk += f"J{i+1}: {angle:.1f}°\n"
        ax1.text2D(0.02, 0.98, info_fk, transform=ax1.transAxes,
                  fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        info_ik = f"IK Solution (deg):\n"
        for i, angle in enumerate(ik_data['solution']['joint_angles']):
            info_ik += f"J{i+1}: {angle:.1f}°\n"
        info_ik += f"\nIterations: {ik_data['solution']['iterations']}\n"
        info_ik += f"Success: {ik_data['solution']['success']}"
        ax2.text2D(0.02, 0.98, info_ik, transform=ax2.transAxes,
                  fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        axes = [ax1, ax2]
        
    elif show_fk:
        # Only FK
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ee_pos_fk = fk_data['ee_pose']['position']
        title = f"Target Angles (Forward Kinematics)\n"
        title += f"Angles: {np.round(target_angles, 1)}\n"
        title += f"End Effector: [{ee_pos_fk[0]:.1f}, {ee_pos_fk[1]:.1f}, {ee_pos_fk[2]:.1f}] mm"
        plot_robot_3d(fk_data['positions'], ax, title=title, color='blue',
                     transforms=fk_data['transforms'], show_frames=True, frame_scale=50)
        
        # Add info box
        info = f"Joint Angles (deg):\n"
        for i, angle in enumerate(target_angles):
            info += f"J{i+1}: {angle:.1f}°\n"
        ax.text2D(0.02, 0.98, info, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes = [ax]
        
    else:
        # Only IK
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        error = ik_data['solution']['error']
        title = f"Target Position (Inverse Kinematics)\n"
        title += f"Target: [{target_position[0]:.1f}, {target_position[1]:.1f}, {target_position[2]:.1f}] mm\n"
        title += f"Error: {error:.4f} mm"
        
        if error > 0.01:
            plot_robot_3d(ik_data['positions'], ax, title=title, 
                         target_pos=target_position, color='green',
                         transforms=ik_data['transforms'], show_frames=True, frame_scale=50)
        else:
            plot_robot_3d(ik_data['positions'], ax, title=title, color='green',
                         transforms=ik_data['transforms'], show_frames=True, frame_scale=50)
        
        # Add info box
        info = f"IK Solution (deg):\n"
        for i, angle in enumerate(ik_data['solution']['joint_angles']):
            info += f"J{i+1}: {angle:.1f}°\n"
        info += f"\nError: {error:.6f} mm\n"
        info += f"Iterations: {ik_data['solution']['iterations']}\n"
        info += f"Success: {ik_data['solution']['success']}"
        ax.text2D(0.02, 0.98, info, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes = [ax]
    
    plt.tight_layout()
    return fig, axes

if __name__ == "__main__":
    # Main visualization based on user input
    
    fig, axes = visualize_dual_mode(
        target_position=TARGET_POSITION if USE_TARGET_POSITION else None,
        target_angles=TARGET_ANGLES if USE_TARGET_ANGLES else None,
        use_target_pos=USE_TARGET_POSITION,
        use_target_angles=USE_TARGET_ANGLES,
        ik_method=IK_METHOD,
        show_comparison=SHOW_COMPARISON
    )
    
    if fig is not None:
        plt.show()