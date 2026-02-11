"""
Forward Kinematics Module for 6-DOF Robot
DH Parameters: [a, d, theta, alpha] (angles in degrees)
"""

import numpy as np

# DH Parameters for the robot
DH_PARAMS = np.array([
    #[0, 0, 0, 0],           # J1
    [0, 211, 0, 90],        # J2
    [20.503, 0, 180, 90],   # J3
    [0, 219.3, -180, 90],   # J4
    [180, 0, 90, 0],        # J5
    [0, 24, 90, 90],        # J6
    [0, 150, 0, 0]           # End effector
])

def dh_transform(a, d, theta, alpha):
    """
    Create DH transformation matrix using modified DH convention
    
    Parameters:
    -----------
    a : float
        Link length (mm)
    d : float
        Link offset (mm)
    theta : float
        Joint angle (degrees)
    alpha : float
        Link twist (degrees)
    
    Returns:
    --------
    T : ndarray (4x4)
        Homogeneous transformation matrix
    """
    theta_rad = np.deg2rad(theta)
    alpha_rad = np.deg2rad(alpha)
    
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    
    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    
    return T

def forward_kinematics(joint_angles, dh_params=None):
    """
    Calculate forward kinematics for the robot
    
    Parameters:
    -----------
    joint_angles : list or ndarray
        Joint angles in degrees [q1, q2, q3, q4, q5, q6]
        Can also include 7th element (0) for end effector
    dh_params : ndarray, optional
        DH parameters table. If None, uses default DH_PARAMS
    
    Returns:
    --------
    positions : ndarray
        Array of (x, y, z) positions for base and all joints
    transforms : list
        List of cumulative transformation matrices
    end_effector_pose : dict
        Dictionary with 'position' (x,y,z) and 'orientation' (rotation matrix)
    """
    if dh_params is None:
        dh_params = DH_PARAMS
    
    # Ensure joint_angles has correct length
    if len(joint_angles) < len(dh_params):
        joint_angles = list(joint_angles) + [0] * (len(dh_params) - len(joint_angles))
    
    transforms = []
    T = np.eye(4)  # Start with identity (world/base frame)
    
    positions = [T[:3, 3]]  # Store base position
    
    for i, (dh, q) in enumerate(zip(dh_params, joint_angles)):
        a, d, theta, alpha = dh
        # Add joint angle to theta offset
        total_theta = theta + q
        
        # Calculate transformation
        T_i = dh_transform(a, d, total_theta, alpha)
        T = T @ T_i
        
        transforms.append(T.copy())
        positions.append(T[:3, 3])
    
    # End effector pose
    end_effector_pose = {
        'position': T[:3, 3],
        'orientation': T[:3, :3],
        'transform': T
    }
    
    return np.array(positions), transforms, end_effector_pose

def get_end_effector_position(joint_angles):
    """
    Get only the end effector position (x, y, z)
    
    Parameters:
    -----------
    joint_angles : list or ndarray
        Joint angles in degrees
    
    Returns:
    --------
    position : ndarray
        End effector position [x, y, z] in mm
    """
    _, _, ee_pose = forward_kinematics(joint_angles)
    return ee_pose['position']

def print_fk_summary(joint_angles):
    """
    Print summary of forward kinematics calculation
    """
    positions, transforms, ee_pose = forward_kinematics(joint_angles)
    
    print("=" * 60)
    print("FORWARD KINEMATICS SUMMARY")
    print("=" * 60)
    print(f"\nJoint Angles (degrees): {joint_angles[:6]}")
    print("\nJoint Positions (mm):")
    print("-" * 60)
    
    for i, pos in enumerate(positions):
        if i == 0:
            print(f"Base:      [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
        elif i == len(positions) - 1:
            print(f"End Eff:   [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
        else:
            print(f"Joint {i}:   [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
    
    print("\n" + "=" * 60)
    print(f"End Effector Position: [{ee_pose['position'][0]:.3f}, "
          f"{ee_pose['position'][1]:.3f}, {ee_pose['position'][2]:.3f}] mm")
    print("=" * 60)
    
    return positions, transforms, ee_pose

if __name__ == "__main__":
    # Test forward kinematics
    print("\nTest 1: Home Position (all joints at 0°)")
    test_angles_1 = [0, 0, 0, 0, 0, 0]
    print_fk_summary(test_angles_1)
    
    print("\n\nTest 2: Sample Configuration")
    test_angles_2 = [30, 45, -20, 15, 30, -15]
    print_fk_summary(test_angles_2)