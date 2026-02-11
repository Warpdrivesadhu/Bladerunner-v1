
"""
Inverse Kinematics Module for 6-DOF Robot
Uses numerical optimization to find joint angles for target position
Automatically retries until solution is within joint limits
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from forward_kinematics import forward_kinematics, get_end_effector_position, DH_PARAMS

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default joint limits (degrees)
DEFAULT_JOINT_LIMITS = [
    (-90, 60),  # J1
    (-90, 10),  # J2
    (-90, 90),  # J3
    (-120, 65),  # J4
    (-120, 65),  # J5
    (-180, 180),  # J6
]

# Acceptable position error (mm) - CHANGE THIS VALUE
ACCEPTABLE_ERROR_MM = 3.0  # Solutions with error < this are considered successful

# ============================================================================

def check_joint_limits(joint_angles, limits=None):
    """
    Check if joint angles are within limits
    
    Parameters:
    -----------
    joint_angles : array-like
        Joint angles in degrees
    limits : list of tuples, optional
        Joint limits [(min, max), ...] for each joint
        
    Returns:
    --------
    within_limits : bool
        True if all joints are within limits
    violations : list
        List of (joint_index, angle, min, max) for violations
    """
    if limits is None:
        limits = DEFAULT_JOINT_LIMITS
    
    violations = []
    for i, (angle, (min_angle, max_angle)) in enumerate(zip(joint_angles, limits)):
        if angle < min_angle or angle > max_angle:
            violations.append((i, angle, min_angle, max_angle))
    
    return len(violations) == 0, violations

def inverse_kinematics(target_position, initial_guess=None, method='optimization', 
                       orientation=None, bounds=None, max_attempts=20):
    """
    Calculate inverse kinematics to reach target position
    Automatically retries with different initial guesses until solution is within joint limits
    
    Parameters:
    -----------
    target_position : array-like
        Target end effector position [x, y, z] in mm
    initial_guess : array-like, optional
        Initial joint angles guess (degrees). If None, uses [0,0,0,0,0,0]
    method : str
        'optimization' - fast local optimization
        'global' - slower global optimization (more reliable)
    orientation : ndarray (3x3), optional
        Target orientation matrix (not fully implemented)
    bounds : list of tuples, optional
        Joint angle limits [(min, max), ...] for each joint in degrees
    max_attempts : int
        Maximum number of attempts to find valid solution
    
    Returns:
    --------
    solution : dict
        Dictionary containing:
        - 'joint_angles': solution joint angles (degrees)
        - 'position': achieved position
        - 'error': position error (mm)
        - 'success': whether solution was found
        - 'iterations': number of iterations
        - 'attempts': number of attempts made
        - 'within_limits': whether solution is within joint limits
    """
    target_position = np.array(target_position)
    
    if initial_guess is None:
        initial_guess = [0, 0, 0, 0, 0, 0]
    
    # Default joint limits (can be customized)
    if bounds is None:
        bounds = DEFAULT_JOINT_LIMITS
    
    def position_error(joint_angles):
        """Calculate position error"""
        current_pos = get_end_effector_position(joint_angles)
        error = np.linalg.norm(current_pos - target_position)
        return error
    
    def cost_function(joint_angles):
        """Cost function for optimization"""
        pos_error = position_error(joint_angles)
        
        # Add small penalty for large joint movements (smoothness)
        smoothness_penalty = 0.001 * np.sum(np.abs(joint_angles))
        
        # Add penalty for being outside bounds
        limit_penalty = 0
        for angle, (min_ang, max_ang) in zip(joint_angles, bounds):
            if angle < min_ang:
                limit_penalty += 100 * (min_ang - angle)**2
            elif angle > max_ang:
                limit_penalty += 100 * (angle - max_ang)**2
        
        return pos_error + smoothness_penalty + limit_penalty
    
    print(f"Searching for IK solution within joint limits...")
    print(f"Joint limits: {bounds}")
    
    best_solution = None
    best_error = float('inf')
    
    for attempt in range(max_attempts):
        if attempt == 0:
            # First attempt: use provided initial guess
            current_guess = initial_guess
        elif attempt == 1:
            # Second attempt: try home position
            current_guess = [0, 0, 0, 0, 0, 0]
        else:
            # Random initial guesses within bounds
            current_guess = []
            for min_ang, max_ang in bounds:
                # Random angle within bounds, biased toward center
                center = (min_ang + max_ang) / 2
                range_size = (max_ang - min_ang) * 0.5
                random_offset = np.random.uniform(-range_size, range_size)
                current_guess.append(center + random_offset)
        
        if method == 'optimization':
            # Local optimization - fast but may not find global minimum
            result = minimize(
                cost_function,
                current_guess,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            joint_angles = result.x
            iterations = result.nit
            
        elif method == 'global':
            # Global optimization - slower but more reliable
            result = differential_evolution(
                cost_function,
                bounds=bounds,
                maxiter=300,
                atol=1e-6,
                seed=42 + attempt  # Different seed each attempt
            )
            
            joint_angles = result.x
            iterations = result.nit
        
        else:
            raise ValueError("method must be 'optimization' or 'global'")
        
        # Check if this solution is better
        error = position_error(joint_angles)
        within_limits, violations = check_joint_limits(joint_angles, bounds)
        
        if error < best_error:
            best_error = error
            best_solution = {
                'joint_angles': joint_angles,
                'error': error,
                'iterations': iterations,
                'attempt': attempt + 1,
                'within_limits': within_limits,
                'violations': violations
            }
        
        # If we found a good solution within limits, use it
        if within_limits and error < ACCEPTABLE_ERROR_MM:
            print(f"✓ Found valid solution on attempt {attempt + 1}")
            achieved_position = get_end_effector_position(joint_angles)
            
            solution = {
                'joint_angles': joint_angles,
                'position': achieved_position,
                'target': target_position,
                'error': error,
                'success': True,
                'iterations': iterations,
                'attempts': attempt + 1,
                'within_limits': True,
                'method': method
            }
            return solution
    
    # If we get here, we didn't find a solution within limits
    # Return the best solution we found
    print(f"⚠ Could not find solution within limits after {max_attempts} attempts")
    
    if best_solution:
        achieved_position = get_end_effector_position(best_solution['joint_angles'])
        
        print(f"Returning best solution found:")
        print(f"  Position error: {best_solution['error']:.4f} mm")
        print(f"  Within limits: {best_solution['within_limits']}")
        
        if not best_solution['within_limits']:
            print(f"  Limit violations:")
            for joint_idx, angle, min_ang, max_ang in best_solution['violations']:
                print(f"    J{joint_idx+1}: {angle:.2f}° outside [{min_ang}°, {max_ang}°]")
        
        solution = {
            'joint_angles': best_solution['joint_angles'],
            'position': achieved_position,
            'target': target_position,
            'error': best_solution['error'],
            'success': best_solution['error'] < ACCEPTABLE_ERROR_MM,
            'iterations': best_solution['iterations'],
            'attempts': max_attempts,
            'within_limits': best_solution['within_limits'],
            'method': method
        }
        return solution
    
    # Complete failure
    solution = {
        'joint_angles': initial_guess,
        'position': get_end_effector_position(initial_guess),
        'target': target_position,
        'error': position_error(initial_guess),
        'success': False,
        'iterations': 0,
        'attempts': max_attempts,
        'within_limits': False,
        'method': method
    }
    return solution

def inverse_kinematics_multiple_solutions(target_position, num_attempts=5, bounds=None):
    """
    Try to find multiple IK solutions by using different initial guesses
    
    Parameters:
    -----------
    target_position : array-like
        Target position [x, y, z]
    num_attempts : int
        Number of different initial guesses to try
    bounds : list of tuples, optional
        Joint limits
    
    Returns:
    --------
    solutions : list
        List of solution dictionaries sorted by error
    """
    solutions = []
    
    # Try different initial guesses
    np.random.seed(42)
    for i in range(num_attempts):
        if i == 0:
            # First attempt: home position
            initial_guess = [0, 0, 0, 0, 0, 0]
        else:
            # Random initial guesses
            initial_guess = np.random.uniform(-90, 90, 6)
        
        sol = inverse_kinematics(target_position, initial_guess=initial_guess, 
                                method='optimization', bounds=bounds)
        
        if sol['success']:
            # Check if this is a unique solution
            is_unique = True
            for existing_sol in solutions:
                if np.allclose(sol['joint_angles'], existing_sol['joint_angles'], atol=5.0):
                    is_unique = False
                    break
            
            if is_unique:
                solutions.append(sol)
    
    # Sort by error
    solutions.sort(key=lambda x: x['error'])
    
    return solutions

def print_ik_summary(solution):
    """Print summary of IK solution"""
    print("=" * 60)
    print("INVERSE KINEMATICS SOLUTION")
    print("=" * 60)
    print(f"\nTarget Position:   [{solution['target'][0]:8.3f}, "
          f"{solution['target'][1]:8.3f}, {solution['target'][2]:8.3f}] mm")
    print(f"Achieved Position: [{solution['position'][0]:8.3f}, "
          f"{solution['position'][1]:8.3f}, {solution['position'][2]:8.3f}] mm")
    print(f"\nPosition Error: {solution['error']:.6f} mm")
    print(f"Success: {solution['success']}")
    print(f"Within Limits: {solution.get('within_limits', 'N/A')}")
    print(f"Method: {solution['method']}")
    print(f"Iterations: {solution['iterations']}")
    print(f"Attempts: {solution.get('attempts', 1)}")
    
    print(f"\nJoint Angles (degrees):")
    print("-" * 60)
    for i, angle in enumerate(solution['joint_angles']):
        print(f"  Joint {i+1}: {angle:8.3f}°")
    print("=" * 60)

if __name__ == "__main__":
    # Test inverse kinematics
    
    print("\nTest 1: IK for known position (from FK test)")
    # First, get a target position from FK
    test_angles = [30, 45, -20, 15, 30, -15]
    target_pos = get_end_effector_position(test_angles)
    print(f"\nOriginal Joint Angles: {test_angles}")
    print(f"Target Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] mm")
    
    # Solve IK
    solution = inverse_kinematics(target_pos, method='optimization')
    print_ik_summary(solution)
    
    print("\n\nTest 2: IK for custom position")
    custom_target = [100, 150, 300]  # Custom XYZ position
    solution2 = inverse_kinematics(custom_target, method='global')
    print_ik_summary(solution2)
    
    print("\n\nTest 3: Finding multiple solutions")
    solutions = inverse_kinematics_multiple_solutions(target_pos, num_attempts=10)
    print(f"\nFound {len(solutions)} unique solutions:")
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i+1}: Error = {sol['error']:.6f} mm")
        print(f"  Angles: {np.round(sol['joint_angles'], 2)}")