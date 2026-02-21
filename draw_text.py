"""
Text Drawing Module
Generate waypoints to draw text in 3D space
"""

import numpy as np
from motion_executor import MotionExecutor

# Letter definitions using line segments (normalized 0-1 coordinates)
LETTERS = {
    'H': [
        [(0, 0), (0, 1)],    # Left vertical
        
         [(0, 0.5), (1, 0.5)],
         [(1, 1), (1, 0)],    # Right vertical
        # Horizontal bar
    ],
    'E': [
        [(0, 0), (0, 1)],    # Vertical
        [(0, 1), (1, 1)],    # Top horizontal
        [(0, 0.5), (0.8, 0.5)], # Middle horizontal
        [(0, 0), (1, 0)]     # Bottom horizontal
    ],
    'L': [
        [(0, 0), (0, 1)],    # Vertical
        [(0, 0), (1, 0)]     # Bottom horizontal
    ],
    'O': [
        [(0, 0), (0, 1)],    # Left
        [(0, 1), (1, 1)],    # Top
        [(1, 1), (1, 0)],    # Right
        [(1, 0), (0, 0)]     # Bottom
    ],
    'W': [
        [(0, 1), (0.25, 0)], # Left diagonal down
        [(0.25, 0), (0.5, 0.6)], # Left diagonal up
        [(0.5, 0.6), (0.75, 0)], # Right diagonal down
        [(0.75, 0), (1, 1)]  # Right diagonal up
    ],
    'R': [
        [(0, 0), (0, 1)],    # Vertical
        [(0, 1), (1, 1)],    # Top horizontal
        [(1, 1), (1, 0.5)],  # Right side top
        [(1, 0.5), (0, 0.5)], # Middle horizontal
        [(0.5, 0.5), (1, 0)] # Diagonal leg
    ],
    'D': [
        [(0, 0), (0, 1)],    # Vertical
        [(0, 1), (0.7, 1)],  # Top
        [(0.7, 1), (1, 0.7)], # Top curve
        [(1, 0.7), (1, 0.3)], # Right side
        [(1, 0.3), (0.7, 0)], # Bottom curve
        [(0.7, 0), (0, 0)]   # Bottom
    ],
    ' ': []  # Space (no lines)
}

def generate_text_waypoints(text, plane='YZ', x_pos=150, scale=30, spacing=40, 
                           orientation=None, points_per_segment=10):
    """
    Generate waypoints to draw text
    
    Parameters:
    -----------
    text : str
        Text to draw (e.g., "HELLO WORLD")
    plane : str
        Which plane to draw on: 'YZ', 'XY', or 'XZ'
    x_pos : float
        Fixed position on the perpendicular axis (mm)
    scale : float
        Size of each letter (mm)
    spacing : float
        Space between letters (mm)
    orientation : list, optional
        Fixed orientation [roll, pitch, yaw] in degrees
    points_per_segment : int
        Number of points per line segment
        
    Returns:
    --------
    waypoints : list of dicts
        List of waypoint poses
    """
    text = text.upper()
    waypoints = []
    
    current_offset = -400
    
    for char in text:
        if char not in LETTERS:
            print(f"Warning: Character '{char}' not defined, skipping")
            continue
        
        letter_strokes = LETTERS[char]
        
        if char == ' ':
            # Just add spacing
            current_offset += spacing * 0.5
            continue
        
        # Draw each stroke of the letter
        for stroke_idx, stroke in enumerate(letter_strokes):
            start_norm, end_norm = stroke
            
            # Scale and position the stroke
            if plane == 'YZ':
                # Draw on YZ plane (X is fixed)
                start = [
                    x_pos,
                    current_offset + start_norm[0] * scale,
                    start_norm[1] * scale + 250  # Base height
                ]
                end = [
                    x_pos,
                    current_offset + end_norm[0] * scale,
                    end_norm[1] * scale + 250
                ]
            elif plane == 'XY':
                # Draw on XY plane (Z is fixed)
                start = [
                    current_offset + start_norm[0] * scale,
                    start_norm[1] * scale,
                    x_pos
                ]
                end = [
                    current_offset + end_norm[0] * scale,
                    end_norm[1] * scale,
                    x_pos
                ]
            elif plane == 'XZ':
                # Draw on XZ plane (Y is fixed)
                start = [
                    current_offset + start_norm[0] * scale,
                    x_pos,
                    start_norm[1] * scale + 250
                ]
                end = [
                    current_offset + end_norm[0] * scale,
                    x_pos,
                    end_norm[1] * scale + 250
                ]
            
            # If this is the first stroke of the letter, move to start position
            # (pen up - could add a retract here for real drawing)
            if stroke_idx == 0:
                waypoint = {'position': start}
                if orientation:
                    waypoint['euler'] = orientation
                waypoints.append(waypoint)
            
            # Interpolate points along the stroke
            for i in range(points_per_segment):
                t = i / (points_per_segment - 1)
                point = [
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t,
                    start[2] + (end[2] - start[2]) * t
                ]
                waypoint = {'position': point}
                if orientation:
                    waypoint['euler'] = orientation
                waypoints.append(waypoint)
        
        # Move to next letter position
        current_offset += scale + spacing
    
    return waypoints

def draw_hello_world(executor, plane='YZ', x_pos=150, scale=30, spacing=40,
                    orientation=None, maintain_orientation=False):
    """
    Draw "HELLO WORLD" using the robot
    
    Parameters:
    -----------
    executor : MotionExecutor
        Motion executor instance
    plane : str
        Which plane to draw on: 'YZ', 'XY', or 'XZ'
    x_pos : float
        Fixed position on perpendicular axis (mm)
    scale : float
        Size of letters (mm)
    spacing : float
        Space between letters (mm)
    orientation : list, optional
        Fixed orientation [roll, pitch, yaw]
    maintain_orientation : bool
        If True, maintains current orientation
    """
    print("\n" + "="*70)
    print("DRAWING: HELLO WORLD")
    print("="*70)
    print(f"Plane: {plane}")
    print(f"Scale: {scale}mm per letter")
    print(f"Spacing: {spacing}mm between letters")
    
    # Generate waypoints
    waypoints = generate_text_waypoints(
        "HELLO WORLD",
        plane=plane,
        x_pos=x_pos,
        scale=scale,
        spacing=spacing,
        orientation=orientation,
        points_per_segment=2
    )
    
    print(f"Generated {len(waypoints)} waypoints")
    
    # Execute the trajectory
    return executor.execute_trajectory(waypoints, ik_method='optimization')

# Example usage
if __name__ == "__main__":
    from motion_executor import MotionExecutor
    
    # Initialize executor
    executor = MotionExecutor(use_robot=True)
    
    # Method 1: Draw on YZ plane (default)
    print("\n### Drawing HELLO WORLD on YZ plane ###")
    draw_hello_world(
        executor,
        plane='YZ',
        x_pos=150,      # X coordinate stays at 150mm
        scale=30,       # Each letter is 30mm tall
        spacing=10,     # 10mm between letters
        orientation=[0, 0, 0]  # End effector pointing straight down
    )
    
    # Method 2: Draw on XY plane (horizontal)
    print("\n### Drawing HELLO WORLD on XY plane ###")
    draw_hello_world(
        executor,
        plane='XY',
        x_pos=300,      # Z coordinate stays at 300mm
        scale=25,
        spacing=8
    )
    
    # Method 3: Custom text on YZ plane
    print("\n### Drawing custom text ###")
    custom_waypoints = generate_text_waypoints(
        "HELLO",
        plane='YZ',
        x_pos=150,
        scale=40,
        spacing=15,
        orientation=[0, 45, 0],  # Tilted 45 degrees
        points_per_segment=15
    )
    
    executor.execute_custom_waypoints(custom_waypoints)
    
    # Cleanup
    executor.disconnect()
    print("\n" + "="*70)
    print("Text drawing complete!")
    print("="*70)
