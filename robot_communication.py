"""
Robot Communication Module
Send joint angles or target positions to ESP32 via serial
Always shows simulation before sending to real robot
"""

import serial
import time
import numpy as np
from forward_kinematics import get_end_effector_position
from inverse_kinematics import inverse_kinematics, DEFAULT_JOINT_LIMITS
from robot_simulator import RobotSimulator

# ============================================================================
# CONFIGURATION
# ============================================================================

SERIAL_PORT = '/dev/tty.usbserial-0001'  # Change this to your port
BAUD_RATE = 9600  # Match ESP32 Serial.begin(9600)
TIMEOUT = 1  # seconds
SIMULATION_DURATION = 2.0  # seconds

# Joint limits (degrees) - imported from inverse_kinematics
JOINT_LIMITS = DEFAULT_JOINT_LIMITS

# ============================================================================

class RobotController:
    """
    Controller for sending commands to the robot via serial
    Always simulates movement before sending to real robot
    """
    
    def __init__(self, port=SERIAL_PORT, baud_rate=BAUD_RATE, timeout=TIMEOUT):
        """
        Initialize robot controller with simulation
        
        Parameters:
        -----------
        port : str
            Serial port (e.g., '/dev/tty.usbserial-0001', 'COM3')
        baud_rate : int
            Baud rate (must match ESP32 setting)
        timeout : float
            Serial timeout in seconds
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self.current_angles = [0, 0, 0, 0, 0, 0]
        
        # Initialize simulator (always enabled)
        self.simulator = RobotSimulator(self.current_angles)
        print("🎬 Simulator initialized - all movements will be animated")
        
    def connect(self):
        """Establish serial connection to robot"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for ESP32 to reset
            
            # Clear any startup messages
            while self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                print(f"ESP32: {line}")
            
            self.connected = True
            print(f"✓ Connected to robot on {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect: {e}")
            print("  Running in simulation-only mode")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.connected = False
            print("Disconnected from robot")
    
    def _check_joint_limits(self, angles):
        """
        Check if angles are within limits and ask user if not
        
        Parameters:
        -----------
        angles : list or array
            Joint angles to check
            
        Returns:
        --------
        proceed : bool
            True if user wants to proceed
        """
        violations = []
        for i, (angle, (min_angle, max_angle)) in enumerate(zip(angles, JOINT_LIMITS)):
            if angle < min_angle or angle > max_angle:
                violations.append((i+1, angle, min_angle, max_angle))
        
        if violations:
            print("\n⚠️  WARNING: Joint limit violations detected!")
            print("-" * 60)
            for joint, angle, min_val, max_val in violations:
                print(f"  J{joint}: {angle:.1f}° is outside limits [{min_val}°, {max_val}°]")
            print("-" * 60)
            
            response = input("Continue anyway? (yes/no): ").strip().lower()
            return response in ['yes', 'y']
        
        return True
    
    def send_joint_angles(self, angles, wait_for_response=True, skip_confirmation=False):
        """
        Send joint angles to robot (always simulates first)
        
        Parameters:
        -----------
        angles : list or array
            Joint angles in degrees [j1, j2, j3, j4, j5, j6]
        wait_for_response : bool
            Whether to wait for ESP32 response
        skip_confirmation : bool
            Skip asking for confirmation (for trajectory execution)
            
        Returns:
        --------
        success : bool
            True if command sent successfully
        """
        # Validate angles
        if len(angles) != 6:
            print(f"✗ Error: Expected 6 angles, got {len(angles)}")
            return False
        
        angles = [float(a) for a in angles]
        
        # Check joint limits
        if not self._check_joint_limits(angles):
            print("❌ Movement cancelled by user")
            return False
        
        # ALWAYS simulate movement first (unless skipping for trajectories)
        if not skip_confirmation:
            print("\n🎬 Simulating movement...")
            self.simulator.move_to_angles(angles, duration=SIMULATION_DURATION, animate=True)
        
        # Update simulator's current position
        self.current_angles = list(angles)
        
        # If not connected, stop here (simulation only)
        if not self.connected:
            if not skip_confirmation:
                print("ℹ️  Not connected to robot - simulation only")
            return True
        
        # Ask before sending to real robot (unless skipping)
        if not skip_confirmation:
            print("\n" + "="*70)
            response = input("Send this command to real robot? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Command NOT sent to robot")
                return False
        
        # Format command: "ANGLES,j1,j2,j3,j4,j5,j6\n"
        command = "ANGLES," + ",".join([f"{a:.2f}" for a in angles]) + "\n"
        
        try:
            # Send command
            self.serial.write(command.encode('utf-8'))
            if not skip_confirmation:
                print(f"→ Sent to robot: {command.strip()}")
            
            if wait_for_response:
                # Read response
                time.sleep(0.1)
                while self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line and not skip_confirmation:
                        print(f"← ESP32: {line}")
            
            if not skip_confirmation:
                print("✓ Command sent successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error sending command: {e}")
            return False
    
    def send_target_position(self, target_xyz, ik_method='global', wait_for_response=True):
        """
        Send target XYZ position (solves IK with joint limit checking, simulates, then sends)
        
        Parameters:
        -----------
        target_xyz : list or array
            Target position [x, y, z] in mm
        ik_method : str
            IK solving method ('optimization' or 'global')
        wait_for_response : bool
            Whether to wait for ESP32 response
            
        Returns:
        --------
        success : bool
            True if command sent successfully
        solution : dict
            IK solution details
        """
        print(f"\n🎯 Target Position: {target_xyz} mm")
        print(f"   Solving IK using '{ik_method}' method...")
        
        # Solve IK with automatic retry until within joint limits
        solution = inverse_kinematics(target_xyz, 
                                     initial_guess=self.current_angles,
                                     method=ik_method,
                                     bounds=JOINT_LIMITS,
                                     max_attempts=20)
        
        if not solution['success']:
            print(f"✗ IK failed: Error = {solution['error']:.2f} mm")
            return False, solution
        
        print(f"✓ IK solved: Error = {solution['error']:.6f} mm")
        print(f"   Joint Angles: {np.round(solution['joint_angles'], 1)}°")
        print(f"   Within Limits: {solution['within_limits']}")
        print(f"   Attempts: {solution['attempts']}")
        
        if not solution['within_limits']:
            print(f"⚠ Warning: Best solution found is outside joint limits")
        
        # Send the joint angles (this will simulate and ask for confirmation)
        success = self.send_joint_angles(solution['joint_angles'], wait_for_response)
        
        return success, solution
    
    def home(self, wait_for_response=True):
        """Move robot to home position (all joints at 0°)"""
        print("\n🏠 Homing robot...")
        return self.send_joint_angles([0, 0, 0, 0, 0, 0], wait_for_response)
    
    def get_current_angles(self):
        """Get current joint angles (last commanded position)"""
        return self.current_angles.copy()
    
    def get_current_end_effector(self):
        """Get current end effector position based on current joint angles"""
        return get_end_effector_position(self.current_angles)
    
    def visualize_current_state(self):
        """Show current robot configuration"""
        self.simulator.visualize_current_state()

# ============================================================================
# Helper Functions
# ============================================================================

def find_serial_ports():
    """
    Find available serial ports
    
    Returns:
    --------
    ports : list
        List of available serial port names
    """
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    available = [port.device for port in ports]
    
    if available:
        print("Available serial ports:")
        for port in available:
            print(f"  - {port}")
    else:
        print("No serial ports found")
    
    return available

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ROBOT CONTROLLER WITH SIMULATION")
    print("="*70)
    
    # Find available ports
    print("\n1. Searching for serial ports...")
    find_serial_ports()
    
    # Create controller (simulator always enabled)
    print(f"\n2. Initializing controller...")
    robot = RobotController(SERIAL_PORT)
    
    # Try to connect (optional - can run simulation only)
    print(f"\n3. Attempting connection to {SERIAL_PORT}...")
    robot.connect()
    print("   (If connection failed, will run in simulation-only mode)")
    
    try:
        # Test 1: Move to specific angles
        print("\n" + "="*70)
        print("TEST 1: Move to Specific Joint Angles")
        print("="*70)
        test_angles = [45, 30, 60, 20, 90, 45]
        print(f"Target angles: {test_angles}")
        robot.send_joint_angles(test_angles)
        
        # Test 2: Move to target position (IK)
        print("\n" + "="*70)
        print("TEST 2: Move to Target XYZ Position")
        print("="*70)
        target_pos = [100, 150, 300]
        success, solution = robot.send_target_position(target_pos, ik_method='global')
        
        # Test 3: Test joint limits
        print("\n" + "="*70)
        print("TEST 3: Test Joint Limits (will ask for confirmation)")
        print("="*70)
        extreme_angles = [200, 30, 60, 20, 90, 45]  # J1 exceeds limit
        print(f"Attempting angles with violation: {extreme_angles}")
        robot.send_joint_angles(extreme_angles)
        
        # Test 4: Return home
        print("\n" + "="*70)
        print("TEST 4: Return to Home")
        print("="*70)
        robot.home()
        
        # Show current status
        print("\n" + "="*70)
        print("Final Status:")
        print("="*70)
        print(f"Joint Angles: {np.round(robot.get_current_angles(), 2)}°")
        print(f"End Effector: {np.round(robot.get_current_end_effector(), 2)} mm")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\n" + "="*70)
        robot.disconnect()
        print("="*70)