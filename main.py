import os
import time
import pygame
import numpy as np
from piper_sdk import *

# Choose the appropriate controller based on the inverse kinematics algorithm
# from src.gamepad_pin import RoboticArmController
from src.gamepad_curobo import RoboticArmController
# from src.gamepad_limit import RoboticArmController
# from src.gamepad_no_limit import RoboticArmController


# Inherit RoboticArmController class, add physical robot arm control function
class Teleop(RoboticArmController):
    """Teleoperation class with physical robot arm control functionality."""
    
    def __init__(self, interface: C_PiperInterface_V2 = None, urdf_path: str = None, mesh_path: str = None, root_name: str = None, target_link_name: str = None):
        super().__init__(urdf_path, mesh_path, root_name, target_link_name)
        self.interface = interface

    def _go_home(self):
        """Return to home position with physical robot arm control."""
        if self.arm_connected and self.arm_enabled:
            self.interface.JointCtrl(0, 0, 0, 0, 0, 0)
            self.joint_angles = np.zeros(6)
            self._joint_to_pose()

    def _connect_and_enable_arm(self):
        """Connect and enable physical robot arm."""
        if not self.arm_connected:
            self.interface.ConnectPort()
            self.arm_connected = True
        if not self.arm_enabled:
            while not self.interface.EnablePiper():
                time.sleep(0.01)
            self.arm_enabled = True
            move_code = 0x01
            if self.low_level_mode == "pose":
                move_code = 0x00
            self.interface.ModeCtrl(0x01, move_code, self.movement_speeds[self.movement_speed_index], self.command_mode)
            time.sleep(0.1)
            self._go_home()

    def _go_home_and_disable(self):
        """Return to home and disable physical robot arm."""
        if self.arm_connected and self.arm_enabled:
            self._go_home()
            self.interface.GripperCtrl(0, 1000, 0, 0)
            time.sleep(2)
            self.interface.MotionCtrl_1(0x01, 0, 0)
            time.sleep(1)
            self.interface.MotionCtrl_1(0x02, 0, 0)
            self.arm_enabled = False
            self.arm_connected = False

    def _change_to_0x00_mode(self):
        """Change to 0x00 command mode."""
        self._go_home_and_disable()
        time.sleep(0.1)
        move_code = 0x01
        if self.low_level_mode == "pose":
            move_code = 0x00
        self.interface.ModeCtrl(0x01, move_code, self.movement_speeds[self.movement_speed_index], 0x00)
        time.sleep(1)
        while not self.interface.EnablePiper():
            time.sleep(0.01)
        time.sleep(0.1)
        self.interface.ModeCtrl(0x01, move_code, self.movement_speeds[self.movement_speed_index], 0x00)
        time.sleep(0.01)
        joints = np.round(np.degrees(self.joint_angles[:6]) * 1000).astype(int).tolist()
        self.interface.JointCtrl(*joints)
        self.arm_connected = True
        self.arm_enabled = True

    def _toggle_command_mode(self):
        """Toggle command mode between 0x00 and 0xAD."""
        if self.command_mode == 0x00:
            move_code = 0x01
            if self.low_level_mode == "pose":
                move_code = 0x00
            self.interface.ModeCtrl(0x01, move_code, self.movement_speeds[self.movement_speed_index], 0xAD)
            self.command_mode = 0xAD
        elif self.command_mode == 0xAD:
            self._change_to_0x00_mode()
            self.command_mode = 0x00

def get_current_path():
    """Get current path"""
    return os.path.dirname(os.path.realpath(__file__))

def main():
    """Main function for robotic arm teleoperation."""
    urdf_path = os.path.join(get_current_path(), "piper/piper.urdf")
    mesh_path = os.path.join(get_current_path(), "piper/meshes/")

    # Initialize low-level interface
    robot = C_PiperInterface_V2()

    # Initialize control class
    controller = Teleop(robot, urdf_path, mesh_path, "/base_link", "link6")

    t1 = time.time()

    try:
        while True:
            # Update control state
            controller.update()

            # Get current state
            state = controller.get_state()

            # Print status
            controller.print_state()

            # Control physical robot arm movement
            if state["arm_connected"] and state["arm_enabled"]:
                move_speed = state["movement_speed"]
                cmd_mode = state["command_mode"]
                low_level_mode = state["low_level_mode"]
                
                if low_level_mode == "joint":
                    # Low-level joint control
                    joints = state["joints"]
                    joints_ctl = np.round(np.degrees(joints[:6]) * 1000).astype(int).tolist()
                    robot.ModeCtrl(0x01, 0x01, move_speed, cmd_mode)
                    robot.JointCtrl(*joints_ctl)
                elif low_level_mode == "pose":
                    # Low-level pose control
                    xyz_rpy = state["xyz_rpy"]
                    xyz_rpy[:3] = np.round(xyz_rpy[:3] * 1e6)
                    xyz_rpy[3:] = np.round(xyz_rpy[3:] * 1000)
                    xyz_rpy = xyz_rpy.astype(int).tolist()
                    robot.ModeCtrl(0x01, 0x00, move_speed, cmd_mode)
                    robot.EndPoseCtrl(*xyz_rpy)

                # Low-level gripper control
                gripper_state = state["gripper"]
                gripper_value = int(controller.gripper_max_width * gripper_state * 1e4)
                robot.GripperCtrl(gripper_value, 3000, 0x01, 0)

            t2 = time.time()
            print(f"Update time: {(t2 - t1) *1000:.3f}ms")
            t1 = t2

            pygame.time.wait(5)  # Control loop frequency

    except KeyboardInterrupt:
        print("\nProgram exited")
        pygame.quit()

if __name__ == "__main__":
    # Activate can0 interface
    os.system("sudo ip link set can0 up type can bitrate 1000000")

    # Run main function
    main()