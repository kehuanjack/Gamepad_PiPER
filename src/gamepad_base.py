import os
import time
import viser
import queue
import pygame
import yourdfpy
import numpy as np
import multiprocessing as mp
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation as R


def visualization_process(urdf, root_name, joint_queue: mp.Queue, shutdown_event):
    """Independent process running Viser visualization server"""
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    urdf_vis = ViserUrdf(server, urdf, root_node_name=root_name)

    try:
        while not shutdown_event.is_set():
            try:
                joints_deg = joint_queue.get(timeout=0.1)
                for _ in range(joint_queue.qsize()):
                    joints_deg = joint_queue.get_nowait()
                urdf_vis.update_cfg(joints_deg)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in visualization process: {e}")
    except KeyboardInterrupt:
        pass


class Button:
    """Button class for handling button press events."""
    
    def __init__(self):
        self.last_state = False
        self.triggered = False

    def update(self, current_state):
        # Detect button state changes
        if current_state and not self.last_state:  # Detect rising edge (from released to pressed)
            self.triggered = True
            self.last_state = current_state
            return True

        self.last_state = current_state
        self.triggered = False
        return False


class GamepadBase:
    """Base class for gamepad control of robotic arm."""
    
    def __init__(self, urdf_path: str = None, mesh_path: str = None, root_name: str = None):
        pygame.init()
        pygame.joystick.init()

        # Detect system platform
        self.system_platform = pygame.display.get_driver()

        # Robot arm status
        self.joint_angles = np.zeros(6)
        self.up_level_mode = "joint"
        self.low_level_mode = "joint"
        self.command_mode = 0x00

        # Deadzone settings
        self.deadzone = 0.1

        # Step control: joint angle (rad), gripper percentage, translation (m), rotation (deg)
        self.joint_angle_step = 0.5 * np.pi / 180.0
        self.gripper_step = 1.0
        self.translation_step = 0.001
        self.rotation_step = 0.5

        # Gripper parameter settings
        self.gripper_max_width = 0.07   # Gripper maximum stroke (m)
        self.gripper_state = 0          # Gripper opening and closing degree (%)

        # Gripper control state
        self.last_left_trigger = 0                  # Last value of left trigger
        self.last_right_trigger = 0                 # Last value of right trigger
        self.left_trigger_ready_for_jump = False    # Whether left trigger is ready for jump
        self.right_trigger_ready_for_jump = False   # Whether right trigger is ready for jump
        self.just_jumped_with_left = False          # Just jumped with left trigger
        self.just_jumped_with_right = False         # Just jumped with right trigger
        self.gripper_initialized = False            # Whether gripper is initialized

        # Speed control
        self.speed_factors = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.speed_factor_index = 2     # Default 1.0x
        self.movement_speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.movement_speed_index = 9   # Default 100%

        # Saved positions
        self.saved_positions = []
        self.current_position_index = -1
        self.playback_reversed = False  # Playback order flag

        # Set button and joystick mapping according to platform
        self._setup_platform_mappings()

        # Robot arm connection status
        self.arm_connected = False
        self.arm_enabled = False

        # Button long press detection
        self.x_pressed_time = 0
        self.a_pressed_time = 0
        self.start_pressed_time = 0
        self.lb_pressed_time = 0
        self.rb_pressed_time = 0
        self.long_press_threshold = 0.3  # Long press time threshold (seconds)

        # Joystick connection status
        self.joystick = None
        self.joystick_connected = False

        # Create button objects
        self.buttons = {
            'a': Button(), 'b': Button(), 'x': Button(), 'y': Button(),
            'lb': Button(), 'rb': Button(), 'back': Button(), 'start': Button(),
            'home': Button(), 'l3': Button(), 'r3': Button(),
        }

        # Initialize visualization process
        self.urdf = None
        self.visualization_process = None
        self.joint_queue = None
        self.shutdown_event = None

        if urdf_path is not None and mesh_path is not None and root_name is not None:
            try:
                self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_path)

                # Set multiprocess startup method
                mp.set_start_method('spawn')

                # Create queues and events for inter-process communication
                self.joint_queue = mp.Queue(maxsize=10)
                self.shutdown_event = mp.Event()

                # Start visualization process
                self.visualization_process = mp.Process(
                    target=visualization_process,
                    args=(self.urdf, root_name, self.joint_queue, self.shutdown_event),
                    daemon=True
                )
                self.visualization_process.start()

            except Exception as e:
                self.urdf = None

        # Kinematic properties - to be implemented by subclasses
        self.kinematic = None
        self.joint_limits = None
        self.xyz_wxyz = np.zeros(7)
        self.xyz_rpy = np.zeros(6)

    def _setup_platform_mappings(self):
        """Set button and joystick mapping according to platform"""
        if self.system_platform == "windows":
            self.button_map = {
                'a': 0, 'b': 1, 'x': 2, 'y': 3,
                'lb': 4, 'rb': 5, 'back': 6, 'start': 7,
                'l3': 8, 'r3': 9, 'home': 10,
            }
            self.axis_map = {
                'left_x': 0, 'left_y': 1, 'right_x': 2, 'right_y': 3,
                'left_trigger': 4, 'right_trigger': 5,
            }
            self.hat_map = {'dpad': 0}
        else:
            self.button_map = {
                'a': 0, 'b': 1, 'x': 2, 'y': 3,
                'lb': 4, 'rb': 5, 'back': 6, 'start': 7,
                'home': 8, 'l3': 9, 'r3': 10,
            }
            self.axis_map = {
                'left_x': 0, 'left_y': 1, 'right_x': 3, 'right_y': 4,
                'left_trigger': 2, 'right_trigger': 5,
            }
            self.hat_map = {'dpad': 0}

    def _handle_joystick_events(self):
        """Handle joystick connection/disconnection events"""
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                try:
                    self.joystick = pygame.joystick.Joystick(event.device_index)
                    self.joystick.init()
                    self.joystick_connected = True
                    self._update_vis()
                except Exception as e:
                    self.joystick = None
                    self.joystick_connected = False

            elif event.type == pygame.JOYDEVICEREMOVED:
                self.joystick = None
                self.joystick_connected = False

        return self.joystick_connected

    def _apply_deadzone(self, value):
        """Apply deadzone filtering"""
        return value if abs(value) > self.deadzone else 0

    def _get_axis_value(self, axis_name):
        """Get axis value"""
        if axis_name not in self.axis_map:
            return 0

        axis_index = self.axis_map[axis_name]
        if axis_index >= self.joystick.get_numaxes():
            return 0

        value = self.joystick.get_axis(axis_index)

        # Trigger range is -1.0 to 1.0, need to convert to 0.0 to 1.0
        if axis_name in ['left_trigger', 'right_trigger']:
            return (value + 1) / 2

        return value

    def _get_hat_value(self, hat_name):
        """Get hat switch value"""
        if hat_name not in self.hat_map:
            return (0, 0)

        hat_index = self.hat_map[hat_name]
        if hat_index >= self.joystick.get_numhats():
            return (0, 0)

        return self.joystick.get_hat(hat_index)

    def _update_vis(self) -> None:
        """Update visualization with current joint angles (radians)."""
        if self.joint_queue is None:
            return

        joints = self.joint_angles.copy()
        gripper_value = self.gripper_max_width * self.gripper_state * 1e-2 / 2
        joints = np.append(joints, gripper_value)
        joints = np.append(joints, -gripper_value)

        try:
            joints_list = joints.tolist() if hasattr(joints, 'tolist') else joints
            self.joint_queue.put_nowait(joints_list)
        except queue.Full:
            try:
                self.joint_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                joints_list = joints.tolist() if hasattr(joints, 'tolist') else joints
                self.joint_queue.put_nowait(joints_list)
            except queue.Full:
                pass
    
    def _wxyz_to_xyzw(self, wxyz):
        """Convert wxyz to xyzw"""
        return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    
    def _xyzw_to_wxyz(self, xyzw):
        """Convert xyzw to wxyz"""
        return [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]

    def _joint_to_pose(self):
        """Joint angle to end pose - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _joint_to_pose")

    def _pose_to_joint(self, xyz, orientation):
        """End pose to joint angle - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _pose_to_joint")

    def _update_joint_mode(self):
        """Control in joint mode"""
        # Get current speed factor and movement speed
        speed_factor = self.speed_factors[self.speed_factor_index]

        # Left joystick: J1 (base rotation) and J2 (arm lifting)
        left_x = self._apply_deadzone(self._get_axis_value('left_x'))
        left_y = self._apply_deadzone(self._get_axis_value('left_y'))

        self.joint_angles[0] -= left_x * self.joint_angle_step * speed_factor  # J1
        self.joint_angles[1] -= left_y * self.joint_angle_step * speed_factor  # J2

        # Right joystick: J3 (forearm) and J6 (wrist rotation)
        right_x = self._apply_deadzone(self._get_axis_value('right_x'))
        right_y = self._apply_deadzone(self._get_axis_value('right_y'))

        self.joint_angles[2] += right_y * self.joint_angle_step * speed_factor  # J3
        self.joint_angles[5] += right_x * self.joint_angle_step * speed_factor  # J6

        # Direction keys: J4 (wrist yaw) and J5 (wrist pitch)
        hat = self._get_hat_value('dpad')
        self.joint_angles[3] += hat[0] * self.joint_angle_step * speed_factor  # J4
        self.joint_angles[4] -= hat[1] * self.joint_angle_step * speed_factor  # J5

        if left_x or left_y or right_x or right_y or hat[0] or hat[1]:
            # Limit joint angles within reasonable range
            for i in range(len(self.joint_angles)):
                self.joint_angles[i] = max(self.joint_limits[i][0], min(self.joint_limits[i][1], self.joint_angles[i]))

            self._joint_to_pose()
            self._update_vis()

    def _update_pose_mode(self):
        """Control in pose coordinate mode (based on end coordinate system)"""
        # Get current speed factor
        speed_factor = self.speed_factors[self.speed_factor_index]

        # Left joystick: X and Y movement in end coordinate system
        left_x = self._apply_deadzone(self._get_axis_value('left_x'))
        left_y = self._apply_deadzone(self._get_axis_value('left_y'))

        # Right joystick: Z movement and rotation around Z in end coordinate system
        right_x = self._apply_deadzone(self._get_axis_value('right_x'))
        right_y = self._apply_deadzone(self._get_axis_value('right_y'))

        # Direction keys: Rotation around X and Y in end coordinate system
        hat = self._get_hat_value('dpad')

        # Calculate displacement and rotation increments in end coordinate system
        d_local = np.array([left_y, -left_x, -right_y]) * self.translation_step * speed_factor
        r_local = np.array([hat[0], -hat[1], right_x]) * self.rotation_step * speed_factor

        # If there is any input, update inverse kinematics
        if np.any(d_local) or np.any(r_local):
            # Get current pose
            current_position = self.xyz_wxyz[0:3]
            current_orientation = R.from_quat(self._wxyz_to_xyzw(self.xyz_wxyz[3:]))

            # Calculate new pose (based on end coordinate system)
            R_local = R.from_euler('xyz', r_local, degrees=True)
            new_orientation = current_orientation * R_local
            d_world = current_orientation.apply(d_local)
            new_position = current_position + d_world

            self._pose_to_joint(new_position, new_orientation)
            self._update_vis()

    def _update_gripper(self):
        """Update gripper status (ensure that after jump, it must be released to increase or decrease normally)"""
        # Get current speed factor and movement speed
        speed_factor = self.speed_factors[self.speed_factor_index]

        # Left trigger: close gripper
        left_trigger = self._get_axis_value('left_trigger')  # Already converted to 0-1 range

        # Right trigger: open gripper
        right_trigger = self._get_axis_value('right_trigger')  # Already converted to 0-1 range

        # Initialization at first run
        if not self.gripper_initialized:
            self.last_left_trigger = left_trigger
            self.last_right_trigger = right_trigger
            self.gripper_initialized = True
            return  # Skip first update to avoid initial jump

        # Check if trigger is released
        left_released = left_trigger == 0 and self.last_left_trigger > 0
        right_released = right_trigger == 0 and self.last_right_trigger > 0

        # If the trigger is released, set the flag ready for jump
        if left_released:
            self.left_trigger_ready_for_jump = True
            self.just_jumped_with_left = False  # Reset jump state
        if right_released:
            self.right_trigger_ready_for_jump = True
            self.just_jumped_with_right = False  # Reset jump state

        # Update last trigger value
        self.last_left_trigger = left_trigger
        self.last_right_trigger = right_trigger

        # Handle left trigger (reduce gripper)
        if left_trigger > 0:
            if self.just_jumped_with_left:
                # Just jumped with left trigger, need to release before operating
                pass
            elif self.gripper_state > 0 and not self.just_jumped_with_right:
                # Normal reduction (provided no jump with right trigger)
                self.gripper_state = max(0, self.gripper_state - left_trigger * self.gripper_step * speed_factor)
                self.left_trigger_ready_for_jump = False  # Reset jump ready flag
                self._update_vis()
            elif self.gripper_state == 0 and self.left_trigger_ready_for_jump and not self.just_jumped_with_right:
                # Jump from 0% to 100%
                self.gripper_state = 100
                self.left_trigger_ready_for_jump = False
                self.just_jumped_with_left = True  # Mark just jumped with left trigger
                self._update_vis()

        # Handle right trigger (increase gripper)
        if right_trigger > 0:
            if self.just_jumped_with_right:
                # Just jumped with right trigger, need to release before operating
                pass
            elif self.gripper_state < 100 and not self.just_jumped_with_left:
                # Normal increase (provided no jump with left trigger)
                self.gripper_state = min(100, self.gripper_state + right_trigger * self.gripper_step * speed_factor)
                self.right_trigger_ready_for_jump = False  # Reset jump ready flag
                self._update_vis()
            elif self.gripper_state == 100 and self.right_trigger_ready_for_jump and not self.just_jumped_with_left:
                # Jump from 100% to 0%
                self.gripper_state = 0
                self.right_trigger_ready_for_jump = False
                self.just_jumped_with_right = True  # Mark just jumped with right trigger
                self._update_vis()

    def _check_button_long_press(self):
        """Check button long press"""
        current_time = time.time()

        # Check X button
        if 'x' in self.button_map:
            x_pressed = self.joystick.get_button(self.button_map['x'])
            if x_pressed and self.x_pressed_time == 0:
                self.x_pressed_time = current_time
            elif not x_pressed and self.x_pressed_time > 0:
                press_duration = current_time - self.x_pressed_time
                if press_duration >= self.long_press_threshold:
                    self._clear_all_positions()
                else:
                    self._toggle_playback_order()
                self.x_pressed_time = 0

        if 'a' in self.button_map:
            a_pressed = self.joystick.get_button(self.button_map['a'])
            if a_pressed and self.a_pressed_time == 0:
                self.a_pressed_time = current_time
            elif not a_pressed and self.a_pressed_time > 0:
                press_duration = current_time - self.a_pressed_time
                if press_duration >= self.long_press_threshold:
                    self._clear_current_position()
                else:
                    self._save_position()
                self.a_pressed_time = 0

        if 'start' in self.button_map:
            start_pressed = self.joystick.get_button(self.button_map['start'])
            if start_pressed and self.start_pressed_time == 0:
                self.start_pressed_time = current_time
            elif not start_pressed and self.start_pressed_time > 0:
                press_duration = current_time - self.start_pressed_time
                if press_duration >= self.long_press_threshold:
                    self._toggle_low_level_mode() 
                else:
                    self._toggle_up_level_mode()
                self.start_pressed_time = 0

        # Check LB button
        if 'lb' in self.button_map:
            lb_pressed = self.joystick.get_button(self.button_map['lb'])
            if lb_pressed and self.lb_pressed_time == 0:
                self.lb_pressed_time = current_time
            elif not lb_pressed and self.lb_pressed_time > 0:
                press_duration = current_time - self.lb_pressed_time
                if press_duration >= self.long_press_threshold:
                    self._decrease_speed_factor()
                else:
                    self._increase_speed_factor()
                self.lb_pressed_time = 0

        # Check RB button
        if 'rb' in self.button_map:
            rb_pressed = self.joystick.get_button(self.button_map['rb'])
            if rb_pressed and self.rb_pressed_time == 0:
                self.rb_pressed_time = current_time
            elif not rb_pressed and self.rb_pressed_time > 0:
                press_duration = current_time - self.rb_pressed_time
                if press_duration >= self.long_press_threshold:
                    self._decrease_movement_speed()
                else:
                    self._increase_movement_speed()
                self.rb_pressed_time = 0

    def _handle_button_events(self, button_events):
        """Handle all button events"""
        if self.arm_connected:
            if 'b' in button_events:
                self._restore_previous_position()

            if 'y' in button_events:
                self._go_home()
                self._update_vis()

            if 'back' in button_events:
                self._toggle_command_mode()

        if 'home' in button_events:
            self._toggle_arm_connection()
            self._update_vis()

    def _increase_speed_factor(self):
        """Increase speed factor (support cycle)"""
        if self.speed_factor_index < len(self.speed_factors) - 1:
            self.speed_factor_index += 1
        else:
            self.speed_factor_index = 0  # Cycle to beginning

    def _decrease_speed_factor(self):
        """Decrease speed factor (support cycle)"""
        if self.speed_factor_index > 0:
            self.speed_factor_index -= 1
        else:
            self.speed_factor_index = len(self.speed_factors) - 1  # Cycle to end

    def _increase_movement_speed(self):
        """Increase movement speed (support cycle)"""
        if self.movement_speed_index < len(self.movement_speeds) - 1:
            self.movement_speed_index += 1
        else:
            self.movement_speed_index = 0  # Cycle to beginning

    def _decrease_movement_speed(self):
        """Decrease movement speed (support cycle)"""
        if self.movement_speed_index > 0:
            self.movement_speed_index -= 1
        else:
            self.movement_speed_index = len(self.movement_speeds) - 1  # Cycle to end

    def _toggle_arm_connection(self):
        """Toggle robot arm connection status"""
        if not self.arm_connected:
            self._connect_and_enable_arm()
        else:
            self._go_home_and_disable()

    def _toggle_playback_order(self):
        """Toggle position playback order"""
        self.playback_reversed = not self.playback_reversed

    def _clear_current_position(self):
        """Clear current saved position"""
        if self.current_position_index >= 0:
            self.saved_positions.pop(self.current_position_index)
            self.current_position_index = -1

    def _clear_all_positions(self):
        """Clear all saved positions"""
        self.saved_positions.clear()
        self.current_position_index = -1

    def _save_position(self):
        """Save current position"""
        position = {
            "joints": self.joint_angles.copy(),
            "gripper": self.gripper_state
        }
        self.saved_positions.append(position)
        self.current_position_index = len(self.saved_positions) - 1

    def _restore_previous_position(self):
        """Restore previous saved position"""
        if not self.saved_positions:
            return

        if self.playback_reversed:
            # Reverse order playback
            if self.current_position_index < len(self.saved_positions) - 1:
                self.current_position_index += 1
            else:
                self.current_position_index = 0
        else:
            # Sequential playback
            if self.current_position_index > 0:
                self.current_position_index -= 1
            else:
                self.current_position_index = len(self.saved_positions) - 1

        position = self.saved_positions[self.current_position_index]
        self.joint_angles = position["joints"].copy()
        self.gripper_state = position["gripper"]
        self._joint_to_pose()
        self._update_vis()

    def _toggle_up_level_mode(self):
        """Toggle up_level_mode (joint/pose)"""
        if self.up_level_mode == "joint":
            self.up_level_mode = "pose"
        elif self.up_level_mode == "pose":
            self.up_level_mode = "joint"

    def _toggle_low_level_mode(self):
        """Toggle low_level_mode (joint/pose)"""
        if self.low_level_mode == "joint":
            self.low_level_mode = "pose"
        elif self.low_level_mode == "pose":
            self.low_level_mode = "joint"

    def _go_home(self):
        """Return to home position"""
        self.joint_angles = np.zeros(6)
        self._joint_to_pose()

    def _connect_and_enable_arm(self):
        """Connect and enable robot arm"""
        self.arm_connected = True
        self.arm_enabled = True

    def _go_home_and_disable(self):
        """Return to home and disable reset robot arm"""
        self._go_home()
        self.arm_enabled = False
        self.arm_connected = False

    def _toggle_command_mode(self):
        """Toggle command mode"""
        if self.command_mode == 0x00:
            self.command_mode = 0xAD
        else:
            self.command_mode = 0x00

    def update(self):
        """Update joystick input and calculate robot arm status"""
        # Check joystick connection status
        if not self._handle_joystick_events():
            return

        try:
            # Update all button states
            button_events = {}
            for name, button in self.buttons.items():
                if name not in self.button_map:
                    continue

                button_id = self.button_map[name]
                if button_id >= self.joystick.get_numbuttons():
                    continue

                pressed = button.update(self.joystick.get_button(button_id))
                if pressed:
                    button_events[name] = True

            # Handle button events
            self._handle_button_events(button_events)

            # Check button long press
            self._check_button_long_press()

            if self.arm_connected and self.arm_enabled:
                # Update robot arm control
                if self.up_level_mode == "joint":
                    self._update_joint_mode()
                elif self.up_level_mode == "pose":
                    self._update_pose_mode()

                # Update gripper
                self._update_gripper()

        except pygame.error:
            self.joystick_connected = False
            self.joystick = None

    def get_state(self):
        """Get current status of robot arm"""
        return {
            "joints": self.joint_angles.copy(),
            "xyz_rpy": self.xyz_rpy.copy(),
            "gripper": self.gripper_state,
            "speed_factor": self.speed_factors[self.speed_factor_index],
            "movement_speed": self.movement_speeds[self.movement_speed_index],
            "arm_connected": self.arm_connected,
            "arm_enabled": self.arm_enabled,
            "command_mode": self.command_mode,
            "up_level_mode": self.up_level_mode,
            "low_level_mode": self.low_level_mode,
            "joystick_connected": self.joystick_connected
        }

    def print_state(self):
        """Print current status of robot arm"""

         # Clear screen and print status information
        os.system('cls' if os.name == 'nt' else 'clear')

        arm_status = "Connected" if self.arm_connected else "Disconnected"
        arm_enabled = "Enabled" if self.arm_enabled else "Disabled"
        joy_status = "Connected" if self.joystick_connected else "Disconnected"
        cmd_mode = f"0x{self.command_mode:02X}"
        playback_order = "Sequence" if self.playback_reversed else "Reverse"
        current_position = f"#{self.current_position_index + 1}" if self.saved_positions else "None"

        print("=== Robot Arm Control Status ===")
        print(f"Joint angles: {[f'{np.degrees(a):6.1f}' for a in self.joint_angles]}")
        print(f"End pose: {[f'{p:6.3f}' for p in self.xyz_rpy]}")
        print(f"Gripper status: {self.gripper_state:5.1f}%")
        print(f"Command mode: {cmd_mode}")
        print(f"up_level_mode: {self.up_level_mode:8}")
        print(f"low_level_mode: {self.low_level_mode:8}")
        print(f"Speed factor: x{self.speed_factors[self.speed_factor_index]}")
        print(f"Movement speed: {self.movement_speeds[self.movement_speed_index]}%")
        print(f"Robot arm status: {arm_status}/{arm_enabled}")
        print(f"Joystick status: {joy_status}")
        print(f"Position playback order: {playback_order}")
        print(f"Saved positions count: {len(self.saved_positions)}")
        print(f"Current position index: {current_position}")
        print("\n=== Button instructions ===")
        print("Home: Connect/Disconnect robot arm")
        print("Back: Toggle command mode")
        print("Start: Short press toggle up_level_mode, long press toggle low_level_mode")
        print("Y: Return to home")
        print("A: Short press Save position, long press clear current position")
        print("B: Restore position")
        print("X: Short press toggle order, long press clear all positions")
        print("LB: Short press increase speed, long press decrease")
        print("RB: Short press increase movement speed, long press decrease")
        print("LT: Close gripper")
        print("RT: Open gripper")