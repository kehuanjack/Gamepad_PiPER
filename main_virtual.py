import os
import time
import pygame

# Choose the appropriate controller based on the inverse kinematics algorithm
# from src.gamepad_pin import RoboticArmController
# from src.gamepad_curobo import RoboticArmController
from src.gamepad_trac_ik import RoboticArmController
# from src.gamepad_limit import RoboticArmController
# from src.gamepad_no_limit import RoboticArmController


class Teleop(RoboticArmController):
    def __init__(self, urdf_path: str = None, mesh_path: str = None, root_name: str = None, target_link_name: str = None):
            # controller.print_state()
        super().__init__(urdf_path, mesh_path, root_name, target_link_name = target_link_name)

def get_current_path():
    """Get current path"""
    return os.path.dirname(os.path.realpath(__file__))

def main():
    """Main function for robotic arm teleoperation."""
    urdf_path = os.path.join(get_current_path(), "piper/piper.urdf")
    mesh_path = os.path.join(get_current_path(), "piper/meshes/")

    # Initialize control class
    controller = Teleop(urdf_path, mesh_path, "/base_link", "link6")

    t1 = time.time()

    try:
        while True:
            # Update control state
            controller.update()

            # Print status
            controller.print_state()

            t2 = time.time()
            print(f"Update time: {(t2 - t1) *1000:.3f}ms")
            t1 = t2

            pygame.time.wait(5)  # Control loop frequency

    except KeyboardInterrupt:
        print("\nProgram exited")
        pygame.quit()

if __name__ == "__main__":
    main()