"""
Robot Forward and Inverse Kinematics Solver using cuRobo.

To use this module, you need to install the following dependencies:
    - CUDA 11.8 (Recommended)
    - conda create -n test_curobo python=3.10.*
    - conda activate test_curobo
    - sudo apt install git-lfs
    - git clone https://github.com/NVlabs/curobo.git && cd curobo
    - pip3 install "numpy<2.0"
    - pip3 install "torch==2.0.0"
    - pip3 install -e . --no-build-isolation
    - pip3 install pytest lark
    - python3 -m pytest .

References:
    - Curobo library: https://github.com/NVlabs/curobo
    - Curobo documentation: https://curobo.org/get_started/1_install_instructions.html
"""

from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class Kinematic:
    """
    A class that combines forward and inverse kinematics for a robot (based on curobo).
    Input and output are both basic Python list types.
    """

    def __init__(self, urdf_path: str = None, target_link_name: str = "link6"):
        """
        Initialize the kinematics solver.

        Args:
            urdf_path: Path to the robot urdf file.
            target_link_name: End link name.
        """
        self.tensor_args = TensorDeviceType()

        # curobo robot configuration
        self.robot_cfg = RobotConfig.from_basic(urdf_path, "base_link", target_link_name , self.tensor_args)

        # curobo IK configuration
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.01,
            position_threshold=0.0011,
            num_seeds=100,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)

        # Get joint information
        self.joints_cfg = self.robot_cfg.kinematics.get_joint_limits()

        # Number of joints and limits
        self.num_joints = len(self.joints_cfg.joint_names)
        self.joint_limits = [i for i in zip(*self.joints_cfg.position.tolist())]

        self.joints_zero = [0] * self.num_joints
        self.xyz_wxyz_zero = self.solve_fk(self.joints_zero)
        self.solve_ik(self.xyz_wxyz_zero[:3], self.xyz_wxyz_zero[3:])

    def solve_fk(self, joint_angles: list) -> list:
        """
        Forward kinematics: input joint angle list, output end pose (xyz + wxyz).

        Args:
            joint_angles: Joint angle list (unit: radians).

        Returns:
            [x, y, z, w, x, y, z] list
        """
        # Convert to torch tensor
        q_tensor = self.tensor_args.to_device(joint_angles)
        kin_state = self.ik_solver.fk(q_tensor)
        # Output as list
        xyz = kin_state.ee_position[0].tolist()
        wxyz = kin_state.ee_quaternion[0].tolist()
        return xyz + wxyz

    def solve_ik(
        self,
        target_position: list,
        target_wxyz: list,
        initial_guess: list = None,
    ) -> list:
        """
        Inverse kinematics: input target end pose (xyz + wxyz), output joint angle list.

        Args:
            target_position: Target position [x, y, z].
            target_wxyz: Target quaternion [w, x, y, z].
            initial_guess: Initial guess joint angle list.

        Returns:
            Joint angle list (unit: radians), or None if no solution found.
        """
        # Create goal Pose
        pos_tensor = self.tensor_args.to_device(target_position)
        quat_tensor = self.tensor_args.to_device(target_wxyz)
        goal = Pose(pos_tensor, quat_tensor)

        # Initial guess (optional)
        seed_config = None
        if initial_guess is not None:
            seed_config = self.tensor_args.to_device(initial_guess).unsqueeze(0).unsqueeze(0)

        # Call curobo IK
        result = self.ik_solver.solve_single(goal, seed_config=seed_config)
        if result.success[0]:
            return result.solution[0][0].tolist()
        else:
            return None

# Test Demo
if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    urdf_path = os.path.join(current_dir, "piper/piper.urdf")
    solver = Kinematic(urdf_path=urdf_path)

    # Test FK
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fk_solution = solver.solve_fk(joint_angles)
    print(f"FK solution: {fk_solution}")

    # Test IK
    target_position = fk_solution[:3]
    target_wxyz = fk_solution[3:]
    ik_solution = solver.solve_ik(target_position, target_wxyz)
    print(f"IK solution: {ik_solution}")

    # Calculate time
    import time
    t1 = time.time()
    while True:
        ik_solution = solver.solve_ik(target_position, target_wxyz)
        print(f"IK solution: {ik_solution}")
        t2 = time.time()
        print(f"time: {(t2 - t1) * 1000:.2f} ms")
        t1 = t2