"""
Robot Forward and Inverse Kinematics Solver using PyRoKi.

This module provides a combined forward and inverse kinematics solver for robotic arms.
It utilizes JAX for high-performance numerical computing and automatic differentiation,
and the PyRoKi library for robot-specific kinematics computations.

To use this module, you need to install the following dependencies:
    - conda create -n test_pyroki python=3.10.*
    - conda activate test_pyroki
    - pip3 install yourdfpy
    - pip3 install "numpy<=1.26.0"
    - pip3 install pyroki@git+https://github.com/chungmin99/pyroki.git@f234516

References:
    - JAX documentation: https://jax.readthedocs.io
    - PyRoKi library: https://github.com/chungmin99/pyroki
"""

import jax
import jaxls
import jaxlie
import numpy as onp
import pyroki as pk
import jax.numpy as jnp
import jax_dataclasses as jdc

try:
    import yourdfpy
except ImportError:
    yourdfpy = None


class Kinematic:
    """
    A class that combines forward and inverse kinematics for a robot.

    Provides methods to compute forward kinematics (FK) for determining end-effector
    pose from joint angles, and inverse kinematics (IK) for finding joint angles
    that achieve a desired end-effector pose.
    """

    def __init__(self, urdf_path: str = None, robot: pk.Robot = None, target_link_name: str = None, ik_max_iters: int = 100):
        """
        Initialize the solver with a robot model.

        Args:
            urdf_path: Path to the URDF file describing the robot.
            robot: Pre-constructed PyRoKi Robot object.
            target_link_name: Default link name for FK/IK computations.
            ik_max_iters: Maximum iterations for the IK solver.
        """
        if robot is not None and isinstance(robot, pk.Robot):
            self.robot = robot
        elif urdf_path is not None and yourdfpy is not None:
            self.robot = pk.Robot.from_urdf(yourdfpy.URDF.load(urdf_path))
        else:
            raise ValueError("Either urdf_path or robot must be provided")

        if target_link_name is not None and target_link_name not in self.robot.links.names:
            raise ValueError(f"Link '{target_link_name}' not found in the robot model")

        self.target_link_name = target_link_name
        self.ik_max_iters = ik_max_iters

        # Get number of joints
        self.num_joints = self.robot.joints.num_actuated_joints

        # Initialize joint zero and previous solution
        self.joints_zero = onp.zeros(self.num_joints)
        self.previous_solution = self.joints_zero.copy()

        # Store joint limits
        self.joint_limits = list(zip(self.robot.joints.lower_limits.tolist(), self.robot.joints.upper_limits.tolist()))

        self.xyz_wxyz_zero = self.solve_fk(self.joints_zero)
        self.solve_ik(self.xyz_wxyz_zero[:3], self.xyz_wxyz_zero[3:])

    def solve_fk(self, joint_angles: onp.ndarray, target_link_name: str = None) -> onp.ndarray:
        """
        Compute forward kinematics for a specific link of the robot.

        Args:
            joint_angles: Array of joint angles in radians.
                         Shape: (robot.joints.actuated_count,)
            target_link_name: Name of the link for which to compute FK.

        Returns:
            Array containing position and orientation in [x, y, z, qw, qx, qy, qz] format.
        """
        assert joint_angles.shape == (self.robot.joints.num_actuated_joints,)

        if target_link_name is None:
            target_link_name = self.target_link_name
            if target_link_name is None:
                raise ValueError("target_link_name must be specified")
        target_link_index = self.robot.links.names.index(target_link_name)

        wxyz_xyz = self.robot.forward_kinematics(jnp.array(joint_angles))[target_link_index]
        return onp.concatenate((wxyz_xyz[4:7], wxyz_xyz[0:4]))

    def solve_ik(
        self,
        target_position: onp.ndarray,
        target_wxyz: onp.ndarray,
        target_link_name: str = None,
        initial_guess: onp.ndarray = None,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
        limit_weight: float = 100.0,
        smoothness_weight: float = 0.2
    ) -> onp.ndarray:
        """
        Solve inverse kinematics for a desired end-effector pose.

        Args:
            target_position: Target position as [x, y, z] in meters.
            target_wxyz: Target orientation as a quaternion in wxyz format.
            target_link_name: Name of the link to be controlled.
            initial_guess: Optional initial guess for joint angles(rad).
            pos_weight: Weight for position error.
            ori_weight: Weight for orientation error.
            limit_weight: Weight for joint limit constraint.
            smoothness_weight: Weight for smoothness constraint.

        Returns:
            Joint angles(rad) that achieve the desired pose.
        """
        assert target_position.shape == (3,) and target_wxyz.shape == (4,)

        if target_link_name is None:
            target_link_name = self.target_link_name
            if target_link_name is None:
                raise ValueError("target_link_name must be specified")
        target_link_index = self.robot.links.names.index(target_link_name)

        cfg = self._solve_ik_jax(
            self.robot,
            jnp.array(target_position),
            jnp.array(target_wxyz),
            jnp.array(target_link_index),
            initial_guess,
            pos_weight,
            ori_weight,
            limit_weight,
            smoothness_weight,
            termination=jaxls.TerminationConfig(max_iterations=self.ik_max_iters)
        )
        assert cfg.shape == (self.robot.joints.num_actuated_joints,)
        return onp.array(cfg)

    @staticmethod
    @jdc.jit
    def _solve_ik_jax(
        robot: pk.Robot,
        target_position: jax.Array,
        target_wxyz: jax.Array,
        target_link_index: jax.Array,
        initial_guess: onp.ndarray,
        pos_weight: float,
        ori_weight: float,
        limit_weight: float,
        smoothness_weight: float,
        termination: jaxls.TerminationConfig = jaxls.TerminationConfig()
    ) -> jax.Array:
        """JAX-accelerated internal method for inverse kinematics optimization."""
        joint_var = robot.joint_var_cls(0)
        target_rotation = jaxlie.SO3(target_wxyz)
        target_transform = jaxlie.SE3.from_rotation_and_translation(target_rotation, target_position)

        factors = [
            pk.costs.pose_cost_analytic_jac(
                robot, joint_var, target_transform, target_link_index,
                pos_weight=pos_weight, ori_weight=ori_weight
            ),
            pk.costs.limit_cost(robot, joint_var, weight=limit_weight),
            pk.costs.smoothness_cost(
                robot.joint_var_cls(0), robot.joint_var_cls(1),
                jnp.array([smoothness_weight])[None]
            ),
        ]

        if initial_guess is not None:
            initial_guess = jaxls.VarValues.make([joint_var.with_value(jnp.array(initial_guess))])

        sol = (
            jaxls.LeastSquaresProblem(factors, [joint_var])
            .analyze()
            .solve(
                initial_vals=initial_guess,
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                termination=termination
            )
        )

        return sol[joint_var]

# Test Demo
if __name__ == "__main__":
    try:
        urdf_path = r"piper/piper.urdf"
        solver = Kinematic(urdf_path=urdf_path, target_link_name="link6", ik_max_iters=100)

        joint_angles = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fk_solution = solver.solve_fk(joint_angles)
        print(f"FK solution: {fk_solution.tolist()}")

        import time
        from scipy.spatial.transform import Rotation as R

        while True:
            t1 = time.time()
            target_position = onp.array([0.0561278, 0.0, 0.213266])
            target_orientation = R.from_euler(
                'xyz', [0.0, 85.008, 0.0], degrees=True
            ).as_quat()
            target_orientation = onp.array([target_orientation[3], target_orientation[0], target_orientation[1], target_orientation[2]])
            
            ik_solution = solver.solve_ik(target_position, target_orientation, "link6")
            t2 = time.time()
            print(f"Computation time: {(t2 - t1) * 1000:.4f} ms")

    except Exception as e:
        print(f"Error: {e}")