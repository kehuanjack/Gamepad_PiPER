"""
Trac-IK based Kinematic Solver for Robot Arm Control.

This module provides a forward and inverse kinematics solver using Trac-IK library,
designed to be used with the teleoperation system.
"""

import numpy as np
from trac_ik import TracIK
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R


class Kinematic:
    """
    A kinematic solver using Trac-IK for forward and inverse kinematics.
    
    This class provides methods to compute forward kinematics (FK) for determining 
    end-effector pose from joint angles, and inverse kinematics (IK) for finding 
    joint angles that achieve a desired end-effector pose.
    """
    
    def __init__(
        self, 
        urdf_path: str,
        base_link_name: str = "base_link",
        target_link_name: str = "link6",
        timeout: float = 0.005,
        epsilon: float = 0.00001,
        solver_type: str = "Speed"
    ):
        """
        Initialize the Trac-IK kinematic solver.

        Args:
            urdf_path: Path to the URDF file describing the robot.
            base_link_name: Starting link of the kinematic chain.
            target_link_name: End effector link of the kinematic chain.
            timeout: Timeout in seconds for IK calls.
            epsilon: Error epsilon for IK solutions.
            solver_type: Type of solver ('Speed', 'Distance', 'Manip1', 'Manip2')
        """
        # Initialize Trac-IK solver
        self.ik_solver = TracIK(
            base_link_name=base_link_name,
            tip_link_name=target_link_name,
            urdf_path=urdf_path,
            timeout=timeout,
            epsilon=epsilon,
            solver_type=solver_type
        )
        
        # Store configuration
        self.base_link = base_link_name
        self.tip_link = target_link_name
        self.urdf_path = urdf_path
        
        # Get joint limits
        self.lower_limits, self.upper_limits = self.ik_solver.joint_limits
        self.joint_limits = list(zip(self.lower_limits.tolist(), self.upper_limits.tolist()))
        
        # Get number of joints
        self.num_joints = len(self.lower_limits)
        
        # Initialize joint zero position and previous solution
        self.joints_zero = np.zeros(self.num_joints)
        self.previous_solution = self.joints_zero.copy()
        
        # Compute zero pose for reference
        self.xyz_wxyz_zero = self.solve_fk(self.joints_zero)
        
        print(f"Trac-IK Kinematic Solver initialized:")
        print(f"  Base link: {base_link_name}")
        print(f"  Tip link: {target_link_name}")
        print(f"  Number of joints: {self.num_joints}")
        print(f"  Joint limits: {self.joint_limits}")

    def solve_fk(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for the robot's end effector.

        Args:
            joint_angles: Array of joint angles in radians.
                        Shape: (num_joints,)

        Returns:
            Array containing position and orientation in [x, y, z, qw, qx, qy, qz] format.
        """
        assert joint_angles.shape == (self.num_joints,), \
            f"Expected joint_angles shape ({self.num_joints},), got {joint_angles.shape}"
        
        # Compute forward kinematics using Trac-IK
        position, rotation_matrix = self.ik_solver.fk(joint_angles)
        
        # Convert rotation matrix to quaternion (wxyz format)
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        
        # Convert to [w, x, y, z] format
        wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        # Return [x, y, z, w, x, y, z]
        return np.concatenate([position, wxyz])

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        use_previous_solution: bool = True
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for a desired end-effector pose.

        Args:
            target_position: Target position as [x, y, z] in meters.
            target_wxyz: Target orientation as a quaternion in wxyz format.
            initial_guess: Optional initial guess for joint angles (radians).
            use_previous_solution: Whether to use previous solution as seed if no initial_guess provided.

        Returns:
            Joint angles (radians) that achieve the desired pose, or None if no solution found.
        """
        assert target_position.shape == (3,), f"Expected target_position shape (3,), got {target_position.shape}"
        assert target_wxyz.shape == (4,), f"Expected target_wxyz shape (4,), got {target_wxyz.shape}"
        
        # Prepare seed joint values
        if initial_guess is not None:
            seed_joints = initial_guess
        elif use_previous_solution:
            seed_joints = self.previous_solution
        else:
            seed_joints = self.joints_zero
        
        # Convert quaternion (wxyz) to rotation matrix
        # Trac-IK expects quaternion in [x, y, z, w] format
        xyzw = np.array([target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]])
        rotation = R.from_quat(xyzw)
        target_rotmat = rotation.as_matrix()
        
        # Solve inverse kinematics
        result = self.ik_solver.ik(target_position, target_rotmat, seed_jnt_values=seed_joints)
        
        # Update previous solution if successful
        if result is not None:
            self.previous_solution = result.copy()
            result = result.tolist()
            
        return result

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint lower and upper limits."""
        return self.lower_limits.copy(), self.upper_limits.copy()

    def get_zero_pose(self) -> np.ndarray:
        """Get the end-effector pose at zero joint angles."""
        return self.xyz_wxyz_zero.copy()

    def check_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """Check if joint angles are within limits."""
        return np.all(joint_angles >= self.lower_limits) and np.all(joint_angles <= self.upper_limits)


# Test Demo
if __name__ == "__main__":
    try:
        # Test the Trac-IK kinematic solver
        urdf_path = "piper/piper.urdf"
        solver = Kinematic(urdf_path=urdf_path, tip_link_name="link6")
        
        # Test forward kinematics
        joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fk_solution = solver.solve_fk(joint_angles)
        print(f"FK solution: {fk_solution.tolist()}")
        
        # Test inverse kinematics
        import time
        from scipy.spatial.transform import Rotation as R
        
        target_position = np.array([0.1, 0.0, 0.27])
        target_orientation = R.from_euler('xyz', [0, 85, 0], degrees=True).as_quat(scalar_first=True)
        
        t1 = time.time()
        ik_solution = solver.solve_ik(target_position, target_orientation)
        t2 = time.time()
        
        if ik_solution is not None:
            print(f"IK solution: {ik_solution.tolist()}")
            print(f"Computation time: {(t2 - t1) * 1000:.4f} ms")
            
            # Verify solution
            verified_pose = solver.solve_fk(ik_solution)
            print(f"Verified pose: {verified_pose.tolist()}")
        else:
            print("No IK solution found")
            
    except Exception as e:
        print(f"Error: {e}")