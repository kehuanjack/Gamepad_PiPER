from .gamepad_base import GamepadBase, np, R
from .kinematic_trac_ik import Kinematic


class RoboticArmController(GamepadBase):
    """Robotic arm controller using Trac-IK kinematics."""

    def __init__(self, urdf_path: str = None, mesh_path: str = None, root_name: str = None, 
                 base_link_name: str = "base_link", target_link_name: str = "link6"):
        """
        Initialize robotic arm controller with Trac-IK kinematics.
        
        Args:
            urdf_path: Path to the URDF file
            mesh_path: Path to mesh files directory
            root_name: Root node name for visualization
            base_link_name: Base link name for kinematic chain
            target_link_name: Target end-effector link name
        """
        super().__init__(urdf_path, mesh_path, root_name)
        
        # Create kinematic object using Trac-IK
        self.kinematic = Kinematic(
            urdf_path=urdf_path,
            base_link_name=base_link_name,
            target_link_name=target_link_name,
            timeout=0.005,
            epsilon=0.00001,
            solver_type="Speed"
        )
        
        # Set joint limits from kinematic solver
        self.joint_limits = self.kinematic.joint_limits.copy()
        
        # Initialize joint angles and pose
        self._joint_to_pose()

    def _joint_to_pose(self):
        """Convert joint angles to end-effector pose using Trac-IK forward kinematics"""
        try:
            # Get current joint angles (first 6 joints for 6-DOF arm)
            joints = self.joint_angles.copy()
            
            # Solve forward kinematics
            self.xyz_wxyz = self.kinematic.solve_fk(joints)
            
            # Convert quaternion to RPY for display
            quat_xyzw = [self.xyz_wxyz[4], self.xyz_wxyz[5], self.xyz_wxyz[6], self.xyz_wxyz[3]]  # Convert to xyzw
            self.xyz_rpy = np.concatenate((
                self.xyz_wxyz[0:3], 
                R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
            ))
            
        except Exception as e:
            print(f"Error in forward kinematics: {e}")
            # Fallback to default pose
            self.xyz_wxyz = np.zeros(7)
            self.xyz_rpy = np.zeros(6)

    def _pose_to_joint(self, xyz, orientation):
        """Convert end-effector pose to joint angles using Trac-IK inverse kinematics"""
        try:
            # Convert orientation to wxyz quaternion format
            quat_xyzw = orientation.as_quat()  # [x, y, z, w]
            target_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to wxyz
            
            # Solve inverse kinematics
            ik_solution = self.kinematic.solve_ik(
                target_position=xyz,
                target_wxyz=target_wxyz,
                initial_guess=self.joint_angles,  # Use current joint angles as initial guess
                use_previous_solution=True
            )
            
            if ik_solution is not None:
                # Update joint angles with IK solution
                self.joint_angles = np.array(ik_solution)
                
                # Update pose for consistency
                self.xyz_wxyz = np.concatenate((xyz, target_wxyz))
                self.xyz_rpy = np.concatenate((xyz, orientation.as_euler('xyz', degrees=True)))
            else:
                print("IK solution not found - maintaining current joint configuration")
                
        except Exception as e:
            print(f"Error in inverse kinematics: {e}")