from .gamepad_base import GamepadBase, np, R
from .kinematic_curobo import Kinematic


class RoboticArmController(GamepadBase):
    """Robotic arm controller using cuRobo kinematics."""
    
    def __init__(self, urdf_path: str = None, mesh_path: str = None, root_name: str = None, target_link_name: str = None):
        super().__init__(urdf_path, mesh_path, root_name)
        
        # Create kinematic object
        self.kinematic = Kinematic(urdf_path, target_link_name=target_link_name)
        self.joint_limits = self.kinematic.joint_limits.copy()
        self._joint_to_pose()

    def _joint_to_pose(self):
        """Joint angle to end pose"""
        joints = self.joint_angles.tolist() if hasattr(self.joint_angles, 'tolist') else self.joint_angles
        self.xyz_wxyz = self.kinematic.solve_fk(joints)
        self.xyz_rpy = np.concatenate((
            self.xyz_wxyz[0:3],
            R.from_quat(self._wxyz_to_xyzw(self.xyz_wxyz[3:])).as_euler('xyz', degrees=True)
        ))

    def _pose_to_joint(self, xyz, orientation):
        """End pose to joint angle"""
        wxyz = self._xyzw_to_wxyz(orientation.as_quat())
        result = self.kinematic.solve_ik(xyz, wxyz)
        if result is not None:
            joint_angles = np.array(result[:6])
            # 计算关节1角度变化量
            if np.abs(joint_angles[0] - self.joint_angles[0]) < np.radians(170):
                self.joint_angles = joint_angles
                self.xyz_wxyz = np.concatenate((xyz, wxyz))
                self.xyz_rpy = np.concatenate((self.xyz_wxyz[0:3], orientation.as_euler('xyz', degrees=True)))
            else:
                print("Warning: IK solution has large jump, ignoring.")