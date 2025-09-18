from .gamepad_base import GamepadBase, np, R
from .kinematic_pin import Arm_IK


class RoboticArmController(GamepadBase):
    """Robotic arm controller using Pinocchio kinematics."""
    
    def __init__(self, urdf_path: str = None, mesh_path: str = None, root_name: str = None, target_link_name: str = None):
        super().__init__(urdf_path, mesh_path, root_name)
        
        # Create kinematic object
        self.kinematic = Arm_IK(urdf_path)
        self.joint_limits = self.kinematic.joint_limits.copy()
        self._joint_to_pose()

    def _joint_to_pose(self):
        """Joint angle to end pose"""
        joints = np.zeros(self.kinematic.model.nq)
        joints[:6] = self.joint_angles
        self.xyz_rpy = self.kinematic.get_fk(joints)
        self.xyz_wxyz = np.concatenate((self.xyz_rpy[:3], self._xyzw_to_wxyz(R.from_euler('xyz', self.xyz_rpy[3:]).as_quat())))

    def _pose_to_joint(self, xyz, orientation):
        """End pose to joint angle"""
        euler = orientation.as_euler('xyz')
        result = self.kinematic.get_ik_solution(xyz[0], xyz[1], xyz[2], euler[0], euler[1], euler[2])
        if result is not None:
            self.joint_angles = result[:6]
            self.xyz_rpy = np.concatenate((xyz, euler))
            self.xyz_wxyz = np.concatenate((xyz, self._xyzw_to_wxyz(orientation.as_quat())))