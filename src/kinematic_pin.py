"""
Robot Forward and Inverse Kinematics Solver using Pinocchio.

To use this module, you need to install the following dependencies:
    - conda create -n test_pinocchio python=3.9.*
    - conda activate test_pinocchio
    - conda install pinocchio=3.6.0 -c conda-forge
    - pip install meshcat
    - pip install casadi

References:
    - Pinocchio library: https://github.com/stack-of-tasks/pinocchio
"""

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from scipy.spatial.transform import Rotation as R


class Arm_IK:
    """Inverse kinematics solver for robotic arm using Pinocchio and CasADi."""
    
    def __init__(self, urdf_path: str = None):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
    
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.model = self.reduced_robot.model
        
        q = pin.Quaternion(1, 0, 0, 0)
        self.model.addFrame(
            pin.Frame('ee',
                      self.model.getJointId('joint6'),
                      pin.SE3(q, np.array([0.0, 0.0, 0.0])),
                      pin.FrameType.OP_FRAME)
        )

        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("ee")

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        self.joint_limits = list(zip(self.reduced_robot.model.lowerPositionLimit.tolist(), self.reduced_robot.model.upperPositionLimit.tolist()))

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        """Calculate inverse kinematics for target pose."""
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            return self.opti.debug.value(self.var_q), '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        """Check for self-collision in the robot configuration."""
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return collision

    def get_fk(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate forward kinematics for given joint angles.

        Args:
            q: joint angles(rad) of the arm

        Returns:
            xyz_rpy: xyz(m), rpy(rad)
        """
        pin.framesForwardKinematics(self.model, self.data, q)
        frame = self.data.oMf[self.ee_frame_id]
        return np.concatenate([frame.translation, pin.rpy.matrixToRpy(frame.rotation)])

    def get_ik_solution(self, x, y, z, roll, pitch, yaw) -> np.ndarray:
        """Get inverse kinematics solution for target position and orientation."""
        q = R.from_euler('xyz', (roll, pitch, yaw)).as_quat()
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous,0)
        
        if get_result:
            return sol_q
        else:
            return None

# Test Demo
if __name__ == "__main__":
    ik = Arm_IK("piper/piper.urdf")
    sol_q = np.zeros(ik.model.nq)
    xyz_rpy = ik.get_fk(sol_q)
    print(xyz_rpy)
    xyz_rpy[0] = 1.0
    sol_q = ik.get_ik_solution(*xyz_rpy)
    print(sol_q)

    xyz_rpy = ik.get_fk(sol_q)
    print(xyz_rpy)