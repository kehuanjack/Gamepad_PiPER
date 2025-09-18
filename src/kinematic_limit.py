"""
Robot Forward and Inverse Kinematics Solver using PyRoKi.

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

        # Precompile JAX functions for better performance
        self._compile_jax_functions()

        # Calculate FK and IK at zero point
        self.xyz_wxyz_zero = self.solve_fk(self.joints_zero)
        self.solve_ik(self.xyz_wxyz_zero[:3], self.xyz_wxyz_zero[3:])

    def _compile_jax_functions(self):
        """Precompile JAX functions for better performance"""
        self._solve_ik_jax_compiled = jax.jit(
            lambda target_pos, target_ori, link_idx, initial_guess: self._solve_ik_jax(
                self.robot, target_pos, target_ori, link_idx, initial_guess,
                60.0, 40.0, 100.0, 0.0, jaxls.TerminationConfig(max_iterations=self.ik_max_iters)
            )
        )

    def solve_fk(self, joint_angles: onp.ndarray, target_link_name: str = None) -> onp.ndarray:
        """
        Calculate forward kinematics for a specific link of the robot.

        Args:
            joint_angles: Array of joint angles (radians).
                         Shape: (robot.joints.actuated_count,)
            target_link_name: Link name for FK calculation.

        Returns:
            Array containing position and orientation in format [x, y, z, qw, qx, qy, qz].
        """
        assert joint_angles.shape == (self.num_joints,)

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
        pos_tolerance: float = 0.0011,
        ori_tolerance: float = 0.01,
        joint_change_weight: float = 0.5,
        use_multiple_initial_guesses: bool = True,
        num_initial_guesses: int = 10
    ) -> onp.ndarray:
        """
        Solve inverse kinematics for desired end effector pose.

        Args:
            target_position: Target position [x, y, z] (meters).
            target_wxyz: Target orientation, quaternion, wxyz format.
            target_link_name: Link name to control.
            initial_guess: Optional initial guess for joint angles (radians).
            pos_tolerance: Maximum allowed position error (meters).
            ori_tolerance: Maximum allowed orientation error (radians).
            joint_change_weight: Weight to penalize large joint changes.
            use_multiple_initial_guesses: Whether to use multiple initial guesses.
            num_initial_guesses: Number of different initial guesses to try.

        Returns:
            Joint angles (radians) to achieve desired pose, or None if no solution found.
        """
        assert target_position.shape == (3,) and target_wxyz.shape == (4,)

        if target_link_name is None:
            target_link_name = self.target_link_name
            if target_link_name is None:
                raise ValueError("target_link_name must be specified")
        target_link_index = self.robot.links.names.index(target_link_name)

        # Generate multiple initial guesses
        if use_multiple_initial_guesses:
            initial_guesses = self._generate_initial_guesses(
                initial_guess, num_initial_guesses
            )
        else:
            initial_guesses = [initial_guess if initial_guess is not None else self.previous_solution]

        # Convert to JAX array for parallel processing
        jax_target_position = jnp.array(target_position)
        jax_target_wxyz = jnp.array(target_wxyz)
        jax_target_link_index = jnp.array(target_link_index)
        jax_initial_guesses = jnp.array(initial_guesses)

        # Use JAX's parallel processing to solve IK for all initial guesses
        try:
            # Use JAX's vmap to solve IK in parallel
            solutions = self._solve_ik_parallel(
                jax_target_position,
                jax_target_wxyz,
                jax_target_link_index,
                jax_initial_guesses,
            )

            # Convert back to numpy for evaluation
            solutions_np = onp.array(solutions)

            # Evaluate all solutions
            evaluation_results = self._evaluate_solutions(
                solutions_np, target_position, target_wxyz, target_link_name
            )

            # Find the best solution
            best_solution = self._select_best_solution(
                solutions_np, evaluation_results,
                pos_tolerance, ori_tolerance, joint_change_weight
            )

            if best_solution is not None:
                self.previous_solution = best_solution.copy()
                return best_solution
            else:
                return None

        except Exception as e:
            print(f"IK parallel solving failed: {e}")
            # If parallel fails, fall back to sequential processing
            return self._solve_ik_sequential(
                target_position, target_wxyz, target_link_index,
                initial_guesses, pos_tolerance, ori_tolerance, joint_change_weight
            )

    def _solve_ik_parallel(
        self,
        target_position: jax.Array,
        target_wxyz: jax.Array,
        target_link_index: jax.Array,
        initial_guesses: jax.Array,
    ) -> jax.Array:
        """Use JAX to solve IK for multiple initial guesses in parallel"""
        batched_ik_solver = jax.vmap(
            lambda initial_guess: self._solve_ik_jax_compiled(
                target_position, target_wxyz, target_link_index, initial_guess
            ),
            in_axes=0, out_axes=0
        )

        return batched_ik_solver(initial_guesses)

    def _evaluate_solutions(
        self,
        solutions: onp.ndarray,
        target_position: onp.ndarray,
        target_wxyz: onp.ndarray,
        target_link_name: str
    ) -> dict:
        """Evaluate the quality of multiple solutions"""
        results = {
            'pos_errors': [],
            'ori_errors': [],
            'joint_changes': []
        }

        for i, solution in enumerate(solutions):
            try:
                # Calculate forward kinematics
                fk_result = self.solve_fk(solution, target_link_name)
                achieved_pos = fk_result[0:3]
                achieved_ori = fk_result[3:7]

                # Calculate position error
                pos_error = onp.linalg.norm(achieved_pos - target_position)

                # Calculate orientation error (angle between quaternions)
                dot_product = onp.clip(onp.dot(achieved_ori, target_wxyz), -1.0, 1.0)
                ori_error = 2 * onp.arccos(onp.abs(dot_product))

                # Calculate joint change
                joint_change = onp.linalg.norm(solution[:6] - self.previous_solution[:6])

                results['pos_errors'].append(pos_error)
                results['ori_errors'].append(ori_error)
                results['joint_changes'].append(joint_change)

                # print(f"========sol: {i}========")
                # print(f"sol: {solution[:6].tolist()}")
                # print(f"pos: {pos_error}, ori: {ori_error}, jc: {joint_change}")

            except Exception as e:
                # If evaluation fails, use large error value
                results['pos_errors'].append(float('inf'))
                results['ori_errors'].append(float('inf'))
                results['joint_changes'].append(float('inf'))

        # Convert to numpy array
        for key in results:
            results[key] = onp.array(results[key])

        return results

    def _select_best_solution(
        self,
        solutions: onp.ndarray,
        evaluation_results: dict,
        pos_tolerance: float,
        ori_tolerance: float,
        joint_change_weight: float
    ) -> onp.ndarray:
        """Select the best solution based on error metrics"""
        pos_errors = evaluation_results['pos_errors']
        ori_errors = evaluation_results['ori_errors']
        joint_changes = evaluation_results['joint_changes']

        # Find solutions that meet the tolerance
        valid_mask = (pos_errors <= pos_tolerance) & (ori_errors <= ori_tolerance)
        valid_indices = onp.where(valid_mask)[0]

        if len(valid_indices) > 0:
            # Calculate scores for valid solutions
            scores = (
                pos_errors[valid_indices] +
                ori_errors[valid_indices] +
                joint_change_weight * joint_changes[valid_indices]
            )

            # Select the solution with the best score
            best_index = valid_indices[onp.argmin(scores)]

            # for i in valid_indices:
            #     print(f"========sol: {i}========")
            #     print(f"sol: {onp.degrees(solutions[i][:6]).tolist()}")
            #     print(f"pos: {pos_errors[i]}, ori: {onp.degrees(ori_errors[i])}, jc: {onp.degrees(joint_changes[i])}")
            # print(f"best_index: {best_index}\n")
            return solutions[best_index]
        else:
            return None

    def _solve_ik_sequential(
        self,
        target_position: onp.ndarray,
        target_wxyz: onp.ndarray,
        target_link_index: int,
        initial_guesses: list,
        pos_tolerance: float,
        ori_tolerance: float,
        joint_change_weight: float
    ) -> onp.ndarray:
        """Fallback method for sequential IK solving"""
        best_solution = None
        best_score = float('inf')

        for guess in initial_guesses:
            try:
                cfg = self._solve_ik_jax_compiled(
                    jnp.array(target_position),
                    jnp.array(target_wxyz),
                    jnp.array(target_link_index),
                    jnp.array(guess)
                )

                # Evaluate solution
                fk_result = self.solve_fk(cfg)
                achieved_pos = fk_result[0:3]
                achieved_ori = fk_result[3:7]

                # Calculate error
                pos_error = onp.linalg.norm(achieved_pos - target_position)
                dot_product = onp.clip(onp.dot(achieved_ori, target_wxyz), -1.0, 1.0)
                ori_error = 2 * onp.arccos(onp.abs(dot_product))

                # Calculate joint change
                joint_change = onp.linalg.norm(cfg[:6] - self.previous_solution[:6])

                # Calculate score
                score = pos_error + ori_error + joint_change_weight * joint_change

                if pos_error <= pos_tolerance and ori_error <= ori_tolerance and score < best_score:
                    best_score = score
                    best_solution = cfg

            except Exception as e:
                continue

        return best_solution

    def _generate_initial_guesses(self, user_guess, num_guesses):
        """Generate multiple initial guesses for IK solver"""
        guesses = []
        added_guesses_set = set()  # Used to track added guesses and avoid duplication

        # Define a function to check if a guess already exists
        def is_duplicate(guess, existing_set, tolerance=1e-5):
            """Check if the guess duplicates existing guesses within tolerance"""
            for existing in existing_set:
                if onp.allclose(guess, existing, atol=tolerance):
                    return True
            return False

        # 1. Add zero position
        if not is_duplicate(self.joints_zero, added_guesses_set):
            guesses.append(self.joints_zero.copy())
            added_guesses_set.add(tuple(self.joints_zero))
            if len(guesses) == num_guesses:
                return guesses

        # 2. Add user-provided guess (if any)
        if user_guess is not None and not is_duplicate(user_guess, added_guesses_set):
            guesses.append(user_guess)
            added_guesses_set.add(tuple(user_guess))  # Use tuple to store in set
            if len(guesses) == num_guesses:
                return guesses

        # 3. Add previous solution
        if not is_duplicate(self.previous_solution, added_guesses_set):
            guesses.append(self.previous_solution.copy())
            added_guesses_set.add(tuple(self.previous_solution))
            if len(guesses) == num_guesses:
                return guesses

        # 4. Add a set of all joint minimum values
        # min_guess = onp.array([limit[0] for limit in self.joint_limits])
        # if not is_duplicate(min_guess, added_guesses_set):
        #     guesses.append(min_guess)
        #     added_guesses_set.add(tuple(min_guess))

        # 5. Add a set of all joint maximum values
        # max_guess = onp.array([limit[1] for limit in self.joint_limits])
        # if not is_duplicate(max_guess, added_guesses_set):
        #     guesses.append(max_guess)
        #     added_guesses_set.add(tuple(max_guess))

        # 6. Add intermediate value (average of min and max)
        mid_guess = onp.array([(limit[0] + limit[1]) / 2 for limit in self.joint_limits])
        if not is_duplicate(mid_guess, added_guesses_set):
            guesses.append(mid_guess)
            added_guesses_set.add(tuple(mid_guess))
            if len(guesses) == num_guesses:
                return guesses

        # 7. Add random guesses to fill remaining positions
        remaining_slots = num_guesses - len(guesses)
        if remaining_slots > 0:
            for _ in range(remaining_slots):
                # Generate random guesses until a non-duplicate is found
                max_attempts = 100  # Prevent infinite loop
                for attempt in range(max_attempts):
                    random_guess = onp.zeros_like(self.previous_solution)
                    for i, (lower, upper) in enumerate(self.joint_limits):
                        random_guess[i] = onp.random.uniform(lower, upper)

                    if not is_duplicate(random_guess, added_guesses_set):
                        guesses.append(random_guess)
                        added_guesses_set.add(tuple(random_guess))
                        break
                    elif attempt == max_attempts - 1:
                        # If unable to generate a non-duplicate guess, add a slightly perturbed version
                        perturbation = onp.random.normal(0, 0.01, size=random_guess.shape)
                        perturbed_guess = random_guess + perturbation
                        # Ensure perturbed guess is still within joint limits
                        for i, (lower, upper) in enumerate(self.joint_limits):
                            perturbed_guess[i] = onp.clip(perturbed_guess[i], lower, upper)

                        if not is_duplicate(perturbed_guess, added_guesses_set):
                            guesses.append(perturbed_guess)
                            added_guesses_set.add(tuple(perturbed_guess))
                        else:
                            # If still duplicate, add directly
                            guesses.append(random_guess)
                            added_guesses_set.add(tuple(random_guess))

        return guesses[:num_guesses]  # Ensure not to exceed the requested number

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
        """JAX-accelerated inverse kinematics optimization internal method"""
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
        solver = Kinematic(urdf_path=urdf_path, target_link_name="link6")

        joint_angles = onp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fk_solution = solver.solve_fk(joint_angles)
        print(f"FK solution: {fk_solution.tolist()}")

        import time
        from scipy.spatial.transform import Rotation as R

        while True:
            t1 = time.time()
            target_position = onp.array([0.2561278, 0.0, 0.613266])
            target_orientation = R.from_euler(
                'xyz', [0.0, 85.008, 0.0], degrees=True
            ).as_quat()
            target_orientation = onp.array([target_orientation[3], target_orientation[0], target_orientation[1], target_orientation[2]])

            ik_solution = solver.solve_ik(
                target_position,
                target_orientation,
                "link6"
            )

            t2 = time.time()

            if ik_solution is not None:
                print(f"Calculation time: {(t2 - t1) * 1000:.4f} ms")
            else:
                print("No IK solution found")

    except Exception as e:
        print(f"Error: {e}")