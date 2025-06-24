"""
NMPC controller implementation based on do-mpc - enhanced version supports Simulator
Use do-mpc library to simplify the development of NMPC controllers, providing a more stable and efficient implementation
"""

# pylint: disable=invalid-name
# cspell:ignore dompc vertcat radau mterm lterm rterm ndarray fmin fmax idas abstol reltol
# cspell: ignore nlpsol ipopt print_level max_iter acceptable_tol acceptable_obj_change_tol tol
# cspell: ignore cvodes mu_strategy hessian_approximation limited_memory_max_history alpha_for_y recalc_y max_wall_time print_time

import numpy as np
import casadi as ca
import do_mpc

from config import parameters as params
from airship.model import AirshipCasADiSymbolic
from airship.thrust import thrust_params_to_force_torque
from airship.observer import NMPCDisturbanceObserver


class DoMPCAirshipController:
    """
    Airship NMPC controller based on do-mpc - enhanced version

    Advantages:
    1. Simplified MPC settings and configuration
    2. Built-in solver configuration and optimization
    3. Better numerical stability
    4. Automatic handling of constraints and boundaries
    5. Built-in visualization and analysis tools
    6. Support do-mpc Simulator integration
    """

    def __init__(self, use_disturbance_compensation=True, create_simulator=True):
        """
        Initialize the controller based on do-mpc

        Args:
            use_disturbance_compensation: Whether to enable disturbance compensation
            create_simulator: Whether to create do-mpc Simulator
        """
        self.use_disturbance_compensation = use_disturbance_compensation
        self.params = params
        self.create_simulator = create_simulator

        # Initialize the disturbance observer
        if self.use_disturbance_compensation:
            self.disturbance_observer = NMPCDisturbanceObserver()
            self.disturbance_compensation_factor = getattr(params, 'do_compensation_gain', 0.9)
            self.last_disturbance_estimate = np.zeros(6)

        # Create the do-mpc model
        self.model = self._create_model()

        # Create the MPC controller
        self.mpc = self._create_mpc_controller()

        # Create the estimator
        self.estimator = self._create_estimator()

        # Create the Simulator (if needed)
        if self.create_simulator:
            self.simulator = self._create_simulator()
        else:
            self.simulator = None

        # Initialize the controller
        self._setup_initial_conditions()

        # Store the last control input
        self.last_control = np.array([5.0, 0.0, 0.0])

    def _create_model(self):
        """
        Create do-mpc model - enhanced version

        Returns:
            do_mpc.model.Model: Airship dynamics model
        """
        # Create the model type (continuous time)
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # Define the state variables - 12-dimensional state vector
        pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))      # Position [x, y, z]
        att = model.set_variable(var_type='_x', var_name='att', shape=(3, 1))      # Attitude [phi, theta, psi]
        vel = model.set_variable(var_type='_x', var_name='vel', shape=(3, 1))      # Linear velocity [u, v, w]
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(3, 1))  # Angular velocity [p, q, r]

        # Define the control input - 3-dimensional control vector
        T = model.set_variable(var_type='_u', var_name='T')          # Thrust magnitude
        mu = model.set_variable(var_type='_u', var_name='mu')        # Horizontal deflection angle
        nu = model.set_variable(var_type='_u', var_name='nu')        # Vertical deflection angle

        # Define the reference trajectory parameters
        pos_ref = model.set_variable(var_type='_p', var_name='pos_ref', shape=(3, 1))
        att_ref = model.set_variable(var_type='_p', var_name='att_ref', shape=(3, 1))
        vel_ref = model.set_variable(var_type='_p', var_name='vel_ref', shape=(3, 1))
        omega_ref = model.set_variable(var_type='_p', var_name='omega_ref', shape=(3, 1))

        # Define the disturbance variable (for Simulator)
        disturbance = model.set_variable(var_type='_p', var_name='disturbance', shape=(6, 1))

        # Use the existing symbolic dynamic model
        symbolic_model = AirshipCasADiSymbolic(self.params)

        # Combine the state vector
        X_state = ca.vertcat(pos, att, vel, omega)
        U_control = ca.vertcat(T, mu, nu)

        # Get the dynamic equations (including disturbance)
        X_dot = symbolic_model.rhs_symbolic(X_state, U_control, external_disturbance=disturbance)

        # Add numerical stability - limit the size of the derivative
        max_derivative = 1e5
        X_dot = ca.fmin(ca.fmax(X_dot, -max_derivative), max_derivative)

        # Decompose the state derivative
        pos_dot = X_dot[0:3]
        att_dot = X_dot[3:6]
        vel_dot = X_dot[6:9]
        omega_dot = X_dot[9:12]

        # Set the differential equation
        model.set_rhs('pos', pos_dot)
        model.set_rhs('att', att_dot)
        model.set_rhs('vel', vel_dot)
        model.set_rhs('omega', omega_dot)

        # Set the state expression (for the objective function)
        position_error = pos - pos_ref
        attitude_error = att - att_ref
        velocity_error = vel - vel_ref
        angular_error = omega - omega_ref

        # Auxiliary expression
        model.set_expression('pos_error', position_error)
        model.set_expression('att_error', attitude_error)
        model.set_expression('vel_error', velocity_error)
        model.set_expression('ang_error', angular_error)

        # Add the measurement output (for the estimator)
        model.set_expression('y_meas', X_state)  # Assume the state is completely measurable

        # Complete the model setup
        model.setup()

        return model

    def _create_mpc_controller(self):
        """
        Create MPC controller - enhanced version

        Returns:
            do_mpc.controller.MPC: Configured MPC controller
        """
        mpc = do_mpc.controller.MPC(self.model)

        # MPC settings
        setup_mpc = {
            'n_horizon': min(params.N_HORIZON, 8), # Prediction horizon length (how many steps the controller predicts into the future)
            'n_robust': 1, # Robust control (for handling model uncertainty)
            'open_loop': 0, # Whether to enable open-loop control (not considering current state)
            't_step': params.DT, # Time step (time interval for each step)
            'state_discretization': 'collocation',  # State discretization method
            'collocation_type': 'radau',  # Use radau, more stable
            'collocation_deg': 2,
            'collocation_ni': 1,  # Reduce number of internal points
            'store_full_solution': False, #True,
            # Optimized solver options
            'nlpsol_opts': {
                'ipopt.print_level': 0,
                'ipopt.max_iter': 50,  # Reduce maximum iterations
                'ipopt.acceptable_tol': 1e-3,  # Relax tolerance
                'ipopt.acceptable_obj_change_tol': 1e-3,
                'ipopt.tol': 1e-3,
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.hessian_approximation': 'limited-memory',
                'ipopt.limited_memory_max_history': 5,  # Reduce history
                'ipopt.alpha_for_y': 'primal',
                'ipopt.recalc_y': 'yes',
                'ipopt.max_wall_time': 3.0,  # Further limit solving time
                'ipopt.warm_start_init_point': 'yes',  # Enable warm start
                'print_time': 0
            }
        }

        mpc.set_param(**setup_mpc)

        # Initialize default values for reference trajectory and disturbance parameters, which will be updated in the simulator according to actual reference trajectory and disturbance
        mpc.set_uncertainty_values(
            pos_ref=np.zeros((3, 1)),
            att_ref=np.zeros((3, 1)),
            vel_ref=np.zeros((3, 1)),
            omega_ref=np.zeros((3, 1)),
            disturbance=np.zeros((6, 1))  # Add disturbance parameters
        )

        # Define the objective function of the controller
        # Scale the weight matrix to avoid numerical problems
        Q_scaled = params.Q * 0.01  # Reduce the state weight
        Qf_scaled = params.Qf * 0.01
        R_scaled = params.R * 100.0  # Increase the control weight to promote smoothness

        # Terminal cost - represents the state error at the end of the prediction horizon, excluding the control input
        mterm = (self.model.aux['pos_error'].T @ Qf_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Qf_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Qf_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Qf_scaled[9:12, 9:12] @ self.model.aux['ang_error'])

        # Stage cost - represents the state error and control input cost at each step
        lterm = (self.model.aux['pos_error'].T @ Q_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Q_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Q_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Q_scaled[9:12, 9:12] @ self.model.aux['ang_error'] +
                 self.model.u['T']**2 * R_scaled[0, 0] +
                 self.model.u['mu']**2 * R_scaled[1, 1] +
                 self.model.u['nu']**2 * R_scaled[2, 2])

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Control input regularization
        mpc.set_rterm(T=0.1, mu=0.1, nu=0.1)

        # Set the constraints for the control input and state
        self._set_mpc_constraints(mpc)

        # Complete the MPC setup
        mpc.setup()

        return mpc

    def _set_mpc_constraints(self, mpc):
        """Set the MPC constraints

        Args:
            mpc: MPC controller
        """
        # Control input constraint
        mpc.bounds['lower', '_u', 'T'] = max(params.T_MIN, 0.1)  # Avoid zero thrust
        mpc.bounds['upper', '_u', 'T'] = params.T_MAX
        mpc.bounds['lower', '_u', 'mu'] = params.MU_MIN
        mpc.bounds['upper', '_u', 'mu'] = params.MU_MAX
        mpc.bounds['lower', '_u', 'nu'] = params.NU_MIN
        mpc.bounds['upper', '_u', 'nu'] = params.NU_MAX

        # State constraint
        max_position = 200.0  # Reduce to a reasonable range
        max_angle = np.pi/2   # Limit the attitude angle to prevent singularity
        max_velocity = 30.0
        max_angular_velocity = np.pi/2  # Reasonable angular velocity limit

        # Position constraint
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # Attitude constraint
        mpc.bounds['lower', '_x', 'att'] = -max_angle
        mpc.bounds['upper', '_x', 'att'] = max_angle
        mpc.bounds['lower', '_x', 'att', 1] = -max_angle/6  # Limit the pitch angle theta
        mpc.bounds['upper', '_x', 'att', 1] = max_angle/6

        # Velocity constraint
        mpc.bounds['lower', '_x', 'vel'] = -max_velocity
        mpc.bounds['upper', '_x', 'vel'] = max_velocity
        mpc.bounds['lower', '_x', 'omega'] = -max_angular_velocity
        mpc.bounds['upper', '_x', 'omega'] = max_angular_velocity

    def _create_estimator(self):
        """
        Create the state estimator
        Imagine you are driving an airship, but you cannot directly see all the states of the airship (like its speed, attitude, etc.).
        You can only get some measurements through sensors, like position and speed.
        At this time, you need a helper to infer the complete state of the airship based on these measurements and the mathematical model of the system.
        This helper is the state estimator.
        In this method, do-mpc provides a simple state feedback estimator (StateFeedback),
        it assumes that the state of the system is completely measurable (i.e., the sensors can directly measure all state variables).
        Therefore, the work of this estimator is very simple: directly use the measurements as the current state of the system.

        Returns:
            do_mpc.estimator.StateFeedback: State feedback estimator
        """
        estimator = do_mpc.estimator.StateFeedback(self.model)
        return estimator

    def _create_simulator(self):
        """
        Create do-mpc Simulator

        Returns:
            do_mpc.simulator.Simulator: Configured simulator
        """
        if not self.create_simulator:
            return None

        # Create simulator object - the simulator will calculate the state changes of the system over time based on this model
        simulator = do_mpc.simulator.Simulator(self.model)

        # Simulator settings
        setup_simulator = {
            't_step': params.DT,
            'integration_tool': 'cvodes', #'idas',  # Use IDAS integrator
            'abstol': 1e-6,
            'reltol': 1e-4,
        }

        simulator.set_param(**setup_simulator)

        # Define a disturbance function to simulate external disturbances (such as wind, airflow, etc.)
        def disturbance_func(t_now):
            """Define time-varying disturbance
            This function will return a disturbance vector based on the current time t_now
            """
            try:
                delta = params.disturbance_delta(t_now)
                return delta.reshape(-1, 1)
            except Exception: # pylint: disable=broad-exception-caught
                return np.zeros((6, 1))

        # Set parameter template - define the parameter template of the simulator, including reference trajectory and disturbance, these parameters will be dynamically updated during the simulation process
        p_template = simulator.get_p_template()
        p_template['pos_ref'] = np.zeros((3, 1))
        p_template['att_ref'] = np.zeros((3, 1))
        p_template['vel_ref'] = np.zeros((3, 1))
        p_template['omega_ref'] = np.zeros((3, 1))
        p_template['disturbance'] = disturbance_func

        def p_fun(t_now):
            """
            This function will be called at each simulation step and return the parameter values at the current time
            """
            _ = t_now
            return p_template

        simulator.set_p_fun(p_fun)

        # Complete Simulator setup
        simulator.setup()

        return simulator

    def _setup_initial_conditions(self):
        """
        Set the initial conditions and initial guesses for the controller, estimator, and simulator, so that the system can start running from a reasonable initial state
        1. Set the initial state of the airship:
            like the initial position, speed, attitude, etc.
        2. Give an initial control guess:
            like the initial thrust and deflection angle, to ensure that the airship does not lose control at the beginning.
        This method is to set these initial conditions for the flight simulator, so that the controller, estimator, and simulator know where to start.
        """
        # Set the initial state of the airship, including the initial position, attitude, speed, and angular velocity of the system
        x0 = params.X0.copy()

        # More strict numerical cleaning to ensure the numerical validity of the initial state, avoiding NaN or infinite values
        # x0 = np.nan_to_num(x0, nan=0.0, posinf=10.0, neginf=-10.0)

        # Limit the initial angle to prevent singularity
        # x0[3:6] = np.clip(x0[3:6], -np.pi/2, np.pi/2)

        # Recombine the initial state vector into the column vector format expected by do-mpc
        _x0 = np.concatenate([
            x0[0:3].reshape(-1, 1),   # pos
            x0[3:6].reshape(-1, 1),   # att
            x0[6:9].reshape(-1, 1),   # vel
            x0[9:12].reshape(-1, 1)   # omega
        ])

        # Set the initial state _x0 to the controller (mpc), estimator (estimator), and simulator (simulator).
        # So that each component knows the initial state of the system.
        self.mpc.x0 = _x0
        self.estimator.x0 = _x0
        if self.simulator is not None:
            self.simulator.x0 = _x0

        # Improved initial guess - whether the initial guess value for each prediction is the same?
        # First prediction: will use the initial guess value below
        # Second prediction and subsequent predictions:
        #       In subsequent predictions, the controller will update the initial guess value based on the result of the previous optimization.
        # Specifically:
        #      1. The initial guess value of the control input will be inherited from the result of the previous optimization.
        #      2. The initial guess value of the state will be updated based on the feedback from the simulator or the actual system.
        try:
            # Set a conservative initial control input guess
            u0 = np.array([[5.0], [0.0], [0.0]])  # Stable thrust, zero torque

            # Set the initial guess value for the entire prediction horizon
            for k in range(self.mpc.settings.n_horizon):
                self.mpc.u0[k] = u0
                self.mpc.x0[k+1] = _x0  # Keep the state stable

            self.mpc.set_initial_guess()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Set initial guess failed: {e}") # pylint: disable=line-too-long

    def step(self, current_state, reference_trajectory, t_current=0.0):
        """
        Execute one step of MPC control

        Args:
            current_state: Current state [12x1]
            reference_trajectory: Reference trajectory dictionary
            t_current: Current time

        Returns:
            control_input: Control input [T, mu, nu]
        """
        _ = t_current
        try:
            # Update current state
            current_x = np.concatenate([
                current_state[0:3].reshape(-1, 1),   # pos
                current_state[3:6].reshape(-1, 1),   # att
                current_state[6:9].reshape(-1, 1),   # vel
                current_state[9:12].reshape(-1, 1)   # omega
            ])

            # Check input validity
            if np.any(np.isnan(current_x)) or np.any(np.isinf(current_x)):
                print("Warning: The input state contains NaN or infinity value")
                return np.array([5.0, 0.0, 0.0])

            # Update reference trajectory parameters
            reference_params = {
                'pos_ref': reference_trajectory['position'].reshape(-1, 1),
                'att_ref': reference_trajectory['attitude'].reshape(-1, 1),
                'vel_ref': reference_trajectory['velocity'].reshape(-1, 1),
                'omega_ref': reference_trajectory['angular_velocity'].reshape(-1, 1)
            }

            # Add disturbance parameters
            reference_params['disturbance'] = np.zeros((6, 1))

            # Pass reference trajectory and disturbance information to the controller
            self.mpc.set_uncertainty_values(**reference_params)

            # Disturbance compensation
            if self.use_disturbance_compensation:
                self._update_disturbance_compensation(current_state, reference_trajectory)

            # Execute MPC solving
            u_mpc = self.mpc.make_step(current_x)

            # Extract control input
            control_input = self._extract_control_input(u_mpc)

            # Save control input
            self.last_control = control_input

            return control_input

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"MPC step failed: {e}")
            # Return safe default control
            safe_control = np.array([5.0, 0.0, 0.0])
            self.last_control = safe_control
            return safe_control

    def _update_disturbance_compensation(self, current_state, reference_trajectory):
        """Update the disturbance compensation

        """
        # Calculate the error
        pos_error = current_state[0:3] - reference_trajectory['position']
        att_error = current_state[3:6] - reference_trajectory['attitude']
        vel_error = current_state[6:9] - reference_trajectory['velocity']
        ang_error = current_state[9:12] - reference_trajectory['angular_velocity']

        e1 = np.concatenate([pos_error, att_error])
        e2 = np.concatenate([vel_error, ang_error])

        # Update the disturbance estimate
        gamma = current_state[3:6]
        tau = thrust_params_to_force_torque(self.last_control, self.params.rp_r, self.params.rp_l)

        # Update the disturbance observer
        delta_hat = self.disturbance_observer.update(params.DT, e1, e2, tau, gamma)
        self.last_disturbance_estimate = delta_hat

    def _extract_control_input(self, u_mpc):
        """
        Convert the control input obtained from MPC optimization to a usable control quantity
        1. Convert the control input obtained from MPC optimization to a usable control quantity
        2. Check the validity of the control input, ensuring that it does not contain NaN or infinite values
        3. Limit the range of the control input, ensuring that it conforms to the physical constraints
        4. Return the final control input
        """
        try:
            if hasattr(u_mpc, 'full'):
                u_array = u_mpc.full().flatten()
                control_input = np.array([
                    float(u_array[0]),  # T
                    float(u_array[1]),  # mu
                    float(u_array[2])   # nu
                ])
            elif isinstance(u_mpc, np.ndarray):
                control_input = np.array([
                    float(u_mpc[0]),
                    float(u_mpc[1]),
                    float(u_mpc[2])
                ])
            else:
                u_flat = np.array(u_mpc).flatten()
                control_input = np.array([
                    float(u_flat[0]),
                    float(u_flat[1]),
                    float(u_flat[2])
                ])
        except (IndexError, ValueError, TypeError):
            print("Warning: Control input cannot be extracted correctly")
            control_input = np.array([5.0, 0.0, 0.0])

        # Check if the control input contains invalid values (NaN or infinity)
        if np.any(np.isnan(control_input)) or np.any(np.isinf(control_input)):
            print("Warning: The control input contains NaN or infinity value")
            return np.array([5.0, 0.0, 0.0])

        # Limit the range of the control input, ensuring that it conforms to the physical constraints
        control_input[0] = np.clip(control_input[0], params.T_MIN, params.T_MAX)
        control_input[1] = np.clip(control_input[1], params.MU_MIN, params.MU_MAX)
        control_input[2] = np.clip(control_input[2], params.NU_MIN, params.NU_MAX)

        return control_input

    def get_prediction(self):
        """
        Get the MPC prediction result
        Return the prediction result
        If the prediction result is empty, return None
        """
        try:
            if hasattr(self.mpc, 'data') and self.mpc.data is not None:
                # Use the public interface to get the prediction data
                prediction_data = {
                    'states': self.mpc.data.prediction(('_x', 'pos')),
                    'controls': self.mpc.data.prediction(('_u', 'T'))
                }
                return prediction_data
            else:
                return {'states': None, 'controls': None}
        except Exception: # pylint: disable=broad-exception-caught
            return {'states': None, 'controls': None}

    def get_current_disturbance_estimate(self):
        """
        Get current disturbance estimate
        """
        if self.use_disturbance_compensation and self.last_disturbance_estimate is not None:
            try:
                # Handle the CasADi DM object
                if hasattr(self.last_disturbance_estimate, 'full'):
                    return self.last_disturbance_estimate.full().flatten()
                # Handle the numpy array
                elif hasattr(self.last_disturbance_estimate, 'flatten'):
                    return self.last_disturbance_estimate.flatten()
                # Handle other types
                else:
                    return np.array(self.last_disturbance_estimate).flatten()
            except (AttributeError, ValueError):
                return np.zeros(6)
        else:
            return np.zeros(6)

    def reset(self):
        """Reset the controller
        1. Reset the disturbance observer
        2. Reset the initial conditions
        3. Clear the history data
        4. Reset the controller
        5. Reset the estimator
        6. Reset the simulator
        """
        if self.use_disturbance_compensation:
            self.disturbance_observer.reset()
            self.last_disturbance_estimate = np.zeros(6)

        # Reset the initial conditions
        self._setup_initial_conditions()

        # Clear the history data
        self.mpc.reset_history()
        self.estimator.reset_history()
        if self.simulator is not None:
            self.simulator.reset_history()


# Auxiliary function: trajectory format conversion
def convert_trajectory_format(yc, yc_dot):
    """
    Convert the trajectory format to the format required by the do-mpc controller

    Args:
        yc: Reference state [position (3) + attitude (3)]
        yc_dot: Reference state derivative [position derivative (3) + attitude derivative (3)]

    Returns:
        dict: Formatted reference trajectory
    """
    return {
        'position': yc[0:3],
        'attitude': yc[3:6],
        'velocity': yc_dot[0:3],
        'angular_velocity': yc_dot[3:6]
    }
