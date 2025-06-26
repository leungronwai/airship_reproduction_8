"""
NMPC Controller Implementation based on do-mpc
"""

# pylint: disable=invalid-name
# cspell:ignore dompc vertcat radau mterm lterm rterm ndarray fmin fmax idas abstol reltol Kalman
# cspell: ignore nlpsol ipopt print_level max_iter acceptable_tol acceptable_obj_change_tol tol opcua casadi
# cspell: ignore cvodes mu_strategy hessian_approximation limited_memory_max_history alpha_for_y recalc_y max_wall_time print_time

# standard library
import logging
import warnings

# third-party library
import numpy as np
import casadi as ca
import do_mpc


# local module
from src.airship_modeling.airship_dynamic import AirshipCasADiSymbolic
from src.airship_modeling.thrust_vectoring import thrust_params_to_force_torque
from src.airship_modeling.observer import DisturbanceObserver
from src.config import parameters as params

warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.sysid")
warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.opcua")
# set up logger
logger = logging.getLogger(__name__)



class do_mpc_controller:
    """
    Airship NMPC Controller based on do-mpc
    """

    def __init__(self, use_disturbance_compensation=True, create_simulator=True):
        """
        Initialize do-mpc based controller

        Args:
            use_disturbance_compensation: Whether to enable disturbance compensation
            create_simulator: Whether to create do-mpc Simulator
        """
        self.use_disturbance_compensation = use_disturbance_compensation
        self.params = params
        self.create_simulator = create_simulator

        # Initialize disturbance observer
        if use_disturbance_compensation:
            self.disturbance_compensation_factor = getattr(params, 'do_compensation_gain', 0.9)
            self.last_disturbance_estimate = np.zeros(6)
            self.disturbance_observer = DisturbanceObserver()
        else:
            self.disturbance_observer = None

        # Create do-mpc model
        self.model = self._create_model()

        # Create MPC controller
        self.mpc = self._create_mpc_controller()

        # Create estimator
        self.estimator = self._create_estimator()

        # Create Simulator (if needed)
        if create_simulator:
            self.simulator = self._create_simulator()
        else:
            self.simulator = None

        # Initialize controller
        self._setup_initial_conditions()

        # Store last control input
        self.last_control = np.array([5.0, 0.0, 0.0])

    def _create_model(self):
        """
        Create do-mpc model

        Returns:
            do_mpc.model.Model: Airship dynamics model
        """
        # Create model type (continuous time)
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # Define state variables - 12-dimensional state vector
        pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))      # Position [x, y, z]
        att = model.set_variable(var_type='_x', var_name='att', shape=(3, 1))      # Attitude [phi, theta, psi]
        vel = model.set_variable(var_type='_x', var_name='vel', shape=(3, 1))      # Linear velocity [u, v, w]
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(3, 1))  # Angular velocity [p, q, r]

        # Define control inputs - 3-dimensional control vector
        T = model.set_variable(var_type='_u', var_name='T')          # Thrust magnitude
        mu = model.set_variable(var_type='_u', var_name='mu')        # Horizontal deflection angle
        nu = model.set_variable(var_type='_u', var_name='nu')        # Vertical deflection angle

        # Define reference trajectory parameters
        pos_ref = model.set_variable(var_type='_p', var_name='pos_ref', shape=(3, 1))
        att_ref = model.set_variable(var_type='_p', var_name='att_ref', shape=(3, 1))
        vel_ref = model.set_variable(var_type='_p', var_name='vel_ref', shape=(3, 1))
        omega_ref = model.set_variable(var_type='_p', var_name='omega_ref', shape=(3, 1))

        # Define disturbance variables (for Simulator)
        disturbance = model.set_variable(var_type='_p', var_name='disturbance', shape=(6, 1))


        # Use existing symbolic dynamics model
        symbolic_model = AirshipCasADiSymbolic(self.params)

        # Combine state vector
        X_state = ca.vertcat(pos, att, vel, omega)
        Thrust_paras = ca.vertcat(T, mu, nu)

        # Get dynamics equations (including disturbance)
        X_dot = symbolic_model.rhs_symbolic(X_state, Thrust_paras, external_disturbance=disturbance)

        # Add numerical stability - limit derivative magnitudes
        max_derivative = 1e6
        X_dot = ca.fmin(ca.fmax(X_dot, -max_derivative), max_derivative)

        # Decompose state derivatives
        pos_dot = X_dot[0:3]
        att_dot = X_dot[3:6]
        vel_dot = X_dot[6:9]
        omega_dot = X_dot[9:12]

        # Set differential equations
        model.set_rhs('pos', pos_dot)
        model.set_rhs('att', att_dot)
        model.set_rhs('vel', vel_dot)
        model.set_rhs('omega', omega_dot)

        # state error  (for cost function)
        position_error = pos - pos_ref
        attitude_error = att - att_ref
        velocity_error = vel - vel_ref
        angular_error = omega - omega_ref

        # Auxiliary expressions （Custom intermediate variable expressions）
        model.set_expression('pos_error', position_error)
        model.set_expression('att_error', attitude_error)
        model.set_expression('vel_error', velocity_error)
        model.set_expression('ang_error', angular_error)

        # Add measurement output (for estimator) # Assume full state observability
        # The value that the sensor can measure and the output variable that the system can "observe"
        model.set_expression('y_meas', X_state)

        # Complete model setup
        model.setup()

        return model

    def _create_mpc_controller(self):
        """
        Create MPC controller

        Returns:
            do_mpc.controller.MPC: Configured MPC controller
        """
        mpc = do_mpc.controller.MPC(self.model)

        # MPC setup
        setup_mpc = {
            'n_horizon': min(params.N_HORIZON, 8),
            'n_robust': 1,
            'open_loop': 0,
            't_step': params.DT,
            'state_discretization': 'collocation',  # Back to collocation
            'collocation_type': 'radau',  # Change to legendre for better stability
            'collocation_deg': 2,
            'collocation_ni': 1,  # Reduce internal point count
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
                'ipopt.limited_memory_max_history': 5,  # Reduce history records
                'ipopt.alpha_for_y': 'primal',
                'ipopt.recalc_y': 'yes',
                'ipopt.max_wall_time': 3.0,  # Further limit solving time
                'ipopt.warm_start_init_point': 'yes',  # Enable warm start
                'print_time': 0
            }
        }

        mpc.set_param(**setup_mpc)

        # Set uncertainty values (disturbance compensation)
        mpc.set_uncertainty_values(
            pos_ref=np.zeros((3, 1)),
            att_ref=np.zeros((3, 1)),
            vel_ref=np.zeros((3, 1)),
            omega_ref=np.zeros((3, 1)),
            disturbance=np.zeros((6, 1))  # Add disturbance parameters
        )

        # Scale weight matrices to avoid numerical issues
        Q_scaled = params.Q * 0.01  # Reduce state weights
        Qf_scaled = params.Qf * 0.01
        _R_scaled = params.R * 100.0  # Increase control weights to promote smoothness



        # Terminal cost - only includes state errors, not control inputs
        mterm = (self.model.aux['pos_error'].T @ Qf_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Qf_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Qf_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Qf_scaled[9:12, 9:12] @ self.model.aux['ang_error'])

        # Stage cost - includes state errors and control inputs
        lterm = (self.model.aux['pos_error'].T @ Q_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Q_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Q_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Q_scaled[9:12, 9:12] @ self.model.aux['ang_error'] )
        # + self.model.u['T']**2 * R_scaled[0, 0] + self.model.u['mu']**2 * R_scaled[1, 1] + self.model.u['nu']**2 * R_scaled[2, 2]

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Setting the penalty weight for the control input
        # in the objective function this is the "smoothness constraint" of control
        mpc.set_rterm(T=0.1, mu=0.1, nu=0.1) # means: rterm = 0.1 * T^2 + 0.1 * mu^2 + 0.1 * nu^2

        # Set constraints
        self._set_mpc_constraints(mpc)

        # Complete MPC setup
        mpc.setup()

        return mpc

    def _set_mpc_constraints(self, mpc):
        """Set MPC constraints"""
        # Control input constraints
        mpc.bounds['lower', '_u', 'T'] = max(params.T_MIN, 0.1)  # Avoid zero thrust
        mpc.bounds['upper', '_u', 'T'] = params.T_MAX
        mpc.bounds['lower', '_u', 'mu'] = params.MU_MIN
        mpc.bounds['upper', '_u', 'mu'] = params.MU_MAX
        mpc.bounds['lower', '_u', 'nu'] = params.NU_MIN
        mpc.bounds['upper', '_u', 'nu'] = params.NU_MAX

        # State constraints
        max_position = 200.0  # Reduce to reasonable range
        max_angle = np.pi/2   # Limit attitude angles to prevent singularities
        max_velocity = 30.0
        max_angular_velocity = np.pi/2  # Reasonable angular velocity limits

        # Position constraints
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # Attitude constraints
        mpc.bounds['lower', '_x', 'att'] = -max_angle
        mpc.bounds['upper', '_x', 'att'] = max_angle
        mpc.bounds['lower', '_x', 'att', 1] = -max_angle/6  # Pitch angle theta limits
        mpc.bounds['upper', '_x', 'att', 1] = max_angle/6

        # Velocity constraints
        mpc.bounds['lower', '_x', 'vel'] = -max_velocity
        mpc.bounds['upper', '_x', 'vel'] = max_velocity
        mpc.bounds['lower', '_x', 'omega'] = -max_angular_velocity
        mpc.bounds['upper', '_x', 'omega'] = max_angular_velocity

    def _create_estimator(self):
        """
        Create an estimator that assumes "the complete system state can be directly observed"
        to pass the current state to the MPC controller for optimization
        If you replace it with a more complex estimator (such as the Extended Kalman Filter) in the future,
        it will be replaced here

        Returns:
                State feedback estimator
        """
        estimator = do_mpc.estimator.StateFeedback(self.model)
        return estimator


    def _create_simulator(self):
        """
        Create do-mpc Simulator - Enhanced version for standalone use

        Returns:
            do_mpc.simulator.Simulator: Configured simulator
        """
        if not self.create_simulator:
            return None

        simulator = do_mpc.simulator.Simulator(self.model)

        # Enhanced simulator setup for stability
        setup_simulator = {
            't_step': params.DT,
            'integration_tool': 'cvodes',  # Use CVODES for better stability
            'abstol': 1e-8,  # Tighter absolute tolerance
            'reltol': 1e-6,  # Tighter relative tolerance
            'max_step_size': params.DT / 10,  # Limit maximum step size
            'min_step_size': params.DT / 1000,  # Set minimum step size
        }

        simulator.set_param(**setup_simulator)

        # Enhanced parameter function with error handling
        def disturbance_func(t_now):
            """Define time-varying disturbance with error handling"""
            try:
                delta = params.disturbance_delta(t_now)
                # Ensure proper shape
                if delta.shape[0] != 6:
                    raise ValueError(f"Disturbance must be 6-dimensional, got {delta.shape}")
                return delta.reshape(-1, 1)
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.warning("Disturbance function failed at t=%.3f: %s", t_now, e)
                return np.zeros((6, 1))

        #
        p_template = simulator.get_p_template()
        p_template['pos_ref'] = np.zeros((3, 1))
        p_template['att_ref'] = np.zeros((3, 1))
        p_template['vel_ref'] = np.zeros((3, 1))
        p_template['omega_ref'] = np.zeros((3, 1))
        p_template['disturbance'] = np.zeros((6, 1))

        def p_fun(t_now):
            """
            Tell the simulator which values should be used at each simulation moment for disturbance
            return the current disturbance based on time to be used in simulation
            """
            try:
                # Update disturbance for current time
                p_template['disturbance'] = disturbance_func(t_now)
                return p_template
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error("Parameter function failed at t=%.3f: %s", t_now, e)
                # Return safe default parameters
                safe_template = simulator.get_p_template()
                for key in safe_template.keys():
                    if 'ref' in key:
                        safe_template[key] = np.zeros((3, 1))
                    elif key == 'disturbance':
                        safe_template[key] = np.zeros((6, 1))
                return safe_template

        simulator.set_p_fun(p_fun)

        # Complete Simulator setup with validation
        try:
            simulator.setup()
            logger.info("do-mpc Simulator setup completed successfully")
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Failed to setup do-mpc Simulator: %s", e)
            raise RuntimeError(f"Simulator setup failed: {e}") # pylint: disable=raise-missing-from

        return simulator

    def _setup_initial_conditions(self):
        """
        Setting the initial state and initial guess (initial value) of the MPC controller,
        estimator and simulator at the beginning of simulation/operation,
        which is the "startup preparation" of the entire control process.

        """
        # Set initial state
        x0 = params.X0.copy()

        # Stricter numerical cleaning
        # x0 = np.nan_to_num(x0, nan=0.0, posinf=10.0, neginf=-10.0)

        # Limit initial angles to avoid singularities
        x0[3:6] = np.clip(x0[3:6], -np.pi/2, np.pi/2)

        try:
            # set initial state for MPC
            self.mpc.x0['pos'] = x0[0:3].reshape(-1, 1)
            self.mpc.x0['att'] = x0[3:6].reshape(-1, 1)
            self.mpc.x0['vel'] = x0[6:9].reshape(-1, 1)
            self.mpc.x0['omega'] = x0[9:12].reshape(-1, 1)

            # set initial control guess
            self.mpc.u0['T'] = 10.0    # reasonable thrust value
            self.mpc.u0['mu'] = 0.0    # zero deflection angle
            self.mpc.u0['nu'] = 0.0    # zero deflection angle

            # set initial guess
            self.mpc.set_initial_guess()

            # set initial state for estimator
            self.estimator.x0['pos'] = x0[0:3].reshape(-1, 1)
            self.estimator.x0['att'] = x0[3:6].reshape(-1, 1)
            self.estimator.x0['vel'] = x0[6:9].reshape(-1, 1)
            self.estimator.x0['omega'] = x0[9:12].reshape(-1, 1)

            # set initial state for simulator (if exists)
            if self.simulator is not None:
                self.simulator.x0['pos'] = x0[0:3].reshape(-1, 1)
                self.simulator.x0['att'] = x0[3:6].reshape(-1, 1)
                self.simulator.x0['vel'] = x0[6:9].reshape(-1, 1)
                self.simulator.x0['omega'] = x0[9:12].reshape(-1, 1)

            logger.info("Initial conditions set successfully")

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Failed to set initial conditions: %s", e)
            # try basic setting
            try:
                self.mpc.set_initial_guess()
                logger.info("Basic initial guess set")
            except Exception as e2: # pylint: disable=broad-exception-caught
                logger.error("Failed to set even basic initial guess: %s", e2)

    def step(self, current_state, reference_trajectory, t_current=0.0):
        """
        Execute one step of MPC control

        Args:
            current_state: Current state [12x1]
            reference_trajectory: Reference trajectory dictionary [position, attitude, velocity, angular_velocity]
            t_current: Current time

        Returns:
            control_params for thrust: Control input [T, mu, nu]
        """
        _ = t_current
        try:
            # Check input validity
            if np.any(np.isnan(current_state)) or np.any(np.isinf(current_state)):
                print("Warning: Input state contains NaN or infinite values")
                return np.array([5.0, 0.0, 0.0])

            # Update current state - use dictionary format instead of vector format
            self.mpc.x0['pos'] = current_state[0:3].reshape(-1, 1)
            self.mpc.x0['att'] = current_state[3:6].reshape(-1, 1)
            self.mpc.x0['vel'] = current_state[6:9].reshape(-1, 1)
            self.mpc.x0['omega'] = current_state[9:12].reshape(-1, 1)

            # Update reference trajectory parameters
            reference_params = {
                'pos_ref': reference_trajectory['position'].reshape(-1, 1),
                'att_ref': reference_trajectory['attitude'].reshape(-1, 1),
                'vel_ref': reference_trajectory['velocity'].reshape(-1, 1),
                'omega_ref': reference_trajectory['angular_velocity'].reshape(-1, 1)
            }

            # Add disturbance parameters
            if self.use_disturbance_compensation and self.last_disturbance_estimate is not None:
                reference_params['disturbance'] = self.last_disturbance_estimate.reshape(6, 1)
            else:
                reference_params['disturbance'] = np.zeros((6, 1))

            # Before the execution of each MPC control step, the reference trajectory
            # and disturbance value at the current moment are passed to the _p parameter variable in the model
            self.mpc.set_uncertainty_values(**reference_params)

            # Disturbance compensation
            if self.use_disturbance_compensation:
                self._update_disturbance_compensation(current_state, reference_trajectory)

            # Under the current state current x, solve for the optimal control input u
            # 执行 MPC 求解 - 传入字典格式的状态
            try:
                u_mpc = self.mpc.make_step(self.mpc.x0)
            except Exception as mpc_error: # pylint: disable=broad-exception-caught
                logger.warning("MPC make_step failed: %s, using previous control", mpc_error)
                return self.last_control if hasattr(self, 'last_control') else np.array([5.0, 0.0, 0.0])

            # It is usually a data structure of casadi (such as casadi.DM),
            # and you will convert it to numpy.array later to extract control instructions here
            control_input_params = self._extract_control_input(u_mpc)

            # Save control input
            self.last_control = control_input_params

            return control_input_params  # [T, mu, nu]

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"MPC step failed: {e}")
            # Return safe default control
            safe_control_params = np.array([5.0, 0.0, 0.0])
            self.last_control = safe_control_params
            return safe_control_params

    def _update_disturbance_compensation(self, current_state, reference_trajectory):
        """Update disturbance compensation"""
        # Calculate errors
        pos_error = current_state[0:3] - reference_trajectory['position']
        att_error = current_state[3:6] - reference_trajectory['attitude']
        vel_error = current_state[6:9] - reference_trajectory['velocity']
        ang_error = current_state[9:12] - reference_trajectory['angular_velocity']

        e1 = np.concatenate([pos_error, att_error])
        e2 = np.concatenate([vel_error, ang_error])

        # Update disturbance estimation
        gamma = current_state[3:6]
        Thrust_Force_torque = thrust_params_to_force_torque(self.last_control, self.params.rp_r, self.params.rp_l)

        # Update disturbance observer
        delta_hat = self.disturbance_observer.update(params.DT, e1, e2, Thrust_Force_torque, gamma)
        self.last_disturbance_estimate = delta_hat

    def _extract_control_input(self, u_mpc):
        """
        Extract control input from casadi structure to numpy array
        Args:
            u_mpc: Control input params from MPC (is casadi structure)
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
            print("Warning: Unable to properly extract control input")
            control_input = np.array([5.0, 0.0, 0.0])

        # Check validity and limit range
        if np.any(np.isnan(control_input)) or np.any(np.isinf(control_input)):
            print("Warning: Control input contains NaN or infinite values")
            return np.array([5.0, 0.0, 0.0])

        control_input[0] = np.clip(control_input[0], params.T_MIN, params.T_MAX)
        control_input[1] = np.clip(control_input[1], params.MU_MIN, params.MU_MAX)
        control_input[2] = np.clip(control_input[2], params.NU_MIN, params.NU_MAX)

        return control_input

    def get_prediction(self):
        """
        Obtain the state trajectory and control input trajectory predicted by the do-mpc controller

        prediction_data = {
        'pos': self.mpc.data.prediction(('_x', 'pos')),
        'att': self.mpc.data.prediction(('_x', 'att')),
        'vel': self.mpc.data.prediction(('_x', 'vel')),
        'omega': self.mpc.data.prediction(('_x', 'omega')),
        'T': self.mpc.data.prediction(('_u', 'T')),
        'mu': self.mpc.data.prediction(('_u', 'mu')),
        'nu': self.mpc.data.prediction(('_u', 'nu'))
}
        """
        try:
            if hasattr(self.mpc, 'data') and self.mpc.data is not None:
                # Use public interface to get prediction data
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
        """Get current disturbance estimate"""
        if self.use_disturbance_compensation and self.last_disturbance_estimate is not None:
            try:
                # Handle CasADi DM objects
                if hasattr(self.last_disturbance_estimate, 'full'):
                    return self.last_disturbance_estimate.full().flatten()
                # Handle numpy arrays
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
        """
        Reset the internal state of the controller to
        enable the entire MPC control system to "restart" its operation
        """
        if self.use_disturbance_compensation and self.disturbance_observer is not None:
            self.disturbance_observer.reset()
            self.last_disturbance_estimate = np.zeros(6)

        # Reset initial conditions
        self._setup_initial_conditions()

        # Clear history data
        self.mpc.reset_history()
        self.estimator.reset_history()
        if self.simulator is not None:
            self.simulator.reset_history()
