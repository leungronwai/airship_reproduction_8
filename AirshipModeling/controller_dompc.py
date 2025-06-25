"""
NMPC Controller Implementation based on do-mpc        # Create do-mpc model
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

        # Store last control input with Simulator support
Uses do-mpc library to simplify NMPC controller development, providing more stable and efficient implementation
"""

# pylint: disable=invalid-name
# cspell:ignore dompc vertcat radau mterm lterm rterm ndarray fmin fmax idas abstol reltol
# cspell: ignore nlpsol ipopt print_level max_iter acceptable_tol acceptable_obj_change_tol tol
# cspell: ignore cvodes mu_strategy hessian_approximation limited_memory_max_history alpha_for_y recalc_y max_wall_time print_time

import numpy as np
import casadi as ca
import do_mpc

from config import parameters as params
from AirshipModeling.airship_dynamic import AirshipCasADiSymbolic
from AirshipModeling.thrust_vectoring import thrust_params_to_force_torque



class DoMPCAirshipController:
    """
    Airship NMPC Controller based on do-mpc

    Advantages:
    1. Simplified MPC setup and configuration
    2. Built-in solver configuration and optimization
    3. Better numerical stability
    4. Automatic constraint and boundary handling
    5. Built-in visualization and analysis tools
    6. Support for do-mpc Simulator integration
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
        Create do-mpc model - Enhanced version

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
        U_control = ca.vertcat(T, mu, nu)

        # Get dynamics equations (including disturbance)
        X_dot = symbolic_model.rhs_symbolic(X_state, U_control, external_disturbance=disturbance)

        # Add numerical stability - limit derivative magnitudes
        max_derivative = 1e5
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

        # Set state expressions (for objective function)
        position_error = pos - pos_ref
        attitude_error = att - att_ref
        velocity_error = vel - vel_ref
        angular_error = omega - omega_ref

        # Auxiliary expressions
        model.set_expression('pos_error', position_error)
        model.set_expression('att_error', attitude_error)
        model.set_expression('vel_error', velocity_error)
        model.set_expression('ang_error', angular_error)

        # Add measurement output (for estimator)
        model.set_expression('y_meas', X_state)  # Assume full state observability

        # Complete model setup
        model.setup()

        return model

    def _create_mpc_controller(self):
        """
        Create MPC controller - Enhanced version

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

        # Improved objective function - numerical scaling
        # Scale weight matrices to avoid numerical issues
        Q_scaled = params.Q * 0.01  # Reduce state weights
        Qf_scaled = params.Qf * 0.01
        R_scaled = params.R * 100.0  # Increase control weights to promote smoothness



        # Terminal cost - only includes state errors, not control inputs
        mterm = (self.model.aux['pos_error'].T @ Qf_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Qf_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Qf_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Qf_scaled[9:12, 9:12] @ self.model.aux['ang_error'])

        # Stage cost - includes state errors and control inputs
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
        Create state estimator

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

        simulator = do_mpc.simulator.Simulator(self.model)

        # Simulator setup
        setup_simulator = {
            't_step': params.DT,
            'integration_tool': 'cvodes', #'idas',  # Use IDAS integrator
            'abstol': 1e-6,
            'reltol': 1e-4,
        }

        simulator.set_param(**setup_simulator)

        # Set disturbance function
        def disturbance_func(t_now):
            """Define time-varying disturbance"""
            try:
                delta = params.disturbance_delta(t_now)
                return delta.reshape(-1, 1)
            except Exception: # pylint: disable=broad-exception-caught
                return np.zeros((6, 1))

        # Set parameter function
        p_template = simulator.get_p_template()
        p_template['pos_ref'] = np.zeros((3, 1))
        p_template['att_ref'] = np.zeros((3, 1))
        p_template['vel_ref'] = np.zeros((3, 1))
        p_template['omega_ref'] = np.zeros((3, 1))
        p_template['disturbance'] = disturbance_func

        def p_fun(t_now):
            """Parameter function"""
            _ = t_now
            return p_template

        simulator.set_p_fun(p_fun)

        # Complete Simulator setup
        simulator.setup()

        return simulator

    def _setup_initial_conditions(self):
        """Set initial conditions"""
        # Set initial state
        x0 = params.X0.copy()

        # Stricter numerical cleaning
        # x0 = np.nan_to_num(x0, nan=0.0, posinf=10.0, neginf=-10.0)

        # Limit initial angles to avoid singularities
        # x0[3:6] = np.clip(x0[3:6], -np.pi/2, np.pi/2)

        # Reconstruct state vector to do-mpc expected format
        _x0 = np.concatenate([
            x0[0:3].reshape(-1, 1),   # pos
            x0[3:6].reshape(-1, 1),   # att
            x0[6:9].reshape(-1, 1),   # vel
            x0[9:12].reshape(-1, 1)   # omega
        ])

        # Set initial states for all components
        self.mpc.x0 = _x0
        self.estimator.x0 = _x0
        if self.simulator is not None:
            self.simulator.x0 = _x0

        # Improved initial guess
        try:
            # Set conservative initial control guess
            u0 = np.array([[5.0], [0.0], [0.0]])  # Stable thrust, zero moment

            # Set initial guess for entire prediction horizon
            for k in range(self.mpc.settings.n_horizon):
                self.mpc.u0[k] = u0
                self.mpc.x0[k+1] = _x0  # Keep states stable

            self.mpc.set_initial_guess()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Failed to set initial guess: {e}")

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
                print("Warning: Input state contains NaN or infinite values")
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

            # Set reference trajectory
            self.mpc.set_uncertainty_values(**reference_params)

            # Disturbance compensation
            if self.use_disturbance_compensation:
                self._update_disturbance_compensation(current_state, reference_trajectory)

            # Execute MPC solution
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
        tau = thrust_params_to_force_torque(self.last_control, self.params.rp_r, self.params.rp_l)

        # Update disturbance observer
        delta_hat = self.disturbance_observer.update(params.DT, e1, e2, tau, gamma)
        self.last_disturbance_estimate = delta_hat

    def _extract_control_input(self, u_mpc):
        """Extract control input"""
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
        """Get MPC prediction results"""
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
        """Reset controller"""
        if self.use_disturbance_compensation:
            self.disturbance_observer.reset()
            self.last_disturbance_estimate = np.zeros(6)

        # Reset initial conditions
        self._setup_initial_conditions()

        # Clear history data
        self.mpc.reset_history()
        self.estimator.reset_history()
        if self.simulator is not None:
            self.simulator.reset_history()


# Auxiliary function: trajectory format conversion
def convert_trajectory_format(yc, yc_dot):
    """
    Convert trajectory format to do-mpc controller required format

    Args:
        yc: Reference state [position (3) + attitude (3)]
        yc_dot: Reference state derivatives [position derivatives (3) + attitude derivatives (3)]

    Returns:
        dict: Formatted reference trajectory
    """
    return {
        'position': yc[0:3],
        'attitude': yc[3:6],
        'velocity': yc_dot[0:3],
        'angular_velocity': yc_dot[3:6]
    }
