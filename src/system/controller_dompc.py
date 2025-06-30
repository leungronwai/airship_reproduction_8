"""
NMPC Controller Implementation based on do-mpc
"""

# pylint: disable=invalid-name
# cspell:ignore dompc vertcat radau mterm lterm rterm ndarray fmin fmax idas abstol reltol Kalman symvar
# cspell: ignore nlpsol ipopt print_level max_iter acceptable_tol acceptable_obj_change_tol tol opcua casadi NMPC
# cspell: ignore cvodes mu_strategy hessian_approximation limited_memory_max_history alpha_for_y recalc_y max_wall_time print_time

import warnings
# third-party library
import numpy as np
import casadi as ca
import do_mpc


# local module
from src.system.airship_dynamic import AirshipCasADiSymbolic
from src.system.trajectory_ref import Trajectory

from src.config import parameters as params

warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.sysid")
warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.opcua")





class DoMpcConfig:
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




        # Create do-mpc model
        self.model = self.create_model()

        # Create MPC controller
        self.mpc = self.create_mpc_controller()






        # Store last control input
        self.last_control = np.array([5.0, 0.0, 0.0])

        # Create reference trajectory
        self.trajectory = Trajectory()

    def create_model(self, symvar_type='MX'):
        """
        Create do-mpc model
        here is placeholder for symbolic variable type

        Returns:
            do_mpc.model.Model: Airship dynamics model
        """
        # Create model type (continuous time)
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type, symvar_type)

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

    def create_mpc_controller(self, silence_solver=False):
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

        if silence_solver:
            mpc.set_param(nlpsol_opts={'ipopt.print_level': 0}) # print_level = 0 means no print


        # 设置各分量的缩放因子
        pos_weight = 10
        att_weight = 5
        vel_weight = 1
        ang_weight = 0.5

        Q_scaled = params.Q.copy()
        Qf_scaled = params.Qf.copy()

        # 缩放各类状态误差
        Q_scaled[0:3, 0:3] *= pos_weight  # 增强位置误差权重（x, y, z）
        Q_scaled[3:6, 3:6] *= att_weight
        Q_scaled[6:9, 6:9] *= vel_weight
        Q_scaled[9:12, 9:12] *= ang_weight

        Qf_scaled[0:3, 0:3] *= pos_weight * 2    # 终端位置更强调
        Qf_scaled[3:6, 3:6] *= att_weight * 2
        Qf_scaled[6:9, 6:9] *= vel_weight * 0.2    # 减弱速度/角速度误差在代价函数中的影响
        Qf_scaled[9:12, 9:12] *= ang_weight * 0.2



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
        mpc.set_rterm(T=1, mu=1, nu=1) # means: rterm = 1 * T^2 + 1 * mu^2 + 1 * nu^2

        # Set constraints
        self._set_mpc_constraints(mpc)

        # Complete MPC setup
        mpc.setup()

        return mpc

    @staticmethod
    def _set_mpc_constraints(mpc):
        """Set MPC constraints"""
        # === Control input constraints ===
        # Thrust: avoid zero, ensure minimum lift
        mpc.bounds['lower', '_u', 'T'] = max(params.T_MIN, 5.0)
        mpc.bounds['upper', '_u', 'T'] = params.T_MAX

        # Deflection angles: mu and nu (typically ±30°)
        mpc.bounds['lower', '_u', 'mu'] = params.MU_MIN
        mpc.bounds['upper', '_u', 'mu'] = params.MU_MAX
        mpc.bounds['lower', '_u', 'nu'] = params.NU_MIN
        mpc.bounds['upper', '_u', 'nu'] = params.NU_MAX

        # === State constraints ===
        max_position = 50.0                         # Maximum range from origin in meters
        max_angle = np.deg2rad(45)                  # Limit roll/yaw to ±45°
        max_pitch = np.deg2rad(30)                  # Limit pitch tighter ±30°
        max_velocity = 10.0                         # Limit linear velocity (m/s)
        max_angular_velocity = np.deg2rad(45)       # Limit angular velocity (rad/s)

        # Position constraints
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # Attitude constraints (Euler angles: [roll, pitch, yaw])
        mpc.bounds['lower', '_x', 'att'] = -max_angle
        mpc.bounds['upper', '_x', 'att'] = max_angle

        # Pitch angle (theta, index 1) more restrictive
        mpc.bounds['lower', '_x', 'att', 1] = -max_pitch
        mpc.bounds['upper', '_x', 'att', 1] = max_pitch

        # Linear velocity constraints
        mpc.bounds['lower', '_x', 'vel'] = -max_velocity
        mpc.bounds['upper', '_x', 'vel'] = max_velocity

        # Angular velocity constraints
        mpc.bounds['lower', '_x', 'omega'] = -max_angular_velocity
        mpc.bounds['upper', '_x', 'omega'] = max_angular_velocity





    def create_simulator(self):
        """
        Create do-mpc Simulator with clean structure

        Returns:
            do_mpc.simulator.Simulator: Configured simulator
        """
        if not self.create_simulator:
            return None

        simulator = do_mpc.simulator.Simulator(self.model)

        # Configure simulator parameters
        simulator_params = {
            't_step': params.DT,
            'integration_tool': 'cvodes',
            'abstol': 1e-8,
            'reltol': 1e-6,
            'max_step_size': params.DT / 5,
            'min_step_size': params.DT / 500,
        }
        simulator.set_param(**simulator_params)

        # # Set initial state
        tvp_num = simulator.get_tvp_template()
        tvp_num['pos_ref'] = np.zeros((3, 1))  # Initial position [x, y, z]
        tvp_num['att_ref'] = np.zeros((3, 1))  # Initial attitude [phi, theta, psi]
        tvp_num['vel_ref'] = np.zeros((3, 1))  # Initial velocity [vx, vy, vz]
        tvp_num['omega_ref'] = np.zeros((3, 1))  # Initial angular velocity [p, q, r]
        tvp_num['disturbance'] = np.zeros((6, 1))  # Initial disturbance [d1, d2, d3, d4, d5, d6]


        def tvp_fun(t_now):
            """Update disturbance parameters for current simulation time"""
            yc, yc_dot, _, _, _ = self.trajectory.get_spiral_trajectory(t_now)
            tvp_num['pos_ref'] = yc[0:3].reshape(-1, 1)  # 位置参考 [x, y, z]
            tvp_num['att_ref'] = yc[3:6].reshape(-1, 1)  # 姿态参考 [phi, theta, psi]
            tvp_num['vel_ref'] = yc_dot[0:3].reshape(-1, 1)  # 速度参考 [vx, vy, vz]
            tvp_num['omega_ref'] = yc_dot[3:6].reshape(-1, 1)  # 角速度参考 [p, q, r]
            tvp_num['disturbance'] = params.disturbance_delta(t_now).reshape(-1, 1)
            return tvp_num

        simulator.set_tvp_fun(tvp_fun)

        simulator.setup()

        print("do-mpc Simulator setup completed successfully")
        return simulator
