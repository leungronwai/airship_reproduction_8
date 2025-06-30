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
from do_mpc.model import Model


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
        # Create reference trajectory
        self.trajectory = Trajectory()

        self.use_disturbance_compensation = use_disturbance_compensation
        self.params = params




        # Create do-mpc model
        self.model = self.create_model()

        # Create MPC controller
        self.mpc = self.create_mpc_controller(self.model)






        # Store last control input
        self.last_control = np.array([5.0, 0.0, 0.0])



    def create_model(self, symvar_type='MX'):
        """
        Create do-mpc model
        here is placeholder for symbolic variable type

        Returns:
            do_mpc.model.Model: Airship dynamics model
        """
        # Create model type (continuous time)
        model_type = 'continuous'
        model = Model(model_type, symvar_type)
        # model = do_mpc.model.Model(model_type, symvar_type)

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
        pos_ref = model.set_variable(var_type='_tvp', var_name='pos_ref', shape=(3, 1))
        att_ref = model.set_variable(var_type='_tvp', var_name='att_ref', shape=(3, 1))
        vel_ref = model.set_variable(var_type='_tvp', var_name='vel_ref', shape=(3, 1))
        omega_ref = model.set_variable(var_type='_tvp', var_name='omega_ref', shape=(3, 1))

        # Define disturbance variables (for Simulator)
        disturbance = model.set_variable(var_type='_tvp', var_name='disturbance', shape=(6, 1))


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

        # Wrap yaw angle error to [-pi, pi] directly
        # 使用 np.arctan2(np.sin(angle), np.cos(angle)) 将 yaw 角度 wrap 到 [-π, π] 范围内
        yaw_error = ca.atan2(ca.sin(att[2] - att_ref[2]), ca.cos(att[2] - att_ref[2]))
        attitude_error = ca.vertcat(att[0] - att_ref[0], att[1] - att_ref[1], yaw_error)

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

    def create_mpc_controller(self, silence_solver=True):
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
        pos_weight = 5
        att_weight = 20
        vel_weight = 10
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

        # Setting the penalty weight for the control input # 增加控制输入变化率的惩罚
        # in the objective function this is the "smoothness constraint" of control
        mpc.set_rterm(T=20, mu=20, nu=20) # means: rterm = 1 * T^2 + 1 * mu^2 + 1 * nu^2

        # Set constraints
        self._set_mpc_constraints(mpc)

        # # === Assign TVP (Time-Varying Parameters) ===
        tvp_template = mpc.get_tvp_template()


        def tvp_fun(t_now):
            """Update reference trajectory parameters."""
            yc, yc_dot, _, _, _ = self.trajectory.get_spiral_trajectory(t_now)

            tvp_current = tvp_template()

            # 使用正确的 TVP 访问方式 - 基于调试输出的结构
            tvp_current['_tvp', :, 'pos_ref'] = yc[0:3].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'att_ref'] = yc[3:6].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'vel_ref'] = yc_dot[0:3].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'omega_ref'] = yc_dot[3:6].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'disturbance'] = params.disturbance_delta(t_now).reshape(-1, 1).astype(float)

            return tvp_current


        mpc.set_tvp_fun(tvp_fun)

        # Complete MPC setup
        mpc.setup()



        # === 调试：打印约束信息 ===
        print("\n=== MPC Bounds Debug Info ===")
        try:
            bounds_info = mpc.bounds
            print("Velocity bounds structure:")

            # 正确的访问方式：(bound_type, var_type, var_name, index)
            print("Lower bounds:")
            try:
                lower_vel_0 = bounds_info['lower', '_x', 'vel', 0]
                lower_vel_1 = bounds_info['lower', '_x', 'vel', 1]
                lower_vel_2 = bounds_info['lower', '_x', 'vel', 2]
                print(f"  vel[0] lower: {lower_vel_0}")
                print(f"  vel[1] lower: {lower_vel_1}")
                print(f"  vel[2] lower: {lower_vel_2}")
            except Exception as e:
                print(f"  Error accessing lower velocity bounds: {e}")

            print("Upper bounds:")
            try:
                upper_vel_0 = bounds_info['upper', '_x', 'vel', 0]
                upper_vel_1 = bounds_info['upper', '_x', 'vel', 1]
                upper_vel_2 = bounds_info['upper', '_x', 'vel', 2]
                print(f"  vel[0] upper: {upper_vel_0}")
                print(f"  vel[1] upper: {upper_vel_1}")
                print(f"  vel[2] upper: {upper_vel_2}")
            except Exception as e:
                print(f"  Error accessing upper velocity bounds: {e}")

        except Exception as e:
            print(f"Error accessing bounds info: {e}")



        # 设置更平滑的初始控制猜测 - 使用悬停推力
        mpc.u0['T'] = 8  # 使用精确的悬停推力而不是固定值
        mpc.u0['mu'] = 0.0  # 初始无偏转
        mpc.u0['nu'] = 0.0  # 初始无偏转

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
        max_position = 5000.0                       # 适合飞艇的范围
        max_roll_pitch = np.deg2rad(25)             # 滚转/俯仰角约束
        max_yaw = np.pi                             # 偏航角可以 360°
        max_velocity = 20.0                         # 飞艇速度 (30-100 km/h)
        max_vertical_velocity = 8.0                  # 垂直速度更小
        max_angular_velocity = np.deg2rad(15)       # 角速度约束 (转弯)

        # Position constraints
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # Attitude constraints (Euler angles: [roll, pitch, yaw])
        mpc.bounds['lower', '_x', 'att', 0] = -max_roll_pitch  # Roll
        mpc.bounds['upper', '_x', 'att', 0] = max_roll_pitch
        mpc.bounds['lower', '_x', 'att', 1] = -max_roll_pitch  # Pitch
        mpc.bounds['upper', '_x', 'att', 1] = max_roll_pitch
        mpc.bounds['lower', '_x', 'att', 2] = -max_yaw         # Yaw
        mpc.bounds['upper', '_x', 'att', 2] = max_yaw

        # Linear velocity constraints - 分别设置每个分量
        mpc.bounds['lower', '_x', 'vel', 0] = -max_velocity      # u (X 方向速度)
        mpc.bounds['upper', '_x', 'vel', 0] = max_velocity
        mpc.bounds['lower', '_x', 'vel', 1] = -max_velocity      # v (Y 方向速度)
        mpc.bounds['upper', '_x', 'vel', 1] = max_velocity
        mpc.bounds['lower', '_x', 'vel', 2] = -max_vertical_velocity  # w (Z 方向速度)
        mpc.bounds['upper', '_x', 'vel', 2] = max_vertical_velocity


        # Angular velocity constraints - 分别设置每个分量
        max_roll_pitch_rate = np.deg2rad(10)  # Roll/Pitch角速度
        max_yaw_rate = np.deg2rad(8)          # Yaw 角速度

        # Angular velocity constraints
        mpc.bounds['lower', '_x', 'omega', 0] = -max_roll_pitch_rate  # p
        mpc.bounds['upper', '_x', 'omega', 0] = max_roll_pitch_rate
        mpc.bounds['lower', '_x', 'omega', 1] = -max_roll_pitch_rate  # q
        mpc.bounds['upper', '_x', 'omega', 1] = max_roll_pitch_rate
        mpc.bounds['lower', '_x', 'omega', 2] = -max_yaw_rate         # r
        mpc.bounds['upper', '_x', 'omega', 2] = max_yaw_rate

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
        # 使用 simulator 的 TVP template
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_now):
            """Update disturbance parameters for current simulation time"""
            yc, yc_dot, _, _, _ = self.trajectory.get_spiral_trajectory(t_now)

            tvp_current = tvp_template()

            tvp_current['pos_ref'] = yc[0:3].reshape(-1, 1).astype(float)
            tvp_current['att_ref'] = yc[3:6].reshape(-1, 1).astype(float)
            tvp_current['vel_ref'] = yc_dot[0:3].reshape(-1, 1).astype(float)
            tvp_current['omega_ref'] = yc_dot[3:6].reshape(-1, 1).astype(float)
            tvp_current['disturbance'] = params.disturbance_delta(t_now).reshape(-1, 1).astype(float)

            return tvp_current

        simulator.set_tvp_fun(tvp_fun)

        simulator.setup()

        print("do-mpc Simulator setup completed successfully")
        return simulator
