"""
NMPC Controller Implementation based on do-mpc
"""

# pylint: disable=invalid-name
# cspell:ignore dompc vertcat radau mterm lterm rterm ndarray fmin fmax idas abstol reltol Kalman symvar
# cspell: ignore nlpsol ipopt print_level max_iter acceptable_tol acceptable_obj_change_tol tol opcua casadi NMPC
# cspell: ignore cvodes mu_strategy hessian_approximation limited_memory_max_history alpha_for_y recalc_y max_wall_time print_time

import warnings

import casadi as ca
import do_mpc
# third-party library
import numpy as np
from do_mpc.model import Model

# local module
from src.system.airship_dynamic import AirshipCasADiSymbolic
from src.system.trajectory_ref import Trajectory

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
        # --- 控制器参数 ---
        self.DT = 1  # 仿真步长 (s)
        self.N_HORIZON = 18  # 预测时域长度

        # --- 权重矩阵 ---
        self.Q = np.diag([1, 1, 1,  # 位置权重
                          1, 1, 1,  # 姿态权重
                          0, 0, 0,  # 线速度权重
                          0, 0, 0])  # 角速度权重

        self.R = np.diag([0.1, 0.1, 0.1])  # 控制输入权重

        self.Qf = np.diag([1, 1, 1,  # 终端位置权重
                           1, 1, 1,  # 终端姿态权重
                           0, 0, 0,  # 终端线速度权重
                           0, 0, 0])  # 终端角速度权重

        # Create reference trajectory
        self.trajectory = Trajectory()

        self.use_disturbance_compensation = use_disturbance_compensation

        # Create do-mpc model
        self.model = self.create_model()

        # Create MPC controller
        self.mpc = self.create_mpc_controller(self.model)

    def disturbance_delta(self, t):
        """ Define external disturbance vector"""
        d = np.zeros(6)
        # d[0] = 0.5 + 2 * np.sin(0.1 * t)
        # d[1] = 0.4 + 1.5 * np.cos(0.1 * t)
        # d[2] = 0.6 + 1.5 * np.sin(0.1 * t)
        # d[3] = 1.5 + 2 * np.sin(0.1 * t)
        # d[4] = 1.5 + 1.5 * np.sin(0.1 * t)
        # d[5] = 1.5 + 2 * np.cos(0.1 * t)
        return d

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
        pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))  # Position [x, y, z]
        att = model.set_variable(var_type='_x', var_name='att', shape=(3, 1))  # Attitude [phi, theta, psi]
        vel = model.set_variable(var_type='_x', var_name='vel', shape=(3, 1))  # Linear velocity [u, v, w]
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(3, 1))  # Angular velocity [p, q, r]

        # Define control inputs - 3-dimensional control vector
        T = model.set_variable(var_type='_u', var_name='T')  # Thrust magnitude
        mu = model.set_variable(var_type='_u', var_name='mu')  # Horizontal deflection angle
        nu = model.set_variable(var_type='_u', var_name='nu')  # Vertical deflection angle

        # Define reference trajectory parameters
        pos_ref = model.set_variable(var_type='_tvp', var_name='pos_ref', shape=(3, 1))
        att_ref = model.set_variable(var_type='_tvp', var_name='att_ref', shape=(3, 1))
        vel_ref = model.set_variable(var_type='_tvp', var_name='vel_ref', shape=(3, 1))
        omega_ref = model.set_variable(var_type='_tvp', var_name='omega_ref', shape=(3, 1))

        # Define disturbance variables (for Simulator)
        disturbance = model.set_variable(var_type='_tvp', var_name='disturbance', shape=(6, 1))

        # Use existing symbolic dynamics model
        symbolic_model = AirshipCasADiSymbolic()

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
        print("do-mpc model created successfully.")

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
            'n_horizon': 10,
            'n_robust': 1,
            't_step': self.DT,
            'store_full_solution': True,
            'nlpsol_opts': {
                'ipopt.print_level': 0,  # 减少输出以避免过多的 NaN 警告
                'ipopt.max_iter': 300,
                'ipopt.tol': 1e-4,  # 放宽容差
                'ipopt.acceptable_tol': 1e-3,
                'ipopt.constr_viol_tol': 1e-4,
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.hessian_approximation': 'limited-memory',
                'print_time': 0,
                'verbose': False,
            }
        }

        mpc.set_param(**setup_mpc)

        # suppress solver output
        if silence_solver:
            mpc.settings.supress_ipopt_output()

        mpc.set_param(nlpsol_opts={'ipopt.print_level': 0})  # print_level = 0 means no print

        # 设置各分量的缩放因子
        pos_weight = 100
        att_weight = 100
        vel_weight = 0
        ang_weight = 0

        Q_scaled = self.Q.copy()
        Qf_scaled = self.Qf.copy()

        # 缩放各类状态误差
        Q_scaled[0:3, 0:3] *= pos_weight  # 增强位置误差权重（x, y, z）
        Q_scaled[3:6, 3:6] *= att_weight
        Q_scaled[6:9, 6:9] *= vel_weight
        Q_scaled[9:12, 9:12] *= ang_weight

        Qf_scaled[0:3, 0:3] *= pos_weight * 2  # 终端位置更强调
        Qf_scaled[3:6, 3:6] *= att_weight * 2
        Qf_scaled[6:9, 6:9] *= vel_weight * 2  # 减弱速度/角速度误差在代价函数中的影响
        Qf_scaled[9:12, 9:12] *= ang_weight * 2

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
                 200 * self.model.u['T'] ** 2 +
                 50 * self.model.u['mu'] ** 2 +
                 50 * self.model.u['nu'] ** 2)

        # + self.model.u['T']**2 * R_scaled[0, 0] + self.model.u['mu']**2 * R_scaled[1, 1] + self.model.u['nu']**2 * R_scaled[2, 2]

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Setting the penalty weight for the control input # 增加控制输入变化率的惩罚
        # in the objective function this is the "smoothness constraint" of control
        mpc.set_rterm(T=20, mu=20, nu=20)  # means: rterm = 1 * T^2 + 1 * mu^2 + 1 * nu^2

        # === Control input constraints ===
        # Thrust: avoid zero, ensure minimum lift
        mpc.bounds['lower', '_u', 'T'] = 5.0  # 最小推力
        mpc.bounds['upper', '_u', 'T'] = 20.0  # 最大推力

        # Deflection angles: mu and nu (typically ±30°)
        mpc.bounds['lower', '_u', 'mu'] = -np.deg2rad(30)  # 水平偏转角最小值 (-30°)
        mpc.bounds['upper', '_u', 'mu'] = np.deg2rad(30)  # 水平偏转角最大值 (+30°)
        mpc.bounds['lower', '_u', 'nu'] = -np.deg2rad(30)  # 垂直偏转角最小值 (-30°)
        mpc.bounds['upper', '_u', 'nu'] = np.deg2rad(30)  # 垂直偏转角最大值 (+30°)

        # === State constraints ===
        # Position constraints - 调整 Y 坐标约束以包含 0-500m 范围
        mpc.bounds['lower', '_x', 'pos', 0] = -5000.0  # x 位置最小值
        mpc.bounds['upper', '_x', 'pos', 0] = 5000.0  # x 位置最大值
        mpc.bounds['lower', '_x', 'pos', 1] = -100.0  # y 位置最小值（允许稍微偏离）
        mpc.bounds['upper', '_x', 'pos', 1] = 600.0  # y 位置最大值（允许稍微超出 500m）
        mpc.bounds['lower', '_x', 'pos', 2] = -25000.0  # z 位置最小值 (允许高达 25km)
        mpc.bounds['upper', '_x', 'pos', 2] = 1000.0  # z 位置最大值 (地面以下 1km)

        # Attitude constraints (Euler angles: [roll, pitch, yaw])
        mpc.bounds['lower', '_x', 'att', 0] = -np.deg2rad(25)  # Roll 最小值 (-25°)
        mpc.bounds['upper', '_x', 'att', 0] = np.deg2rad(25)  # Roll 最大值 (+25°)
        mpc.bounds['lower', '_x', 'att', 1] = -np.deg2rad(25)  # Pitch 最小值 (-25°)
        mpc.bounds['upper', '_x', 'att', 1] = np.deg2rad(25)  # Pitch 最大值 (+25°)
        mpc.bounds['lower', '_x', 'att', 2] = -np.pi  # Yaw 最小值 (-180°)
        mpc.bounds['upper', '_x', 'att', 2] = np.pi  # Yaw 最大值 (+180°)

        # Linear velocity constraints
        mpc.bounds['lower', '_x', 'vel', 0] = -20.0  # u (X 方向速度最小值)
        mpc.bounds['upper', '_x', 'vel', 0] = 20.0  # u (X 方向速度最大值)
        mpc.bounds['lower', '_x', 'vel', 1] = -20.0  # v (Y 方向速度最小值)
        mpc.bounds['upper', '_x', 'vel', 1] = 20.0  # v (Y 方向速度最大值)
        mpc.bounds['lower', '_x', 'vel', 2] = -3.0  # w (Z 方向速度最小值)
        mpc.bounds['upper', '_x', 'vel', 2] = 3.0  # w (Z 方向速度最大值)

        # Angular velocity constraints
        mpc.bounds['lower', '_x', 'omega', 0] = -np.deg2rad(10)  # p (Roll rate 最小值 -10°/s)
        mpc.bounds['upper', '_x', 'omega', 0] = np.deg2rad(10)  # p (Roll rate 最大值 +10°/s)
        mpc.bounds['lower', '_x', 'omega', 1] = -np.deg2rad(10)  # q (Pitch rate 最小值 -10°/s)
        mpc.bounds['upper', '_x', 'omega', 1] = np.deg2rad(10)  # q (Pitch rate 最大值 +10°/s)
        mpc.bounds['lower', '_x', 'omega', 2] = -np.deg2rad(8)  # r (Yaw rate 最小值 -8°/s)
        mpc.bounds['upper', '_x', 'omega', 2] = np.deg2rad(8)  # r (Yaw rate 最大值 +8°/s)

        # # 状态变量缩放
        # mpc.scaling['_x', 'pos', 0] = 1000.0  # x 位置缩放：1km -> 1
        # mpc.scaling['_x', 'pos', 1] = 1000.0  # y 位置缩放：1km -> 1
        # mpc.scaling['_x', 'pos', 2] = 1000.0  # z 位置缩放：1km -> 1
        #
        # mpc.scaling['_x', 'att', 0] = 1.0     # roll 缩放：保持原始值
        # mpc.scaling['_x', 'att', 1] = 1.0     # pitch 缩放：保持原始值
        # mpc.scaling['_x', 'att', 2] = 1.0     # yaw 缩放：保持原始值
        #
        # mpc.scaling['_x', 'vel', 0] = 10.0    # x 速度缩放：10m/s -> 1
        # mpc.scaling['_x', 'vel', 1] = 10.0    # y 速度缩放：10m/s -> 1
        # mpc.scaling['_x', 'vel', 2] = 5.0     # z 速度缩放：5m/s -> 1
        #
        # mpc.scaling['_x', 'omega', 0] = 0.1   # roll rate 缩放：0.1rad/s -> 1
        # mpc.scaling['_x', 'omega', 1] = 0.1   # pitch rate 缩放：0.1rad/s -> 1
        # mpc.scaling['_x', 'omega', 2] = 0.1   # yaw rate 缩放：0.1rad/s -> 1
        #
        # # 控制输入缩放
        # mpc.scaling['_u', 'T'] = 10.0         # 推力缩放：10N -> 1
        # mpc.scaling['_u', 'mu'] = 1.0         # 水平偏转角缩放：保持原始值
        # mpc.scaling['_u', 'nu'] = 1.0         # 垂直偏转角缩放：保持原始值

        # # === Assign TVP (Time-Varying Parameters) ===
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            """Update reference trajectory parameters."""
            # 使用直线轨迹替代螺旋轨迹
            yc, yc_dot, _, _, _ = self.trajectory.get_straight_line_trajectory(t_now)

            tvp_current = tvp_template()

            tvp_current['_tvp', :, 'pos_ref'] = yc[0:3].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'att_ref'] = yc[3:6].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'vel_ref'] = yc_dot[0:3].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'omega_ref'] = yc_dot[3:6].reshape(-1, 1).astype(float)
            tvp_current['_tvp', :, 'disturbance'] = self.disturbance_delta(t_now).reshape(-1, 1).astype(float)

            return tvp_current

        mpc.set_tvp_fun(tvp_fun)

        # Complete MPC setup
        mpc.setup()

        return mpc

    def create_simulator(self, model):
        """
        Create do-mpc Simulator with clean structure

        Returns:
            do_mpc.simulator.Simulator: Configured simulator
        """
        if not self.create_simulator:
            return None

        simulator = do_mpc.simulator.Simulator(model)

        # Configure simulator parameters
        simulator_params = {
            't_step': self.DT,
            'integration_tool': 'cvodes',
            'abstol': 1e-8,
            'reltol': 1e-6,
            'max_step_size': self.DT / 5,
            'min_step_size': self.DT / 500,
        }
        simulator.set_param(**simulator_params)

        # # Set initial state
        # 使用 simulator 的 TVP template
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_now):
            """Update disturbance parameters for current simulation time"""
            # 使用直线轨迹
            yc, yc_dot, _, _, _ = self.trajectory.get_straight_line_trajectory(t_now)

            tvp_template['pos_ref'] = yc[0:3].reshape(-1, 1).astype(float)
            tvp_template['att_ref'] = yc[3:6].reshape(-1, 1).astype(float)
            tvp_template['vel_ref'] = yc_dot[0:3].reshape(-1, 1).astype(float)
            tvp_template['omega_ref'] = yc_dot[3:6].reshape(-1, 1).astype(float)
            tvp_template['disturbance'] = self.disturbance_delta(t_now).reshape(-1, 1).astype(float)

            return tvp_template

        simulator.set_tvp_fun(tvp_fun)

        simulator.setup()

        print("do-mpc Simulator setup completed successfully")
        return simulator
