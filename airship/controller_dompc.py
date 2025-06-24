"""
基于 do-mpc 的 NMPC 控制器实现 - 增强版支持 Simulator / NMPC controller implementation based on do-mpc - enhanced version supports Simulator`
使用 do-mpc 库简化 NMPC 控制器的开发，提供更稳定和高效的实现 / Use do-mpc library to simplify the development of NMPC controllers, providing a more stable and efficient implementation
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
    基于 do-mpc 的气艇 NMPC 控制器 - 增强版 / Airship NMPC controller based on do-mpc - enhanced version

    优势： / Advantages:
    1. 简化的 MPC 设置和配置 / Simplified MPC settings and configuration
    2. 内置的求解器配置和优化 / Built-in solver configuration and optimization
    3. 更好的数值稳定性 / Better numerical stability
    4. 自动处理约束和边界 / Automatic handling of constraints and boundaries
    5. 内置的可视化和分析工具 / Built-in visualization and analysis tools
    6. 支持 do-mpc Simulator 集成 / Support do-mpc Simulator integration
    """

    def __init__(self, use_disturbance_compensation=True, create_simulator=True):
        """
        初始化基于 do-mpc 的控制器 / Initialize the controller based on do-mpc

        Args:
            use_disturbance_compensation: 是否启用扰动补偿 / Whether to enable disturbance compensation
            create_simulator: 是否创建 do-mpc Simulator / Whether to create do-mpc Simulator
        """
        self.use_disturbance_compensation = use_disturbance_compensation
        self.params = params
        self.create_simulator = create_simulator

        # 初始化扰动观测器 / Initialize the disturbance observer
        if self.use_disturbance_compensation:
            self.disturbance_observer = NMPCDisturbanceObserver()
            self.disturbance_compensation_factor = getattr(params, 'do_compensation_gain', 0.9)
            self.last_disturbance_estimate = np.zeros(6)

        # 创建 do-mpc 模型 / Create the do-mpc model
        self.model = self._create_model()

        # 创建 MPC 控制器 / Create the MPC controller
        self.mpc = self._create_mpc_controller()

        # 创建估计器 / Create the estimator
        self.estimator = self._create_estimator()

        # 创建 Simulator (如果需要) / Create the Simulator (if needed)
        if self.create_simulator:
            self.simulator = self._create_simulator()
        else:
            self.simulator = None

        # 初始化控制器 / Initialize the controller
        self._setup_initial_conditions()

        # 存储上一次控制输入 / Store the last control input
        self.last_control = np.array([5.0, 0.0, 0.0])

    def _create_model(self):
        """
        创建 do-mpc 模型 - 增强版

        Returns:
            do_mpc.model.Model: 气艇动力学模型
        """
        # 创建模型类型（连续时间） / Create the model type (continuous time)
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # 定义状态变量 - 12 维状态向量 / Define the state variables - 12-dimensional state vector
        pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))      # 位置 [x, y, z] / Position [x, y, z]
        att = model.set_variable(var_type='_x', var_name='att', shape=(3, 1))      # 姿态 [phi, theta, psi] / Attitude [phi, theta, psi]
        vel = model.set_variable(var_type='_x', var_name='vel', shape=(3, 1))      # 线速度 [u, v, w] / Linear velocity [u, v, w]
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(3, 1))  # 角速度 [p, q, r] / Angular velocity [p, q, r]

        # 定义控制输入 - 3 维控制向量 / Define the control input - 3-dimensional control vector
        T = model.set_variable(var_type='_u', var_name='T')          # 推力大小
        mu = model.set_variable(var_type='_u', var_name='mu')        # 水平偏转角
        nu = model.set_variable(var_type='_u', var_name='nu')        # 垂直偏转角

        # 定义参考轨迹参数 / Define the reference trajectory parameters
        pos_ref = model.set_variable(var_type='_p', var_name='pos_ref', shape=(3, 1))
        att_ref = model.set_variable(var_type='_p', var_name='att_ref', shape=(3, 1))
        vel_ref = model.set_variable(var_type='_p', var_name='vel_ref', shape=(3, 1))
        omega_ref = model.set_variable(var_type='_p', var_name='omega_ref', shape=(3, 1))

        # 定义扰动变量（用于 Simulator） / Define the disturbance variable (for Simulator)
        disturbance = model.set_variable(var_type='_p', var_name='disturbance', shape=(6, 1))

        # 使用现有的符号化动力学模型 / Use the existing symbolic dynamic model
        symbolic_model = AirshipCasADiSymbolic(self.params)

        # 组合状态向量 / Combine the state vector
        X_state = ca.vertcat(pos, att, vel, omega)
        U_control = ca.vertcat(T, mu, nu)

        # 获取动力学方程（包含扰动） / Get the dynamic equations (including disturbance)
        X_dot = symbolic_model.rhs_symbolic(X_state, U_control, external_disturbance=disturbance)

        # 添加数值稳定性 - 限制导数的大小 / Add numerical stability - limit the size of the derivative
        max_derivative = 1e5
        X_dot = ca.fmin(ca.fmax(X_dot, -max_derivative), max_derivative)

        # 分解状态导数 / Decompose the state derivative
        pos_dot = X_dot[0:3]
        att_dot = X_dot[3:6]
        vel_dot = X_dot[6:9]
        omega_dot = X_dot[9:12]

        # 设置微分方程 / Set the differential equation
        model.set_rhs('pos', pos_dot)
        model.set_rhs('att', att_dot)
        model.set_rhs('vel', vel_dot)
        model.set_rhs('omega', omega_dot)

        # 设置状态表达式（用于目标函数） / Set the state expression (for the objective function)
        position_error = pos - pos_ref
        attitude_error = att - att_ref
        velocity_error = vel - vel_ref
        angular_error = omega - omega_ref

        # 辅助表达式 / Auxiliary expression
        model.set_expression('pos_error', position_error)
        model.set_expression('att_error', attitude_error)
        model.set_expression('vel_error', velocity_error)
        model.set_expression('ang_error', angular_error)

        # 添加测量输出（用于估计器） / Add the measurement output (for the estimator)
        model.set_expression('y_meas', X_state)  # 假设状态完全可测量 / Assume the state is completely measurable

        # 完成模型设置 / Complete the model setup
        model.setup()

        return model

    def _create_mpc_controller(self):
        """
        创建 MPC 控制器 - 增强版

        Returns:
            do_mpc.controller.MPC: 配置好的 MPC 控制器
        """
        mpc = do_mpc.controller.MPC(self.model)

        # MPC 设置
        setup_mpc = {
            'n_horizon': min(params.N_HORIZON, 8), # 预测时域的长度（即控制器预测未来多少步）
            'n_robust': 1, # 鲁棒性控制（用于处理模型不确定性）
            'open_loop': 0, # 是否启用开环控制（即不考虑当前状态）
            't_step': params.DT, # 时间步长（每一步的时间间隔）
            'state_discretization': 'collocation',  # 状态离散化方法
            'collocation_type': 'radau',  # 改用 legendre，更稳定
            'collocation_deg': 2,
            'collocation_ni': 1,  # 减少内部点数量
            'store_full_solution': False, #True,
            # 优化的求解器选项
            'nlpsol_opts': {
                'ipopt.print_level': 0,
                'ipopt.max_iter': 50,  # 减少最大迭代次数
                'ipopt.acceptable_tol': 1e-3,  # 放宽容差
                'ipopt.acceptable_obj_change_tol': 1e-3,
                'ipopt.tol': 1e-3,
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.hessian_approximation': 'limited-memory',
                'ipopt.limited_memory_max_history': 5,  # 减少历史记录
                'ipopt.alpha_for_y': 'primal',
                'ipopt.recalc_y': 'yes',
                'ipopt.max_wall_time': 3.0,  # 进一步限制求解时间
                'ipopt.warm_start_init_point': 'yes',  # 启用热启动
                'print_time': 0
            }
        }

        mpc.set_param(**setup_mpc)

        # 初始化参考轨迹和扰动参数的默认值，在仿真器中会根据实际的参考轨迹和扰动进行更新  / Initialize the default values of the reference trajectory and disturbance parameters, which will be updated in the simulator according to the actual reference trajectory and disturbance
        mpc.set_uncertainty_values(
            pos_ref=np.zeros((3, 1)),
            att_ref=np.zeros((3, 1)),
            vel_ref=np.zeros((3, 1)),
            omega_ref=np.zeros((3, 1)),
            disturbance=np.zeros((6, 1))  # 添加扰动参数 / Add disturbance parameters
        )

        # 定义控制器的目标函数 / Define the objective function of the controller
        # 缩放权重矩阵，避免数值问题 / Scale the weight matrix to avoid numerical problems
        Q_scaled = params.Q * 0.01  # 缩小状态权重 / Reduce the state weight
        Qf_scaled = params.Qf * 0.01
        R_scaled = params.R * 100.0  # 增大控制权重，促进平滑 / Increase the control weight to promote smoothness

        # 终端代价 - 表示预测时域末端的状态误差，不包含控制输入 / Terminal cost - represents the state error at the end of the prediction horizon, excluding the control input
        mterm = (self.model.aux['pos_error'].T @ Qf_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Qf_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Qf_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Qf_scaled[9:12, 9:12] @ self.model.aux['ang_error'])

        # 阶段代价 - 表示每一步的状态误差和控制输入的代价 / Stage cost - represents the state error and control input cost at each step
        lterm = (self.model.aux['pos_error'].T @ Q_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Q_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Q_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Q_scaled[9:12, 9:12] @ self.model.aux['ang_error'] +
                 self.model.u['T']**2 * R_scaled[0, 0] +
                 self.model.u['mu']**2 * R_scaled[1, 1] +
                 self.model.u['nu']**2 * R_scaled[2, 2])

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # 控制输入正则化 / Control input regularization
        mpc.set_rterm(T=0.1, mu=0.1, nu=0.1)

        # 设置控制输入和状态的约束条件 / Set the constraints for the control input and state
        self._set_mpc_constraints(mpc)

        # 完成 MPC 设置 / Complete the MPC setup
        mpc.setup()

        return mpc

    def _set_mpc_constraints(self, mpc):
        """设置 MPC 约束 / Set the MPC constraints

        Args:
            mpc: MPC 控制器 / MPC controller
        """
        # 控制输入约束 / Control input constraint
        mpc.bounds['lower', '_u', 'T'] = max(params.T_MIN, 0.1)  # 避免零推力 / Avoid zero thrust
        mpc.bounds['upper', '_u', 'T'] = params.T_MAX
        mpc.bounds['lower', '_u', 'mu'] = params.MU_MIN
        mpc.bounds['upper', '_u', 'mu'] = params.MU_MAX
        mpc.bounds['lower', '_u', 'nu'] = params.NU_MIN
        mpc.bounds['upper', '_u', 'nu'] = params.NU_MAX

        # 状态约束 / State constraint
        max_position = 200.0  # 减小到合理范围 / Reduce to a reasonable range
        max_angle = np.pi/2   # 限制姿态角防止奇点 / Limit the attitude angle to prevent singularity
        max_velocity = 30.0
        max_angular_velocity = np.pi/2  # 合理的角速度限制 / Reasonable angular velocity limit

        # 位置约束
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # 姿态约束 / Attitude constraint
        mpc.bounds['lower', '_x', 'att'] = -max_angle
        mpc.bounds['upper', '_x', 'att'] = max_angle
        mpc.bounds['lower', '_x', 'att', 1] = -max_angle/6  # 俯仰角 theta 限制 / Limit the pitch angle theta
        mpc.bounds['upper', '_x', 'att', 1] = max_angle/6

        # 速度约束 / Velocity constraint
        mpc.bounds['lower', '_x', 'vel'] = -max_velocity
        mpc.bounds['upper', '_x', 'vel'] = max_velocity
        mpc.bounds['lower', '_x', 'omega'] = -max_angular_velocity
        mpc.bounds['upper', '_x', 'omega'] = max_angular_velocity

    def _create_estimator(self):
        """
        创建状态估计器 / Create the state estimator
        想象你在驾驶一艘气艇，但你并不能直接看到气艇的所有状态（比如它的速度、姿态等）。 / Imagine you are driving a airship, but you cannot directly see all the states of the airship (like its speed, attitude, etc.).
        你只能通过传感器获取一些测量值，比如位置和速度。 / You can only get some measurements through sensors, like position and speed.
        这时候，你需要一个助手来根据这些测量值和系统的数学模型，推测出气艇的完整状态。 / At this time, you need a helper to infer the complete state of the airship based on these measurements and the mathematical model of the system.
        这个助手就是状态估计器。 / This helper is the state estimator.
        在这个方法中，do-mpc 提供了一个简单的状态反馈估计器（StateFeedback）， / In this method, do-mpc provides a simple state feedback estimator (StateFeedback),
        它假设系统的状态是完全可测量的（即传感器可以直接测量所有状态变量）。 / It assumes that the state of the system is completely measurable (i.e., the sensors can directly measure all state variables).
        因此，这个估计器的工作非常简单：直接将测量值作为系统的当前状态。 / Therefore, the work of this estimator is very simple: directly use the measurements as the current state of the system.

        Returns:
            do_mpc.estimator.StateFeedback: 状态反馈估计器
        """
        estimator = do_mpc.estimator.StateFeedback(self.model)
        return estimator

    def _create_simulator(self):
        """
        创建 do-mpc Simulator

        Returns:
            do_mpc.simulator.Simulator: 配置好的仿真器
        """
        if not self.create_simulator:
            return None

        # 创建仿真器对象 仿真器会基于这个模型计算系统的状态随时间的变化
        simulator = do_mpc.simulator.Simulator(self.model)

        # Simulator 设置
        setup_simulator = {
            't_step': params.DT,
            'integration_tool': 'cvodes', #'idas',  # 使用 IDAS 积分器
            'abstol': 1e-6,
            'reltol': 1e-4,
        }

        simulator.set_param(**setup_simulator)

        # 定义一个扰动函数，用于模拟外部扰动（如风力、气流等）
        def disturbance_func(t_now):
            """定义时变扰动
            这个函数会根据当前时间 t_now 返回一个扰动向量
            """
            try:
                delta = params.disturbance_delta(t_now)
                return delta.reshape(-1, 1)
            except Exception: # pylint: disable=broad-exception-caught
                return np.zeros((6, 1))

        # 设置参数模板 定义仿真器的参数模板，包括参考轨迹和扰动，这些参数会在仿真过程中动态更新
        p_template = simulator.get_p_template()
        p_template['pos_ref'] = np.zeros((3, 1))
        p_template['att_ref'] = np.zeros((3, 1))
        p_template['vel_ref'] = np.zeros((3, 1))
        p_template['omega_ref'] = np.zeros((3, 1))
        p_template['disturbance'] = disturbance_func

        def p_fun(t_now):
            """
            这个函数会在每一步仿真时被调用，返回当前时间的参数值
            """
            _ = t_now
            return p_template

        simulator.set_p_fun(p_fun)

        # 完成 Simulator 设置
        simulator.setup()

        return simulator

    def _setup_initial_conditions(self):
        """
        为控制器、估计器和仿真器设置初始状态和初始猜测值，以便系统能够从一个合理的初始状态开始运行 / Set the initial conditions and initial guesses for the controller, estimator, and simulator, so that the system can start running from a reasonable initial state
        1. 设置飞机的初始状态： / Set the initial state of the airship:
            比如飞机的初始位置、速度、姿态等。 / like the initial position, speed, attitude, etc.
        2. 给出一个初始的控制猜测： / Give an initial control guess:
            比如初始推力和偏转角，确保飞机不会一开始就失控。 / like the initial thrust and deflection angle, to ensure that the airship does not lose control at the beginning.
        这个方法就是在为飞行模拟器设置这些初始条件，让控制器、估计器和仿真器都知道从哪里开始。 / This method is to set these initial conditions for the flight simulator, so that the controller, estimator, and simulator know where to start.
        """
        # 设置飞艇的初始状态 包含系统的初始位置、姿态、速度和角速度 / Set the initial state of the airship, including the initial position, attitude, speed, and angular velocity of the system
        x0 = params.X0.copy()

        # 更严格的数值清理 确保初始状态的数值有效性，避免出现 NaN 或无穷值 / More strict numerical cleaning to ensure the numerical validity of the initial state, avoiding NaN or infinite values
        # x0 = np.nan_to_num(x0, nan=0.0, posinf=10.0, neginf=-10.0)

        # 限制初始角度，避免奇点 / Limit the initial angle to prevent singularity
        # x0[3:6] = np.clip(x0[3:6], -np.pi/2, np.pi/2)

        # 将初始状态向量重新组合为 do-mpc 期望的列向量格式
        _x0 = np.concatenate([
            x0[0:3].reshape(-1, 1),   # pos
            x0[3:6].reshape(-1, 1),   # att
            x0[6:9].reshape(-1, 1),   # vel
            x0[9:12].reshape(-1, 1)   # omega
        ])

        # 将初始状态 _x0 分别设置给控制器（mpc）、估计器（estimator）和仿真器（simulator）。 / Set the initial state _x0 to the controller (mpc), estimator (estimator), and simulator (simulator).
        # 这样每个组件都知道系统的初始状态。 / So that each component knows the initial state of the system.
        self.mpc.x0 = _x0
        self.estimator.x0 = _x0
        if self.simulator is not None:
            self.simulator.x0 = _x0

        # 改进的初始猜测 每次预测的初始猜测值是否相同？ / Improved initial guess - whether the initial guess value for each prediction is the same?
        # 第一次预测：会用到下面的初始猜测值 / First prediction: will use the initial guess value below
        # 第二次预测及后续预测： / Second prediction and subsequent predictions:
        #       在后续的预测中，控制器会根据上一次优化的结果更新初始猜测值。 / In subsequent predictions, the controller will update the initial guess value based on the result of the previous optimization.
        # 具体来说： / Specifically:
        #      1. 控制输入的初始猜测值会从上一次优化的结果中继承。 / The initial guess value of the control input will be inherited from the result of the previous optimization.
        #      2. 状态的初始猜测值会根据仿真器或实际系统的反馈进行更新。 / The initial guess value of the state will be updated based on the feedback from the simulator or the actual system.
        try:
            # 设置保守的初始控制输入猜测 / Set a conservative initial control input guess
            u0 = np.array([[5.0], [0.0], [0.0]])  # 稳定的推力，零力矩 / Stable thrust, zero torque

            # 为整个预测时域设置初始猜测值 / Set the initial guess value for the entire prediction horizon
            for k in range(self.mpc.settings.n_horizon):
                self.mpc.u0[k] = u0
                self.mpc.x0[k+1] = _x0  # 保持状态稳定 / Keep the state stable

            self.mpc.set_initial_guess()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Set initial guess failed: {e}") # pylint: disable=line-too-long

    def step(self, current_state, reference_trajectory, t_current=0.0):
        """
        执行一步 MPC 控制

        Args:
            current_state: 当前状态 [12x1]
            reference_trajectory: 参考轨迹字典
            t_current: 当前时间

        Returns:
            control_input: 控制输入 [T, mu, nu]
        """
        _ = t_current
        try:
            # 更新当前状态
            current_x = np.concatenate([
                current_state[0:3].reshape(-1, 1),   # pos
                current_state[3:6].reshape(-1, 1),   # att
                current_state[6:9].reshape(-1, 1),   # vel
                current_state[9:12].reshape(-1, 1)   # omega
            ])

            # 检查输入有效性
            if np.any(np.isnan(current_x)) or np.any(np.isinf(current_x)):
                print("Warning: The input state contains NaN or infinity value")
                return np.array([5.0, 0.0, 0.0])

            # 更新参考轨迹参数
            reference_params = {
                'pos_ref': reference_trajectory['position'].reshape(-1, 1),
                'att_ref': reference_trajectory['attitude'].reshape(-1, 1),
                'vel_ref': reference_trajectory['velocity'].reshape(-1, 1),
                'omega_ref': reference_trajectory['angular_velocity'].reshape(-1, 1)
            }

            # 添加扰动参数
            reference_params['disturbance'] = np.zeros((6, 1))

            # 将参考轨迹和扰动信息传递给控制器
            self.mpc.set_uncertainty_values(**reference_params)

            # 扰动补偿
            if self.use_disturbance_compensation:
                self._update_disturbance_compensation(current_state, reference_trajectory)

            # 执行 MPC 求解
            u_mpc = self.mpc.make_step(current_x)

            # 提取控制输入
            control_input = self._extract_control_input(u_mpc)

            # 保存控制输入
            self.last_control = control_input

            return control_input

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"MPC step failed: {e}")
            # 返回安全的默认控制
            safe_control = np.array([5.0, 0.0, 0.0])
            self.last_control = safe_control
            return safe_control

    def _update_disturbance_compensation(self, current_state, reference_trajectory):
        """更新扰动补偿 / Update the disturbance compensation

        """
        # 计算误差 / Calculate the error
        pos_error = current_state[0:3] - reference_trajectory['position']
        att_error = current_state[3:6] - reference_trajectory['attitude']
        vel_error = current_state[6:9] - reference_trajectory['velocity']
        ang_error = current_state[9:12] - reference_trajectory['angular_velocity']

        e1 = np.concatenate([pos_error, att_error])
        e2 = np.concatenate([vel_error, ang_error])

        # 更新扰动估计 / Update the disturbance estimate
        gamma = current_state[3:6]
        tau = thrust_params_to_force_torque(self.last_control, self.params.rp_r, self.params.rp_l)

        # 更新扰动观测器 / Update the disturbance observer
        delta_hat = self.disturbance_observer.update(params.DT, e1, e2, tau, gamma)
        self.last_disturbance_estimate = delta_hat

    def _extract_control_input(self, u_mpc):
        """
        将 MPC 优化得到的控制输入转换为可用的控制量 / Convert the control input obtained from MPC optimization to a usable control quantity
        1. 将 MPC 优化得到的控制输入转换为可用的控制量 / Convert the control input obtained from MPC optimization to a usable control quantity
        2. 检查控制输入的有效性，确保其不包含 NaN 或无穷大值 / Check the validity of the control input, ensuring that it does not contain NaN or infinite values
        3. 限制控制输入的范围，确保其符合物理约束 / Limit the range of the control input, ensuring that it conforms to the physical constraints
        4. 返回最终的控制输入
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

        # 检查控制输入是否包含无效值（NaN 或无穷大） / Check if the control input contains invalid values (NaN or infinity)
        if np.any(np.isnan(control_input)) or np.any(np.isinf(control_input)):
            print("Warning: The control input contains NaN or infinity value")
            return np.array([5.0, 0.0, 0.0])

        # 限制控制输入的范围，确保其符合物理约束 / Limit the range of the control input, ensuring that it conforms to the physical constraints
        control_input[0] = np.clip(control_input[0], params.T_MIN, params.T_MAX)
        control_input[1] = np.clip(control_input[1], params.MU_MIN, params.MU_MAX)
        control_input[2] = np.clip(control_input[2], params.NU_MIN, params.NU_MAX)

        return control_input

    def get_prediction(self):
        """
        1. 获取 MPC 预测结果 / Get the MPC prediction result
        2. 返回预测结果 / Return the prediction result
        3. 如果预测结果为空，返回 None / If the prediction result is empty, return None
        """
        try:
            if hasattr(self.mpc, 'data') and self.mpc.data is not None:
                # 使用公共接口获取预测数据 / Use the public interface to get the prediction data
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
        1. 获取当前扰动估计
        """
        if self.use_disturbance_compensation and self.last_disturbance_estimate is not None:
            try:
                # 处理 CasADi DM 对象 / Handle the CasADi DM object
                if hasattr(self.last_disturbance_estimate, 'full'):
                    return self.last_disturbance_estimate.full().flatten()
                # 处理 numpy 数组 / Handle the numpy array
                elif hasattr(self.last_disturbance_estimate, 'flatten'):
                    return self.last_disturbance_estimate.flatten()
                # 处理其他类型
                else:
                    return np.array(self.last_disturbance_estimate).flatten()
            except (AttributeError, ValueError):
                return np.zeros(6)
        else:
            return np.zeros(6)

    def reset(self):
        """重置控制器 / Reset the controller
        1. 重置扰动观测器 / Reset the disturbance observer
        2. 重置初始条件 / Reset the initial conditions
        3. 清除历史数据 / Clear the history data
        4. 重置控制器 / Reset the controller
        5. 重置估计器 / Reset the estimator
        6. 重置仿真器 / Reset the simulator
        """
        if self.use_disturbance_compensation:
            self.disturbance_observer.reset()
            self.last_disturbance_estimate = np.zeros(6)

        # 重置初始条件 / Reset the initial conditions
        self._setup_initial_conditions()

        # 清除历史数据 / Clear the history data
        self.mpc.reset_history()
        self.estimator.reset_history()
        if self.simulator is not None:
            self.simulator.reset_history()


# 辅助函数：轨迹格式转换 / Auxiliary function: trajectory format conversion
def convert_trajectory_format(yc, yc_dot):
    """
    将轨迹格式转换为 do-mpc 控制器所需格式 / Convert the trajectory format to the format required by the do-mpc controller

    Args:
        yc: 参考状态 [位置 (3) + 姿态 (3)] / Reference state [position (3) + attitude (3)]
        yc_dot: 参考状态导数 [位置导数 (3) + 姿态导数 (3)] / Reference state derivative [position derivative (3) + attitude derivative (3)]

    Returns:
        dict: 格式化的参考轨迹
    """
    return {
        'position': yc[0:3],
        'attitude': yc[3:6],
        'velocity': yc_dot[0:3],
        'angular_velocity': yc_dot[3:6]
    }
