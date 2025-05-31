"""
基于 do-mpc 的 NMPC 控制器实现 - 增强版支持 Simulator
使用 do-mpc 库简化 NMPC 控制器的开发，提供更稳定和高效的实现
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
    基于 do-mpc 的气艇 NMPC 控制器 - 增强版

    优势：
    1. 简化的 MPC 设置和配置
    2. 内置的求解器配置和优化
    3. 更好的数值稳定性
    4. 自动处理约束和边界
    5. 内置的可视化和分析工具
    6. 支持 do-mpc Simulator 集成
    """

    def __init__(self, use_disturbance_compensation=True, create_simulator=True):
        """
        初始化基于 do-mpc 的控制器

        Args:
            use_disturbance_compensation: 是否启用扰动补偿
            create_simulator: 是否创建 do-mpc Simulator
        """
        self.use_disturbance_compensation = use_disturbance_compensation
        self.params = params
        self.create_simulator = create_simulator

        # 初始化扰动观测器
        if use_disturbance_compensation:
            self.disturbance_observer = NMPCDisturbanceObserver()
            self.disturbance_compensation_factor = getattr(params, 'do_compensation_gain', 0.9)
            self.last_disturbance_estimate = np.zeros(6)

        # 创建 do-mpc 模型
        self.model = self._create_model()

        # 创建 MPC 控制器
        self.mpc = self._create_mpc_controller()

        # 创建估计器
        self.estimator = self._create_estimator()

        # 创建 Simulator (如果需要)
        if create_simulator:
            self.simulator = self._create_simulator()
        else:
            self.simulator = None

        # 初始化控制器
        self._setup_initial_conditions()

        # 存储上一次控制输入
        self.last_control = np.array([5.0, 0.0, 0.0])

    def _create_model(self):
        """
        创建 do-mpc 模型 - 增强版

        Returns:
            do_mpc.model.Model: 气艇动力学模型
        """
        # 创建模型类型（连续时间）
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # 定义状态变量 - 12 维状态向量
        pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))      # 位置 [x, y, z]
        att = model.set_variable(var_type='_x', var_name='att', shape=(3, 1))      # 姿态 [phi, theta, psi]
        vel = model.set_variable(var_type='_x', var_name='vel', shape=(3, 1))      # 线速度 [u, v, w]
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(3, 1))  # 角速度 [p, q, r]

        # 定义控制输入 - 3 维控制向量
        T = model.set_variable(var_type='_u', var_name='T')          # 推力大小
        mu = model.set_variable(var_type='_u', var_name='mu')        # 水平偏转角
        nu = model.set_variable(var_type='_u', var_name='nu')        # 垂直偏转角

        # 定义参考轨迹参数
        pos_ref = model.set_variable(var_type='_p', var_name='pos_ref', shape=(3, 1))
        att_ref = model.set_variable(var_type='_p', var_name='att_ref', shape=(3, 1))
        vel_ref = model.set_variable(var_type='_p', var_name='vel_ref', shape=(3, 1))
        omega_ref = model.set_variable(var_type='_p', var_name='omega_ref', shape=(3, 1))

        # 定义扰动变量（用于 Simulator）
        disturbance = model.set_variable(var_type='_p', var_name='disturbance', shape=(6, 1))


        # 使用现有的符号化动力学模型
        symbolic_model = AirshipCasADiSymbolic(self.params)

        # 组合状态向量
        X_state = ca.vertcat(pos, att, vel, omega)
        U_control = ca.vertcat(T, mu, nu)

        # 获取动力学方程（包含扰动）
        X_dot = symbolic_model.rhs_symbolic(X_state, U_control, external_disturbance=disturbance)

        # 添加数值稳定性 - 限制导数的大小
        max_derivative = 1e5
        X_dot = ca.fmin(ca.fmax(X_dot, -max_derivative), max_derivative)

        # 分解状态导数
        pos_dot = X_dot[0:3]
        att_dot = X_dot[3:6]
        vel_dot = X_dot[6:9]
        omega_dot = X_dot[9:12]

        # 设置微分方程
        model.set_rhs('pos', pos_dot)
        model.set_rhs('att', att_dot)
        model.set_rhs('vel', vel_dot)
        model.set_rhs('omega', omega_dot)

        # 设置状态表达式（用于目标函数）
        position_error = pos - pos_ref
        attitude_error = att - att_ref
        velocity_error = vel - vel_ref
        angular_error = omega - omega_ref

        # 辅助表达式
        model.set_expression('pos_error', position_error)
        model.set_expression('att_error', attitude_error)
        model.set_expression('vel_error', velocity_error)
        model.set_expression('ang_error', angular_error)

        # 添加测量输出（用于估计器）
        model.set_expression('y_meas', X_state)  # 假设状态完全可测量

        # 完成模型设置
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
            'n_horizon': min(params.N_HORIZON, 8),
            'n_robust': 1,
            'open_loop': 0,
            't_step': params.DT,
            'state_discretization': 'collocation',  # 回到 collocation
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

        # 设置不确定性值（扰动补偿）
        mpc.set_uncertainty_values(
            pos_ref=np.zeros((3, 1)),
            att_ref=np.zeros((3, 1)),
            vel_ref=np.zeros((3, 1)),
            omega_ref=np.zeros((3, 1)),
            disturbance=np.zeros((6, 1))  # 添加扰动参数
        )

        # 改进的目标函数 - 数值缩放
        # 缩放权重矩阵，避免数值问题
        Q_scaled = params.Q * 0.01  # 缩小状态权重
        Qf_scaled = params.Qf * 0.01
        R_scaled = params.R * 100.0  # 增大控制权重，促进平滑



        # 终端代价 - 只包含状态误差，不包含控制输入
        mterm = (self.model.aux['pos_error'].T @ Qf_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Qf_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Qf_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Qf_scaled[9:12, 9:12] @ self.model.aux['ang_error'])

        # 阶段代价 - 包含状态误差和控制输入
        lterm = (self.model.aux['pos_error'].T @ Q_scaled[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ Q_scaled[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ Q_scaled[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ Q_scaled[9:12, 9:12] @ self.model.aux['ang_error'] +
                 self.model.u['T']**2 * R_scaled[0, 0] +
                 self.model.u['mu']**2 * R_scaled[1, 1] +
                 self.model.u['nu']**2 * R_scaled[2, 2])

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # 控制输入正则化
        mpc.set_rterm(T=0.1, mu=0.1, nu=0.1)

        # 设置约束
        self._set_mpc_constraints(mpc)

        # 完成 MPC 设置
        mpc.setup()

        return mpc

    def _set_mpc_constraints(self, mpc):
        """设置 MPC 约束"""
        # 控制输入约束
        mpc.bounds['lower', '_u', 'T'] = max(params.T_MIN, 0.1)  # 避免零推力
        mpc.bounds['upper', '_u', 'T'] = params.T_MAX
        mpc.bounds['lower', '_u', 'mu'] = params.MU_MIN
        mpc.bounds['upper', '_u', 'mu'] = params.MU_MAX
        mpc.bounds['lower', '_u', 'nu'] = params.NU_MIN
        mpc.bounds['upper', '_u', 'nu'] = params.NU_MAX

        # 状态约束
        max_position = 200.0  # 减小到合理范围
        max_angle = np.pi/2   # 限制姿态角防止奇点
        max_velocity = 30.0
        max_angular_velocity = np.pi/2  # 合理的角速度限制

        # 位置约束
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # 姿态约束
        mpc.bounds['lower', '_x', 'att'] = -max_angle
        mpc.bounds['upper', '_x', 'att'] = max_angle
        mpc.bounds['lower', '_x', 'att', 1] = -max_angle/6  # 俯仰角 theta 限制
        mpc.bounds['upper', '_x', 'att', 1] = max_angle/6

        # 速度约束
        mpc.bounds['lower', '_x', 'vel'] = -max_velocity
        mpc.bounds['upper', '_x', 'vel'] = max_velocity
        mpc.bounds['lower', '_x', 'omega'] = -max_angular_velocity
        mpc.bounds['upper', '_x', 'omega'] = max_angular_velocity

    def _create_estimator(self):
        """
        创建状态估计器

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

        simulator = do_mpc.simulator.Simulator(self.model)

        # Simulator 设置
        setup_simulator = {
            't_step': params.DT,
            'integration_tool': 'cvodes', #'idas',  # 使用 IDAS 积分器
            'abstol': 1e-6,
            'reltol': 1e-4,
        }

        simulator.set_param(**setup_simulator)

        # 设置扰动函数
        def disturbance_func(t_now):
            """定义时变扰动"""
            try:
                delta = params.disturbance_delta(t_now)
                return delta.reshape(-1, 1)
            except Exception: # pylint: disable=broad-exception-caught
                return np.zeros((6, 1))

        # 设置参数函数
        p_template = simulator.get_p_template()
        p_template['pos_ref'] = np.zeros((3, 1))
        p_template['att_ref'] = np.zeros((3, 1))
        p_template['vel_ref'] = np.zeros((3, 1))
        p_template['omega_ref'] = np.zeros((3, 1))
        p_template['disturbance'] = disturbance_func

        def p_fun(t_now):
            """参数函数"""
            _ = t_now
            return p_template

        simulator.set_p_fun(p_fun)

        # 完成 Simulator 设置
        simulator.setup()

        return simulator

    def _setup_initial_conditions(self):
        """设置初始条件"""
        # 设置初始状态
        x0 = params.X0.copy()

        # 更严格的数值清理
        # x0 = np.nan_to_num(x0, nan=0.0, posinf=10.0, neginf=-10.0)

        # 限制初始角度，避免奇点
        # x0[3:6] = np.clip(x0[3:6], -np.pi/2, np.pi/2)

        # 将状态向量重新组合为 do-mpc 期望的格式
        _x0 = np.concatenate([
            x0[0:3].reshape(-1, 1),   # pos
            x0[3:6].reshape(-1, 1),   # att
            x0[6:9].reshape(-1, 1),   # vel
            x0[9:12].reshape(-1, 1)   # omega
        ])

        # 设置各组件的初始状态
        self.mpc.x0 = _x0
        self.estimator.x0 = _x0
        if self.simulator is not None:
            self.simulator.x0 = _x0

        # 改进的初始猜测
        try:
            # 设置保守的初始控制猜测
            u0 = np.array([[5.0], [0.0], [0.0]])  # 稳定的推力，零力矩

            # 为整个预测长度设置初始猜测
            for k in range(self.mpc.settings.n_horizon):
                self.mpc.u0[k] = u0
                self.mpc.x0[k+1] = _x0  # 保持状态稳定

            self.mpc.set_initial_guess()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"设置初始猜测失败：{e}")

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
                print("警告：输入状态包含 NaN 或无穷大值")
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

            # 设置参考轨迹
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
            print(f"MPC 步骤失败：{e}")
            # 返回安全的默认控制
            safe_control = np.array([5.0, 0.0, 0.0])
            self.last_control = safe_control
            return safe_control

    def _update_disturbance_compensation(self, current_state, reference_trajectory):
        """更新扰动补偿"""
        # 计算误差
        pos_error = current_state[0:3] - reference_trajectory['position']
        att_error = current_state[3:6] - reference_trajectory['attitude']
        vel_error = current_state[6:9] - reference_trajectory['velocity']
        ang_error = current_state[9:12] - reference_trajectory['angular_velocity']

        e1 = np.concatenate([pos_error, att_error])
        e2 = np.concatenate([vel_error, ang_error])

        # 更新扰动估计
        gamma = current_state[3:6]
        tau = thrust_params_to_force_torque(self.last_control, self.params.rp_r, self.params.rp_l)

        # 更新扰动观测器
        delta_hat = self.disturbance_observer.update(params.DT, e1, e2, tau, gamma)
        self.last_disturbance_estimate = delta_hat

    def _extract_control_input(self, u_mpc):
        """提取控制输入"""
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
            print("警告：无法正确提取控制输入")
            control_input = np.array([5.0, 0.0, 0.0])

        # 检查有效性并限制范围
        if np.any(np.isnan(control_input)) or np.any(np.isinf(control_input)):
            print("警告：控制输入包含 NaN 或无穷大值")
            return np.array([5.0, 0.0, 0.0])

        control_input[0] = np.clip(control_input[0], params.T_MIN, params.T_MAX)
        control_input[1] = np.clip(control_input[1], params.MU_MIN, params.MU_MAX)
        control_input[2] = np.clip(control_input[2], params.NU_MIN, params.NU_MAX)

        return control_input

    def get_prediction(self):
        """获取 MPC 预测结果"""
        try:
            if hasattr(self.mpc, 'data') and self.mpc.data is not None:
                # 使用公共接口获取预测数据
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
        """获取当前扰动估计"""
        if self.use_disturbance_compensation and self.last_disturbance_estimate is not None:
            try:
                # 处理 CasADi DM 对象
                if hasattr(self.last_disturbance_estimate, 'full'):
                    return self.last_disturbance_estimate.full().flatten()
                # 处理 numpy 数组
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
        """重置控制器"""
        if self.use_disturbance_compensation:
            self.disturbance_observer.reset()
            self.last_disturbance_estimate = np.zeros(6)

        # 重置初始条件
        self._setup_initial_conditions()

        # 清除历史数据
        self.mpc.reset_history()
        self.estimator.reset_history()
        if self.simulator is not None:
            self.simulator.reset_history()


# 辅助函数：轨迹格式转换
def convert_trajectory_format(yc, yc_dot):
    """
    将轨迹格式转换为 do-mpc 控制器所需格式

    Args:
        yc: 参考状态 [位置 (3) + 姿态 (3)]
        yc_dot: 参考状态导数 [位置导数 (3) + 姿态导数 (3)]

    Returns:
        dict: 格式化的参考轨迹
    """
    return {
        'position': yc[0:3],
        'attitude': yc[3:6],
        'velocity': yc_dot[0:3],
        'angular_velocity': yc_dot[3:6]
    }
