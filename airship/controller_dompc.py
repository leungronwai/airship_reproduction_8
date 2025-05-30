"""
基于 do-mpc 的 NMPC 控制器实现
使用 do-mpc 库简化 NMPC 控制器的开发，提供更稳定和高效的实现
"""

# pylint: disable=invalid-name
# cspell:ignore dompc vertcat radau mterm lterm rterm

import numpy as np
import casadi as ca
import do_mpc

from config import parameters as params
from airship.model import AirshipCasADiSymbolic
from airship.thrust import thrust_params_to_force_torque
from airship.observer import NMPCDisturbanceObserver


class DoMPCAirshipController:
    """
    基于 do-mpc 的气艇 NMPC 控制器

    优势：
    1. 简化的 MPC 设置和配置
    2. 内置的求解器配置和优化
    3. 更好的数值稳定性
    4. 自动处理约束和边界
    5. 内置的可视化和分析工具
    """

    def __init__(self, use_disturbance_compensation=True):
        """
        初始化基于 do-mpc 的控制器

        Args:
            use_disturbance_compensation: 是否启用扰动补偿
        """
        self.use_disturbance_compensation = use_disturbance_compensation
        self.params = params

        # 初始化扰动观测器
        if use_disturbance_compensation:
            self.disturbance_observer = NMPCDisturbanceObserver()
            self.disturbance_compensation_factor = getattr(params, 'do_compensation_gain', 0.9)
            self.last_disturbance_estimate = np.zeros(6)

        # 创建 do-mpc 模型
        self.model = self._create_model()

        # 创建 MPC 控制器
        self.mpc = self._create_mpc_controller()

        # 创建估计器（可选）
        self.estimator = self._create_estimator()

        # 初始化控制器
        self._setup_initial_conditions()

        # 存储上一次控制输入
        self.last_control = np.array([5.0, 0.0, 0.0])

    def _create_model(self):
        """
        创建 do-mpc 模型

        Returns:
            do_mpc.model.Model: 气艇动力学模型
        """
        # 创建模型类型（连续时间）
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # 定义状态变量 - 12 维状态向量
        # 位置和姿态
        pos = model.set_variable(var_type='_x', var_name='pos', shape=(3, 1))      # 位置 [x, y, z]
        att = model.set_variable(var_type='_x', var_name='att', shape=(3, 1))      # 姿态 [phi, theta, psi]

        # 速度和角速度
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

        # 使用现有的符号化动力学模型
        symbolic_model = AirshipCasADiSymbolic(self.params)

        # 组合状态向量
        X_state = ca.vertcat(pos, att, vel, omega)
        U_control = ca.vertcat(T, mu, nu)

        # 获取动力学方程
        X_dot = symbolic_model.rhs_symbolic(X_state, U_control)

        # 分解状态导数
        pos_dot = X_dot[0:3]
        att_dot = X_dot[3:6]
        vel_dot = X_dot[6:9]
        omega_dot = X_dot[9:12]

        # 设置微分方程 the ODE for each state is set:
        model.set_rhs('pos', pos_dot)
        model.set_rhs('att', att_dot)
        model.set_rhs('vel', vel_dot)
        model.set_rhs('omega', omega_dot)

        # 设置状态表达式（用于目标函数）
        position_error = pos - pos_ref
        attitude_error = att - att_ref
        velocity_error = vel - vel_ref
        angular_error = omega - omega_ref

        model.set_expression('pos_error', position_error)
        model.set_expression('att_error', attitude_error)
        model.set_expression('vel_error', velocity_error)
        model.set_expression('ang_error', angular_error)

        # 完成模型设置
        model.setup()

        return model

    def _create_mpc_controller(self):
        """
        创建 MPC 控制器

        Returns:
            do_mpc.controller.MPC: 配置好的 MPC 控制器
        """
        mpc = do_mpc.controller.MPC(self.model)

        # MPC 设置
        setup_mpc = {
            'n_horizon': params.N_HORIZON,           # 预测时域
            'n_robust': 1,                           # 鲁棒性参数
            'open_loop': 0,                          # 闭环控制
            't_step': params.DT,                     # 时间步长
            'state_discretization': 'collocation',   # 离散化方法
            'collocation_type': 'radau',             # 配点法类型
            'collocation_deg': 2,                    # 配点法阶数
            'collocation_ni': 2,                     # 内部节点数
            'store_full_solution': True,             # 存储完整解
        }

        mpc.set_param(**setup_mpc)

        # 设置目标函数
        # 终端代价
        mterm = (self.model.aux['pos_error'].T @ params.Qf[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ params.Qf[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ params.Qf[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ params.Qf[9:12, 9:12] @ self.model.aux['ang_error'])

        # 阶段代价（包含状态误差和控制输入代价）
        lterm = (self.model.aux['pos_error'].T @ params.Q[0:3, 0:3] @ self.model.aux['pos_error'] +
                 self.model.aux['att_error'].T @ params.Q[3:6, 3:6] @ self.model.aux['att_error'] +
                 self.model.aux['vel_error'].T @ params.Q[6:9, 6:9] @ self.model.aux['vel_error'] +
                 self.model.aux['ang_error'].T @ params.Q[9:12, 9:12] @ self.model.aux['ang_error'] +
                 self.model.u['T']**2 * params.R[0, 0] +
                 self.model.u['mu']**2 * params.R[1, 1] +
                 self.model.u['nu']**2 * params.R[2, 2])

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # 设置约束
        # 控制输入约束
        mpc.bounds['lower', '_u', 'T'] = params.T_MIN
        mpc.bounds['upper', '_u', 'T'] = params.T_MAX
        mpc.bounds['lower', '_u', 'mu'] = params.MU_MIN
        mpc.bounds['upper', '_u', 'mu'] = params.MU_MAX
        mpc.bounds['lower', '_u', 'nu'] = params.NU_MIN
        mpc.bounds['upper', '_u', 'nu'] = params.NU_MAX

        # 状态约束（可选）
        max_position = 1e5
        max_angle = np.pi
        max_velocity = 50.0
        max_angular_velocity = np.pi

        # 位置约束
        mpc.bounds['lower', '_x', 'pos'] = -max_position
        mpc.bounds['upper', '_x', 'pos'] = max_position

        # 姿态约束
        mpc.bounds['lower', '_x', 'att'] = -max_angle
        mpc.bounds['upper', '_x', 'att'] = max_angle
        mpc.bounds['lower', '_x', 'att', 1] = -max_angle/2  # theta 限制避免奇点
        mpc.bounds['upper', '_x', 'att', 1] = max_angle/2

        # 速度约束
        mpc.bounds['lower', '_x', 'vel'] = -max_velocity
        mpc.bounds['upper', '_x', 'vel'] = max_velocity
        mpc.bounds['lower', '_x', 'omega'] = -max_angular_velocity
        mpc.bounds['upper', '_x', 'omega'] = max_angular_velocity

        # 完成 MPC 设置
        mpc.setup()

        return mpc

    def _create_estimator(self):
        """
        创建状态估计器（如果需要）

        Returns:
            do_mpc.estimator.StateFeedback: 状态反馈估计器
        """
        estimator = do_mpc.estimator.StateFeedback(self.model)
        return estimator

    def _setup_initial_conditions(self):
        """设置初始条件"""
        # 设置初始状态
        x0 = params.X0

        # 分解初始状态
        pos_init = x0[0:3].reshape(-1, 1)
        att_init = x0[3:6].reshape(-1, 1)
        vel_init = x0[6:9].reshape(-1, 1)
        omega_init = x0[9:12].reshape(-1, 1)

        # 设置 MPC 初始状态
        self.mpc.x0 = {
            'pos': pos_init,
            'att': att_init,
            'vel': vel_init,
            'omega': omega_init
        }

        # 设置估计器初始状态
        self.estimator.x0 = self.mpc.x0.copy()

        # 设置初始参考值（零参考）
        self.mpc.set_initial_guess()

    def step(self, current_state, reference_trajectory, t_current=0.0):
        """
        执行一步 MPC 控制

        Args:
            current_state: 当前状态 [12x1]
            reference_trajectory: 参考轨迹字典，包含 'position', 'attitude', 'velocity', 'angular_velocity'
            t_current: 当前时间

        Returns:
            control_input: 控制输入 [T, mu, nu]
        """
        try:
            # 更新当前状态
            current_x = {
                'pos': current_state[0:3].reshape(-1, 1),
                'att': current_state[3:6].reshape(-1, 1),
                'vel': current_state[6:9].reshape(-1, 1),
                'omega': current_state[9:12].reshape(-1, 1)
            }

            # 更新参考轨迹参数
            reference_params = {
                'pos_ref': reference_trajectory['position'].reshape(-1, 1),
                'att_ref': reference_trajectory['attitude'].reshape(-1, 1),
                'vel_ref': reference_trajectory['velocity'].reshape(-1, 1),
                'omega_ref': reference_trajectory['angular_velocity'].reshape(-1, 1)
            }

            # 扰动补偿
            if self.use_disturbance_compensation:
                # 计算误差
                pos_error = current_state[0:3] - reference_trajectory['position']
                att_error = current_state[3:6] - reference_trajectory['attitude']
                vel_error = current_state[6:9] - reference_trajectory['velocity']
                ang_error = current_state[9:12] - reference_trajectory['angular_velocity']

                e1 = np.concatenate([pos_error, att_error])
                e2 = np.concatenate([vel_error, ang_error])

                # 更新扰动估计
                gamma = current_state[3:6]

                # 获取上一次的控制输入（近似）
                tau = thrust_params_to_force_torque(self.last_control,
                                                  self.params.rp_r,
                                                  self.params.rp_l)

                # 更新扰动观测器
                delta_hat = self.disturbance_observer.update(
                    params.DT, e1, e2, tau, gamma
                )

                self.last_disturbance_estimate = delta_hat

            # 设置参数
            self.mpc.set_param(**reference_params)

            # 执行 MPC 求解
            u_mpc = self.mpc.make_step(current_x)

            # 提取控制输入
            control_input = np.array([
                float(u_mpc['T']),
                float(u_mpc['mu']),
                float(u_mpc['nu'])
            ])

            # 保存控制输入供下次使用
            self.last_control = control_input

            return control_input

        except Exception as e:     # pylint: disable=broad-except
            print(f"MPC 步骤失败：{e}")
            # 返回安全的默认控制输入
            return np.array([5.0, 0.0, 0.0])

    def get_prediction(self):
        """
        获取 MPC 预测结果

        Returns:
            dict: 预测状态和控制序列
        """
        try:
            prediction = self.mpc.data.prediction
            return {
                'states': prediction[('_x')],
                'controls': prediction[('_u')],
                'time': prediction[('_time')]
            }
        except:
            return {'states': None, 'controls': None, 'time': None}

    def get_current_disturbance_estimate(self):
        """获取当前扰动估计"""
        if self.use_disturbance_compensation:
            return self.last_disturbance_estimate.flatten()
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
