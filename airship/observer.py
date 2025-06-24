"""
Fixed-time Disturbance Observer (DO) for Airship
扰动观测器模块，支持 NMPC 控制策略下的扰动补偿。 / Disturbance observer module, supporting disturbance compensation under the NMPC control strategy.
"""


# cspell:ignore R_block coeff
# pylint: disable=invalid-name




import numpy as np
import casadi as ca
from config import parameters as params
from .utils import R_block



# 在 airship/observer.py 中修改 NMPCDisturbanceObserver 类

class NMPCDisturbanceObserver:
    """
    估计扰动：通过观测气艇的状态误差（位置误差、速度误差等），估计外部扰动的大小和方向。 / Estimate the disturbance: by observing the state error of the airship (position error, velocity error, etc.), estimate the magnitude and direction of the external disturbance.
    补偿扰动：将估计的扰动值反馈给控制器，用于补偿外部扰动对气艇轨迹跟踪的影响。 / Compensate the disturbance: feed the estimated disturbance value back to the controller, used to compensate the influence of the external disturbance on the airship trajectory tracking.
    支持 NMPC 控制器：提供符号化的扰动观测器方程，用于 NMPC 控制器的预测模型。 / Support NMPC controller: provide a symbolic disturbance observer equation for the prediction model of the NMPC controller.
    专门为 NMPC 控制器设计的扰动观测器，使用 CasADi 符号计算 / A disturbance observer specifically designed for the NMPC controller, using CasADi symbolic calculation
    """
    def __init__(self):
        # 基本观测器参数（从 params 模块获取或使用默认值） / Basic observer parameters (obtained from the params module or using default values)
        self.l1 = params.l1 if hasattr(params, 'l1') else 2.0
        self.l2 = params.l2 if hasattr(params, 'l2') else 1.0
        self.l3 = params.l3 if hasattr(params, 'l3') else 1.5
        self.l4 = params.l4 if hasattr(params, 'l4') else 2.0
        self.l5 = params.l5 if hasattr(params, 'l5') else 1.0
        self.beta1 = params.beta1 if hasattr(params, 'beta1') else 0.5
        self.beta2 = params.beta2 if hasattr(params, 'beta2') else 1.5
        self.M = params.M_cfg
        self.M_inv = params.M_inv

        # 初始化观测器状态 / Initialize the observer state
        self.z1_hat = np.zeros(6) # 位置和速度误差的估计 / Estimation of position and velocity errors
        self.e2_hat = np.zeros(6) # 速度误差的估计 / Estimation of velocity errors
        self.delta_hat = np.zeros(6) # 扰动估计 / Estimation of disturbance

        # 用于扰动滤波的参数 / Parameters for disturbance filtering
        self.filter_coeff = params.do_filter_coeff if hasattr(params, 'do_filter_coeff') else 0.7
        self.prev_delta_hat = np.zeros(6)

        # 扰动补偿增益 / Disturbance compensation gain
        self.compensation_gain = params.do_compensation_gain if hasattr(params, 'do_compensation_gain') else 0.9

        # 历史记录 / History
        self.history = []

        # 创建 CasADi 符号函数版本，用于 NMPC 预测 / Create a CasADi symbolic function version, used for NMPC prediction
        self._create_symbolic_observer()

    def _create_symbolic_observer(self):
        """创建 CasADi 符号版本的观测器方程，用于 NMPC 控制器的预测模型 / Create a CasADi symbolic version of the observer equation, used for the prediction model of the NMPC controller
        # 注意：本函数没有返回值，是因为它构造了一个 CasADi 符号函数 / Note: This function does not return a value because it constructs a CasADi symbolic function
        # observer_update_func, 并将其绑定到类变量 self.observer_update_func, / Bind the observer_update_func to the class variable self.observer_update_func,
        # 供其他方法（如 update()）调用。 / Note: This function does not return a value because it constructs a CasADi symbolic function
        args:
            None
        returns:
            None
        """
        # 定义符号变量 / Define symbolic variables
        # __import__() 是 Python 的底层函数，用于 动态导入模块，等价于 import casadi as ca
        # ca = __import__("casadi") # 导入 casadi 库
        e1_sym = ca.SX.sym("e1", 6) # 位置/姿态误差向量 / Position/attitude error vector
        e2_sym = ca.SX.sym("e2", 6) # 速度和角速度误差。 / Velocity and angular velocity error.
        tau_sym = ca.SX.sym("tau", 6) # 控制输入（力和力矩） / Control input (force and torque)
        gamma_sym = ca.SX.sym("gamma", 3) # 姿态角（欧拉角） / Attitude angle (Euler angle)
        dt_sym = ca.SX.sym("dt", 1) # 时间步长 / Time step
        z1_hat_sym = ca.SX.sym("z1_hat", 6) # 观测器的内部状态 / Internal state of the observer
        e2_hat_sym = ca.SX.sym("e2_hat", 6) # 观测器的内部状态 / Internal state of the observer

        # 构建观测器方程 / Build the observer equation
        R_sym = R_block(gamma_sym)  # 假设 R_block 已经支持 CasADi 符号 / Assume R_block already supports CasADi symbolic
        RM_inv_sym = R_sym @ self.M_inv # 旋转矩阵和质量矩阵的组合，用于将控制输入转换到速度误差的变化率。 / Combination of rotation matrix and mass matrix, used to convert the control input to the rate of change of velocity error.

        # e2_hat 更新 / Update e2_hat
        e2_hat_dot_sym = -self.l1 * e2_hat_sym + RM_inv_sym @ tau_sym
        e2_hat_next_sym = e2_hat_sym + e2_hat_dot_sym * dt_sym

        # z1 和 z2 计算 / Calculate z1 and z2
        z1_sym = e2_sym - e2_hat_sym
        z2_sym = self.l2 * z1_sym

        # z1_hat 更新 (完整版，包括非线性项) / Update z1_hat (full version, including nonlinear terms)
        # 定义符号版的 sig 函数 / Define the symbolic version of the sig function
        def sig_sym(x, alpha):
            """
            符号版的 sig 函数，用于替代 np.sign(x) * (np.abs(x) ** alpha)
            sig(x, alpha) = sign(x) * |x|^alpha
            计算带符号的幂函数，用于构造非线性项（如滑模控制等）
            Args:
                x (_type_): _description_
                alpha (_type_): _description_

            Returns:
                _type_: _description_
            """
            return ca.sign(x) * (ca.fabs(x) ** alpha)  # np.pow(ca.fabs(x), alpha)

        # z1_hat 更新 / Update z1_hat
        z1_hat_dot_sym = (-self.l1 * z1_hat_sym + z1_sym + self.l3 * z2_sym +
                          self.l4 * sig_sym(z1_hat_sym, self.beta1) +
                          self.l5 * sig_sym(z1_hat_sym, self.beta2))
        z1_hat_next_sym = z1_hat_sym + z1_hat_dot_sym * dt_sym

        # 计算扰动估计 / Calculate the disturbance estimate
        delta_star_hat_sym = (z2_sym + self.l1 * self.l2 * z1_hat_sym) / self.l2
        # f_term 在 NMPC 模型中通常由控制器内部处理 / f_term is usually handled internally by the controller in the NMPC model
        f_term_sym = ca.SX.zeros(6)  # 简化处理，由控制器提供实际值 / Simplified processing, provided by the controller

        # 计算扰动估计 / Calculate the disturbance estimate
        delta_hat_raw_sym = self.M @ ca.transpose(R_sym) @ (delta_star_hat_sym - self.l1 * e2_sym - f_term_sym)

        # 创建更新函数、生成符号化函数 / Create the update function, generate the symbolic function
        self.observer_update_func = ca.Function(
            'observer_update',
            [e1_sym, e2_sym, tau_sym, gamma_sym, dt_sym, z1_hat_sym, e2_hat_sym],
            [z1_hat_next_sym, e2_hat_next_sym, delta_hat_raw_sym],
            ['e1', 'e2', 'tau', 'gamma', 'dt', 'z1_hat', 'e2_hat'],
            ['z1_hat_next', 'e2_hat_next', 'delta_hat_raw']
        )

    def update(self, dt, e1, e2, tau, gamma, f_func=None):
        """
        更新扰动估计值 / Update the disturbance estimate

        参数： / Parameters:
            dt: 时间步长 / Time step
            e1: 位置/姿态误差向量 / Position/attitude error vector
            e2: 速度/角速度误差向量 / Velocity/angular velocity error vector
            tau: 控制输入（力和力矩） / Control input (force and torque)
            gamma: 姿态角（欧拉角） / Attitude angle (Euler angle)
            f_func: 计算 f(e1,e2) 的函数 (可选) / Function to calculate f(e1,e2) (optional)

        返回： / Returns:
            delta_hat_compensated: 可直接用于补偿的扰动向量 / Disturbance vector that can be directly used for compensation
        """
        # 更新内部状态 / Update the internal state
        z1_hat_next, e2_hat_next, delta_hat_raw = self.observer_update_func(
            e1, e2, tau, gamma, dt, self.z1_hat, self.e2_hat
        )

        self.z1_hat = z1_hat_next
        self.e2_hat = e2_hat_next

        # 如果提供了 f_func，则使用它计算 f_term / If f_func is provided, use it to calculate f_term
        if f_func is not None:
            # 计算旋转矩阵 / Calculate the rotation matrix
            R = R_block(gamma)
            z1 = e2 - self.e2_hat
            z2 = self.l2 * z1
            delta_star_hat = (z2 + self.l1 * self.l2 * self.z1_hat) / self.l2

            # 计算 f(e1, e2) 项 / Calculate the f(e1, e2) term
            f_term = f_func(e1, e2)

            # 重新计算扰动估计（考虑 f_term） / Recalculate the disturbance estimate (considering f_term)
            delta_hat_raw = self.M @ R.T @ (delta_star_hat - self.l1 * e2 - f_term)

        # 应用低通滤波 / Apply low-pass filter
        self.delta_hat = self.filter_coeff * self.prev_delta_hat + (1 - self.filter_coeff) * delta_hat_raw
        self.prev_delta_hat = self.delta_hat

        # 应用补偿增益/可直接用于补偿的扰动估计值 / Apply compensation gain / Disturbance estimate that can be directly used for compensation
        delta_hat_compensated = self.delta_hat * self.compensation_gain

        # 记录历史 / Record history
        self.history.append({
            'delta_hat_raw': np.array(delta_hat_raw).flatten(),
            'delta_hat_filtered': self.delta_hat.full().flatten(),
            'delta_hat_compensated': delta_hat_compensated.full().flatten()
        })

        return delta_hat_compensated

    def get_current_disturbance_estimate(self):
        """返回当前扰动估计值 / Return the current disturbance estimate

        Returns:
            _type_: _description_
        """
        return self.delta_hat

    def get_compensated_estimate(self):
        """返回考虑补偿系数的扰动估计值 / Return the disturbance estimate considering the compensation coefficient
           返回考虑补偿增益的扰动估计值 / Return the disturbance estimate considering the compensation gain
        """
        return self.delta_hat * self.compensation_gain

    def reset(self):
        """重置观测器状态 / Reset the observer state
        用于重置观测器的状态变量和历史记录 / Used to reset the state variables and history of the observer
        """
        self.z1_hat = np.zeros(6)
        self.e2_hat = np.zeros(6)
        self.delta_hat = np.zeros(6)
        self.prev_delta_hat = np.zeros(6)
        self.history = []
