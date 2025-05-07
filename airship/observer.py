# observer.py
import numpy as np
from .utils import sig, R_block
from config import parameters as params


class FixedTimeDO:
    def __init__(self):
        # 从参数文件加载参数 (Load parameters from params file)
        self.l1 = params.l1
        self.l2 = params.l2
        self.l3 = params.l3
        self.l4 = params.l4
        self.l5 = params.l5
        self.beta1 = params.beta1
        self.beta2 = params.beta2
        self.M = params.M_cfg
        self.M_inv = params.M_inv

        # 初始化观测器状态 (Initialize observer states)
        self.z1_hat = np.zeros(6)
        self.e2_hat = np.zeros(6)  # Auxiliary state from Eq. 23
        self.delta_hat = np.zeros(6)  # Estimated disturbance delta

    def update(self, dt, e1, e2, tau, gamma, f_func):
        """更新扰动估计值"""
        # 获取当前旋转矩阵 (Get current rotation matrix)
        R = R_block(gamma)
        RM_inv = R @ self.M_inv

        # 更新辅助状态 e2_hat (Eq. 23)
        e2_hat_dot = -self.l1 * self.e2_hat + RM_inv @ tau
        self.e2_hat = self.e2_hat + e2_hat_dot * dt

        # 计算辅助变量 z1, z2 (Helper variables z1, z2)
        z1 = e2 - self.e2_hat
        z2 = self.l2 * z1

        # 更新观测器状态 z1_hat (Eq. 25 first line)
        # ż̂₁ = -l₁ẑ₁ + z₂/l₂ + l₃z₂ + l₄sig^β₁(ẑ₁) + l₅sig^β₂(ẑ₁)
        z1_hat_dot = -self.l1 * self.z1_hat + z1 + self.l3 * z2 + self.l4 * sig(self.z1_hat, self.beta1) + self.l5 * sig(self.z1_hat, self.beta2)
        self.z1_hat = self.z1_hat + z1_hat_dot * dt

        # 计算扰动估计 δ̂* (Eq. 25 second line, rearranged)
        # δ̂* = (z₂ + l₁l₂ẑ₁)/l₂
        delta_star_hat = (z2 + self.l1 * self.l2 * self.z1_hat) / self.l2

        # 计算 f(e1, e2) 项 (Calculate f(e1, e2) term)
        # f_term = f_func(e1, e2, gamma, R, Rc, xc, xc_dot) # Requires many args
        # The observer needs f(e1,e2) associated with Eq.22: δ* = l1e2 + f(e1,e2) + RM⁻¹δ
        # δ̂ = MR⁻¹(δ̂* - l₁e₂ - f(e1, e2)) (from Eq. 25 third line)
        f_term = f_func(e1, e2)  # Pass necessary calculated 'f' value

        # 计算最终扰动估计 δ̂ (Eq. 25 third line)
        self.delta_hat = self.M @ R.T @ (delta_star_hat - self.l1 * e2 - f_term)

        return self.delta_hat

    def get_estimate(self):
        return self.delta_hat
