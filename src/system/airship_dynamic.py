# model.py
"""
飞艇动力学模型模块 (model.py)
参考文献 Error-constrained fixed-time trajectory tracking control for a stratospheric airship with disturbances
"""
# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot
# cspell:ignore arctan RUDT RUDB ELVL ELVR unmodeled casadi
# pylint: disable=too-many-locals


import casadi as ca
import numpy as np


class AirshipCasADiSymbolic:
    """
    飞艇符号化模型类
    """

    def __init__(self):
        # 物理参数

        self.m = 2934  # 质量 [kg]
        self.Volume = 35705  # 体积 [m^3]
        self.g = 9.74  # 重力加速度 [m/s^2]
        self.S_ref = self.Volume ** (2 / 3)  # 参考面积 [m^2]
        self.L_ref = 88.70  # 参考长度 [m]

        # 转动惯量
        self.Ix = 393187
        self.Iy = 1224880
        self.Iz = 939666
        self.Ixz = -62882
        self.I0 = np.diag([self.Ix, self.Iy, self.Iz])  # 惯性矩阵

        # 几何参数
        self.airship_a1 = 88.70 / 2  # 前椭球半长轴 [m]
        self.airship_a2 = 88.70 / 2  # 后椭球半长轴 [m]
        self.airship_b = 27.7 / 2  # 半短轴 [m]

        # 距离向量
        self.xg = 0  # 重心
        self.yg = 0
        self.zg = 2.66
        self.rg = np.array([self.xg, self.yg, self.zg])
        self.rb = np.array([0, 0, 0])  # 浮心
        self.rp_r = np.array([0 * self.airship_a1, self.airship_b, 3])  # 右侧推力点
        self.rp_l = np.array([0 * self.airship_a1, -self.airship_b, 3])  # 左侧推力点

        # 气动参数

        self.rho = 0.0822  # ~20km 高度的空气密度 [kg/m^3]

        # 气动系数
        self.C_l1 = 2.4e4 / 28.8
        self.C_m1, self.C_m2, self.C_m3, self.C_m4 = 7.7e4 / 28.8, 7.7e4 / 28.8, 7.7e4 / 28.8, 7.7e4 / 28.8
        self.C_n1, self.C_n2, self.C_n3, self.C_n4 = 7.7e4 / 28.8, 7.7e4 / 28.8, 7.7e4 / 28.8, 7.7e4 / 28.8
        self.C_x1, self.C_x2 = 657.0 / 28.8, 657.0 / 28.8
        self.C_y1, self.C_y2, self.C_y3, self.C_y4 = 657.0 / 28.8, 657.0 / 28.8, 657.0 / 28.8, 657.0 / 28.8
        self.C_z1, self.C_z2, self.C_z3, self.C_z4 = 657.0 / 28.8, 657.0 / 28.8, 657.0 / 28.8, 657.0 / 28.8

    # ---------- 旋转矩阵 ----------
    @staticmethod
    def R_zeta(gamma):
        """返回 3x3 旋转矩阵 R_zeta(gamma)"""
        phi, theta, psi = gamma[0], gamma[1], gamma[2]
        return ca.vertcat(
            ca.horzcat(
                ca.cos(theta) * ca.cos(psi),
                ca.sin(phi) * ca.sin(theta) * ca.cos(psi) - ca.cos(phi) * ca.sin(psi),
                ca.cos(phi) * ca.sin(theta) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi)
            ),
            ca.horzcat(
                ca.cos(theta) * ca.sin(psi),
                ca.sin(phi) * ca.sin(theta) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi),
                ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi)
            ),
            ca.horzcat(
                -ca.sin(theta),
                ca.sin(phi) * ca.cos(theta),
                ca.cos(phi) * ca.cos(theta)
            )
        )

    @staticmethod
    def R_gamma(gamma):
        """返回 3x3 姿态到角速度映射矩阵 R_gamma(gamma)"""
        phi, theta, psi = gamma[0], gamma[1], gamma[2]
        return ca.vertcat(
            ca.horzcat(1, ca.sin(phi) * ca.sin(theta) / ca.cos(theta), ca.cos(phi) * ca.sin(theta) / ca.cos(theta)),
            ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
            ca.horzcat(0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta))
        )

    @staticmethod
    def R_y_inv(gamma):
        """返回 3x3 矩阵 R_y_inv（若需要）"""
        phi, theta, psi = gamma[0], gamma[1], gamma[2]
        return ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(-ca.sin(phi) * ca.sin(theta) / ca.cos(theta), ca.cos(phi), -ca.sin(phi) / ca.cos(theta)),
            ca.horzcat(-ca.cos(phi) * ca.sin(theta) / ca.cos(theta), -ca.sin(phi), ca.cos(phi) / ca.cos(theta))
        )

    def rhs_symbolic(self, X, thrust_params, t=None, external_disturbance=None):
        """

        参数：
            X: 12x1 casadi SX 状态向量 [zeta, gamma, v, omega]
            Thrust_paras:
            - 3x1 [T, μ, v] 用于将直接推力参数转换为力/力矩
            t: 时间 (可选)
            external_disturbance: 可选的外部扰动 (6x1)

        返回：
            dX/dt 作为 casadi SX 12x1
        """

        _ = t

        # === 解构状态量 ===
        _zeta = X[0:3]  # ERF 中的位置
        gamma = X[3:6]  # 姿态 (欧拉角)
        v = X[6:9]  # BRF 中的线速度
        omega = X[9:12]  # BRF 中的角速度

        # ===========================================================================
        # ==                               运动学                                   ==
        # ===========================================================================
        R_block = ca.diagcat(self.R_zeta(gamma), self.R_gamma(gamma))
        y_dot = R_block @ ca.vertcat(v, omega)  # y_dot = [zeta_dot, gamma_dot] eq.(5)

        # ===========================================================================
        # ==                               动力学                                   ==
        # ===========================================================================
        # ===  (Calculate Added Mass/Inertia) ===
        k1, k2, k3 = 0.17, 0.83, 0.52
        m_air = self.rho * self.Volume  # displaced air mass
        M_added = m_air * ca.diag([k1, k2, k2])  # eq.[M'] eq 42
        I_added = m_air * ca.diag([0.0, k3, k3])  # eq.[I'_0] eq 42
        # === Skew matrix of CG ===
        rG_skew = ca.vertcat(
            ca.horzcat(0, -self.zg, self.yg),
            ca.horzcat(self.zg, 0, -self.xg),
            ca.horzcat(-self.yg, self.xg, 0)
        )
        # === Rigid Body Inertia Matrix ===
        M_rigid_add = ca.vertcat(
            ca.horzcat(self.m * ca.MX_eye(3) + M_added, -self.m * rG_skew),
            ca.horzcat(self.m * rG_skew, self.I0 + I_added)
        )

        # ===  # 计算科里奥利力和离心力项 (eq.10) ===
        N1 = (self.m * ca.MX.eye(3) + M_added) @ ca.cross(omega, v) + self.m * ca.cross(omega, ca.cross(omega, self.rg))
        N2 = ca.cross(omega, self.I0 @ omega) + self.m * ca.cross(self.rg, ca.cross(omega, v))
        N_term = ca.vertcat(N1, N2)

        # =======================重力作用力和力矩 gravity======================================
        Rz = self.R_zeta(gamma)
        fg_earth = ca.vertcat(0, 0, self.m * self.g)  # 地球坐标系中的重力
        fg_BRF = Rz.T @ fg_earth  # 将重力向量旋转到机体坐标系
        mg_BRF = ca.cross(self.rg, fg_BRF)  # 重力在 CG 处产生的力矩 (rg 是 CV->CG)

        # =======================浮力作用力和力矩======================================
        F_buoy_earth = ca.vertcat(0, 0, -self.rho * self.Volume * self.g)
        fb_BRF = Rz.T @ F_buoy_earth
        mb_BRF = ca.cross(self.rb, fb_BRF)  # 浮力在 CB 处产生的力矩 (假设作用点在 CV，因此力臂为 -rb)

        # =======================气动力和气动力矩======================================
        #    风速和相对速度计算
        V_wind_ERF = np.array([0.0, 0.0, 0.0])  # 地球坐标系中的风速
        V_wind_BRF = Rz.T @ V_wind_ERF  # 将风速转换到机体坐标系

        # 动压
        q_dyn = 0.5 * self.rho * ca.norm_2(v - V_wind_BRF) ** 2

        # 计算相对速度
        u_rel, v_rel_body, w_rel = (v - V_wind_BRF)[0], (v - V_wind_BRF)[1], (v - V_wind_BRF)[2]
        # 攻角和侧滑角
        alpha = ca.atan2(w_rel, u_rel)  # calculate relative wind speed magnitude (if not provided)
        V_rel_mag = ca.sqrt(u_rel ** 2 + v_rel_body ** 2 + w_rel ** 2)
        beta = ca.asin(v_rel_body / (V_rel_mag + 1e-6))  # calculate side slip angle

        # 使用提取的函数计算气动力和气动力矩
        X_a = -q_dyn * (self.C_x1 * ca.cos(alpha) ** 2 * ca.cos(beta) ** 2 + self.C_x2 * ca.sin(2 * alpha) * ca.sin(
            alpha / 2))
        Y_a = -q_dyn * (
                self.C_y1 * ca.cos(beta / 2) * ca.sin(2 * beta) + self.C_y2 * ca.sin(2 * beta) + self.C_y3 * ca.sin(
            beta) * ca.sin(ca.fabs(beta)))
        Z_a = -q_dyn * (self.C_z1 * ca.cos(alpha / 2) * ca.sin(2 * alpha) + self.C_z2 * ca.sin(
            2 * alpha) + self.C_z3 * ca.sin(alpha) * ca.sin(ca.fabs(alpha)))
        L_a = q_dyn * self.C_l1 * ca.sin(beta) * ca.sin(ca.fabs(beta))
        M_a = -q_dyn * (self.C_m1 * ca.cos(alpha / 2) * ca.sin(2 * alpha) + self.C_m2 * ca.sin(
            2 * alpha) + self.C_m3 * ca.sin(alpha) * ca.sin(ca.fabs(alpha)))
        N_a = q_dyn * (
                self.C_n1 * ca.cos(beta / 2) * ca.sin(2 * beta) + self.C_n2 * ca.sin(2 * beta) + self.C_n3 * ca.sin(
            beta) * ca.sin(ca.fabs(beta)))

        fa_BRF, ma_BRF = ca.vertcat(X_a, Y_a, Z_a), ca.vertcat(L_a, M_a, N_a)

        # =======================推力和推力矩 thrust_vetors_controller======================================
        # 将推力参数转换为力和力矩
        T_mag = thrust_params[0]
        mu = thrust_params[1]
        nu = thrust_params[2]

        # 计算右侧推力向量
        thrust_vector_r = ca.vertcat(
            T_mag * ca.cos(mu) * ca.cos(nu),
            T_mag * ca.sin(mu),
            T_mag * ca.cos(mu) * ca.sin(nu)
        )

        # 计算左侧推力向量
        thrust_vector_l = ca.vertcat(
            T_mag * ca.cos(mu) * ca.cos(nu),
            T_mag * ca.sin(mu),
            T_mag * ca.cos(mu) * ca.sin(nu)
        )

        # 总推力
        T_total = thrust_vector_r + thrust_vector_l

        # 计算力矩
        tau_r = ca.cross(self.rp_r, thrust_vector_r)
        tau_l = ca.cross(self.rp_l, thrust_vector_l)
        tau_vec = tau_r + tau_l

        # 组合力和力矩
        Thrust_Force_torque = ca.vertcat(T_total, tau_vec)

        Thrust_Force =  Thrust_Force_torque[0:3]  # BRF 中的推力向量 [340, 0, 0]  #
        Thrust_torque = Thrust_Force_torque[3:6]  # [0, 0, 0]

        # =======================合并力和力矩======================================
        F_forces = (fg_BRF + fb_BRF) + fa_BRF + Thrust_Force
        F_torques = (mg_BRF + mb_BRF) + ma_BRF + Thrust_torque
        F_term = ca.vertcat(F_forces, F_torques)

        # --- 如果提供了外部扰动，则添加 ---
        if external_disturbance is not None:
            F_term = F_term + external_disturbance

        # --- 动力学方程：Mx_dot + N = F ---
        x_dot = ca.inv(M_rigid_add) @ (F_term - N_term)

        # --- 合并状态导数 ---
        dxdt = ca.vertcat(y_dot, x_dot)

        return dxdt
