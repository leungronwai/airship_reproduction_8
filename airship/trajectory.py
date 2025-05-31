"""
Trajectory generation module (trajectory.py)
"""
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# cspell:ignore R_zeta R_y_inv Rc_z Rc_y_inv ddot arctan2 linalg xdot phiddot phidot    psiddot
# cspell:ignore phidot phiddot psidot psiddot thetaddot ydot

import numpy as np
from airship.utils import R_zeta, R_y_inv


class Trajectory:
    """
    轨迹生成模块


        参数：
            t: 当前时间
            start_point: 起点坐标 [x, y, z]，默认为原点
            end_point: 终点坐标 [x, y, z]，默认为 [5000, 5000, -19000]
            speed: 飞行速度，单位 m/s
            hover_at_end: 到达终点后是否悬停，否则继续沿直线飞行

        返回：
            yc, yc_dot, yc_ddot, xc, xc_dot: 期望状态及导数


        说明：
            - yc:

                表示期望状态向量，包含飞艇的期望位置和姿态。
                具体包括：
                    位置：[x, y, z]，即飞艇在空间中的期望位置。
                    姿态：[φ, θ, ψ]，即飞艇的期望姿态角（横滚角、俯仰角、航向角）。
            - yc_dot:

                表示期望状态的一阶导数，即期望速度向量。
                具体包括：
                    线速度：[vx, vy, vz]，即飞艇在空间中的期望线速度。
                    角速度：[ωφ, ωθ, ωψ]，即飞艇的期望角速度。
            - yc_ddot:

                表示期望状态的二阶导数，即期望加速度向量。
                具体包括：
                    线加速度：[ax, ay, az]，即飞艇在空间中的期望线加速度。
                    角加速度：[αφ, αθ, αψ]，即飞艇的期望角加速度。
            - xc:

                表示控制指令向量，包含期望的线速度和角速度。
                具体包括：
                    线速度指令：[vx, vy, vz]，即飞艇的期望线速度。
                    角速度指令：[ωφ, ωθ, ωψ]，即飞艇的期望角速度。
            - xc_dot:

                表示控制指令的一阶导数，即控制指令的变化率。
                具体包括：
                    线速度变化率：[dvx/dt, dvy/dt, dvz/dt]，即线速度的时间变化率。
                    角速度变化率：[dωφ/dt, dωθ/dt, dωψ/dt]，即角速度的时间变化率。
    """
    def __init__(self):
        pass  # No specific initialization needed for this trajectory

    # ┌─────────────────────────────────────────────────────┐
    # │          螺旋轨迹函数 spiral                          │
    # └─────────────────────────────────────────────────────┘

    def get_spiral_trajectory(self, t):

        """
        生成一个螺旋轨迹，带有高度变化
        args:
            t: 当前时间
        returns:
            yc, yc_dot, yc_ddot, xc, xc_dot
        """

        dt_small = 1e-4

        # --- 轨迹参数 ---
        omega = 0.07  # 角速度
        r = 1500  # 半径
        h_max = 2000  # 最大高度

        # 在起始时刻打印起始点信息
        if abs(t) < 1e-3:  # t 接近 0 时
            start_x = r * np.cos(0)  # = r = 1500
            start_y = r * np.sin(0)  # = 0
            start_z = h_max * (1 - np.exp(0))  # = 0
            print(f"【螺旋轨迹】起始点位置：[{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (米)")
            print(f"【螺旋轨迹】轨迹参数：半径={r}m, 最大高度={h_max}m, 角速度={omega}rad/s")

        # --- 直接计算位置、速度和加速度 ---
        theta = omega * t
        # 位置
        xd = r * np.cos(theta)
        yd = r * np.sin(theta)
        zd = h_max * (1 - np.exp(-theta / 10))
        pos = np.array([xd, yd, zd])

        # 速度
        xd_dot = -r * omega * np.sin(theta)
        yd_dot = r * omega * np.cos(theta)
        zd_dot = h_max * (1 / 10) * np.exp(-theta / 10) * omega
        vel = np.array([xd_dot, yd_dot, zd_dot])

        # 加速度
        xd_ddot = -r * omega**2 * np.cos(theta)
        yd_ddot = -r * omega**2 * np.sin(theta)
        zd_ddot = -h_max * (1 / 10) * omega**2 * np.exp(-theta / 10)
        acc = np.array([xd_ddot, yd_ddot, zd_ddot])

        # 构造位置和速度向量
        zeta_d = pos
        zeta_d_dot = vel
        zeta_d_ddot = acc

        # 计算姿态
        phi_d = 0.0  # 保持零横滚
        theta_d = np.arctan2(-vel[2], np.sqrt(vel[0] ** 2 + vel[1] ** 2))  # 俯仰角
        psi_d = np.arctan2(vel[1], vel[0])  # 航向角
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # 使用数值差分获取姿态导数
        _, gamma_d_plus = self.get_spiral_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_spiral_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # 组合 yc、yc_dot
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # 速度指令 vc, wc
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc))

        # xc_dot 通过符号化导数简化近似
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # 简化处理，假设角速度变化率较小
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # yc_ddot 同样简化处理
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_spiral_pos_att(self, t):
        """
        计算螺旋轨迹在时间 t 的位置和姿态，用于计算导数
        避免递归调用 define_spiral_trajectory
        """
        # --- 轨迹参数 ---
        omega = 0.07  # 角速度 (rad/s)
        r = 1500  # 基础半径 (m)
        h_max = 2000  # 最大高度 (m)

        # --- 位置计算 ---
        theta = omega * t
        xd = r * np.cos(theta)
        yd = r * np.sin(theta)
        zd = h_max * (1 - np.exp(-theta / 10))
        zeta_d = np.array([xd, yd, zd])

        # --- 速度计算（用于姿态确定） ---
        xd_dot = -r * omega * np.sin(theta)
        yd_dot = r * omega * np.cos(theta)
        zd_dot = h_max * (1 / 10) * np.exp(-theta / 10) * omega

        # --- 姿态计算 ---
        phi_d = 0.0  # 保持零横滚
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))  # 俯仰角
        psi_d = np.arctan2(yd_dot, xd_dot)  # 航向角
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ********************* 8 字形轨迹函数 *********************

    def get_figure8_trajectory(self, t):
        """
        生成一个水平 8 字形轨迹，带有平滑的高度变化
        返回 8 字形轨迹的期望状态及其导数

        参数：
            t: 当前时间

        返回：
            yc, yc_dot, yc_ddot, xc, xc_dot: 与 get_desired_state 相同的输出格式
        """
        dt_small = 1e-4

        # --- 轨迹参数 ---
        a = 3000  # 8 字形的宽度
        b = 2000  # 8 字形的高度
        omega = 0.003  # 角速度，控制飞艇在轨迹上的移动速度
        h_center = -19000  # 中心高度
        h_amp = 500  # 高度振荡幅度
        omega_h = 0.002  # 高度变化的角速度

        # 在起始时刻打印起始点信息
        if abs(t) < 1e-3:  # t 接近 0 时
            start_x = a * np.sin(0)  # = 0
            start_y = b * np.sin(0) * np.cos(0)  # = 0
            start_z = h_center + h_amp * np.sin(0)  # = h_center = -19000
            print(f"【8 字形轨迹】起始点位置：[{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (米)")
            print(f"【8 字形轨迹】轨迹参数：宽度={a}m, 高度={b}m, 中心高度={h_center}m, 角速度={omega}rad/s")

        # --- 位置计算 ---
        # 8 字形的参数方程
        xd = a * np.sin(omega * t)
        yd = b * np.sin(omega * t) * np.cos(omega * t)
        zd = h_center + h_amp * np.sin(omega_h * t)
        zeta_d = np.array([xd, yd, zd])

        # --- 速度计算 ---
        xd_dot = a * omega * np.cos(omega * t)
        yd_dot = b * omega * (np.cos(omega * t) * np.cos(omega * t)
                              - np.sin(omega * t) * np.sin(omega * t)
                              )
        zd_dot = h_amp * omega_h * np.cos(omega_h * t)
        zeta_d_dot = np.array([xd_dot, yd_dot, zd_dot])

        # --- 加速度计算（使用数值差分） ---
        # 计算 t+dt 时刻的速度
        xd_dot_plus = a * omega * np.cos(omega * (t + dt_small))
        yd_dot_plus = (
            b * omega * (np.cos(omega * (t + dt_small)) * np.cos(omega * (t + dt_small))
                         - np.sin(omega * (t + dt_small)) * np.sin(omega * (t + dt_small))
                         )
        )
        zd_dot_plus = h_amp * omega_h * np.cos(omega_h * (t + dt_small))

        # 计算 t-dt 时刻的速度
        xd_dot_minus = a * omega * np.cos(omega * (t - dt_small))
        yd_dot_minus = (
            b * omega * (np.cos(omega * (t - dt_small)) * np.cos(omega * (t - dt_small))
               - np.sin(omega * (t - dt_small)) * np.sin(omega * (t - dt_small)))
        )
        zd_dot_minus = h_amp * omega_h * np.cos(omega_h * (t - dt_small))

        # 使用中心差分计算加速度
        zeta_d_ddot = np.array(
            [
                (xd_dot_plus - xd_dot_minus) / (2 * dt_small),
                (yd_dot_plus - yd_dot_minus) / (2 * dt_small),
                (zd_dot_plus - zd_dot_minus) / (2 * dt_small),
            ]
        )

        # --- 姿态计算 ---
        # 计算期望航向角（切线方向）
        phi_d = 0.0  # 保持零横滚
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))  # 俯仰角
        psi_d = np.arctan2(yd_dot, xd_dot)  # 航向角
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- 姿态导数计算 ---
        # 使用数值差分获取姿态导数
        _, gamma_d_plus = self.get_figure8_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_figure8_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # --- 组合 yc、yc_dot ---
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # --- 速度指令 vc, wc ---
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc))

        # --- xc_dot 通过符号化导数简化近似 ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # 简化处理，假设角速度变化率较小
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # --- yc_ddot 同样简化处理 ---
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_figure8_pos_att(self, t):
        """
        仅计算 8 字形轨迹在时间 t 的位置和姿态，用于计算导数
        避免递归调用 get_figure8_trajectory
        """
        # --- 轨迹参数 ---
        a = 3000  # 8 字形的宽度
        b = 2000  # 8 字形的高度
        omega = 0.003  # 角速度，控制飞艇在轨迹上的移动速度
        h_center = -19000  # 中心高度
        h_amp = 500  # 高度振荡幅度
        omega_h = 0.002  # 高度变化的角速度

        # --- 位置 ---
        xd = a * np.sin(omega * t)
        yd = b * np.sin(omega * t) * np.cos(omega * t)
        zd = h_center + h_amp * np.sin(omega_h * t)
        zeta_d = np.array([xd, yd, zd])

        # --- 速度（用于计算姿态） ---
        xd_dot = a * omega * np.cos(omega * t)
        yd_dot = b * omega * (np.cos(omega * t) * np.cos(omega * t)
                              - np.sin(omega * t) * np.sin(omega * t)
                              )
        zd_dot = h_amp * omega_h * np.cos(omega_h * t)

        # --- 姿态 ---
        phi_d = 0.0
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))
        psi_d = np.arctan2(yd_dot, xd_dot)
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ********************* 莱洛曲线轨迹函数 *********************

    def get_lemniscate_trajectory(self, t):
        """
        生成莱洛曲线 (Lemniscate) 轨迹，形似无限符号，带有高度变化

        参数：
            t: 当前时间

        返回：
            yc, yc_dot, yc_ddot, xc, xc_dot: 期望状态及其导数
        """
        dt_small = 1e-4

        # --- 轨迹参数 ---
        a = 2500  # 曲线尺度参数
        omega = 0.004  # 角速度
        h_center = -19000  # 中心高度
        h_amp = 800  # 高度变化幅度
        h_freq = 0.001  # 高度变化频率

        # 在起始时刻打印起始点信息
        if abs(t) < 1e-3:  # t 接近 0 时
            theta = 0
            denom = 1 + np.sin(theta) ** 2  # = 1
            start_x = a * np.cos(theta) / denom  # = a = 2500
            start_y = a * np.sin(theta) * np.cos(theta) / denom  # = 0
            start_z = h_center + h_amp * np.sin(0)  # = h_center = -19000
            print(f"【莱洛曲线轨迹】起始点位置：[{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (米)")
            print(f"【莱洛曲线轨迹】轨迹参数：尺度={a}m, 中心高度={h_center}m, 角速度={omega}rad/s")

        # --- 参数曲线参数 ---
        theta = omega * t
        # 莱洛曲线参数方程
        denom = 1 + np.sin(theta) ** 2
        xd = a * np.cos(theta) / denom
        yd = a * np.sin(theta) * np.cos(theta) / denom
        zd = h_center + h_amp * np.sin(h_freq * t)
        zeta_d = np.array([xd, yd, zd])

        # --- 速度计算（解析导数）---
        xd_dot_num = (
            -a * np.sin(theta) * denom
            - a * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        yd_dot_num = (
            a * (np.cos(theta) ** 2 - np.sin(theta) ** 2) * denom
            - a * np.sin(theta) * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        xd_dot = (xd_dot_num / denom**2) * omega
        yd_dot = (yd_dot_num / denom**2) * omega
        zd_dot = h_amp * h_freq * np.cos(h_freq * t)
        zeta_d_dot = np.array([xd_dot, yd_dot, zd_dot])

        # --- 使用数值差分计算加速度 ---
        # 计算 t+dt 时刻的位置和速度
        theta_plus = omega * (t + dt_small)
        denom_plus = 1 + np.sin(theta_plus) ** 2

        xd_dot_num_plus = (
            -a * np.sin(theta_plus) * denom_plus
            - a * np.cos(theta_plus) * 2 * np.sin(theta_plus) * np.cos(theta_plus)
        )
        yd_dot_num_plus = (
            a * (np.cos(theta_plus) ** 2 - np.sin(theta_plus) ** 2) * denom_plus
            - a * np.sin(theta_plus)
            * np.cos(theta_plus)
            * 2
            * np.sin(theta_plus)
            * np.cos(theta_plus)
        )

        xd_dot_plus = (xd_dot_num_plus / denom_plus**2) * omega
        yd_dot_plus = (yd_dot_num_plus / denom_plus**2) * omega
        zd_dot_plus = h_amp * h_freq * np.cos(h_freq * (t + dt_small))

        # 计算 t-dt 时刻的位置和速度
        theta_minus = omega * (t - dt_small)
        denom_minus = 1 + np.sin(theta_minus) ** 2

        xd_dot_num_minus = (
            -a * np.sin(theta_minus) * denom_minus
            - a * np.cos(theta_minus) * 2 * np.sin(theta_minus) * np.cos(theta_minus)
        )
        yd_dot_num_minus = (
            a * (np.cos(theta_minus) ** 2
            - np.sin(theta_minus) ** 2) * denom_minus
            - (a * np.sin(theta_minus) * np.cos(theta_minus)
                * 2 * np.sin(theta_minus) * np.cos(theta_minus))
        )
        xd_dot_minus = (xd_dot_num_minus / denom_minus**2) * omega
        yd_dot_minus = (yd_dot_num_minus / denom_minus**2) * omega
        zd_dot_minus = h_amp * h_freq * np.cos(h_freq * (t - dt_small))

        # 使用中心差分计算加速度
        zeta_d_ddot = np.array(
            [
                (xd_dot_plus - xd_dot_minus) / (2 * dt_small),
                (yd_dot_plus - yd_dot_minus) / (2 * dt_small),
                (zd_dot_plus - zd_dot_minus) / (2 * dt_small),
            ]
        )

        # --- 姿态计算 ---
        phi_d = 0.0  # 保持零横滚
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))  # 俯仰角
        psi_d = np.arctan2(yd_dot, xd_dot)  # 航向角
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- 姿态导数计算（使用辅助函数） ---
        _, gamma_d_plus = self.get_lemniscate_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_lemniscate_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # --- 组合 yc、yc_dot ---
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # --- 速度指令 vc, wc ---
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc))

        # --- xc_dot 和 yc_ddot ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # 简化处理
        xc_dot = np.concatenate((vc_dot, wc_dot))

        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_lemniscate_pos_att(self, t):
        """计算莱洛曲线在时间 t 的位置和姿态，用于计算导数"""
        # --- 轨迹参数 ---
        a = 2500
        omega = 0.004
        h_center = -19000
        h_amp = 800
        h_freq = 0.001

        # --- 参数曲线参数 ---
        theta = omega * t
        # 莱洛曲线参数方程
        denom = 1 + np.sin(theta) ** 2
        xd = a * np.cos(theta) / denom
        yd = a * np.sin(theta) * np.cos(theta) / denom
        zd = h_center + h_amp * np.sin(h_freq * t)
        zeta_d = np.array([xd, yd, zd])

        # --- 速度计算 ---
        xd_dot_num = (
            -a * np.sin(theta) * denom
            - a * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        yd_dot_num = (
            a * (np.cos(theta) ** 2 - np.sin(theta) ** 2) * denom
            - a * np.sin(theta) * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        xd_dot = (xd_dot_num / denom**2) * omega
        yd_dot = (yd_dot_num / denom**2) * omega
        zd_dot = h_amp * h_freq * np.cos(h_freq * t)

        # --- 姿态 ---
        phi_d = 0.0  # 保持零横滚
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))
        psi_d = np.arctan2(yd_dot, xd_dot)
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ********************* 直线轨迹函数 *********************
    def get_linear_trajectory(self, t, start_point=None, end_point=None, speed=10.0, hover_at_end=True):
        """
        生成一条直线轨迹，从起点飞向终点

        参数：
            t: 当前时间
            start_point: 起点坐标 [x, y, z]，默认为原点
            end_point: 终点坐标 [x, y, z]，默认为 [5000, 5000, -19000]
            speed: 飞行速度，单位 m/s
            hover_at_end: 到达终点后是否悬停，否则继续沿直线飞行

        返回：
            yc, yc_dot, yc_ddot, xc, xc_dot: 期望状态及导数


        说明：
            - yc:

                表示期望状态向量，包含飞艇的期望位置和姿态。
                具体包括：
                    位置：[x, y, z]，即飞艇在空间中的期望位置。
                    姿态：[φ, θ, ψ]，即飞艇的期望姿态角（横滚角、俯仰角、航向角）。
            - yc_dot:

                表示期望状态的一阶导数，即期望速度向量。
                具体包括：
                    线速度：[vx, vy, vz]，即飞艇在空间中的期望线速度。
                    角速度：[ωφ, ωθ, ωψ]，即飞艇的期望角速度。
            - yc_ddot:

                表示期望状态的二阶导数，即期望加速度向量。
                具体包括：
                    线加速度：[ax, ay, az]，即飞艇在空间中的期望线加速度。
                    角加速度：[αφ, αθ, αψ]，即飞艇的期望角加速度。
            - xc:

                表示控制指令向量，包含期望的线速度和角速度。
                具体包括：
                    线速度指令：[vx, vy, vz]，即飞艇的期望线速度。
                    角速度指令：[ωφ, ωθ, ωψ]，即飞艇的期望角速度。
            - xc_dot:

                表示控制指令的一阶导数，即控制指令的变化率。
                具体包括：
                    线速度变化率：[dvx/dt, dvy/dt, dvz/dt]，即线速度的时间变化率。
                    角速度变化率：[dωφ/dt, dωθ/dt, dωψ/dt]，即角速度的时间变化率。
        """
        dt_small = 1e-4

        # --- 轨迹参数 ---
        if start_point is None:
            start_point = np.array([0.0, 0.0, -19000.0])  # 默认起点
        if end_point is None:
            end_point = np.array([5000.0, 5000.0, -19000.0])  # 默认终点

        # 计算方向向量和距离
        direction = end_point - start_point
        distance = np.linalg.norm(direction)
        unit_direction = direction / max(distance, 1e-10)  # 避免除零

        # 计算总飞行时间
        total_time = distance / speed

        # --- 位置计算 ---
        if t < total_time or not hover_at_end:
            # 还在飞行中，或不需要悬停
            effective_t = t if hover_at_end else t % total_time
            progress = min(effective_t / total_time, 1.0) if hover_at_end else effective_t / total_time

            # 线性插值计算当前位置
            zeta_d = start_point + progress * direction
        else:
            # 已到达终点且需要悬停
            zeta_d = end_point

        # --- 速度计算 ---
        if t < total_time or not hover_at_end:
            # 飞行中的速度恒定
            if not hover_at_end or t < total_time:
                zeta_d_dot = unit_direction * speed
            else:
                # 到达终点后速度为零
                zeta_d_dot = np.zeros(3)
        else:
            # 悬停状态，速度为零
            zeta_d_dot = np.zeros(3)

        # --- 加速度计算（理论上为零，但为兼容性保留） ---
        zeta_d_ddot = np.zeros(3)

        # --- 姿态计算 ---
        # 计算期望航向角（朝向前进方向）
        if np.linalg.norm(zeta_d_dot) > 1e-6:  # 如果有速度
            phi_d = 0.0  # 保持零横滚
            theta_d = np.arctan2(-zeta_d_dot[2], np.sqrt(zeta_d_dot[0] ** 2 + zeta_d_dot[1] ** 2))  # 俯仰角
            psi_d = np.arctan2(zeta_d_dot[1], zeta_d_dot[0])  # 航向角
        else:
            # 悬停状态保持最后的姿态
            phi_d, theta_d, psi_d = self.get_linear_pos_att(max(0, t - dt_small), start_point, end_point, speed, hover_at_end)[1]

        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- 姿态导数计算 ---
        # 使用数值差分获取姿态导数
        _, gamma_d_plus = self.get_linear_pos_att(t + dt_small, start_point, end_point, speed, hover_at_end)
        _, gamma_d_minus = self.get_linear_pos_att(t - dt_small, start_point, end_point, speed, hover_at_end)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # --- 组合 yc、yc_dot ---
        yc = np.concatenate((zeta_d, gamma_d)) # 位置和姿态
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot)) # 线速度和姿态角速度

        # --- 速度指令 vc, wc ---
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc)) # 包含期望的线速度和角速度 (控制指令向量)

        # --- xc_dot 通过符号化导数简化近似 ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  #
        xc_dot = np.concatenate((vc_dot, wc_dot)) # 控制指令的一阶导数，即控制指令的变化率 线速度变化率和角速度变化率

        # --- yc_ddot 同样简化处理 ---
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot)) # 期望状态的二阶导数，即期望线加速度和角加速度向量。

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_linear_pos_att(self, t, start_point, end_point, speed, hover_at_end):
        """
        计算直线轨迹在时间 t 的位置和姿态，用于计算导数
        避免递归调用 get_linear_trajectory
        args:
            t: 当前时间
            start_point: 起点坐标 [x, y, z]
            end_point: 终点坐标 [x, y, z]
            speed: 飞行速度，单位 m/s
            hover_at_end: 到达终点后是否悬停，否则继续沿直线飞行
        return:
            zeta_d: 期望位置
            gamma_d: 期望姿态
        """
        # 计算方向向量和距离
        direction = end_point - start_point
        distance = np.linalg.norm(direction)
        unit_direction = direction / max(distance, 1e-10)  # 避免除零

        # 计算总飞行时间
        total_time = distance / speed

        # --- 位置计算 ---
        if t < total_time or not hover_at_end:
            effective_t = t if hover_at_end else t % total_time
            progress = min(effective_t / total_time, 1.0) if hover_at_end else effective_t / total_time

            # 线性插值计算当前位置
            zeta_d = start_point + progress * direction
        else:
            # 已到达终点且需要悬停
            zeta_d = end_point

        # --- 速度计算（用于计算姿态） ---
        if t < total_time or not hover_at_end:
            if not hover_at_end or t < total_time:
                zeta_d_dot = unit_direction * speed
            else:
                zeta_d_dot = np.zeros(3)
        else:
            zeta_d_dot = np.zeros(3)

        # --- 姿态计算 ---
        if np.linalg.norm(zeta_d_dot) > 1e-6:  # 如果有速度
            phi_d = 0.0  # 保持零横滚
            theta_d = np.arctan2(-zeta_d_dot[2],
                                 np.sqrt(zeta_d_dot[0] ** 2 + zeta_d_dot[1] ** 2))  # 俯仰角
            psi_d = np.arctan2(zeta_d_dot[1], zeta_d_dot[0])  # 航向角
        else:
            # 悬停状态下保持最后的姿态，这里简化为默认姿态
            phi_d = 0.0
            theta_d = 0.0
            # 使用方向向量计算默认航向角
            if np.linalg.norm(direction[:2]) > 1e-6:
                psi_d = np.arctan2(direction[1], direction[0])
            else:
                psi_d = 0.0  # 默认航向角

        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d
