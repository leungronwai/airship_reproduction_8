# trajectory.py
import numpy as np
from airship.utils import R_zeta, R_y_inv


class Trajectory:
    def __init__(self):
        pass # No specific initialization needed for this trajectory






    def get_desired_state(self, t):
        """计算给定时间t的期望状态及其导数
        Calculate the expected state and its derivative at the given time t
        """
        dt_small = 1e-4

        # --- 期望位置 desired position zeta_d ---
        xd = 2000 * (np.sin(0.005 * t) + np.cos(0.0025 * t))
        yd = 2000 * (np.sin(0.0025 * t) + np.cos(0.005 * t))
        zd = -0.1 * t - 19000
        zeta_d = np.array([xd, yd, zd])

        # --- 期望位置一阶导数 zeta_d_dot ---
        xd_dot = 2000 * (0.005 * np.cos(0.005 * t) - 0.0025 * np.sin(0.0025 * t))
        yd_dot = 2000 * (0.0025 * np.cos(0.0025 * t) - 0.005 * np.sin(0.005 * t))
        zd_dot = -0.1
        zeta_d_dot = np.array([xd_dot, yd_dot, zd_dot])

        # --- 期望位置二阶导数 zeta_d_ddot (用数值差分计算) ---
        zeta_d_dot_plus = np.array([
            2000 * (0.005 * np.cos(0.005 * (t + dt_small)) - 0.0025 * np.sin(0.0025 * (t + dt_small))),
            2000 * (0.0025 * np.cos(0.0025 * (t + dt_small)) - 0.005 * np.sin(0.005 * (t + dt_small))),
            -0.1
        ])
        zeta_d_dot_minus = np.array([
            2000 * (0.005 * np.cos(0.005 * (t - dt_small)) - 0.0025 * np.sin(0.0025 * (t - dt_small))),
            2000 * (0.0025 * np.cos(0.0025 * (t - dt_small)) - 0.005 * np.sin(0.005 * (t - dt_small))),
            -0.1
        ])
        zeta_d_ddot = (zeta_d_dot_plus - zeta_d_dot_minus) / (2 * dt_small)

        # --- 姿态 gamma_d ---
        phi_d = 0.0
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot ** 2 + yd_dot ** 2))
        psi_d = np.arctan2(yd_dot, xd_dot)
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- 姿态一阶导数 gamma_d_dot（通过辅助函数数值差分） ---
        _, gamma_d_plus = self.get_desired_state_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_desired_state_pos_att(t - dt_small)
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

        # --- xc_dot 通过 zeta_d_ddot 和 gamma_d_ddot 数值差分简化近似 ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # 如需精度可进一步实现 gamma_d_ddot
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # --- yc_ddot 同样简化处理 ---
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot




    @staticmethod
    def define_spiral_trajectory(t):
        """Define the spiral trajectory function """

        def trajectory(t):
            # 参数 parameters
            omega = 0.05  # 角速度  angular velocity  (rad/s)
            r = 100  # 基础半径  Basic radius (m)
            h_max = 150  # 最大高度 maximum altitude (m)

            # position pos(t)
            # 位置 pos(t)
            theta = omega * t
            pos = np.array([
                r * np.cos(theta),  # x
                r * np.sin(theta),  # y
                h_max * (1 - np.exp(-theta / 10))  # z：通过指数项渐进上升
            ])

            # velocity vel(t)
            # 速度 vel(t)
            vel = np.array([
                -r * omega * np.sin(theta),  # dx/dt
                r * omega * np.cos(theta),  # dy/dt
                h_max * (1 / 10) * np.exp(-theta / 10) * omega  # dz/dt
            ])

            # acceleration acc(t)
            # 加速度 acc(t)
            acc = np.array([
                -r * omega ** 2 * np.cos(theta),  # d^2x/dt^2
                -r * omega ** 2 * np.sin(theta),  # d^2y/dt^2
                -h_max * (1 / 100) * np.exp(-theta / 10) * omega ** 2  # d^2z/dt^2
            ])

            return pos, vel, acc

        return trajectory

    @staticmethod
    def get_desired_state_pos_att(t):
        """ 
        calculate the desired position and attitude at time t to avoid recursion
        仅计算位置和姿态以避免递归
        """
        xd = 2000 * (np.sin(0.005 * t) + np.cos(0.0025 * t))
        yd = 2000 * (np.sin(0.0025 * t) + np.cos(0.005 * t))
        zd = -0.1 * t - 19000
        zeta_d = np.array([xd, yd, zd])

        xd_dot = 2000 * (0.005 * np.cos(0.005 * t) - 0.0025 * np.sin(0.0025 * t))
        yd_dot = 2000 * (0.0025 * np.cos(0.0025 * t) - 0.005 * np.sin(0.005 * t))
        zd_dot = -0.1

        phi_d = 0.0
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))
        psi_d = np.arctan2(yd_dot, xd_dot)
        gamma_d = np.array([phi_d, theta_d, psi_d])
        return zeta_d, gamma_d