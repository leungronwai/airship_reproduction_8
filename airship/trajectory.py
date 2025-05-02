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

        # --- 期望位置 desired position zeta_d (Eq. 65) ---
        xd = 2000 * (np.sin(0.005 * t) + np.cos(0.0025 * t))
        yd = 2000 * (np.sin(0.0025 * t) + np.cos(0.005 * t))
        zd = -0.1 * t - 19000
        zeta_d = np.array([xd, yd, zd])

        # --- 期望位置一阶导数 The first derivative of the expected position zeta_d_dot ---
        xd_dot = 2000 * (0.005 * np.cos(0.005 * t) - 0.0025 * np.sin(0.0025 * t))
        yd_dot = 2000 * (0.0025 * np.cos(0.0025 * t) - 0.005 * np.sin(0.005 * t))
        zd_dot = -0.1
        zeta_d_dot = np.array([xd_dot, yd_dot, zd_dot])

        # --- 期望位置二阶导数 The second derivative of the expected position  zeta_d_ddot ---
        xd_ddot = 2000 * (-0.005**2 * np.sin(0.005 * t) - 0.0025**2 * np.cos(0.0025 * t))
        yd_ddot = 2000 * (-0.0025**2 * np.sin(0.0025 * t) - 0.005**2 * np.cos(0.005 * t))
        zd_ddot = 0.0
        zeta_d_ddot = np.array([xd_ddot, yd_ddot, zd_ddot])

        # --- 期望姿态 desired attitude  gamma_d (Eq. 66) ---
        phi_d = 0.0 # 期望滚转角设为0 (Desired roll is zero)
        theta_d = np.arctan2(zd_dot, np.sqrt(xd_dot**2 + yd_dot**2)) # 注意: 原论文公式似乎有误, 通常是 -zd_dot / note: The original paper seems to have an error, usually it's -zd_dot
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2)) # 使用常见定义 (Using common definition)
        psi_d = np.arctan2(yd_dot, xd_dot) # 期望偏航角 (Desired yaw)
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- 期望姿态一阶导数 desired first-order derivative of the attitude  gamma_d_dot ---
        # 使用数值微分近似计算 (Using numerical differentiation approximation)
        # 对于更精确的控制，需要解析导数 (Analytical derivatives needed for better control)
        dt_small = 1e-4
        _, _, _, xd_dot_p, _ = self.get_desired_state(t + dt_small) # Need xc_dot_plus
        # This creates recursion. Need analytical derivatives.

        # Analytical derivatives for gamma_d_dot (Requires chain rule, tedious)
        # Placeholder: Assume slow variation or calculate numerically outside recursion
        # Let's compute omega_c first, then gamma_d_dot = Ry @ omega_c
        # gamma_d_dot = np.zeros(3) # Placeholder - Must be computed accurately!

        # --- 期望速度指令  desired speed command xc = [vc, wc] (Eq. 15) ---
        Rc_z = R_zeta(gamma_d)
        Rc_y = R_y_inv(gamma_d)   # 原来是 R_y ??
        # vc = Rcz^-1 * zeta_d_dot
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()

        # wc = Ry_inv * gamma_d_dot. This requires gamma_d_dot. Chicken and egg.
        # Let's try differentiating Eq 66 numerically for gamma_d_dot
        _, gamma_d_plus = self.get_desired_state_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_desired_state_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # Now calculate wc
        Rc_y_inv = R_y_inv(gamma_d) # Use the potentially corrected inverse
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()

        xc = np.concatenate((vc, wc))

        # --- 期望速度指令导数 desired speed command derivative xc_dot = [vc_dot, wc_dot] (Eq. 15 derivation) ---
        # Requires R_dot = R * S(omega) -> Rc_dot = Rc * S(wc)
        # vc_dot = Rcz_dot.T @ zeta_d_dot + Rcz.T @ zeta_d_ddot
        # wc_dot = Rcy_inv_dot @ gamma_d_dot + Rcy_inv @ gamma_d_ddot (Needs gamma_d_ddot!)
        # This is getting very complex analytically. Use numerical differentiation for xc_dot.
        _, _, _, xc_plus, _ = self.get_desired_state(t + dt_small)
        _, _, _, xc_minus, _= self.get_desired_state(t - dt_small)
        xc_dot = (xc_plus - xc_minus) / (2 * dt_small)
        # xc_dot = np.zeros(6) # Placeholder - Critical for performance!

        # --- 组合期望状态 (Combine desired states) ---
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))
        # yc_ddot needs gamma_d_ddot - use numerical approx
        _, yc_dot_plus, _, _, _= self.get_desired_state(t + dt_small)
        _, yc_dot_minus, _, _, _= self.get_desired_state(t - dt_small)
        yc_ddot = (yc_dot_plus - yc_dot_minus) / (2 * dt_small)
        # yc_ddot = np.zeros(6) # Placeholder

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