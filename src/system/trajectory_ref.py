"""
Trajectory generation module (trajectory.py)
"""
# pylint: disable=invalid-name
# pylint: disable=line-too-long


import numpy as np

from src.system.rotation_matrices import R_zeta, R_y_inv


class Trajectory:
    """
    Trajectory generation module
    """

    def __init__(self):
        self.omega = 0.008  # 螺旋轨迹参数（保留）
        self.r = 1500
        self.h_max = 1000

        # 水平直线轨迹参数 (Z 轴向下为正) - 修改为 20km 高度
        self.start_point = np.array([0.0, 0.0, -20000.0])    # 起点 [x, y, z] - 负 Z 表示在地面上方 20km
        self.end_point = np.array([1.64393562e+03, 00.0, -20000.0])   # 终点 [x, y, z] - 保持相同高度
        self.speed = 15.0                                     # 飞行速度 [m/s]

    # ┌─────────────────────────────────────────────────────┐
    # │          Spiral trajectory function                  │
    # └─────────────────────────────────────────────────────┘

    def get_spiral_trajectory(self, t):

        """
        Generate a spiral trajectory with altitude variation

        Args:
            t: Current time
        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot
        """

        dt_small = 1e-4

        # --- Trajectory parameters ---
        omega = self.omega  # Angular velocity
        r = self.r  # Radius
        h_max = self.h_max  # Maximum height

        # Print starting point information at initial time
        if abs(t) < 1e-3:  # When t approaches 0
            start_x = r * np.cos(0)  # = r = 1500
            start_y = r * np.sin(0)  # = 0
            start_z = h_max * (1 - np.exp(0))  # = 0
            print(
                f"[Spiral Trajectory] Starting point position: [{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (meters)")
            print(
                f"[Spiral Trajectory] Trajectory parameters: radius={r}m, max height={h_max}m, angular velocity={omega}rad/s")

        # --- Directly calculate position, velocity and acceleration ---
        theta = omega * t
        # Position
        xd = r * np.cos(theta)
        yd = r * np.sin(theta)
        zd = h_max * (1 - np.exp(-theta / 10))
        pos = np.array([xd, yd, zd])

        # Velocity
        xd_dot = -r * omega * np.sin(theta)
        yd_dot = r * omega * np.cos(theta)
        zd_dot = h_max * (1 / 10) * np.exp(-theta / 10) * omega
        vel = np.array([xd_dot, yd_dot, zd_dot])

        # Acceleration
        print("111")
        xd_ddot = -r * omega ** 2 * np.cos(theta)
        yd_ddot = -r * omega ** 2 * np.sin(theta)
        zd_ddot = -h_max * (1 / 10) * omega ** 2 * np.exp(-theta / 10)
        acc = np.array([xd_ddot, yd_ddot, zd_ddot])

        # Construct position and velocity vectors
        zeta_d = pos.flatten()
        zeta_d_dot = vel.flatten()
        zeta_d_ddot = acc.flatten()

        # Calculate attitude
        phi_d = 0.0  # Maintain zero roll
        theta_d = float(np.arctan2(-vel[2], np.sqrt(vel[0] ** 2 + vel[1] ** 2)))  # Pitch angle
        psi_d = float(np.arctan2(vel[1], vel[0]))  # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # Use numerical differentiation to get attitude derivatives
        _, gamma_d_plus = self.get_spiral_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_spiral_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # Combine yc, yc_dot
        yc = np.concatenate(
            (zeta_d, gamma_d))  # !!!!!!!! position (zeta_d): [x, y, z] + attitude (gamma_d): [phi, theta, psi]
        yc_dot = np.concatenate((zeta_d_dot,
                                 gamma_d_dot))  # !!!!!! velocity (zeta_d_dot): [vx, vy, vz] + angular velocity (gamma_d_dot): [p, q, r]

        # Velocity commands vc, wc
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.reshape((-1, 1))
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.reshape((-1, 1))
        xc = np.concatenate((vc, wc))

        # xc_dot simplified approximation through symbolic derivatives
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.reshape((-1, 1))
        wc_dot = np.zeros(3).reshape(-1, 1)  # Simplified processing, assuming small angular velocity change rate
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # yc_ddot simplified processing
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_spiral_pos_att(self, t):
        """
        Calculate the position and attitude of the spiral trajectory at time t, used for derivative calculation
        Avoid recursive calls to define_spiral_trajectory
        """
        # --- Trajectory parameters ---
        omega = self.omega  # Angular velocity (rad/s)
        r = self.r  # Basic radius (m)
        h_max = self.h_max  # Maximum height (m)

        # --- Position calculation ---
        theta = omega * t
        xd = r * np.cos(theta)
        yd = r * np.sin(theta)
        zd = h_max * (1 - np.exp(-theta / 10))
        zeta_d = np.array([xd, yd, zd])

        # --- Velocity calculation (for attitude determination) ---
        xd_dot = -r * omega * np.sin(theta)
        yd_dot = r * omega * np.cos(theta)
        zd_dot = h_max * (1 / 10) * np.exp(-theta / 10) * omega

        # --- Attitude calculation ---
        phi_d = 0.0  # Maintain zero roll
        theta_d = float(np.arctan2(-zd_dot, np.sqrt(xd_dot ** 2 + yd_dot ** 2)))  # Pitch angle
        psi_d = float(np.arctan2(yd_dot, xd_dot))  # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    def get_straight_line_trajectory(self, t):
        """
        生成水平直线轨迹 (考虑 Z 轴向下为正)
        
        Args:
            t: 当前时间
        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot
        """
        dt_small = 1e-4

        # 计算轨迹向量和总距离
        trajectory_vector = self.end_point - self.start_point
        total_distance = np.linalg.norm(trajectory_vector)
        unit_direction = trajectory_vector / total_distance

        # 计算飞行总时间
        total_time = total_distance / self.speed

        # 在初始时刻打印轨迹信息
        if abs(t) < 1e-3:
            print(
                f"[Straight Line Trajectory] Starting point: [{self.start_point[0]:.1f}, {self.start_point[1]:.1f}, {self.start_point[2]:.1f}] (meters)")
            print(
                f"[Straight Line Trajectory] Ending point: [{self.end_point[0]:.1f}, {self.end_point[1]:.1f}, {self.end_point[2]:.1f}] (meters)")
            print(
                f"[Straight Line Trajectory] Distance: {total_distance:.1f}m, Speed: {self.speed:.1f}m/s, Total time: {total_time:.1f}s")
            print(
                f"[Straight Line Trajectory] Note: Z-axis positive downward, altitude = {-self.start_point[2]:.1f}m above ground")

        # 限制时间在合理范围内
        if t > total_time:
            # 到达终点后保持静止
            pos = self.end_point.copy()
            vel = np.zeros(3)
            acc = np.zeros(3)
        else:
            # 当前位置、速度和加速度
            distance_traveled = self.speed * t
            pos = self.start_point + unit_direction * distance_traveled
            vel = unit_direction * self.speed
            acc = np.zeros(3)  # 匀速直线运动，加速度为零

        # 确保严格水平飞行 (Z 坐标和 Z 速度保持恒定)
        pos[2] = self.start_point[2]  # 强制 Z 坐标保持在起点高度
        vel[2] = 0.0  # 强制 Z 方向速度为 0 (水平飞行)
        acc[2] = 0.0  # 强制 Z 方向加速度为 0

        # 构造位置和速度向量
        zeta_d = pos.flatten()
        zeta_d_dot = vel.flatten()
        zeta_d_ddot = acc.flatten()

        # 计算姿态（面向运动方向，保持水平）
        phi_d = 0.0  # 保持零横滚
        theta_d = 0.0  # 强制保持水平（零俯仰）

        # 当速度很小时，保持当前姿态
        if np.linalg.norm(vel[:2]) < 1e-3:  # 只考虑水平速度分量
            psi_d = 0.0  # 面向 X 轴正方向
        else:
            # 偏航角：基于 X, Y 方向速度分量
            psi_d = float(np.arctan2(vel[1], vel[0]))

        gamma_d = np.array([phi_d, theta_d, psi_d])

        # 使用数值微分计算姿态导数
        _, gamma_d_plus = self.get_straight_line_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_straight_line_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # 组合 yc, yc_dot
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # 速度命令 vc, wc
        from src.system.rotation_matrices import R_zeta, R_y_inv
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.reshape((-1, 1))
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.reshape((-1, 1))
        xc = np.concatenate((vc, wc))

        # xc_dot 简化近似
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.reshape((-1, 1))
        wc_dot = np.zeros(3).reshape(-1, 1)
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # yc_ddot 简化处理
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_straight_line_pos_att(self, t):
        """
        计算水平直线轨迹在时间 t 的位置和姿态 (Z 轴向下为正)
        """
        # 计算轨迹向量和总距离
        trajectory_vector = self.end_point - self.start_point
        total_distance = np.linalg.norm(trajectory_vector)
        unit_direction = trajectory_vector / total_distance

        # 计算飞行总时间
        total_time = total_distance / self.speed

        # 位置计算
        if t > total_time:
            zeta_d = self.end_point.copy()
            vel = np.zeros(3)
        else:
            distance_traveled = self.speed * t
            zeta_d = self.start_point + unit_direction * distance_traveled
            vel = unit_direction * self.speed

        # 确保严格水平飞行
        zeta_d[2] = self.start_point[2]  # 强制 Z 坐标保持恒定
        vel[2] = 0.0  # 强制 Z 方向速度为 0

        # 姿态计算 - 水平飞行
        phi_d = 0.0  # 保持零横滚
        theta_d = 0.0  # 保持水平（零俯仰）

        if np.linalg.norm(vel[:2]) < 1e-3:  # 只考虑水平速度分量
            psi_d = 0.0  # 面向 X 轴正方向
        else:
            psi_d = float(np.arctan2(vel[1], vel[0]))  # 面向运动方向

        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ...existing code...
    # 保留原有的螺旋轨迹函数
