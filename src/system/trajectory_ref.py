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
        # 螺旋轨迹参数
        self.omega = 0.008  # 角速度（rad/s）
        self.r = 2000  # 半径（m）
        self.h_max = 1000  # 最大高度（m）

        # 保留直线轨迹参数（备用）
        self.start_point = np.array([0.0, 0.0, -20000.0])
        self.end_point = np.array([1.64393562e+03, 200.0, -20000.0])
        self.speed = 15.0  # 飞行速度 [m/s]



    # ┌─────────────────────────────────────────────────────┐
    # │          Circular trajectory function               │
    # └─────────────────────────────────────────────────────┘

    def get_circular_trajectory(self, t):
        """
        生成水平圆形轨迹（Z 坐标保持恒定）

        Args:
            t: 当前时间
        Returns:
            yc, yc_dot
        """
        # 圆形轨迹参数
        omega = self.omega  # 角速度
        r = self.r  # 半径
        height = -20000.0  # 固定高度（20km）

        # 计算位置
        theta = omega * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = height  # 固定高度
        pos = np.array([x, y, z])

        # 计算速度
        x_dot = -r * omega * np.sin(theta)
        y_dot = r * omega * np.cos(theta)
        z_dot = 0.0  # Z 方向速度为零
        vel = np.array([x_dot, y_dot, z_dot])

        # 计算姿态
        phi = 0.0  # 横滚角保持为零
        theta_att = 0.0  # 俯仰角保持为零
        psi = np.arctan2(y_dot, x_dot)  # 偏航角指向速度方向
        att = np.array([phi, theta_att, psi])

        # 组合位置和姿态
        yc = np.concatenate((pos, att))

        # 组合速度和角速度
        angular_velocity = np.array([0.0, 0.0, omega])  # 假设角速度只有绕 Z 轴的分量
        yc_dot = np.concatenate((vel, angular_velocity))

        return yc, yc_dot


