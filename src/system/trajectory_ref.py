"""
Trajectory generation module (trajectory.py)
"""
# pylint: disable=invalid-name
# pylint: disable=line-too-long

import numpy as np


class Trajectory:
    """
    Trajectory generation module
    """

    def __init__(self):
        # 参考轨迹初始状态值
        pass

    def get_helix_trajectory(self, t):
        """
        生成上升螺旋轨迹

        Args:
            t: 当前时间 [s]
        Returns:
            yc: 参考位置和姿态 [x, y, z, phi, theta, psi]
            yc_dot: 参考速度和角速度 [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]
        """
        # 螺旋轨迹参数
        omega = 0.01  # 角速度 [rad/s]
        r = 2000.0  # 半径 [m]



        # 位置计算：[500sin(0.01t), 500cos(0.01t), -2t -19800.0]
        x = float(r * np.sin(omega * t))
        y = float(r * np.cos(omega * t))
        z = float(-2.0 * t -19800.0)
        pos_ref = np.array([x, y, z])

        # 速度计算 (位置的导数)
        x_dot = float(r * omega * np.cos(omega * t))
        y_dot = float(-r * omega * np.sin(omega * t))
        z_dot = -2.0
        vel_ref = np.array([x_dot, y_dot, z_dot])

        # 计算姿态 (简化计算)
        phi = 0.0
        theta = float(np.arctan2(z_dot, np.sqrt(x_dot ** 2 + y_dot ** 2)))
        psi = float(np.arctan2(y_dot, x_dot))
        att_ref = np.array([phi, theta, psi])



        # 组合速度和角速度
        omega_body_ref = np.array([0.0, 0.0, omega])


        return pos_ref, att_ref, vel_ref, omega_body_ref
