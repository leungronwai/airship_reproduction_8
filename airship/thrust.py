# airship/thrust.py
"""
推力参数转换模块。
提供将推力参数转换为力和力矩的函数。
"""
# cspell:ignore vertcat arctan allclose
# pylint: disable=invalid-name
# pylint: disable=line-too-long
import numpy as np


def thrust_params_to_tau(thrust_params, rp_r, rp_l, use_casadi=False):
    """
    将推力参数转换为力和力矩向量。

    参数：
        thrust_params: [T, μ, v] - 推力大小和方向参数
            T_mag: 推力大小
            mu: 水平面内的推力偏转角
            nu: 垂直面内的推力偏转角
        rp_r: 右侧推力作用点向量 (相对于 CV)
        rp_l: 左侧推力作用点向量 (相对于 CV)
        use_casadi: 是否使用 CasADi (而不是 NumPy)

    返回：
        tau: [Fx, Fy, Fz, Tx, Ty, Tz] - 6 维力和力矩向量
    """
    # 选择合适的数学库
    if use_casadi:
        ca = __import__("casadi")
        T_mag, mu, nu = thrust_params

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
        tau_r = ca.cross(rp_r, thrust_vector_r.flatten()).reshape(3, 1)
        tau_l = ca.cross(rp_l, thrust_vector_l.flatten()).reshape(3, 1)
        tau_vec = tau_r + tau_l

        # 组合力和力矩
        tau = ca.vertcat(T_total, tau_vec)

    else:
        # 使用 NumPy
        T_mag, mu, nu = thrust_params

        # 计算右侧推力向量
        thrust_vector_r = np.array([
            T_mag * np.cos(mu) * np.cos(nu),
            T_mag * np.sin(mu),
            T_mag * np.cos(mu) * np.sin(nu)
        ])

        # 计算左侧推力向量
        thrust_vector_l = np.array([
            T_mag * np.cos(mu) * np.cos(nu),
            T_mag * np.sin(mu),
            T_mag * np.cos(mu) * np.sin(nu)
        ])

        # 总推力
        T_total = thrust_vector_r + thrust_vector_l

        # 计算力矩
        tau_r = np.cross(rp_r, thrust_vector_r)
        tau_l = np.cross(rp_l, thrust_vector_l)
        tau_vec = tau_r + tau_l

        # 组合力和力矩
        tau = np.concatenate([T_total, tau_vec])

    return tau


def calculate_thrust_direction(F_desired):
    """
    根据期望的力矢量计算合适的推力方向参数。

    参数：
        F_desired: 期望的力矢量 [Fx, Fy, Fz]

    返回：
        tuple: (mu, nu) - 水平和垂直偏转角 (弧度)
    """
    # 对于零力矢量，返回默认值 (前向)
    if np.allclose(F_desired, 0):
        return 0.0, 0.0

    # 计算水平偏转角
    mu = np.arctan2(F_desired[1], np.sqrt(F_desired[0]**2 + F_desired[2]**2))

    # 计算垂直偏转角
    nu = np.arctan2(F_desired[2], F_desired[0]) if abs(F_desired[0]) > 1e-6 else np.sign(F_desired[2]) * np.pi/2

    return mu, nu
