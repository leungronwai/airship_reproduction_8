# thrust_vectoring.py
"""
Thrust parameter conversion module.
Provide functions to convert thrust parameters to force and torque vectors for thrust vectoring.
"""
import casadi as ca
# cspell:ignore vertcat arctan allclose casadi
# pylint: disable=invalid-name
# pylint: disable=line-too-long
import numpy as np


def thrust_params_to_force_torque(mode='forward', *args):
    """
    统一的推力矢量控制函数，可以执行两种操作：
    1. 将推力参数转换为力和力矩向量 (正向模式)
    2. 计算实现期望力所需的推力方向参数 (反向模式)

    参数：
        mode: 操作模式 - 'forward' 或 'inverse'
        *args: 基于模式的参数
            - 正向模式 (forward): thrust_params, rp_r, rp_l
              thrust_params: [T, μ, v] - 推力参数
              rp_r: 右侧推力点相对向量 (相对于 CV)
              rp_l: 左侧推力点相对向量 (相对于 CV)
            - 反向模式 (inverse): F_desired
              F_desired: 期望力向量 [Fx, Fy, Fz]

    返回：
        - 正向模式：[Fx, Fy, Fz, Tx, Ty, Tz] - 6 维力和力矩向量
        - 反向模式：(T, mu, nu) - 推力大小和方向角度 (rad)
    """


    thrust_params, rp_r, rp_l = args
    T_mag = thrust_params[0]
    mu = thrust_params[1]  # 恢复使用参数而不是硬编码为 0
    nu = thrust_params[2]  # 恢复使用参数而不是硬编码为 0

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
    tau_r = ca.cross(rp_r, thrust_vector_r)
    tau_l = ca.cross(rp_l, thrust_vector_l)
    tau_vec = tau_r + tau_l

    # 组合力和力矩
    thrust_force_torque = ca.vertcat(T_total, tau_vec)

    return thrust_force_torque


