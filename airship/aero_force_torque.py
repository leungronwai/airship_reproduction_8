# airship/aerodynamics.py
"""
气动力和气动力矩计算模块。
提供计算气艇气动力和力矩的函数。
"""
# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot nlpsol xlabel ylabel zlabel
# cspell:ignore delta_RUDT delta_RUDB delta_ELVL delta_ELVR arctan
# pylint: disable= line-too-long

import numpy as np


def calculate_aero_forces_moments(
        q_dyn, alpha, beta,
        aero_coeffs,
        delta_RUDT=0.0, delta_RUDB=0.0, delta_ELVL=0.0, delta_ELVR=0.0,
        use_casadi=False):
    """
    计算气动力和气动力矩。

    args:
        q_dyn (float): 动压
        alpha (float): 攻角 (rad)
        beta (float): 侧滑角 (rad)
        aero_coeffs (dict): 气动系数字典
        delta_RUDT (float): 上方向舵偏转角 (rad)
        delta_RUDB (float): 下方向舵偏转角 (rad)
        delta_ELVL (float): 左升降舵偏转角 (rad)
        delta_ELVR (float): 右升降舵偏转角 (rad)
        use_casadi (bool): 是否使用 CasADi (而不是 NumPy)

    return:
        tuple: (fa_BRF, ma_BRF) - 气动力和气动力矩向量
    """
    # 选择合适的数学库
    if use_casadi:
        ca = __import__("casadi")
        math_lib = ca  # 使用 CasADi
        vertcat = ca.vertcat
    else:
        math_lib = np  # 使用 NumPy

        def vertcat(*args):
            return np.array([args]).T  # 简单的 vertcat 模拟

    # 提取气动系数
    Cx1 = aero_coeffs["Cx1"]
    Cx2 = aero_coeffs["Cx2"]
    Cy1 = aero_coeffs["Cy1"]
    Cy2 = aero_coeffs["Cy2"]
    Cy3 = aero_coeffs["Cy3"]
    Cy4 = aero_coeffs["Cy4"]
    Cz1 = aero_coeffs["Cz1"]
    Cz2 = aero_coeffs["Cz2"]
    Cz3 = aero_coeffs["Cz3"]
    Cz4 = aero_coeffs["Cz4"]
    Cl1 = aero_coeffs["Cl1"]
    Cl2 = aero_coeffs["Cl2"]
    Cm1 = aero_coeffs["Cm1"]
    Cm2 = aero_coeffs["Cm2"]
    Cm3 = aero_coeffs["Cm3"]
    Cm4 = aero_coeffs["Cm4"]
    Cn1 = aero_coeffs["Cn1"]
    Cn2 = aero_coeffs["Cn2"]
    Cn3 = aero_coeffs["Cn3"]
    Cn4 = aero_coeffs["Cn4"]

    # 预计算三角函数值
    sin_a = math_lib.sin(alpha)
    cos_a = math_lib.cos(alpha)
    sin_b = math_lib.sin(beta)
    cos_b = math_lib.cos(beta)

    if use_casadi:
        sin_abs_a = math_lib.sin(math_lib.fabs(alpha))
        sin_abs_b = math_lib.sin(math_lib.fabs(beta))
    else:
        sin_abs_a = math_lib.sin(math_lib.abs(alpha))
        sin_abs_b = math_lib.sin(math_lib.abs(beta))

    sin_2a = math_lib.sin(2 * alpha)
    sin_2b = math_lib.sin(2 * beta)
    cos_a_half = math_lib.cos(alpha / 2.0)
    sin_a_half = math_lib.sin(alpha / 2.0)
    cos_b_half = math_lib.cos(beta / 2.0)

    # === 计算气动力 ===
    # X 力 (X Force - Eq. 23)
    Fax = q_dyn * (Cx1 * cos_a ** 2 * cos_b ** 2 + Cx2 * sin_2a * sin_a_half)

    # Y 力 (Y Force - Eq. 24)
    Fay = q_dyn * (Cy1 * cos_b_half * sin_2b + Cy2 * sin_2b +
                   Cy3 * sin_b * sin_abs_b + Cy4 * (delta_RUDT + delta_RUDB))

    # Z 力 (Z Force - Eq. 25)
    Faz = q_dyn * (Cz1 * cos_a_half * sin_2a + Cz2 * sin_2a +
                   Cz3 * sin_a * sin_abs_a + Cz4 * (delta_ELVL + delta_ELVR))

    # 组合力
    if use_casadi:
        fa_BRF = vertcat(Fax, Fay, Faz)
    else:
        fa_BRF = np.array([[Fax], [Fay], [Faz]])

    # === 计算气动力矩 ===
    # L 力矩 (Roll Moment - Eq. 26)
    moment_L = q_dyn * (Cl1 * (delta_ELVL - delta_ELVR + delta_RUDB - delta_RUDT) +
                        Cl2 * sin_b * sin_abs_b)

    # M 力矩 (Pitch Moment - Eq. 27)
    moment_M = q_dyn * (Cm1 * cos_a_half * sin_2a + Cm2 * sin_2a +
                        Cm3 * sin_a * sin_abs_a + Cm4 * (delta_ELVL + delta_ELVR))

    # N 力矩 (Yaw Moment - Eq. 28)
    moment_N = q_dyn * (Cn1 * cos_b_half * sin_2b + Cn2 * sin_2b +
                        Cn3 * sin_b * sin_abs_b + Cn4 * (delta_ELVL + delta_ELVR))

    # 组合力矩
    if use_casadi:
        ma_BRF = vertcat(moment_L, moment_M, moment_N)
    else:
        ma_BRF = np.array([[moment_L], [moment_M], [moment_N]])

    return fa_BRF, ma_BRF


def calculate_relative_velocity(v_airship_brf, V_wind_BRF):
    """
    计算相对风速。

    参数：
        v_airship_brf: 体轴系中的地速
        V_wind_BRF: 体轴系中的风速

    返回：
        tuple: (v_rel, u_rel, v_rel_body, w_rel) - 相对风速向量及其分量
    """
    v_rel = v_airship_brf - V_wind_BRF

    # 提取分量
    if isinstance(v_rel, np.ndarray) and v_rel.ndim > 1:
        # 处理列向量
        u_rel = v_rel[0, 0] if v_rel.shape[1] == 1 else v_rel[0]
        v_rel_body = v_rel[1, 0] if v_rel.shape[1] == 1 else v_rel[1]
        w_rel = v_rel[2, 0] if v_rel.shape[1] == 1 else v_rel[2]
    else:
        # 处理普通数组或 CasADi 向量
        u_rel, v_rel_body, w_rel = v_rel[0], v_rel[1], v_rel[2]

    return v_rel, u_rel, v_rel_body, w_rel


def calculate_aoa_sideslip(u_rel, v_rel_body, w_rel, V_rel_mag=None, use_casadi=False):
    """
    计算攻角和侧滑角。aoa is angle of attack

    参数：
        u_rel: 相对风速 X 分量
        v_rel_body: 相对风速 Y 分量
        w_rel: 相对风速 Z 分量
        V_rel_mag: 相对风速大小 (如果为 None 则计算)
        use_casadi: 是否使用 CasADi

    返回：
        tuple: (alpha, beta) - 攻角和侧滑角 (弧度)
    """
    if use_casadi:
        ca = __import__("casadi")
        math_lib = ca
    else:
        math_lib = np

    # 计算攻角
    if use_casadi:
        alpha = math_lib.atan2(w_rel, u_rel)
    else:
        alpha = math_lib.arctan2(w_rel, u_rel) if math_lib.abs(u_rel) > 1e-3 else math_lib.sign(w_rel) * math_lib.pi / 2

    # 计算相对风速大小 (如果未提供)
    if V_rel_mag is None:
        if use_casadi:
            V_rel_mag = math_lib.sqrt(u_rel ** 2 + v_rel_body ** 2 + w_rel ** 2)
        else:
            V_rel_mag = math_lib.sqrt(u_rel ** 2 + v_rel_body ** 2 + w_rel ** 2)
            if V_rel_mag < 1e-3:
                return alpha, 0.0

    # 计算侧滑角
    if use_casadi:
        beta = math_lib.asin(v_rel_body / (V_rel_mag + 1e-6))  # 避免除零
    else:
        beta = math_lib.arcsin(v_rel_body / V_rel_mag) if V_rel_mag > 1e-3 else 0.0

    return alpha, beta
