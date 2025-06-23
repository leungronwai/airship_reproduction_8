# airship/aerodynamics.py
"""
Aerodynamic force and moment calculation module.
Provides functions to calculate aerodynamic forces and moments for airships.
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
    Calculate aerodynamic forces and moments.

    args:
        q_dyn (float): dynamic pressure
        alpha (float): angle of attack (rad)
        beta (float): sideslip angle (rad)
        aero_coeffs (dict): aerodynamic coefficient dictionary
        delta_RUDT (float): rudder up deflection angle (rad)
        delta_RUDB (float): rudder down deflection angle (rad)
        delta_ELVL (float): elevator left deflection angle (rad)
        delta_ELVR (float): elevator right deflection angle (rad)
        use_casadi (bool): whether to use CasADi (instead of NumPy)

    return:
        tuple: (fa_BRF, ma_BRF) - aerodynamic forces and moments vector
    """
    # Select the appropriate math library
    if use_casadi:
        ca = __import__("casadi")
        math_lib = ca  # use CasADi
        vertcat = ca.vertcat
    else:
        math_lib = np  # use NumPy

        def vertcat(*args):
            return np.array([args]).T  # simple vertcat simulation

    # Extract aerodynamic coefficients
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

    # Precompute trigonometric values
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

    # === Calculate aerodynamic forces ===
    # X Force (Eq. 23)
    Fax = q_dyn * (Cx1 * cos_a ** 2 * cos_b ** 2 + Cx2 * sin_2a * sin_a_half)

    # Y Force (Eq. 24)
    Fay = q_dyn * (Cy1 * cos_b_half * sin_2b + Cy2 * sin_2b +
                   Cy3 * sin_b * sin_abs_b + Cy4 * (delta_RUDT + delta_RUDB))

    # Z Force (Eq. 25)
    Faz = q_dyn * (Cz1 * cos_a_half * sin_2a + Cz2 * sin_2a +
                   Cz3 * sin_a * sin_abs_a + Cz4 * (delta_ELVL + delta_ELVR))

    # Combine forces
    if use_casadi:
        fa_BRF = vertcat(Fax, Fay, Faz)
    else:
        fa_BRF = np.array([[Fax], [Fay], [Faz]])

    # === Calculate aerodynamic moments ===
    # Roll Moment (Eq. 26)
    moment_L = q_dyn * (Cl1 * (delta_ELVL - delta_ELVR + delta_RUDB - delta_RUDT) +
                        Cl2 * sin_b * sin_abs_b)

    # Pitch Moment (Eq. 27)
    moment_M = q_dyn * (Cm1 * cos_a_half * sin_2a + Cm2 * sin_2a +
                        Cm3 * sin_a * sin_abs_a + Cm4 * (delta_ELVL + delta_ELVR))

    # Yaw Moment (Eq. 28)
    moment_N = q_dyn * (Cn1 * cos_b_half * sin_2b + Cn2 * sin_2b +
                        Cn3 * sin_b * sin_abs_b + Cn4 * (delta_ELVL + delta_ELVR))

    # Combine moments
    if use_casadi:
        ma_BRF = vertcat(moment_L, moment_M, moment_N)
    else:
        ma_BRF = np.array([[moment_L], [moment_M], [moment_N]])

    return fa_BRF, ma_BRF


def calculate_relative_velocity(v_airship_brf, V_wind_BRF):
    """
    Calculate relative wind speed.

    Parameters:
        v_airship_brf: 体轴系中的地速
        V_wind_BRF: 体轴系中的风速

    Returns:
        tuple: (v_rel, u_rel, v_rel_body, w_rel) - 相对风速向量及其分量
    """
    v_rel = v_airship_brf - V_wind_BRF

    # Extract components
    if isinstance(v_rel, np.ndarray) and v_rel.ndim > 1:
        # Handle column vectors
        u_rel = v_rel[0, 0] if v_rel.shape[1] == 1 else v_rel[0]
        v_rel_body = v_rel[1, 0] if v_rel.shape[1] == 1 else v_rel[1]
        w_rel = v_rel[2, 0] if v_rel.shape[1] == 1 else v_rel[2]
    else:
        # Handle regular arrays or CasADi vectors
        u_rel, v_rel_body, w_rel = v_rel[0], v_rel[1], v_rel[2]

    return v_rel, u_rel, v_rel_body, w_rel


def calculate_aoa_sideslip(u_rel, v_rel_body, w_rel, V_rel_mag=None, use_casadi=False):
    """
    Calculate angle of attack and sideslip angle. AOA is angle of attack

    Parameters:
        u_rel: relative wind speed X component
        v_rel_body: relative wind speed Y component
        w_rel: relative wind speed Z component
        V_rel_mag: relative wind speed magnitude (if None, calculate)
        use_casadi: whether to use CasADi

    Returns:
        tuple: (alpha, beta) - angle of attack and sideslip angle (rad)
    """
    if use_casadi:
        ca = __import__("casadi")
        math_lib = ca
    else:
        math_lib = np

    # Calculate angle of attack
    if use_casadi:
        alpha = math_lib.atan2(w_rel, u_rel)
    else:
        alpha = math_lib.arctan2(w_rel, u_rel) if math_lib.abs(u_rel) > 1e-3 else math_lib.sign(w_rel) * math_lib.pi / 2

    # Calculate relative wind speed magnitude (if not provided)
    if V_rel_mag is None:
        if use_casadi:
            V_rel_mag = math_lib.sqrt(u_rel ** 2 + v_rel_body ** 2 + w_rel ** 2)
        else:
            V_rel_mag = math_lib.sqrt(u_rel ** 2 + v_rel_body ** 2 + w_rel ** 2)
            if V_rel_mag < 1e-3:
                return alpha, 0.0

    # Calculate sideslip angle
    if use_casadi:
        beta = math_lib.asin(v_rel_body / (V_rel_mag + 1e-6))  # avoid division by zero
    else:
        beta = math_lib.arcsin(v_rel_body / V_rel_mag) if V_rel_mag > 1e-3 else 0.0

    return alpha, beta
