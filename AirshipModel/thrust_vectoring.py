# airship/thrust.py
"""
Thrust parameter conversion module.
Provide functions to convert thrust parameters to force and torque vectors for thrust vectoring.
"""
# cspell:ignore vertcat arctan allclose
# pylint: disable=invalid-name
# pylint: disable=line-too-long
import numpy as np


def thrust_params_to_force_torque(thrust_params, rp_r, rp_l, use_casadi=False):
    """
    Convert thrust parameters to force and torque vectors for thrust vectoring.

    args:
        thrust_params: [T, μ, v] - thrust parameters
            T_mag: thrust magnitude
            mu: horizontal thrust deflection angle
            nu: vertical thrust deflection angle
        rp_r: right thrust force point vector (relative to CV)
        rp_l: left thrust force point vector (relative to CV)
        use_casadi: whether to use CasADi (instead of NumPy)

    return:
        tau: [Fx, Fy, Fz, Tx, Ty, Tz] - 6-dimensional force and torque vector
    """
    # select the appropriate math library
    if use_casadi:
        ca = __import__("casadi")
        # use CasADi matrix operations to extract elements
        T_mag = thrust_params[0]
        mu = thrust_params[1]
        nu = thrust_params[2]

        # calculate right thrust vector
        thrust_vector_r = ca.vertcat(
            T_mag * ca.cos(mu) * ca.cos(nu),
            T_mag * ca.sin(mu),
            T_mag * ca.cos(mu) * ca.sin(nu)
        )  # CasADi's vertcat default generates column vector (3x1)

        # calculate left thrust vector
        thrust_vector_l = ca.vertcat(
            T_mag * ca.cos(mu) * ca.cos(nu),
            T_mag * ca.sin(mu),
            T_mag * ca.cos(mu) * ca.sin(nu)
        )  # CasADi's vertcat default generates column vector (3x1)

        # total thrust
        T_total = thrust_vector_r + thrust_vector_l

        # calculate torque
        tau_r = ca.cross(rp_r, thrust_vector_r)  # CasADi's cross supports (3x1) matrices
        tau_l = ca.cross(rp_l, thrust_vector_l)  # CasADi's cross supports (3x1) matrices
        tau_vec = tau_r + tau_l

        # combine forces and torques for thrust vectoring
        thrust_force_torque = ca.vertcat(T_total, tau_vec)

    else:
        # use NumPy
        T_mag, mu, nu = thrust_params

        # calculate right thrust vector
        thrust_vector_r = np.array([
            T_mag * np.cos(mu) * np.cos(nu),
            T_mag * np.sin(mu),
            T_mag * np.cos(mu) * np.sin(nu)
        ]).reshape(3, 1)  # (3,1)

        # calculate left thrust vector
        thrust_vector_l = np.array([
            T_mag * np.cos(mu) * np.cos(nu),
            T_mag * np.sin(mu),
            T_mag * np.cos(mu) * np.sin(nu)
        ]).reshape(3, 1)  #  (3,1)

        # total thrust
        T_total = thrust_vector_r + thrust_vector_l  # (3,1)

        # calculate torque
        tau_r = np.cross(rp_r.flatten(), thrust_vector_r.flatten()).reshape(3, 1)
        tau_l = np.cross(rp_l.flatten(), thrust_vector_l.flatten()).reshape(3, 1)
        tau_vec = tau_r + tau_l

        # combine forces and torques for thrust vectoring
        thrust_force_torque = np.vstack([T_total, tau_vec]).flatten()

    return thrust_force_torque


def calculate_thrust_direction(F_desired):
    """
    Calculate appropriate thrust direction parameters based on the desired force vector.

    args:
        F_desired: desired force vector [Fx, Fy, Fz]

    return:
        tuple: (mu, nu) - horizontal and vertical deflection angles (rad)
    """
    # for zero force vector, return default value (forward)
    if np.allclose(F_desired, 0):
        return 0.0, 0.0

    # calculate horizontal deflection angle
    mu = np.arctan2(F_desired[1], np.sqrt(F_desired[0]**2 + F_desired[2]**2))

    # calculate vertical deflection angle
    nu = np.arctan2(F_desired[2], F_desired[0]) if abs(F_desired[0]) > 1e-6 else np.sign(F_desired[2]) * np.pi/2

    return mu, nu
