# thrust_vectoring.py
"""
Thrust parameter conversion module.
Provide functions to convert thrust parameters to force and torque vectors for thrust vectoring.
"""
# cspell:ignore vertcat arctan allclose casadi
# pylint: disable=invalid-name
# pylint: disable=line-too-long
import numpy as np
import casadi as ca


def thrust_params_to_force_torque(thrust_params, rp_r, rp_l):
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


    T_mag = thrust_params[0]
    mu = 0 #thrust_params[1]
    nu = 0# thrust_params[2]

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

    return thrust_force_torque




def calculate_thrust_direction(F_desired):
    """
    Calculate appropriate thrust direction parameters based on the desired force vector.

    args:
        F_desired: desired force vector [Fx, Fy, Fz]

    return:
        tuple: (mu, nu) - horizontal and vertical deflection angles (rad)
    """

    # calculate horizontal deflection angle
    mu = np.arctan2(F_desired[1], np.sqrt(F_desired[0]**2 + F_desired[2]**2))

    # calculate vertical deflection angle
    nu = np.arctan2(F_desired[2], F_desired[0]) if abs(F_desired[0]) > 1e-6 else np.sign(F_desired[2]) * np.pi/2

    return mu, nu
