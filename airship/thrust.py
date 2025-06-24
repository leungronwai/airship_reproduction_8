# airship/thrust.py
"""
Thrust parameter conversion module.
Provides functions to convert thrust parameters to force and torque.
"""
# cspell:ignore vertcat arctan allclose
# pylint: disable=invalid-name
# pylint: disable=line-too-long
import numpy as np


def thrust_params_to_force_torque(thrust_params, rp_r, rp_l, use_casadi=False):
    """
    Convert thrust parameters to force and torque vectors.

    Args:
        thrust_params: [T, μ, v] - thrust magnitude and direction parameters
            T_mag: thrust magnitude
            mu: thrust deflection angle in horizontal plane
            nu: thrust deflection angle in vertical plane
        rp_r: right thrust application point vector (relative to CV)
        rp_l: left thrust application point vector (relative to CV)
        use_casadi: whether to use CasADi (instead of NumPy)

    Returns:
        tau: [Fx, Fy, Fz, Tx, Ty, Tz] - 6-dimensional force and torque vector
    """
    # Select appropriate math library
    if use_casadi:
        ca = __import__("casadi")
        # Use CasADi matrix operations to extract elements
        T_mag = thrust_params[0]
        mu = thrust_params[1]
        nu = thrust_params[2]

        # Calculate right thrust vector
        thrust_vector_r = ca.vertcat(
            T_mag * ca.cos(mu) * ca.cos(nu),
            T_mag * ca.sin(mu),
            T_mag * ca.cos(mu) * ca.sin(nu)
        )  # CasADi's vertcat generates column vector (3x1) by default

        # Calculate left thrust vector
        thrust_vector_l = ca.vertcat(
            T_mag * ca.cos(mu) * ca.cos(nu),
            T_mag * ca.sin(mu),
            T_mag * ca.cos(mu) * ca.sin(nu)
        )  # CasADi's vertcat generates column vector (3x1) by default

        # Total thrust
        T_total = thrust_vector_r + thrust_vector_l

        # Calculate torque
        tau_r = ca.cross(rp_r, thrust_vector_r)  # CasADi's cross supports (3x1) matrices
        tau_l = ca.cross(rp_l, thrust_vector_l)  # CasADi's cross supports (3x1) matrices
        tau_vec = tau_r + tau_l

        # Combine force and torque
        thrust_force_torque = ca.vertcat(T_total, tau_vec)

    else:
        # Use NumPy
        T_mag, mu, nu = thrust_params

        # Calculate right thrust vector
        thrust_vector_r = np.array([
            T_mag * np.cos(mu) * np.cos(nu),
            T_mag * np.sin(mu),
            T_mag * np.cos(mu) * np.sin(nu)
        ]).reshape(3, 1)  # Ensure it's a column vector (3,1)

        # Calculate left thrust vector
        thrust_vector_l = np.array([
            T_mag * np.cos(mu) * np.cos(nu),
            T_mag * np.sin(mu),
            T_mag * np.cos(mu) * np.sin(nu)
        ]).reshape(3, 1)  # Ensure it's a column vector (3,1)

        # Total thrust
        T_total = thrust_vector_r + thrust_vector_l  # Already (3,1)

        # Calculate torque
        tau_r = np.cross(rp_r.flatten(), thrust_vector_r.flatten()).reshape(3, 1)
        tau_l = np.cross(rp_l.flatten(), thrust_vector_l.flatten()).reshape(3, 1)
        tau_vec = tau_r + tau_l

        # Combine force and torque
        thrust_force_torque = np.vstack([T_total, tau_vec]).flatten()

    return thrust_force_torque


def calculate_thrust_direction(F_desired):
    """
    Calculate appropriate thrust direction parameters based on desired force vector.

    Args:
        F_desired: desired force vector [Fx, Fy, Fz]

    Returns:
        tuple: (mu, nu) - horizontal and vertical deflection angles (radians)
    """
    # For zero force vector, return default values (forward)
    if np.allclose(F_desired, 0):
        return 0.0, 0.0

    # Calculate horizontal deflection angle
    mu = np.arctan2(F_desired[1], np.sqrt(F_desired[0]**2 + F_desired[2]**2))

    # Calculate vertical deflection angle
    nu = np.arctan2(F_desired[2], F_desired[0]) if abs(F_desired[0]) > 1e-6 else np.sign(F_desired[2]) * np.pi/2

    return mu, nu
