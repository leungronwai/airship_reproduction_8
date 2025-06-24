"""
rotation_matrices.py
refer to Nonlinear adaptive trajectory tracking control for a stratospheric airship with parametric uncertainty
    Error-constrained fixed-time trajectory tracking control for a stratospheric airship with disturbances
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# cspell:ignore R_zeta R_y_inv Rc_z Rc_y_inv ddot arctan2 linalg
# cspell:ignore cphi cth cpsi sphi sth spsi

import numpy as np
import casadi as ca



def skew(v):
    """Convert a 3D vector to a skew-symmetric matrix"""
    if v.shape != (3,) and v.shape != (3, 1):
        raise ValueError(f"Input must be a 3D vector, got shape {v.shape}")
    v = v.flatten()  # Ensure it's 1D
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])


def sig(x, alpha):
    """
    Calculate the fractional power of x: sign(x) * |x|^alpha
    # Element-wise operation
    """
    return np.sign(x) * np.power(np.abs(x), alpha)



def R_zeta(gamma):
    """
    Calculate rotation matrix R_zeta (BRF to ERF) - Eq. 6
    Calculate the rotation matrix from Body Reference Frame to Inertial Reference Frame
    args:
        gamma: attitude angles (phi, theta, psi)
    return:
        3x3 rotation matrix R
    """
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    r_zeta = np.array(
        [
            [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
            [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
            [-sth, sphi * cth, cphi * cth],
        ]
    )
    return r_zeta


def R_gamma(gamma):
    """
    Calculate angular velocity transformation matrix R_y - Eq. 7

    Calculate the rotation matrix from Inertial Reference Frame to Body Reference Frame
    args:
        gamma: attitude angles (phi, theta, psi)
    return:
        3x3 rotation matrix R
    """
    phi, theta, _psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    _cth_safe = ca.if_else(ca.fabs(cth) < 1e-6, 1e-6, cth)

    r_gamma = np.array([[1, sphi * sth / cth, cphi * sth / cth],
                      [0, cphi, -sphi],
                      [0, sphi / cth, cphi / cth]])
    return r_gamma



def R_y_inv(gamma):
    """
    Calculate the inverse of angular velocity transformation matrix R_y - Eq. 16
    Calculate the rotation matrix from Body Reference Frame to Inertial Reference Frame
    args:
        gamma: attitude angles (phi, theta, psi)
    return:
        3x3 rotation matrix R
    """
    phi, theta, _psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    # No division by cth in the inverse according to Eq 16
    R_inv = np.array([[1, sphi * sth, cphi * sth],
                      [0, cphi, -sphi],
                      [0, sphi * cth, cphi * cth]])
    # R_y_inv = [[1, 0, -sin(theta)],
    #            [0, cos(phi), sin(phi)cos(theta)],
    #            [0, -sin(phi), cos(phi)cos(theta)]]
    # It seems my manual calculation based on Eq 7 was wrong. Using Eq 16 directly:
    _R_inv_paper = np.array([[1, 0, -sth],
                            [0, cphi, sphi * cth],
                            [0, -sphi, cphi * cth]])
    try:
        R_inv = np.linalg.inv(R_gamma(gamma))
    except np.linalg.LinAlgError:
        print("Warning: R_gamma is singular, cannot compute inverse.")
        R_inv = np.identity(3)  # Fallback or error
    return R_inv


def S_omega(omega):
    """
    This function is redundant
    Calculate the skew-symmetric matrix S(omega) for Coriolis terms - Eq. 14
    args:
        omega: angular velocity (p, q, r)
    return:
        3x3 rotation matrix
    """
    p, q, r = omega[0], omega[1], omega[2]
    S = np.array([[0, -r, q],
                  [r, 0, -p],
                  [-q, p, 0]])
    return S



def R_block(gamma):
    """
    Construct block diagonal rotation/transformation matrix R = diag(Rz, Ry)
    args:
        gamma: attitude angles (phi, theta, psi)
    return:
        6x6 rotation matrix R
    """
    Rz = R_zeta(gamma)
    Ry = R_gamma(gamma)
    R = np.zeros((6, 6))
    R[0:3, 0:3] = Rz
    R[3:6, 3:6] = Ry
    return R



def rk4_step(f, t, X, dt, *args):
    """
    Compute a single integration step using RK4 method.

    Parameters:
        f: Dynamics function in the form f(t, X, *args), returns dX/dt.
        t: Current time.
        X: Current state vector.
        dt: Time step.
        *args: Additional arguments to pass to the dynamics function.

    Returns:
        X_next: State vector at the next time step.
    """
    k1 = f(t, X, *args)
    k2 = f(t + 0.5 * dt, X + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, X + 0.5 * dt * k2, *args)
    k4 = f(t + dt, X + dt * k3, *args)
    X_next = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X_next