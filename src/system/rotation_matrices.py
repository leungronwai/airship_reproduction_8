"""
rotation_matrices.py
refer to Nonlinear adaptive trajectory tracking control for a stratospheric airship with parametric uncertainty
    Error-constrained fixed-time trajectory tracking control for a stratospheric airship with disturbances
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# cspell:ignore R_zeta R_y_inv Rc_z Rc_y_inv ddot arctan2 linalg
# cspell:ignore cphi cth cpsi sphi sth spsi casadi blockcat

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
        3x3 rotation matrix R (CasADi MX type)
    """
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)

    r_zeta = ca.vertcat(
        ca.horzcat(cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi),
        ca.horzcat(cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi),
        ca.horzcat(-sth, sphi * cth, cphi * cth),
    )
    return r_zeta


def R_gamma(gamma):
    """
    Calculate angular velocity transformation matrix R_y - Eq. 7
    Calculate the rotation matrix from Inertial Reference Frame to Body Reference Frame
    args:
        gamma: attitude angles (phi, theta, psi)
    return:
        3x3 rotation matrix R (CasADi MX type)
    """
    phi, theta, _psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)

    # Ensure cth is safe for division
    _cth_safe = ca.if_else(ca.fabs(cth) < 1e-6, 1e-6, cth)

    r_gamma = ca.vertcat(
        ca.horzcat(1, sphi * sth / _cth_safe, cphi * sth / _cth_safe),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi / _cth_safe, cphi / _cth_safe),
    )
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

    R = ca.MX.zeros(3, 3)
    r_block = ca.blockcat(Rz, R , R , Ry)
    return r_block
