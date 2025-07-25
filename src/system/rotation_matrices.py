"""
rotation_matrices.py
refer to Nonlinear adaptive trajectory tracking control for a stratospheric airship with parametric uncertainty
    Error-constrained fixed-time trajectory tracking control for a stratospheric airship with disturbances
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# cspell:ignore R_zeta R_y_inv Rc_z Rc_y_inv ddot arctan2 linalg
# cspell:ignore cphi cth cpsi sphi sth spsi casadi blockcat

import casadi as ca
import numpy as np


def R_zeta(gamma):
    """
    计算旋转矩阵 - Eq. 6
    参数：
        gamma: 姿态角 (phi, theta, psi)
    返回：
        3x3 旋转矩阵 R (CasADi MX 类型)
    """
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    return ca.vertcat(
        ca.horzcat(
            ca.cos(theta) * ca.cos(psi),
            ca.sin(phi) * ca.sin(theta) * ca.cos(psi) - ca.cos(phi) * ca.sin(psi),
            ca.cos(phi) * ca.sin(theta) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi)
        ),
        ca.horzcat(
            ca.cos(theta) * ca.sin(psi),
            ca.sin(phi) * ca.sin(theta) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi),
            ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi)
        ),
        ca.horzcat(
            -ca.sin(theta),
            ca.sin(phi) * ca.cos(theta),
            ca.cos(phi) * ca.cos(theta)
        )
    )


def R_gamma(gamma):
    """
    参数：
        gamma: 姿态角 (phi, theta, psi)
    返回：
        3x3 旋转矩阵 R (CasADi MX 类型)
    """
    phi, theta, _ = gamma[0], gamma[1], gamma[2]
    return ca.vertcat(
        ca.horzcat(1, ca.sin(phi) * ca.sin(theta) / ca.cos(theta),
                   ca.cos(phi) * ca.sin(theta) / ca.cos(theta)),
        ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
        ca.horzcat(0, ca.sin(phi) / ca.cos(theta),
                   ca.cos(phi) / ca.cos(theta))
    )




def R_y_inv(gamma):
    """
    Compute inverse of rotation matrix R_gamma
    
    Args:
        gamma: attitude angles (phi, theta, psi)
    Returns:
        3x3 inverse rotation matrix (CasADi MX type)
    """
    phi, theta, _ = gamma[0], gamma[1], gamma[2]
    
    # Calculate inverse directly based on matrix structure
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(-ca.sin(phi) * ca.sin(theta) / ca.cos(theta), ca.cos(phi), -ca.sin(phi) / ca.cos(theta)),
        ca.horzcat(-ca.cos(phi) * ca.sin(theta) / ca.cos(theta), -ca.sin(phi), ca.cos(phi) / ca.cos(theta))
    )
