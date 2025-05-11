"""
utils.py
refer to Nonlinear adaptive trajectory tracking control for a stratospheric airship with parametric uncertainty
    Error-constrained fixed-time trajectory tracking control for a stratospheric airship with disturbances
"""
import casadi
# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# cspell:ignore R_zeta R_y_inv Rc_z Rc_y_inv ddot arctan2 linalg
# cspell:ignore cphi cth cpsi sphi sth spsi

import numpy as np
import casadi as ca



def skew(v):
    """将 3D 向量转换为反对称矩阵"""
    if v.shape != (3,) and v.shape != (3, 1):
        raise ValueError(f"Input must be a 3D vector, got shape {v.shape}")
    v = v.flatten()  # Ensure it's 1D
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])


def sig(x, alpha):
    """
    计算 x 的分数阶幂次 : sign(x) * |x|^alpha
    # Element-wise operation
    """
    return np.sign(x) * np.power(np.abs(x), alpha)



def R_zeta(gamma):
    """
    计算旋转矩阵 R_zeta (BRF to ERF) - Eq. 6
    计算从体坐标系 (Body Reference Frame) 到惯性坐标系 (Inertial Reference Frame) 的旋转矩阵
    args:
        gamma 是姿态角 (phi, theta, psi)
    return:
        3x3 的旋转矩阵 R
    """
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R_b2i = np.array(
        [
            [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
            [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
            [-sth, sphi * cth, cphi * cth],
        ]
    )
    return R_b2i


def R_y(gamma):
    """
    计算角速度变换矩阵 R_y - Eq. 7

    计算从惯性坐标系 (Inertial Reference Frame) 到体坐标系 (Body Reference Frame) 的旋转矩阵
    args:
        gamma 是姿态角 (phi, theta, psi)
    return:
        3x3 的旋转矩阵 R
    """
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    cth_safe = ca.if_else(ca.fabs(cth) < 1e-6, 1e-6, cth)

    R_i2b = np.array([[1, sphi * sth / cth, cphi * sth / cth],
                      [0, cphi, -sphi],
                      [0, sphi / cth, cphi / cth]])
    return R_i2b



def R_y_inv(gamma):
    """
    计算角速度变换矩阵 R_y 的逆 - Eq. 16
    计算从体坐标系 (Body Reference Frame) 到惯性坐标系 (Inertial Reference Frame) 的旋转矩阵
    args:
        gamma 是姿态角 (phi, theta, psi)
    return:
        3x3 的旋转矩阵 R
    """
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
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
    R_inv_paper = np.array([[1, 0, -sth],
                            [0, cphi, sphi * cth],
                            [0, -sphi, cphi * cth]])
    try:
        R_inv = np.linalg.inv(R_y(gamma))
    except np.linalg.LinAlgError:
        print("Warning: R_y is singular, cannot compute inverse.")
        R_inv = np.identity(3)  # Fallback or error
    return R_inv


def S_omega(omega):
    """
    这个是多余的
    计算用于科里奥利项的反对称矩阵 S(omega) - Eq. 14
    args:
        omega 是角速度 (p, q, r)
    return:
        3x3 的旋转矩阵
    """
    p, q, r = omega[0], omega[1], omega[2]
    S = np.array([[0, -r, q],
                  [r, 0, -p],
                  [-q, p, 0]])
    return S



def R_block(gamma):
    """
    构建块对角旋转/变换矩阵 R = diag(Rz, Ry)
    args:
        gamma 是姿态角 (phi, theta, psi)
    return:
        6x6 的旋转矩阵 R
    """
    Rz = R_zeta(gamma)
    Ry = R_y(gamma)
    R = np.zeros((6, 6))
    R[0:3, 0:3] = Rz
    R[3:6, 3:6] = Ry
    return R



def rk4_step(f, t, X, dt, *args):
    """
    使用 RK4 方法计算单步积分。

    参数：
        f: 动力学函数，形式为 f(t, X, *args)，返回 dX/dt。
        t: 当前时间。
        X: 当前状态向量。
        dt: 时间步长。
        *args: 传递给动力学函数的额外参数。

    返回：
        X_next: 下一时刻的状态向量。
    """
    k1 = f(t, X, *args)
    k2 = f(t + 0.5 * dt, X + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, X + 0.5 * dt * k2, *args)
    k4 = f(t + dt, X + dt * k3, *args)
    X_next = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X_next

