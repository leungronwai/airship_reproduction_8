# utils.py
import numpy as np


def skew(v):
    """将3D向量转换为反对称矩阵"""
    if v.shape != (3,) and v.shape != (3, 1):
        raise ValueError(f"Input must be a 3D vector, got shape {v.shape}")
    v = v.flatten()  # Ensure it's 1D
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def sig(x, alpha):
    """计算 x 的分数阶幂次: sign(x) * |x|^alpha """
    # Element-wise operation
    return np.sign(x) * np.power(np.abs(x), alpha)


def R_zeta(gamma):
    """计算旋转矩阵 R_zeta (BRF to ERF) - Eq. 6"""
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R = np.array([
        [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
        [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
        [-sth, sphi * cth, cphi * cth]
    ])
    return R


def R_y(gamma):
    """计算角速度变换矩阵 R_y - Eq. 7"""
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    if abs(cth) < 1e-6:  # 避免奇异性 (Avoid singularity)
        print("Warning: cos(theta) is close to zero, R_y is singular.")
        # Handle singularity appropriately, maybe saturation or alternative formulation
        cth = 1e-6 * np.sign(cth) if cth != 0 else 1e-6

    R = np.array([
        [1, sphi * sth / cth, cphi * sth / cth],
        [0, cphi, -sphi],
        [0, sphi / cth, cphi / cth]
    ])
    return R


def R_y_inv(gamma):
    """计算角速度变换矩阵 R_y 的逆 - Eq. 16"""
    phi, theta, psi = gamma[0], gamma[1], gamma[2]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)

    # No division by cth in the inverse according to Eq 16
    R_inv = np.array([
        [1, sphi * sth, cphi * sth],
        [0, cphi, -sphi],
        [0, sphi * cth, cphi * cth]
    ])
    # Let's double check Eq 16 from the paper image...
    # R_y_inv = [[1, 0, -sin(theta)],
    #            [0, cos(phi), sin(phi)cos(theta)],
    #            [0, -sin(phi), cos(phi)cos(theta)]]
    # It seems my manual calculation based on Eq 7 was wrong. Using Eq 16 directly:
    R_inv_paper = np.array([
        [1, 0, -sth],
        [0, cphi, sphi * cth],
        [0, -sphi, cphi * cth]
    ])
    # It seems Eq 16 in the OCR might be incorrect or for a different convention.
    # Let's try inverting R_y numerically to verify.
    # If R_y(gamma) is correct (Eq 7), its inverse should be used.
    # Let's assume Eq 7 is correct and compute its inverse numerically for now.
    try:
        R_inv = np.linalg.inv(R_y(gamma))
    except np.linalg.LinAlgError:
        print("Warning: R_y is singular, cannot compute inverse.")
        R_inv = np.identity(3)  # Fallback or error
    return R_inv


def S_omega(omega):
    """计算用于科里奥利项的反对称矩阵 S(omega) - Eq. 14"""
    p, q, r = omega[0], omega[1], omega[2]
    S = np.array([
        [0, -r, q],
        [r, 0, -p],
        [-q, p, 0]
    ])
    return S


def R_block(gamma):
    """构建块对角旋转/变换矩阵 R = diag(Rz, Ry)"""
    Rz = R_zeta(gamma)
    Ry = R_y(gamma)
    R = np.zeros((6, 6))
    R[0:3, 0:3] = Rz
    R[3:6, 3:6] = Ry
    return R
