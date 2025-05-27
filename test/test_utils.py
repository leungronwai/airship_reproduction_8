# pylint: disable=invalid-name
# cspell:ignore allclose linalg eigvals
# cspell:ignore arctan coeffs dalpha
import numpy as np
from airship.utils import skew, R_zeta, R_y, R_block

def test_skew():
    """测试反对称矩阵生成函数"""
    v = np.array([1, 2, 3])
    expected = np.array([
        [0, -3, 2],
        [3, 0, -1],
        [-2, 1, 0]
    ])
    assert np.allclose(skew(v), expected)

def test_R_zeta():
    """测试 R_zeta 旋转矩阵"""
    gamma = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
    R = R_zeta(gamma)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3))  # 验证正交性

def test_R_block():
    """测试 R_block 块对角旋转矩阵"""
    gamma = np.array([np.pi / 6, np.pi / 4, np.pi / 3])
    R = R_block(gamma)
    assert R.shape == (6, 6)
    assert np.allclose(R[:3, :3] @ R[:3, :3].T, np.eye(3))  # 验证正交性