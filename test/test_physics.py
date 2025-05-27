"""
测试物理模块
"""
# pylint: disable=invalid-name
# cspell:ignore arctan ndarray allclose linalg eigvals

import numpy as np
import pytest
from airship.physics import calculate_added_mass_inertia
from config import parameters as params

def test_calculate_added_mass_inertia_output_shape():
    """测试输出矩阵的形状和类型"""
    # 使用配置文件中的参数
    a1 = params.airship_a1
    a2 = params.airship_a2
    b = params.airship_b
    rho_air = params.rho_air

    M_prime, I0_prime, k1, k2, k3 = calculate_added_mass_inertia(a1, a2, b, rho_air)

    # 检查输出形状
    assert isinstance(M_prime, np.ndarray)
    assert isinstance(I0_prime, np.ndarray)
    assert M_prime.shape == (3, 3)
    assert I0_prime.shape == (3, 3)

    # 检查标量输出
    assert isinstance(k1, (float, np.floating))
    assert isinstance(k2, (float, np.floating))
    assert isinstance(k3, (float, np.floating))

def test_calculate_added_mass_inertia_invalid_inputs():
    """测试无效输入参数"""
    a1 = params.airship_a1
    a2 = params.airship_a2
    b = params.airship_b
    rho_air = params.rho_air

    # 测试 b <= 0
    with pytest.raises(ValueError, match="半短轴 b 必须大于 0"):
        calculate_added_mass_inertia(a1, a2, 0, rho_air)



def test_calculate_added_mass_inertia_symmetric():
    """测试输出矩阵的对称性"""
    a1 = params.airship_a1
    a2 = params.airship_a2
    b = params.airship_b
    rho_air = params.rho_air

    M_prime, I0_prime, _, _, _ = calculate_added_mass_inertia(a1, a2, b, rho_air)

    # 检查矩阵是否对称
    assert np.allclose(M_prime, M_prime.T)
    assert np.allclose(I0_prime, I0_prime.T)

def test_calculate_added_mass_inertia_positive_definite():
    """测试输出矩阵的正定性"""
    a1 = params.airship_a1
    a2 = params.airship_a2
    b = params.airship_b
    rho_air = params.rho_air

    M_prime, I0_prime, _, _, _ = calculate_added_mass_inertia(a1, a2, b, rho_air)

    # 检查特征值是否为正
    assert np.all(np.linalg.eigvals(M_prime) > 0)
    assert np.all(np.linalg.eigvals(I0_prime) >= 0)  # I0_prime 可以有零特征值


# pytest test/test_physics.py -v