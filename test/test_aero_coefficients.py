'''
测试气动系数计算函数
'''
# tests/test_aero_coefficients.py
# pylint: disable=invalid-name
# cspell:ignore traj
# cspell:ignore arctan coeffs dalpha

import pytest  # 需要安装 pytest: pip install pytest
import numpy as np
# 假设你的模块文件名为 aero_coefficients.py，并且可以被导入
from config import aero_coefficients  # 使用相对导入（如果 tests 和模块在同一父目录下）


@pytest.fixture
def sample_k_factors():
    """提供默认的 k1 和 k2 测试值"""
    k1 = 0.1
    k2 = 0.2
    return k1, k2


def test_get_aero_coeffs_basic_run(sample_k_factors):
    """测试气动系数计算函数能否基本运行"""
    k1, k2 = sample_k_factors
    coeffs = aero_coefficients.get_aero_coefficients(k1=k1, k2=k2)
    assert isinstance(coeffs, dict)
    # 检查是否包含所有预期的键
    expected_keys = [
        "Cx1",
        "Cx2",
        "Cy1",
        "Cy2",
        "Cy3",
        "Cy4",
        "Cz1",
        "Cz2",
        "Cz3",
        "Cz4",
        "Cl1",
        "Cl2",
        "Cm1",
        "Cm2",
        "Cm3",
        "Cm4",
        "Cn1",
        "Cn2",
        "Cn3",
        "Cn4",
    ]
    for key in expected_keys:
        assert key in coeffs
        assert isinstance(coeffs[key], (float, np.number))  # 检查类型


def test_get_aero_coeffs_zero_k_factors():
    """测试 k1 和 k2 为零的情况"""
    coeffs = aero_coefficients.get_aero_coefficients(k1=0, k2=0)
    assert coeffs["Cx1"] == -(aero_coefficients.CDh0 * aero_coefficients.Sh +
                                aero_coefficients.CDf0 * aero_coefficients.Sf +
                                aero_coefficients.CDg0 * aero_coefficients.Sg)
    assert coeffs["Cx2"] == 0  # 因为 (k2 - k1) 为 0
    assert coeffs["Cy1"] == coeffs["Cx2"]
    assert coeffs["Cz1"] == coeffs["Cx2"]


def test_get_aero_coeffs_negative_k_factors():
    """测试 k1 和 k2 为负值的情况"""
    k1 = -0.1
    k2 = -0.2
    coeffs = aero_coefficients.get_aero_coefficients(k1=k1, k2=k2)
    assert coeffs["Cx2"] == (k2 - k1) * aero_coefficients.eta_k * aero_coefficients.I1 * aero_coefficients.Sh
    assert coeffs["Cy2"] == -0.5 * aero_coefficients.dCL_dalpha_f * aero_coefficients.Sf * aero_coefficients.eta_f


def test_get_aero_coeffs_specific_values():
    """测试特定输入值的输出"""
    k1 = 0.15
    k2 = 0.25
    coeffs = aero_coefficients.get_aero_coefficients(k1=k1, k2=k2)
    expected_cx1 = -(aero_coefficients.CDh0 * aero_coefficients.Sh +
                        aero_coefficients.CDf0 * aero_coefficients.Sf +
                        aero_coefficients.CDg0 * aero_coefficients.Sg)
    expected_cx2 = (k2 - k1) * aero_coefficients.eta_k * aero_coefficients.I1 * aero_coefficients.Sh
    assert abs(coeffs["Cx1"] - expected_cx1) < 1e-6
    assert abs(coeffs["Cx2"] - expected_cx2) < 1e-6


# 可以添加更多测试用例，测试边界条件、无效输入等
