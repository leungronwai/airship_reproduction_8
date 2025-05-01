# tests/test_aero_coefficients.py
import pytest # 需要安装 pytest: pip install pytest
import numpy as np
# 假设你的模块文件名为 aero_coefficients.py，并且可以被导入
from config import aero_coefficients # 使用相对导入（如果 tests 和模块在同一父目录下）  或者 from ..config import aero_coefficients
# 或者如果你的项目结构允许，可以直接: import aero_coefficients
# 如果还需要导入 parameters.py
from config import parameters
# 如果需要导入模型等
# from airship import model
# from airship import controller
# ...等等

# --- 测试 calculate_added_mass_inertia_local ---
def test_added_mass_basic_run():
    """测试附加质量惯性函数能否基本运行"""
    # 可以使用 aero_coefficients 中定义的默认参数，或传入特定测试参数
    try:
        k1, k2, M_prime, I0_prime = aero_coefficients.calculate_added_mass_inertia_local(
            a1=aero_coefficients.airship_a1, # 使用模块中定义的参数
            a2=aero_coefficients.airship_a2,
            b=aero_coefficients.airship_b,
            rho=aero_coefficients.rho_air_at_altitude
        )
        # 基本断言：检查返回类型或非空
        assert isinstance(k1, float)
        assert isinstance(k2, float)
        assert isinstance(M_prime, np.ndarray)
        assert M_prime.shape == (3, 3)
        assert isinstance(I0_prime, np.ndarray)
        assert I0_prime.shape == (3, 3)
        print(f"\n  (test_added_mass_basic_run) k1={k1:.4f}, k2={k2:.4f}") # pytest 会捕获 print 输出
    except ValueError as e:
        pytest.fail(f"计算附加质量时出错: {e}")

def test_added_mass_sphere():
    """测试球体特殊情况"""
    k1, k2, _, _ = aero_coefficients.calculate_added_mass_inertia_local(a1=10, a2=10, b=10, rho=1.0)
    assert abs(k1 - 0.5) < 1e-6
    assert abs(k2 - 0.5) < 1e-6

# --- 测试 get_aero_coefficients ---
# 可以使用 fixture 来提供 k1, k2 的测试值
@pytest.fixture
def sample_k_factors(request):
    # 可以根据 request.param 提供不同的 k 值组合进行测试
    # 这里简单返回默认计算值
    k1, k2, _, _ = aero_coefficients.calculate_added_mass_inertia_local()
    return k1, k2

def test_get_aero_coeffs_basic_run(sample_k_factors):
    """测试气动系数计算函数能否基本运行"""
    k1, k2 = sample_k_factors
    coeffs = aero_coefficients.get_aero_coefficients(k1=k1, k2=k2)
    assert isinstance(coeffs, dict)
    # 检查是否包含所有预期的键
    expected_keys = ['Cx1', 'Cx2', 'Cy1', 'Cy2', 'Cy3', 'Cy4', 'Cz1', 'Cz2', 'Cz3', 'Cz4',
                     'Cl1', 'Cl2', 'Cm1', 'Cm2', 'Cm3', 'Cm4', 'Cn1', 'Cn2', 'Cn3', 'Cn4']
    for key in expected_keys:
        assert key in coeffs
        assert isinstance(coeffs[key], (float, np.number)) # 检查类型
    print(f"\n  (test_get_aero_coeffs_basic_run) Cx1={coeffs['Cx1']:.4f}, Cm1={coeffs['Cm1']:.4f}")

def test_specific_aero_coeff_value(sample_k_factors):
    """测试某个具体系数的值 (假设已知某个预期结果)"""
    k1, k2 = sample_k_factors
    coeffs = aero_coefficients.get_aero_coefficients(k1=k1, k2=k2)
    # 假设我们知道 Cx1 的精确预期值 (需要根据参数精确计算或设定)
    # expected_cx1 = - (aero_coefficients.CDh0 * aero_coefficients.Sh + ...) # 理论计算
    # assert abs(coeffs['Cx1'] - expected_cx1) < 1e-6
    pass # 在实际测试中取消注释并填入预期值

# 可以添加更多测试用例，测试边界条件、无效输入等