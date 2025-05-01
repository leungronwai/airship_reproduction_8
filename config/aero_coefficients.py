# aero_coefficients.py
import numpy as np
from config import parameters as params
# 假设附加质量计算函数也在此文件中或可以导入
# from added_mass_calculator import calculate_added_mass_inertia # 假设可用

# ==============================================================================
#  定义基础几何、环境和气动参数 (Define Basic Geometric, Env, Aero Params)
# ==============================================================================
# 这些值通常来自设计规格、风洞测试或 CFD
# These values typically come from design specs, wind tunnel tests, or CFD

# --- 几何参数 (Geometric Parameters) ---
airship_a1 = params.airship_a1  # [m] 前椭球半长轴 (Front ellipsoid semi-major axis)
airship_a2 = params.airship_a2  # [m] 后椭球半长轴 (Rear ellipsoid semi-major axis)
airship_b  = params.airship_b  # [m] 半短轴 (Semi-minor axis)

Lh = airship_a1 + airship_a2 # [m] 机身总长 (Total hull length) - 假设
L_ref = Lh                 # [m] 参考长度 (Reference length) - 假设
# 体积中心 x 坐标 (Volume Center x-coordinate) - Placeholder,
# 精确值依赖于双椭球的具体组合方式, 这里用平均 a 近似或假设原点在特定位置
xcv = airship_a1 +(3/8)*(airship_a2 - airship_a1)                  # [m] - Placeholder, assume origin or calc if needed Eq.22

# 计算体积 (Calculate Volume)
mean_a = (airship_a1 + airship_a2) / 2.0
V_airship = (4.0/3.0) * np.pi * mean_a * airship_b**2 # [m^3]

Sh = V_airship**(2.0/3.0) # [m^2] 机身参考面积 (Hull reference area)
Sf = 3656.0              # [m^2] 翼/舵面参考面积 (Fin reference area)
Sg = 202.0               # [m^2] 吊舱参考面积 (Gondola reference area)

# 力臂 (Lever Arms)
lf1 = 117.5 # [m] x-dist origin to aero center fins
lf2 = 129.7 # [m] x-dist origin to geom center fins
lf3 = 18.3  # [m] y,z-dist origin to aero center fins (Used for Cl1)
lgx = 29.2  # [m] x-dist origin to aero center gondola
lgz = 40.0  # [m] z-dist origin to aero center gondola (Used for Cl2)

# --- 环境参数 (Environmental Parameters) ---
# 这个值可能需要在仿真开始时从外部传入或在 parameters.py 中定义
rho_air_at_altitude = params.rho_air_at_altitude # [kg/m^3] 空气密度 @ ~20km - Placeholder

# --- 基础气动系数和导数 (Basic Aero Coeffs & Derivatives from Table 2) ---
CDh0 = 0.025; CDf0 = 0.006; CDg0 = 0.01
CDch = 0.5; CDcf = 1.0; CDcg = 1.0
dCL_dalpha_f = 5.73 # (∂CL/∂α)f
dCL_ddelta_f = 1.24 # (∂CL/∂δ)f

# --- 效率和积分因子 (Efficiency & Integral Factors from Table 2) ---
eta_f = 0.29
eta_k = 1.19
# 使用表格中的积分因子值 (Using integral factor values from Table 2)
I1_table = 0.33
I3_table = -0.69
J1_table = 1.31
J2_table = 0.53
# 注意: 如果需要根据几何形状用 Eq. 82-85 计算，取消注释下面的计算部分
# Note: If calculation based on geometry (Eq. 82-85) is preferred, uncomment below

# --- (可选) 根据几何计算积分因子 (Optional: Calculate Integral Factors from Geometry) ---
"""
f = (Lh - airship_a1) / airship_a2
f_sq = f**2
if abs(1.0 - f_sq) < 1e-9:
    sqrt_1_fsq = 0.0
    asin_f = np.sign(f) * np.pi / 2.0
else:
    sqrt_1_fsq = np.sqrt(1.0 - f_sq)
    f_clipped = np.clip(f, -1.0, 1.0)
    asin_f = np.arcsin(f_clipped)

V_pow_2_3 = V_airship**(2.0/3.0)
if abs(V_pow_2_3) < 1e-9: raise ValueError("V^(2/3) is near zero.")
if abs(L_ref) < 1e-9: raise ValueError("Reference length L is near zero.")

I1_geom = (np.pi * airship_b**2 / V_pow_2_3) * (1.0 - f_sq)
term_I3_paren = (airship_a1 - 2.0 * airship_a2 * f**3 - 3.0 * airship_a1 * f_sq)
term_I3_xcv = xcv / L_ref * I1_geom
I3_geom = (np.pi * airship_b**2 / (3.0 * L_ref * V_pow_2_3)) * term_I3_paren - term_I3_xcv
term_J1_paren = (airship_a1 * np.pi / 2.0 + airship_a2 * sqrt_1_fsq + 2.0 * airship_a2 * asin_f)
J1_geom = (airship_b / (2.0 * V_pow_2_3)) * term_J1_paren
term_J2_frac = (airship_a1 - xcv) / L_ref
term_J2_sqrt_pow = (1.0 - f_sq)**(1.5)
term_J2_b = (2.0 * airship_b / (3.0 * L_ref * V_pow_2_3)) * (airship_a2**2 - airship_a1**2 - airship_a2**2 * term_J2_sqrt_pow)
J2_geom = J1_geom * term_J2_frac + term_J2_b

# 选择使用哪个积分因子来源 (Choose which integral factor source to use)
I1 = I1_table # 或 I1 = I1_geom
I3 = I3_table # 或 I3 = I3_geom
J1 = J1_table # 或 J1 = J1_geom
J2 = J2_table # 或 J2 = J2_geom
"""
# 默认使用表格值 (Defaulting to table values)
I1 = I1_table
I3 = I3_table
J1 = J1_table
J2 = J2_table

# --- 附加质量计算 (需要 k1, k2) ---
# k1, k2 需要先通过 calculate_added_mass_inertia 计算得到
# 这里使用占位符，实际应在调用 get_aero_coefficients 前计算好
k1_placeholder = 0.1 # Placeholder - MUST BE CALCULATED/PROVIDED
k2_placeholder = 0.9 # Placeholder - MUST BE CALCULATED/PROVIDED


# ==============================================================================
#  计算高阶气动系数的函数 (Function to Calculate Higher-Order Aero Coeffs)
# ==============================================================================

def get_aero_coefficients(k1=k1_placeholder, k2=k2_placeholder):
    """
    计算气动系数 Cx1...Cn4。
    Calculates aerodynamic coefficients Cx1...Cn4.
    使用此文件顶部定义的全局基础参数。
    Uses the global basic parameters defined at the top of this file.

    Args:
        k1 (float): 附加质量因子 k1.
        k2 (float): 附加质量因子 k2.

    Returns:
        dict: 包含所有计算出的气动系数的字典。
    """
    coeffs = {}

    # 使用模块级定义的参数 (Use module-level defined parameters)
    # Eq. 66-81
    coeffs['Cx1'] = -(CDh0 * Sh + CDf0 * Sf + CDg0 * Sg)
    coeffs['Cx2'] = (k2 - k1) * eta_k * I1 * Sh
    # 假设 Eq 73 (Cz1=Cz4) 和 Eq 70 (Cz2=Cz4) 是笔误，或者 Cz4 依赖于不同导数
    # 按照最直接的解释 Cy4, Cz4 公式计算

    # coeffs['Cz4'] = 0.5 * dCL_ddelta_f * Sf * eta_f # 假设舵面效率相同
    coeffs['Cz1'] = coeffs['Cx2']
    coeffs['Cy1'] = coeffs['Cx2']

    coeffs['Cy2'] = -0.5 * dCL_dalpha_f * Sf * eta_f
    coeffs['Cz2'] = coeffs['Cy2'] #

    coeffs['Cy3'] = -(CDch * J1 * Sh + CDcf * Sf + CDcg * Sg)
    coeffs['Cy4'] = 0.5 * dCL_ddelta_f * Sf * eta_f
    coeffs['Cz4'] = coeffs['Cy4']
    coeffs['Cz3'] = -(CDch * J1 * Sh + CDcf * Sf)

    coeffs['Cl1'] = dCL_ddelta_f * Sf * eta_f * lf3
    coeffs['Cl2'] = -CDcg * Sg * lgz
    coeffs['Cm1'] = -(k1 - k2) * eta_k * I3 * Sh * L_ref
    coeffs['Cm2'] = -0.5 * dCL_dalpha_f * Sf * eta_f * lf1
    coeffs['Cm3'] = -(CDch * J2 * Sh * L_ref + CDcf * Sf * lf2)
    coeffs['Cm4'] = 0.5 * dCL_ddelta_f * Sf * eta_f * lf1
    # Eq. 81 Cnj = -Cmj --- HIGHLY SUSPECT ---
    coeffs['Cn1'] = -coeffs['Cm1']
    coeffs['Cn2'] = -coeffs['Cm2']
    coeffs['Cn3'] = -coeffs['Cm3']
    coeffs['Cn4'] = -coeffs['Cm4'] # <-- 使用了 Cm4

    return coeffs

# ==============================================================================
#  (可选) 计算附加质量和惯性的函数 (Optional: Function to Calculate Added Mass/Inertia)
# ==============================================================================
# 这个函数也可以放在这里，或者放在单独的文件中
def calculate_added_mass_inertia_local(a1=airship_a1, a2=airship_a2, b=airship_b, rho=rho_air_at_altitude):
    """
    在此文件内部计算附加质量和附加惯性矩阵 (仅用于演示)。
    Calculates added mass/inertia internally within this file (for demonstration).
    实际应用中，k1, k2 可能由外部计算并传入 get_aero_coefficients。
    In practice, k1, k2 might be calculated externally and passed to get_aero_coefficients.
    """
    # --- 重复附加质量计算逻辑 (Repeat added mass calculation logic) ---
    if b <= 0: raise ValueError("b must be positive")
    a = (a1 + a2) / 2.0
    if a <= 0: raise ValueError("a must be positive")
    if b > a: print(f"警告: 扁椭球 b={b} > a={a}，附加质量公式可能不准确。")

    V = (4.0 / 3.0) * np.pi * a * b**2
    m_air = rho * V
    tolerance = 1e-9

    if abs(a - b) < tolerance:
        k1 = 0.5; k2 = 0.5; k3 = 0.5
    else:
        a_sq = a**2
        term_inside_sqrt = 1.0 - (b**2 / a_sq)
        if term_inside_sqrt < -tolerance: # 允许小的负数容差
             print(f"警告: 偏心率计算sqrt内部为负 ({term_inside_sqrt:.2e})，假设为球体。")
             k1 = 0.5; k2 = 0.5; k3 = 0.5
        else:
             term_inside_sqrt = max(0, term_inside_sqrt) # 避免负数
             e = np.sqrt(term_inside_sqrt)
             if abs(1.0 - e) < tolerance: raise ValueError("e is near 1.")
             e_sq = e**2

             if abs(e) < tolerance : # 避免 f 和 g 中的除零
                 # 接近球体的情况，用极限或直接设为球体值
                 k1 = 0.5; k2 = 0.5; k3 = 0.5
             else:
                 f_log = np.log((1.0 + e) / (1.0 - e))
                 e_cubed = e**3
                 if abs(e_cubed) < tolerance: raise ValueError("e^3 near zero.")
                 g = (1.0 - e_sq) / e_cubed
                 alpha_prime = 2.0 * g * (f_log / 2.0 - e)
                 if abs(e_sq) < tolerance: raise ValueError("e^2 near zero.")
                 beta_prime = (1.0 / e_sq) - (g * f_log / 2.0)

                 denom_k1 = 2.0 - alpha_prime
                 if abs(denom_k1) < tolerance: raise ValueError("k1 denominator near zero.")
                 k1 = alpha_prime / denom_k1

                 denom_k2 = 2.0 - beta_prime
                 if abs(denom_k2) < tolerance: raise ValueError("k2 denominator near zero.")
                 k2 = beta_prime / denom_k2

                 b_sq = b**2
                 term1_num_k3 = (b_sq - a_sq) * (alpha_prime - beta_prime)
                 term2_den_k3 = 2.0 * (b_sq - a_sq) + (b_sq + a_sq) * (beta_prime - alpha_prime)
                 if abs(term2_den_k3) < tolerance:
                     if abs(a-b) > tolerance:
                         raise ValueError("k3 denominator near zero (non-sphere).")
                     else:
                         k3 = 0.5
                 else:
                     k3 = (1.0 / 5.0) * term1_num_k3 / term2_den_k3

    # --- 返回 k1, k2 (以及可能需要的 M', I0') ---
    M_prime = m_air * np.diag([k1, k2, k2])
    I0_prime = m_air * np.diag([0.0, k3, k3])

    return k1, k2, M_prime, I0_prime


# ==============================================================================
#  主执行部分 (示例) (Main execution part - Example)
# ==============================================================================
if __name__ == "__main__":
    # 这个部分只在直接运行 aero_coefficients.py 时执行，用于测试
    print("--- 测试计算气动系数 ---")
    try:
        # 1. 计算 k1, k2 (或从外部获取)
        k1_calc, k2_calc, _, _ = calculate_added_mass_inertia_local()
        print(f"计算得到的 k1 = {k1_calc:.4f}, k2 = {k2_calc:.4f}")

        # 2. 使用计算得到的 k1, k2 计算气动系数
        aero_coeffs_calculated = get_aero_coefficients(k1=k1_calc, k2=k2_calc)

        print("\n计算得到的气动系数:")
        for coeff, value in aero_coeffs_calculated.items():
            print(f"  {coeff}: {value:.4f}")

    except ValueError as e:
        print(f"\n计算过程中发生错误: {e}")