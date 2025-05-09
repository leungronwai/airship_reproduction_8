'''
aero_coefficients.py
'''

# pylint: disable=invalid-name
# pylint: disable=undefined-variable
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff
import numpy as np


from config import parameters as params



# 假设附加质量计算函数也在此文件中或可以导入
# from added_mass_calculator import calculate_added_mass_inertia # 假设可用

# ==============================================================================
#  定义基础几何、环境和气动参数 (Define Basic Geometric, Env, Aero Params)
# ==============================================================================


# === 几何参数 (Geometric Parameters) ===
airship_a1 = params.airship_a1  # [m] 前椭球半长轴 (Front ellipsoid semi-major axis)
airship_a2 = params.airship_a2  # [m] 后椭球半长轴 (Rear ellipsoid semi-major axis)
airship_b = params.airship_b  # [m] 半短轴 (Semi-minor axis)

Lh = airship_a1 + airship_a2  # [m] 机身总长 (Total hull length) - 假设
L_ref = Lh  # [m] 参考长度 (Reference length) - 假设
# 体积中心 x 坐标 (Volume Center x-coordinate) - Placeholder,
# 精确值依赖于双椭球的具体组合方式，这里用平均 a 近似或假设原点在特定位置 /
xcv = airship_a1 + (3 / 8) * (airship_a2 - airship_a1)  # [m] - Placeholder, assume origin or calc if needed Eq.22

# 计算体积 (Calculate Volume)
mean_a = (airship_a1 + airship_a2) / 2.0
V_airship = (4.0 / 3.0) * np.pi * mean_a * airship_b**2  # [m^3]

Sh = V_airship ** (2.0 / 3.0)  # [m^2] 机身参考面积 (Hull reference area)
Sf = 3656.0  # [m^2] 翼/舵面参考面积 (Fin reference area)
Sg = 202.0  # [m^2] 吊舱参考面积 (Gondola reference area)

# 力臂 (Lever Arms)
lf1 = 117.5  # [m] x-dist origin to aero center fins
lf2 = 129.7  # [m] x-dist origin to geom center fins
lf3 = 18.3  # [m] y,z-dist origin to aero center fins (Used for Cl1)
lgx = 29.2  # [m] x-dist origin to aero center gondola
lgz = 40.0  # [m] z-dist origin to aero center gondola (Used for Cl2)

# === 环境参数 (Environmental Parameters) ===
rho_air_at_altitude = params.rho_air_at_altitude  # [kg/m^3] 空气密度 @ ~20km - Placeholder

# === 基础气动系数和导数 (Basic Aero Coeffs & Derivatives from Table 2) ===
CDh0 = 0.025
CDf0 = 0.006
CDg0 = 0.01
CDch = 0.5
CDcf = 1.0
CDcg = 1.0
dCL_dalpha_f = 5.73  # (∂CL/∂α)f
dCL_ddelta_f = 1.24  # (∂CL/∂δ)f

# === 效率和积分因子 (Efficiency & Integral Factors from Table 2) ===
eta_f = 0.29
eta_k = 1.19
# 使用表格中的积分因子值 (Using integral factor values from Table 2)
I1_table = 0.33
I3_table = -0.69
J1_table = 1.31
J2_table = 0.53
# 注意：如果需要根据几何形状用 Eq. 82-85 计算，取消注释下面的计算部分
# Note: If calculation based on geometry (Eq. 82-85) is preferred, uncomment below


# 默认使用表格值 (Defaulting to table values)
I1 = I1_table
I3 = I3_table
J1 = J1_table
J2 = J2_table





# ==============================================================================
#  计算高阶气动系数的函数 (Function to Calculate Higher-Order Aero Coeffs)
# ==============================================================================


def get_aero_coefficients(k1=0, k2=0):
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
    coeffs["Cx1"] = -(CDh0 * Sh + CDf0 * Sf + CDg0 * Sg)
    coeffs["Cx2"] = (k2 - k1) * eta_k * I1 * Sh
    # 假设 Eq 73 (Cz1=Cz4) 和 Eq 70 (Cz2=Cz4) 是笔误，或者 Cz4 依赖于不同导数 / Assume Cz4 is a typo or Cz4 depends on different derivatives
    # 按照最直接的解释 Cy4, Cz4 公式计算 / Calculate Cy4, Cz4 based on the most direct interpretation

    # coeffs['Cz4'] = 0.5 * dCL_ddelta_f * Sf * eta_f # 假设舵面效率相同 / Assume control surface efficiency is the same
    coeffs["Cz1"] = coeffs["Cx2"]
    coeffs["Cy1"] = coeffs["Cx2"]

    coeffs["Cy2"] = -0.5 * dCL_dalpha_f * Sf * eta_f
    coeffs["Cz2"] = coeffs["Cy2"]  #

    coeffs["Cy3"] = -(CDch * J1 * Sh + CDcf * Sf + CDcg * Sg)
    coeffs["Cy4"] = 0.5 * dCL_ddelta_f * Sf * eta_f
    coeffs["Cz4"] = coeffs["Cy4"]
    coeffs["Cz3"] = -(CDch * J1 * Sh + CDcf * Sf)

    coeffs["Cl1"] = dCL_ddelta_f * Sf * eta_f * lf3
    coeffs["Cl2"] = -CDcg * Sg * lgz
    coeffs["Cm1"] = -(k1 - k2) * eta_k * I3 * Sh * L_ref
    coeffs["Cm2"] = -0.5 * dCL_dalpha_f * Sf * eta_f * lf1
    coeffs["Cm3"] = -(CDch * J2 * Sh * L_ref + CDcf * Sf * lf2)
    coeffs["Cm4"] = 0.5 * dCL_ddelta_f * Sf * eta_f * lf1
    # Eq. 81 Cnj = -Cmj --- HIGHLY SUSPECT ---
    coeffs["Cn1"] = -coeffs["Cm1"]
    coeffs["Cn2"] = -coeffs["Cm2"]
    coeffs["Cn3"] = -coeffs["Cm3"]
    coeffs["Cn4"] = -coeffs["Cm4"]  # <-- 使用了 Cm4 / Using Cm4

    return coeffs





# ==============================================================================
#  主执行部分 (示例) (Main execution part - Example)
# ==============================================================================
if __name__ == "__main__":
    # 这个部分只在直接运行 aero_coefficients.py 时执行，用于测试
    # / This part only runs when aero_coefficients.py is directly executed, for testing
    print("--- 测试计算气动系数 ---")
    try:
        # 使用计算得到的 k1, k2 计算气动系数
        inertia_k1 = params.k1
        inertia_k2 = params.k2
        aero_coeffs_calculated = get_aero_coefficients(k1=inertia_k1, k2=inertia_k2)

        print("\n计算得到的气动系数 / Aerodynamic Coefficients:")
        for coeff, value in aero_coeffs_calculated.items():
            print(f"  {coeff}: {value:.4f}")

    except ValueError as e:
        print(f"\n计算过程中发生错误 / Error during calculation: {e}")
