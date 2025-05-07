# parameters.py

# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg

import numpy as np

from config.aero_coefficients import calculate_added_mass_inertia_local, get_aero_coefficients

# --- 仿真参数 (Simulation Parameters) ---
DT = 0.1  # 仿真步长 (Simulation step size) [s]
T_SPAN = 50  # 仿真总时间 (Total simulation time) [s] # Reduced for faster testing, original paper used longer

# --- 物理参数 (Physical Parameters - Placeholder Values!) ---
# 这些值需要根据具体气艇模型替换 (These values MUST be replaced based on the specific airship model)
m = 9400  # 质量 (Mass) [kg] - Placeholder
Jx, Jy, Jz = 2e6, 5.5e6, 5.5e6  # 转动惯量 (Moments of Inertia) [kg*m^2] - Placeholder
I0 = np.diag([Jx, Jy, Jz])  # 惯性矩阵 (Inertia Matrix)


# --- 几何参数 (airship Geometric Parameters) ---
airship_a1 = 73.50  # [m] - Placeholder
airship_a2 = 62.5  # [m] - Placeholder (如果只有一个 a，则 a1=a2=a)
airship_b = 19.0  # [m] - Placeholder


rc = np.array([[0], [0], [0.50]])  # CV 到 CG 的距离向量 (Distance vector CV to CG) - Placeholder
rb = np.array([[0], [0], [0]])  # CV 到 CB 的距离向量 (Distance vector CV to CB) - Placeholder
rp_r = np.array([[0.8 * airship_a1], [airship_b], [2.0]])  # 推力作用点在前右下方 (Thrust point in front right down) - Placeholder
rp_l = np.array([[0.8 * airship_a1], [-airship_b], [2.0]])  # 推力作用点在前左下方 (Thrust point in front left down) - Placeholder
g = 9.74  # 重力加速度 (Gravitational acceleration) [m/s^2]
Vol_airship = 10700  # 气艇体积 (Airship Volume) [m^3] - Placeholder
rho_air = 0.088  # 空气密度 (Air density at ~20km altitude) [kg/m^3] - Placeholder
S_ref = 500.0  # 参考面积 (Reference Area for Aero) [m^2] - Placeholder
L_ref = 38.0  # 参考长度 (Reference Length for Aero Moments) [m] - Placeholder


# --- 环境参数 (Environmental Parameters) ---
rho_air_at_altitude = rho_air  # [kg/m^3] 空气密度 air dense @ ~20km - Placeholder

"""
# --- Placeholder Aerodynamic Coefficients --- Placeholders for Buoyancy/Aero
# These should ideally be functions of alpha, beta, Reynolds number, Mach number, control deflections
# Using constant placeholders for simplicity demonstration
CD0 = 0.03   # Zero-lift drag coefficient
CDa = 0.1    # Drag coefficient slope vs alpha^2 (example)
CL0 = 0.0    # Lift coefficient at alpha=0
CLa = 2.0    # Lift coefficient slope vs alpha
CYb = -0.5   # Side force slope vs beta
Clb = -0.05  # Roll moment slope vs beta
Clp = -0.4   # Roll damping coefficient vs p
Cma = 1.0    # Pitch moment slope vs alpha
Cmq = -10.0  # Pitch damping coefficient vs q
Cnb = 0.1    # Yaw moment slope vs beta
Cnr = -1.0   # Yaw damping coefficient vs r
"""

# 在 simulation.py 或 parameters.py 的初始化部分
# In simulation.py or parameters.py initialization section




try:
    # 计算 k1, k2 (假设函数在 aero.py 或其他地方)
    # Calculate k1, k2 (assuming functions are defined in aero.py or elsewhere)
    k1_val, k2_val, _, _ = calculate_added_mass_inertia_local()  # 或者从其他来源获取  or from other sources

    # 计算气动系数
    # Calculate aerodynamic coefficients
    AERO_COEFFS = get_aero_coefficients(k1=k1_val, k2=k2_val)

except ValueError as e:
    print(f"初始化气动参数时出错 Error initializing aerodynamic parameters : {e}")
    AERO_COEFFS = None  # 或者设置默认值并停止  or set default values and stop

# 在 parameters.py 中，可以不再定义 Cx1 等变量，In parameters.py, variables like Cx1 can be omitted
# 或者将 AERO_COEFFS 字典保存在 params 模块中 or save AERO_COEFFS dictionary in params module
# params.AERO_COEFFS = AERO_COEFFS


def calculate_added_mass_inertia(a1, a2, b, rho_air_):
    """
    根据双椭球体模型的几何参数计算附加质量和附加惯性矩阵。
    Calculates the added mass and added inertia matrices based on the
    geometric parameters of a double-ellipsoid model.

    参考公式来源：图片中提供的 Eq. 42 - 51
    Reference Equations: Eq. 42 - 51 from the provided image.

    Args:
        a1 (float): 第一个半长轴 (Semi-major axis 1).
        a2 (float): 第二个半长轴 (Semi-major axis 2).
        b (float): 半短轴 (Semi-minor axis).
        rho_air_ (float): 当地空气密度 (Local air density).

    Returns:
        tuple:包含两个 NumPy 数组的元组 (M_prime, I0_prime)
               A tuple containing two NumPy arrays: (M_prime, I0_prime)
               M_prime (np.ndarray): 附加质量矩阵 (Added mass matrix, 3x3).
               I0_prime (np.ndarray): 附加惯性矩阵 (Added inertia matrix, 3x3).

    Raises:
        ValueError: 如果几何参数无效 (If geometric parameters are invalid).
    """

    if b <= 0:
        raise ValueError("半短轴 b 必须大于 0 /Semi-minor axis b must be positive")

    # 计算平均半长轴 (Calculate mean semi-major axis 'a')
    a = (a1 + a2) / 2.0
    if a <= 0:
        raise ValueError("平均半长轴 a 必须大于 0 / Mean semi-major axis a must be positive")

    # 检查是否为长椭球 (Check for prolate spheroid assumption a >= b)
    # 注意：如果 b > a (扁椭球)，偏心率 e 和相关公式定义不同
    # Note: If b > a (oblate), eccentricity and related formulas differ.
    # 这里假设 a >= b，与图片中公式一致
    # Assuming a >= b as consistent with the provided formulas.
    if b > a:
        print(
            f"Warning: The current formula is applicable for prolate spheroids (a >= b), "
            f"but the input is a={a:.3f}, b={b:.3f} (oblate spheroid). "
            f"The result may be inaccurate."
        )
        # print(f"警告：当前公式适用于长椭球 (a >= b)，但输入为 a={a:.3f}, b={b:.3f} (扁椭球)。"
        #   f"结果可能不准确。")
        # 对于扁椭球需要不同的公式或检查源文献
        # Different formulas or source check needed for oblate case.
        # 为避免错误，可以抛出异常或继续计算（结果可能错误） /  Continue calculation (result may be incorrect)
        # raise ValueError("当前公式仅适用于 a >= b 的情况" / Current formula only for a >= b case")

    # 计算体积 (Calculate Volume V - Eq. 43)
    # V = (2.0 / 3.0) * np.pi * (a1 + a2) * b**2
    V = (4.0 / 3.0) * np.pi * a * b**2  # 使用平均值 a 的等效公式 Use equivalent formula with mean value a

    # 计算排开空气的质量 (Calculate mass of displaced air)
    m_air = rho_air_ * V

    # 处理特殊情况：球体 (Handle special case: Sphere)
    tolerance = 1e-9  # 定义一个小的容差 / Define a small tolerance
    if abs(a - b) < tolerance:
        # 对于球体 (For a sphere, a = b, e = 0)
        # k1_ = k2_ = k3_ = 0.5 (标准流体动力学结果 standard hydrodynamic result)
        k1_ = 0.5
        k2_ = 0.5
        k3_ = 0.5
    else:
        # 计算偏心率 (Calculate eccentricity e - Eq. 44)
        # 确保 ensure  a^2 > 0 且 and 1 - (b^2 / a^2) >= 0
        a_sq = a**2
        term_inside_sqrt = 1.0 - (b**2 / a_sq)
        if (1.0 - (b**2 / (a**2))) < 0:
            # 这理论上不应该在 a >= b 时发生，除非有数值误差
            print(f"警告：偏心率计算出现问题 / Eccentricity calculation warning " f"(term = {term_inside_sqrt:.2e})。将 e 设为 0 / set e as 0。")
            _e = 0.0
            k1_ = k2_ = k3_ = 0.5  # 退化为球体情况 / Fallback to sphere case
        else:
            _e = np.sqrt(1.0 - (b**2 / (a**2)))

            # 避免 e 极其接近 1 (避免 f 中的除零) / Avoid e close to 1 (to avoid division by zero in f)
            if abs(1.0 - _e) < tolerance:
                raise ValueError("偏心率 e 接近 1 (b 接近 0)，几何形状无效。/ Eccentricity e approaches 1 (invalid geometry)")

            # 计算中间参数 / Calculate intermediate parameters f, g, alpha_prime, beta_prime
            # Calculate intermediate parameters f, g, alpha_prime, beta_prime

            # f (Eq. 45)
            f = np.log((1.0 + _e) / (1.0 - _e))

            # g (Eq. 46)
            # 避免 e=0 (已在球体情况中处理) / Avoid division by zero for e=0 (handled in sphere case)
            e_sq = _e**2
            e_cubed = _e**3
            if abs(e_cubed) < tolerance:
                # 理论上 e 非零，但数值上可能很小 / Theoretically e is non-zero, but numerically small
                raise ValueError("偏心率 e 的立方接近于零，无法计算 g。 / Eccentricity e cubed is close to zero, cannot calculate g.")
            _g = (1.0 - e_sq) / e_cubed

            # alpha_prime (Eq. 47)
            alpha_prime = 2.0 * _g * (f / 2.0 - _e)

            # beta_prime (Eq. 48)
            if abs(e_sq) < tolerance:
                raise ValueError(
                    "偏心率 e 的平方接近于零，无法计算 beta_prime。" "The square of eccentricity e is close to zero, beta_prime cannot be calculated."
                )
            beta_prime = (1.0 / e_sq) - (_g * f / 2.0)

            # 计算惯性因子 k1, k2, k3 / Calculate inertia factors k1, k2, k3
            # k1 (Eq. 49)
            denominator_k1 = 2.0 - alpha_prime  # denominator 分母   numer 分子，，fraction 分数
            if abs(denominator_k1) < tolerance:
                raise ValueError("计算 k1 时分母接近零。/ Small denominator in k1 calculation")
            k1_ = -alpha_prime / (2.0 - alpha_prime)

            # k2 (Eq. 50)
            denominator_k2 = 2.0 - beta_prime
            if abs(denominator_k2) < tolerance:
                raise ValueError("计算 k2 时分母接近零。/ Small denominator in k2 calculation")
            k2_ = -beta_prime / (2.0 - beta_prime)

            # k3 (Eq. 51)
            a_sq = a**2
            b_sq = b**2
            term1_num_k3 = (b_sq - a_sq) * (alpha_prime - beta_prime)
            term2_den_k3 = 2.0 * (b_sq - a_sq) + (b_sq + a_sq) * (beta_prime - alpha_prime)

            if abs(term2_den_k3) < tolerance:
                # 检查球体情况是否已处理 (e=0 -> a=b -> b^2-a^2 = 0)
                # / Check if sphere case has been handled (e=0 -> a=b -> b^2-a^2 = 0)
                # 如果 a != b 但分母为零，表示可能有其他问题或特殊共振情况
                # / If a != b but denominator is zero, there may be another issue or resonance case.
                if abs(a - b) > tolerance:
                    raise ValueError("计算 k3 时分母接近零 (非球体情况) Small denominator in k3 calculation。")
                else:  # 如果是球体，分子也为零，极限应为 0.5 / If it's a sphere, numerator is also zero, limit should be 0.5
                    k3_ = 0.5
            else:
                k3_ = -(1.0 / 5.0) * term1_num_k3 / term2_den_k3

    # 构建附加质量矩阵 (Construct Added Mass Matrix M_prime - Eq. 42)
    _M_prime = m_air * np.diag([k1_, k2_, k2_])

    # 构建附加惯性矩阵 (Construct Added Inertia Matrix I0' - Eq. 42)
    _I0_prime = m_air * np.diag([0.0, k3_, k3_])  # 注意第一个元素是 0 /note first element is 0

    return _M_prime, _I0_prime


# --- 计算附加质量/惯性 (Calculate Added Mass/Inertia) ---
try:
    M_prime_calculated, I0_prime_calculated = calculate_added_mass_inertia(airship_a1, airship_a2, airship_b, rho_air_at_altitude)
    # 使用计算出的值 / Use calculated values
    M_prime = M_prime_calculated
    I0_prime = I0_prime_calculated
except ValueError as e:
    print(f"错误：计算附加质量/惯性失败 / Error calculating added mass/inertia: {e}")
    # 可以设置默认值或停止仿真 / Set default values or stop simulation
    M_prime = np.zeros((3, 3))  # Fallback
    I0_prime = np.zeros((3, 3))  # Fallback

# 组合惯性矩阵 (Combined Inertia Matrices - Eq. 9)  Configuration
M_cfg = np.zeros((6, 6))
rc_skew = np.array([[0, -rc[2, 0], rc[1, 0]], [rc[2, 0], 0, -rc[0, 0]], [-rc[1, 0], rc[0, 0], 0]])
M_cfg[0:3, 0:3] = m * np.identity(3) + M_prime
M_cfg[0:3, 3:6] = -m * rc_skew
M_cfg[3:6, 0:3] = m * rc_skew
M_cfg[3:6, 3:6] = I0 + I0_prime
M_inv = np.linalg.inv(M_cfg)  # 预计算逆矩阵 (Pre-compute inverse matrix)

# --- 初始条件 (Initial Conditions) ---
zeta0 = np.array([2080, 2400, -18960])  # 初始位置 (Initial Position) [m]
gamma0 = np.array([np.pi / 90, np.pi / 36, np.pi / 18])  # 初始姿态 (Initial Attitude) [rad] (phi, theta, psi)
v0 = np.array([10, 0, 0])  # 初始线速度 (Initial Linear Velocity) [m/s] (u, v, w)
omega0 = np.array([0, 0, 0])  # 初始角速度 (Initial Angular Velocity) [rad/s] (p, q, r)
X0 = np.concatenate((zeta0, gamma0, v0, omega0))  # 完整初始状态向量 (Complete initial state vector)


# --- 扰动函数 (Disturbance Function delta(t) - Eq. 67) ---
def disturbance_delta(t):
    """定义外部扰动向量 / Define external disturbance vector"""
    d = np.zeros(6)
    d[0] = 0.5 + 2 * np.sin(0.1 * t)
    d[1] = 0.4 + 1.5 * np.cos(0.1 * t)
    d[2] = 0.6 + 1.5 * np.sin(0.1 * t)
    d[3] = 1.5 + 2 * np.sin(0.1 * t)
    d[4] = 1.5 + 1.5 * np.sin(0.1 * t)
    d[5] = 1.5 + 2 * np.cos(0.1 * t)
    return 5000 * d  # Paper scales by 5000


# 你需要添加一个表示风速的参数。风速通常在地球参考系（ERF）中定义。它可以是常数，也可以是时间和/或位置的函数。
# / You need to add a parameter representing the wind speed. The wind speed is usually defined in the Earth Reference Frame (ERF).
# It can be a constant, or a function of time and/or position.
# --- 风速定义 (Wind Definition) ---
# 示例：恒定风速在 ERF (例如：西风 5m/s, 南风 2m/s, 无垂直风)
# Example: Constant wind in ERF (e.g., 5m/s West, 2m/s South, 0m/s Vertical)
V_WIND_ERF = np.array([5.0, 2.0, 0.0])

# 或者，定义一个随时间/位置变化的函数
# Alternatively, define a function for time/position varying wind
# def get_wind_erf(t, zeta):
#     # 根据时间和位置计算风速 (Calculate wind based on time/position)
#     wind_x = 5.0 + np.sin(0.01 * t) * 2.0
#     wind_y = 2.0 + np.cos(t / 100 + zeta[0]/50000) * 1.0
#     wind_z = 0.0
#     return np.array([wind_x, wind_y, wind_z])
# V_WIND_FUNC = get_wind_erf # 如果使用函数，则在这里指定 / If using function, specify here




# === NMPC 控制器参数 ===
# 预测视界长度
N_HORIZON = 10

# 状态权重矩阵 (位置和姿态的误差权重)
Q = np.diag([10.0, 10.0, 10.0, 5.0, 5.0, 5.0,
              1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

# 控制输入权重矩阵
R = np.diag([0.1, 0.1, 0.1])

# 终端状态权重矩阵
Qf = np.diag([20.0, 20.0, 20.0, 10.0, 10.0, 10.0,
               2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

# 推力边界 (N)
T_MIN = 0.0
T_MAX = 20.0

# μ参数边界 (rad)
MU_MIN = -np.pi/4
MU_MAX = np.pi/4

# ν参数边界 (rad)
NU_MIN = -np.pi/4
NU_MAX = np.pi/4
