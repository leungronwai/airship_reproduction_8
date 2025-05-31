"""
# parameters.py
"""


# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg

import numpy as np

# 导入物理计算函数
from airship.physics import calculate_added_mass_inertia



# === 仿真参数 (Simulation Parameters) ===
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


# === 环境参数 (Environmental Parameters) ===
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


# === 计算附加质量/惯性 (Calculate Added Mass/Inertia) ===
try:
    M_prime, I0_prime, k1, k2, k3 = calculate_added_mass_inertia(airship_a1, airship_a2, airship_b, rho_air_at_altitude)
except ValueError as e:
    print(f"错误：计算附加质量/惯性失败 / Error calculating added mass/inertia: {e}")
    # 可以设置默认值或停止仿真 / Set default values or stop simulation
    M_prime = np.zeros((3, 3))  # Fallback
    I0_prime = np.zeros((3, 3))  # Fallback

# 组合惯性矩阵 (Combined Inertia Matrices - Eq. 9)  Configuration
M_cfg = np.zeros((6, 6))
rc_skew = np.array([[0, -rc[2, 0],rc[1, 0]],
                    [rc[2, 0], 0, -rc[0, 0]],
                    [-rc[1, 0], rc[0, 0], 0]])
M_cfg[0:3, 0:3] = m * np.identity(3) + M_prime
M_cfg[0:3, 3:6] = -m * rc_skew
M_cfg[3:6, 0:3] = m * rc_skew
M_cfg[3:6, 3:6] = I0 + I0_prime
M_inv = np.linalg.inv(M_cfg)  # 预计算逆矩阵 (Pre-compute inverse matrix)



# === 气动参数 (Aerodynamic Parameters) ===
try:
    # 计算 k1, k2
    # Calculate k1, k2 (assuming functions are defined in aero.py or elsewhere)
    k1_val, k2_val, _, = k1, k2, k3

    from config.aero_coefficients import get_aero_coefficients # noqa: E402 # 延迟导入避免循环依赖
    # from config.aero_coefficients import get_aero_coefficients  # pylint: disable=import-outside-toplevel

    # 计算气动系数
    # Calculate aerodynamic coefficients
    AERO_COEFFS = get_aero_coefficients(k1=k1_val, k2=k2_val)
except ValueError as e:
    print(f"初始化气动参数时出错 Error initializing aerodynamic parameters : {e}")
    AERO_COEFFS = None  # 或者设置默认值并停止  or set default values and stop



# --- 初始条件 (Initial Conditions) ---
def setup_initial_conditions(_trajectory_type="spiral"):
    """
    根据轨迹类型设置初始条件

    参数：
        trajectory_type: "spiral", "figure8", "lemniscate", "linear"

    返回：
        X0: 完整的初始状态向量 [位置，姿态，线速度，角速度]
    """

    if _trajectory_type == "spiral":
        # 螺旋轨迹
        zeta0 = np.array([1500.0, 0.0, 0.0])  # 初始位置 (Initial Position) [m]
        gamma0 = np.array([0.0, 0.0, np.pi/2]) # 初始姿态 (Initial Attitude) [rad] (phi, theta, psi)
        v0 = np.array([0.0, 105.0, 14.0])  # r*omega, h_max/10*omega # 初始线速度 (Initial Linear Velocity) [m/s] (u, v, w)
        omega0 = np.array([0.0, 0.0, 0.07]) # 初始角速度 (Initial Angular Velocity) [rad/s] (p, q, r)

    elif _trajectory_type == "figure8":
        # 8 字形轨迹
        zeta0 = np.array([0.0, 0.0, -19000.0])
        gamma0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([9.0, 6.0, 1.0])  # a*omega, b*omega
        omega0 = np.array([0.0, 0.0, 0.003])

    elif _trajectory_type == "lemniscate":
        # 莱洛曲线轨迹
        zeta0 = np.array([2500.0, 0.0, -19000.0])
        gamma0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, 10.0, 0.8])
        omega0 = np.array([0.0, 0.0, 0.004])

    elif _trajectory_type == "linear":
        # 直线轨迹（默认设置）
        zeta0 = np.array([0.0, 0.0, -19000.0])
        gamma0 = np.array([0.0, 0.0, np.pi/4])  # 45 度朝向终点
        v0 = np.array([7.07, 7.07, 0.0])  # 10m/s分解到x,y方向
        omega0 = np.array([0.0, 0.0, 0.0])

    else:
        raise ValueError(f"不支持的轨迹类型：{_trajectory_type}")

    # 组合完整状态向量
    _X0 = np.concatenate([zeta0, gamma0, v0, omega0])

    print(f"【{_trajectory_type}轨迹】初始条件设置完成：")
    print(f"  初始位置：{zeta0}")
    print(f"  初始姿态：{gamma0} (rad)")
    print(f"  初始线速度：{v0} (m/s)")
    print(f"  初始角速度：{omega0} (rad/s)")

    return _X0

# 使用示例
trajectory_type = "spiral"  # 选择你要跟踪的轨迹类型
X0 = setup_initial_conditions(trajectory_type)






# === 扰动函数 (Disturbance Function delta(t) - Eq. 67) ===
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


# === 风速定义 (Wind Definition) ===
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







# === 扰动观测器参数 (Disturbance Observer Parameters) ===
# 固定时间扰动观测器参数
l1 = 2.0       # 一阶项系数
l2 = 1.0       # z2 计算系数
l3 = 1.5       # z2 反馈系数
l4 = 2.0       # 非线性项系数 1
l5 = 1.0       # 非线性项系数 2
beta1 = 0.5    # 非线性指数 1 (0 < beta1 < 1)
beta2 = 1.5    # 非线性指数 2 (beta2 > 1)

# 扰动补偿参数
do_compensation_gain = 0.9  # 补偿增益
do_filter_coeff = 0.8       # 滤波系数

# 约束边界函数 - 用于 BLF 算法（可选）
def kb_func(t):
    """随时间变化的误差约束"""
    kb = np.zeros(6)
    kb[0:3] = 10.0 + 5.0 * np.exp(-0.05 * t)  # 位置误差约束
    kb[3:6] = 0.2 + 0.1 * np.exp(-0.05 * t)   # 姿态误差约束
    return kb
