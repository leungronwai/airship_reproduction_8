"""
# parameters.py
"""


# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg


# === Third-party Libraries ===
import numpy as np

# import physical calculation functions
from src.system.added_mass_inertia import calculate_added_mass_inertia
from src.system.trajectory_ref import Trajectory





# === simulation parameters (Simulation Parameters) ===
DT = 1  # simulation step size (Simulation step size) [s]
T_SPAN = 20  # simulation total time (Total simulation time) [s] # Reduced for faster testing, original paper used longer

# --- (Physical Parameters - Placeholder Values!) ---
m = 9400  # mass (Mass) [kg] - Placeholder
Jx, Jy, Jz = 2e6, 5.5e6, 5.5e6  # moments of inertia (Moments of Inertia) [kg*m^2] - Placeholder
I0 = np.diag([Jx, Jy, Jz])  # inertia matrix (Inertia Matrix)


# --- airship geometric parameters (airship Geometric Parameters) ---
airship_a1 = 73.50  # [m] - Placeholder
airship_a2 = 62.5  # [m] - Placeholder (if there is only one a, then a1=a2=a)
airship_b = 19.0  # [m] - Placeholder


rc = np.array([[0], [0], [0.50]])  # distance vector CV to CG (Distance vector CV to CG) - Placeholder
rb = np.array([[0], [0], [0]])  # distance vector CV to CB (Distance vector CV to CB) - Placeholder
rp_r = np.array([[0.8 * airship_a1], [airship_b], [2.0]])  # thrust point in front right down (Thrust point in front right down) - Placeholder
rp_l = np.array([[0.8 * airship_a1], [-airship_b], [2.0]])  # thrust point in front left down (Thrust point in front left down) - Placeholder
g = 9.74  #  (Gravitational acceleration) [m/s^2]
Vol_airship = 10700  # (Airship Volume) [m^3] - Placeholder
rho_air = 0.088  # (Air density at ~20km altitude) [kg/m^3] - Placeholder
S_ref = 500.0  # (Reference Area for Aero) [m^2] - Placeholder
L_ref = 38.0  #  (Reference Length for Aero Moments) [m] - Placeholder


# ===  (Environmental Parameters) ===
rho_air_at_altitude = rho_air  # [kg/m^3]  air dense @ ~20km - Placeholder

# ===  (Calculate Added Mass/Inertia) ===
try:
    M_prime, I0_prime, k1, k2, k3 = calculate_added_mass_inertia(airship_a1, airship_a2, airship_b, rho_air_at_altitude)
except ValueError as e:
    print(f" Error calculating added mass/inertia: {e}")
    #  Set default values or stop simulation
    M_prime = np.zeros((3, 3))  # Fallback
    I0_prime = np.zeros((3, 3))  # Fallback

#  (Combined Inertia Matrices - Eq. 9)  Configuration
M_cfg = np.zeros((6, 6))
rc_skew = np.array([[0, -rc[2, 0],rc[1, 0]],
                    [rc[2, 0], 0, -rc[0, 0]],
                    [-rc[1, 0], rc[0, 0], 0]])
M_cfg[0:3, 0:3] = m * np.identity(3) + M_prime
M_cfg[0:3, 3:6] = -m * rc_skew
M_cfg[3:6, 0:3] = m * rc_skew
M_cfg[3:6, 3:6] = I0 + I0_prime
M_inv = np.linalg.inv(M_cfg)  #  (Pre-compute inverse matrix)



# ===  (Aerodynamic Parameters) ===
try:
    # Calculate k1, k2, k3 based on added mass factors
    k1_val, k2_val, _, = k1, k2, k3

    from src.config.aero_coefficients import get_aero_coefficients # noqa: E402

    # Calculate aerodynamic coefficients
    AERO_COEFFS = get_aero_coefficients(k1=k1_val, k2=k2_val)
except ValueError as e:
    print(f"Error initializing aerodynamic parameters : {e}")
    AERO_COEFFS = None  #  set default values and stop



# --- Initial position Conditions ---
def setup_spiral_initial_conditions():
    """
    Setup initial conditions for spiral trajectory

    Returns:
        initial_state_vector: Complete initial state vector [position, attitude, velocity, angular_velocity]
    """
    # 从轨迹生成器获取 t=0 时的期望状态
    trajectory = Trajectory()
    yc, yc_dot, _, _, _ = trajectory.get_spiral_trajectory(0.0)

    # 使用轨迹的初始状态作为初始条件
    # initial_position = yc[0:3]  # Initial position [m]
    # initial_position[2] += 5  # Adjust altitude to 0.5m above ground
    initial_position = np.array([2500.0, 0.0, 0.0])
    initial_attitude = np.array([0.0, 0.0, np.pi / 2])  # Initial attitude [rad] (roll, pitch, yaw) 面朝 Y 方向（90°朝右）
    initial_velocity = np.array([0, 0.0, 0.0])               # 稍有向前速度（缓慢推进）  # Initial linear velocity [m/s] (u, v, w)
    initial_angular_velocity = np.array([0.0, 0.0, 0.00])      # 缓慢右转（偏航角速度）  # Initial angular velocity [rad/s] (p, q, r)

    _X0 = np.concatenate([initial_position, initial_attitude, initial_velocity, initial_angular_velocity])

    print("Spiral trajectory initial conditions:")
    print(f"  Initial position: {initial_position}")
    print(f"  Initial attitude: {initial_attitude} (rad)")
    print(f"  Initial linear velocity: {initial_velocity} (m/s)")
    print(f"  Initial angular velocity: {initial_angular_velocity} (rad/s)")



    return _X0

# Set initial conditions
X0 = setup_spiral_initial_conditions()






# ===  (Disturbance Function delta(t) - Eq. 67) ===
def disturbance_delta(t):
    """ Define external disturbance vector"""
    d = np.zeros(6)
    # d[0] = 0.5 + 2 * np.sin(0.1 * t)
    # d[1] = 0.4 + 1.5 * np.cos(0.1 * t)
    # d[2] = 0.6 + 1.5 * np.sin(0.1 * t)
    # d[3] = 1.5 + 2 * np.sin(0.1 * t)
    # d[4] = 1.5 + 1.5 * np.sin(0.1 * t)
    # d[5] = 1.5 + 2 * np.cos(0.1 * t)
    return d  # Paper scales by 5000


# ===  Wind Definition ===
#  You need to add a parameter representing the wind speed. The wind speed is usually defined in the Earth Reference Frame (ERF).
# It can be a constant, or a function of time and/or position.
# --- (Wind Definition) ---
# Example: Constant wind in ERF (e.g., 5m/s West, 2m/s South, 0m/s Vertical)
V_WIND_ERF = np.array([5.0, 2.0, 0.0])


# Alternatively, define a function for time/position varying wind
# def get_wind_erf(t, zeta):
#     # Calculate wind based on time/position)
#     wind_x = 5.0 + np.sin(0.01 * t) * 2.0
#     wind_y = 2.0 + np.cos(t / 100 + zeta[0]/50000) * 1.0
#     wind_z = 0.0
#     return np.array([wind_x, wind_y, wind_z])
# V_WIND_FUNC = get_wind_erf #  If using function, specify here






# ===  (do-mpc Controller Parameters) ===
#  prediction horizon length
N_HORIZON = 8

#  state weight matrix (position and attitude error weights)
Q = np.diag([10.0, 10.0, 10.0,
            5.0, 5.0, 5.0,
            1.0, 1.0, 1.0,
            0.5, 0.5, 0.5])

# 控制输入权重矩阵 (平衡控制能耗与性能)
R = np.diag([5.0, 10.0, 10.0])  # [推力，水平偏转，垂直偏转]

#  terminal state weight matrix
Qf = np.diag([20.0, 20.0, 20.0,
            10.0, 10.0, 10.0,
            2.0, 2.0, 2.0,
            1.0, 1.0, 1.0])