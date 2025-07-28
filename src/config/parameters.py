"""
# parameters.py
"""

# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg


# === Third-party Libraries ===
import numpy as np

from src.system.trajectory_ref import Trajectory













# # --- Initial position Conditions ---
# def setup_spiral_initial_conditions():
#     """
#     Setup initial conditions for spiral trajectory

#     Returns:
#         initial_state_vector: Complete initial state vector [position, attitude, velocity, angular_velocity]
#     """
#     # 从轨迹生成器获取 t=0 时的期望状态
#     trajectory = Trajectory()
#     yc, _yc_dot, _, _, _ = trajectory.get_straight_line_trajectory(0.0)

#     # 使用轨迹的初始状态作为初始条件
#     initial_position = yc[0:3]  # Initial position [m]
#     # initial_position[2] += 5  # Adjust altitude to 0.5m above ground
#     # initial_position = np.array([2500.0, 0.0, 0.0])
#     initial_attitude = np.array([0.0, 0.0, np.pi / 3])  # Initial attitude [rad] (roll, pitch, yaw) 面朝 Y 方向（90°朝右）
#     initial_velocity = np.array([25, 0.0, 0.0])  # 稍有向前速度（缓慢推进）  # Initial linear velocity [m/s] (u, v, w)
#     initial_angular_velocity = np.array([0.0, 0.0, 0.00])  # 缓慢右转（偏航角速度）  # Initial angular velocity [rad/s] (p, q, r)

#     _X0 = np.concatenate([initial_position, initial_attitude, initial_velocity, initial_angular_velocity])

#     print("Spiral trajectory initial conditions:")
#     print(f"  Initial position: {initial_position}")
#     print(f"  Initial attitude: {initial_attitude} (rad)")
#     print(f"  Initial linear velocity: {initial_velocity} (m/s)")
#     print(f"  Initial angular velocity: {initial_angular_velocity} (rad/s)")

#     return _X0


# # Set initial conditions
# X0 = setup_spiral_initial_conditions()


# ===  (Disturbance Function delta(t) - Eq. 67) ===
def disturbance_delta(t):
    """ Define external disturbance vector"""
    _ = t
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


# Alternatively, define a function for time/position varying wind
# def get_wind_erf(t, zeta):
#     # Calculate wind based on time/position)
#     wind_x = 5.0 + np.sin(0.01 * t) * 2.0
#     wind_y = 2.0 + np.cos(t / 100 + zeta[0]/50000) * 1.0
#     wind_z = 0.0
#     return np.array([wind_x, wind_y, wind_z])
# V_WIND_FUNC = get_wind_erf #  If using function, specify here



