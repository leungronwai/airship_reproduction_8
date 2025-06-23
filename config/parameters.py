"""
# parameters.py
"""


# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg

import numpy as np

# import physical calculation functions
from airship.physics import calculate_added_mass_inertia



# === simulation parameters (Simulation Parameters) ===
DT = 0.1  # simulation step size (Simulation step size) [s]
T_SPAN = 50  # simulation total time (Total simulation time) [s] # Reduced for faster testing, original paper used longer

# --- (Physical Parameters - Placeholder Values!) ---
# These values MUST be replaced based on the specific airship model
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
    #  calculate k1, k2
    # Calculate k1, k2 (assuming functions are defined in aero.py or elsewhere)
    k1_val, k2_val, _, = k1, k2, k3

    from config.aero_coefficients import get_aero_coefficients # noqa: E402 # delay import to avoid circular dependency
    # from config.aero_coefficients import get_aero_coefficients  # pylint: disable=import-outside-toplevel

    #  calculate aerodynamic coefficients
    # Calculate aerodynamic coefficients
    AERO_COEFFS = get_aero_coefficients(k1=k1_val, k2=k2_val)
except ValueError as e:
    print(f"Error initializing aerodynamic parameters : {e}")
    AERO_COEFFS = None  #  set default values and stop



# --- (Initial Conditions) ---
def setup_initial_conditions(_trajectory_type="spiral"):
    """
    set initial conditions based on trajectory type

    args:
        trajectory_type: "spiral", "figure8", "lemniscate", "linear"

    return:
        X0: complete initial state vector [position, attitude, linear velocity, angular velocity]
    """

    if _trajectory_type == "spiral":
        # spiral trajectory
        zeta0 = np.array([1500.0, 0.0, 0.0])  #  (Initial Position) [m]
        gamma0 = np.array([0.0, 0.0, np.pi/2]) #  (Initial Attitude) [rad] (phi, theta, psi)
        v0 = np.array([0.0, 105.0, 14.0])  # r*omega, h_max/10*omega #  (Initial Linear Velocity) [m/s] (u, v, w)
        omega0 = np.array([0.0, 0.0, 0.07]) #  (Initial Angular Velocity) [rad/s] (p, q, r)

    elif _trajectory_type == "figure8":
        # figure 8 trajectory
        zeta0 = np.array([0.0, 0.0, -19000.0])
        gamma0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([9.0, 6.0, 1.0])  # a*omega, b*omega
        omega0 = np.array([0.0, 0.0, 0.003])

    elif _trajectory_type == "lemniscate":
        # lemniscate trajectory
        zeta0 = np.array([2500.0, 0.0, -19000.0])
        gamma0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, 10.0, 0.8])
        omega0 = np.array([0.0, 0.0, 0.004])

    elif _trajectory_type == "linear":
        # linear trajectory (default setting)
        zeta0 = np.array([0.0, 0.0, -19000.0])
        gamma0 = np.array([0.0, 0.0, np.pi/4])  # 45 degree to the end
        v0 = np.array([7.07, 7.07, 0.0])  # 10m/s decomposed into x,y direction
        omega0 = np.array([0.0, 0.0, 0.0])

    else:
        raise ValueError(f"unsupported trajectory type: {_trajectory_type}")

    #  combine complete state vector
    _X0 = np.concatenate([zeta0, gamma0, v0, omega0])

    print(f"[{_trajectory_type} trajectory] initial conditions set:")
    print(f"  initial position: {zeta0}")
    print(f"  initial attitude: {gamma0} (rad)")
    print(f"  initial linear velocity: {v0} (m/s)")
    print(f"  initial angular velocity: {omega0} (rad/s)")

    return _X0

# example
trajectory_type = "spiral"  # select the trajectory type you want to track
X0 = setup_initial_conditions(trajectory_type)






# ===  (Disturbance Function delta(t) - Eq. 67) ===
def disturbance_delta(t):
    """ Define external disturbance vector"""
    d = np.zeros(6)
    d[0] = 0.5 + 2 * np.sin(0.1 * t)
    d[1] = 0.4 + 1.5 * np.cos(0.1 * t)
    d[2] = 0.6 + 1.5 * np.sin(0.1 * t)
    d[3] = 1.5 + 2 * np.sin(0.1 * t)
    d[4] = 1.5 + 1.5 * np.sin(0.1 * t)
    d[5] = 1.5 + 2 * np.cos(0.1 * t)
    return 5000 * d  # Paper scales by 5000


# ===  (Wind Definition) ===
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




# ===  (NMPC Controller Parameters) ===
#  prediction horizon length
N_HORIZON = 10

#  state weight matrix (position and attitude error weights)
Q = np.diag([10.0, 10.0, 10.0, 5.0, 5.0, 5.0,
              1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

#  control input weight matrix
R = np.diag([0.1, 0.1, 0.1])

#  terminal state weight matrix
Qf = np.diag([20.0, 20.0, 20.0, 10.0, 10.0, 10.0,
               2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

#  thrust boundary (N)
T_MIN = 0.0
T_MAX = 20.0

# μ parameter boundary (rad)
MU_MIN = -np.pi/4
MU_MAX = np.pi/4

# ν parameter boundary (rad)
NU_MIN = -np.pi/4
NU_MAX = np.pi/4







# === (Disturbance Observer Parameters) ===
#  fixed time disturbance observer parameters
l1 = 2.0       # first order term coefficient
l2 = 1.0       # z2 calculation coefficient
l3 = 1.5       # z2 feedback coefficient
l4 = 2.0       # nonlinear term coefficient 1
l5 = 1.0       # nonlinear term coefficient 2
beta1 = 0.5    # nonlinear index 1 (0 < beta1 < 1)
beta2 = 1.5    # nonlinear index 2 (beta2 > 1)

#  disturbance compensation parameters
do_compensation_gain = 0.9  # compensation gain
do_filter_coeff = 0.8       # filter coefficient

#  constraint boundary function - for BLF algorithm (optional)
def kb_func(t):
    """error constraint over time"""
    kb = np.zeros(6)
    kb[0:3] = 10.0 + 5.0 * np.exp(-0.05 * t)  # position error constraint
    kb[3:6] = 0.2 + 0.1 * np.exp(-0.05 * t)   # attitude error constraint
    return kb
