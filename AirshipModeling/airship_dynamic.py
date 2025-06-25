# model.py
"""
Airship dynamic model module (model.py)
refer to      Error-constrained fixed-time trajectory tracking control for a stratospheric airship with disturbances
"""
# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot
# cspell:ignore arctan RUDT RUDB ELVL ELVR unmodeled

# === standard libraries ===
import sys
import os

# === third-party libraries ===
import numpy as np
import casadi as ca

# === local modules ===
from AirshipModeling.aero_force_torque import calculate_aero_forces_moments, calculate_relative_velocity, calculate_aoa_sideslip
from AirshipModeling.thrust_vectoring import thrust_params_to_force_torque
from AirshipModeling.rotation_matrices import R_zeta, R_block

# === set the path (if needed) ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))







class AirshipCasADiSymbolic:
    """
    Airship symbolic model class
    """
    def __init__(self, input_params):
        self.params = input_params
        self.m = input_params.m
        self.g = input_params.g
        self.I0 = input_params.I0
        self.M = input_params.M_cfg
        self.M_inv = input_params.M_inv
        self.M_upper_left = self.M[0:3, 0:3]
        self.rc = input_params.rc.flatten()
        self.rb = input_params.rb.flatten()
        self.rp_r = input_params.rp_r.flatten()
        self.rp_l = input_params.rp_l.flatten()
        self.Vol_airship = input_params.Vol_airship
        self.rho_air = input_params.rho_air
        self.S_ref = input_params.S_ref
        self.L_ref = input_params.L_ref
        self.V_wind = input_params.V_WIND_ERF
        self.AERO_COEFFS = input_params.AERO_COEFFS

    def rhs_symbolic(self, X, U, t=None, external_disturbance=None):
        """
        Build symbolic RHS using CasADi.

        Args:
            X: 12x1 casadi SX state vector [zeta, gamma, v, omega]
            U: Control vector - either:
               - 3x1 [T, μ, v] for direct thrust control
               - 6x1 [T, μ, v, delta_RUDT, delta_RUDB, delta_ELVL, delta_ELVR] for full control
            t: Time (optional)
            external_disturbance: Optional external disturbance (6x1)

        Returns:
            dX/dt as casadi SX 12x1
        """

        _ = t

        # === Deconstruct the state ===
        _zeta = X[0:3]  # Position in ERF
        gamma = X[3:6]  # Attitude (Euler angles)
        v = X[6:9]  # Linear velocity in BRF
        omega = X[9:12]  # Angular velocity in BRF

        # === 运动学 (Kinematics) ===
        R = R_block(gamma)  # Combined rotation matrix R = diag(R_zeta, R_gamma) eq.(5)
        y_dot = R @ ca.vertcat(v, omega)  # y_dot = [zeta_dot, gamma_dot] eq.(5)



        # === 动力学 (Dynamics) ===
        #=================================================================
        #                     Coriolis and Centrifugal Effects
        #=================================================================
        # --- Calculate N term (Coriolis and centrifugal effects) --- eq.(10)
        omega_cross_v = ca.cross(omega, v)
        omega_cross_rc = ca.cross(omega, self.rc)
        omega_cross_omega_cross_rc = ca.cross(omega, omega_cross_rc)
        omega_cross_I0_omega = ca.cross(omega, self.I0 @ omega)
        rc_cross_omega_cross_v = ca.cross(self.rc, omega_cross_v)

        N1 = self.M_upper_left @ omega_cross_v + self.m * omega_cross_omega_cross_rc
        N2 = omega_cross_I0_omega + self.m * rc_cross_omega_cross_v
        N_term = ca.vertcat(N1, N2)

        # =============================================================
        #                     Gravity force and moment
        # =============================================================
        Rz = R_zeta(gamma)
        fg_earth = ca.vertcat(0, 0, self.m * self.g)  # Gravity in Earth Frame
        fg_BRF = Rz.T @ fg_earth  # Rotate gravity vector to Body Frame
        mg_BRF = ca.cross(self.rc, fg_BRF)  # Torque due to gravity acting at CG (rc is CV->CG)


        #========================================================================
        #                     Buoyancy force and moment
        #========================================================================
        F_buoy_earth = ca.vertcat(0, 0, -self.rho_air * self.Vol_airship * self.g)
        fb_BRF = Rz.T @ F_buoy_earth
        mb_BRF = ca.cross(-self.rb, fb_BRF)  # Torque due to buoyancy acting at CB (assumed at CV, so arm is -rb)


        #========================================================================
        #                     aerodynamic forces and moments
        #========================================================================
        # === Wind and relative velocity calculation ===
        V_wind_ERF = self.V_wind  # Wind velocity in ERF
        V_wind_BRF = Rz.T @ V_wind_ERF  # Transform wind to BRF

        # Calculate relative velocity
        v_rel_brf, u_rel, v_rel_body, w_rel = calculate_relative_velocity(v, V_wind_BRF)

        # Dynamic pressure
        V_rel_mag = ca.norm_2(v_rel_brf)
        q_dyn = 0.5 * self.rho_air * V_rel_mag**2

        # Angle of attack and sideslip angle
        alpha, beta = calculate_aoa_sideslip(u_rel, v_rel_body, w_rel, V_rel_mag, use_casadi=True)

        # === 获取控制舵面偏转角 (Get Control Surface Deflections) ===

        # Critical Placeholder: These values need to be determined by a Control Allocation module based on tau[3:6] !!!
        delta_RUDT = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_RUDB = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVL = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVR = np.deg2rad(0.0)  # [rad] - Placeholder

        # Use the extracted function to calculate aerodynamic forces and moments
        fa_BRF, ma_BRF = calculate_aero_forces_moments(
            q_dyn, alpha, beta,
            self.AERO_COEFFS,
            delta_RUDT, delta_RUDB, delta_ELVL, delta_ELVR,
            use_casadi=True
        )


        #========================================================================
        #                     Thrust and torque
        #========================================================================
        # check U dimension
        print(U.shape)
        # convert thrust parameters to force and torque
        U_vec = thrust_params_to_force_torque(U, self.rp_r, self.rp_l, use_casadi=True)


        print(U_vec.shape)
        T_total = U_vec[0:3]  # Thrust vectoring in BRF eq.(??)
        tau_vec = U_vec[3:6]



        #==================================================================
        #                     Combine forces and moments
        #==================================================================
        F_forces = fg_BRF - fb_BRF + fa_BRF + T_total
        F_torques = mg_BRF + mb_BRF + ma_BRF + tau_vec
        F_term = ca.vertcat(F_forces, F_torques)

        # --- Add external disturbance if provided ---
        if external_disturbance is not None:
            F_term = F_term + external_disturbance

        # --- Dynamics equation: Mx_dot + N = F ---
        x_dot = self.M_inv @ (F_term - N_term)

        # --- Combine state derivatives ---
        dXdt = ca.vertcat(y_dot, x_dot)
        print(dXdt.shape)

        return dXdt