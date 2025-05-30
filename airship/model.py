# model.py
"""
Airship dynamic model module (model.py)
"""
# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot
# cspell:ignore arctan RUDT RUDB ELVL ELVR unmodeled
# === 标准库 ===
import sys
import os

# === 第三方库 ===
import numpy as np

import casadi as ca

# === 本地模块 ===
from config import parameters as params
from airship.aero_force_torque import calculate_aero_forces_moments, calculate_relative_velocity, calculate_aoa_sideslip
from airship.thrust import thrust_params_to_force_torque
from .utils import skew, R_zeta, R_block



# === 设置路径（如有需要） ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




class Airship:
    """
    气艇模型类
    """

    def __init__(self, initial_state):
        self.X = initial_state  # [zeta, gamma, v, omega] (12x1)
        self.M = params.M_cfg  # Combined inertia matrix from Eq. 9
        self.M_inv = params.M_inv
        self.m = params.m  # Mass
        self.g = params.g  # Gravity
        self.I0 = params.I0  # Inertia matrix I0
        self.M_upper_left = params.M_cfg[0:3, 0:3]  # m*I + M' from Eq. 9
        self.rc_vec = params.rc  # Vector CV->CG (shape (3,1))
        self.rb_vec = params.rb  # Vector CV->CB (shape (3,1))
        self.rp_r_vec = params.rp_r  # 右侧推力作用点向量 (Vector CV->CP Right)
        self.rp_l_vec = params.rp_l  # 左侧推力作用点向量 (Vector CV->CP Left)
        self.rc_skew = skew(self.rc_vec.flatten())  # Skew-symmetric matrix for rc

        # --- 添加计算浮力和气动所需的参数 (Add parameters for buoyancy/aero) ---
        self.Vol_airship = params.Vol_airship  # 体积 Volume
        self.rho_air = params.rho_air  # 空气密度 Air density
        self.S_ref = params.S_ref  # 参考面积 Reference Area
        self.L_ref = params.L_ref  # 参考长度 Reference Length

        # 添加对风速参数的引用
        self.V_wind_erf_const = params.V_WIND_ERF  # 如果风速是常数
        # self.V_wind_func = params.V_WIND_FUNC # 如果风速是函数


    def rhs(self, t, X, tau, disturbance_func):
        """计算状态向量 X 的导数 dX/dt - Right Hand Side"""
        zeta = X[0:3]
        gamma = X[3:6]
        v = X[6:9]  # Linear velocity in BRF [u, v, w]
        omega = X[9:12]  # Angular velocity in BRF [p, q, r]
        x_vec = X[6:12]  # Combined velocity state [v, omega]

        # Ensure v and omega are 1D arrays for calculations
        v_1d = v.flatten()
        omega_1d = omega.flatten()
        rc_1d = self.rc_vec.flatten()  # Vector from CG to CV
        rb_1d = self.rb_vec.flatten()  # Vector from CB to CV
        u, v_body, w = v_1d[0], v_1d[1], v_1d[2]
        p, q, r = omega_1d[0], omega_1d[1], omega_1d[2]

        # --- 运动学 (Kinematics - Eq. 5 / Eq. 12 first part) ---
        # 注意：运动学方程描述的是气艇相对于地面的运动
        R = R_block(gamma)  # Combined rotation matrix diag(R_zeta, R_y)
        y_dot = R @ x_vec  # [zeta_dot, gamma_dot]  (相对于地面)


        # --- 动力学 (Dynamics - Eq. 8 / Eq. 12 second part) ---

        # --- 计算 N 项 (Calculate N term - Eq. 10) ---
        # N = [ N1 ; N2 ] where N1 is 3x1 (forces) and N2 is 3x1 (torques)
        v_col = v.reshape(3, 1)
        omega_col = omega.reshape(3, 1)
        rc_col = self.rc_vec  # Already (3,1)  # Vector from CG to CV

        # Calculate common cross products
        omega_cross_v = np.cross(omega, v, axis=0).reshape(3, 1)  # omega x v
        omega_cross_rc = np.cross(omega, rc_col.flatten(), axis=0).reshape(3, 1)  # omega x rc
        omega_cross_omega_cross_rc = np.cross(omega, omega_cross_rc.flatten(), axis=0).reshape(3, 1)  # omega x (omega x rc)
        omega_cross_I0_omega = np.cross(omega, (self.I0 @ omega_col).flatten(), axis=0).reshape(3, 1)  # omega x (I0*omega)
        rc_cross_omega_cross_v = np.cross(rc_col.flatten(), omega_cross_v.flatten(), axis=0).reshape(3, 1)  # rc x (omega x v)

        # N1 = (m*I + M') * (omega x v) + m * omega x (omega x rc)
        # Note: M_upper_left already contains (m*I + M')
        N1 = self.M_upper_left @ omega_cross_v + self.m * omega_cross_omega_cross_rc

        # N2 = omega x (I0*omega) + m*rc x (omega x v)
        N2 = omega_cross_I0_omega + self.m * rc_cross_omega_cross_v

        N_term = np.vstack((N1, N2)).flatten()  # Combine N1 and N2 into 6x1 vector

        # === 计算 F 项 (Calculate F term - Eq. 11) ===
        # F = [ F_forces ; F_torques ]
        # F_forces = fg - fb + fa
        # F_torques = mg + mb + ma

        # === 计算重力力和力矩 (Calculate Gravity Force and Torque) ===
        # fg: Gravity force in BRF
        Rz = R_zeta(gamma)  # Rotation from BRF to ERF
        gravity_ERF = np.array([[0], [0], [self.m * self.g]])  # Gravity in Earth Frame
        fg_BRF = Rz.T @ gravity_ERF  # Rotate gravity vector to Body Frame

        # mg: Gravity torque in BRF
        # Torque = r_cg x F_g.  Since F_g acts at CG, r_cg = 0.
        mg_BRF = np.cross(rc_1d, fg_BRF.flatten()).reshape(3, 1)  # Torque due to gravity acting at CG (rc is CV->CG)

        # === 计算浮力和浮力矩 (Calculate Buoyancy Force and Torque) ===
        # fb: Buoyancy force in BRF
        # Requires displaced volume V and air density rho_air.
        F_buoyancy_ERF = np.array([[0], [0], [-self.Vol_airship * self.rho_air * self.g]])  # 向上为负 Z
        fb_BRF = Rz.T @ F_buoyancy_ERF  # Rotate buoyancy vector to Body Frame

        # mb: Buoyancy torque in BRF
        # Torque = r_cb x F_b. r_cb = vector from CV to CB. Assume r_cb = -rc_vec.
        # Requires fb_BRF which is assumed zero here.
        # Torque due to buoyancy acting at CB (assumed at CV, so arm is -rb)
        mb_BRF = np.cross(-rb_1d, fb_BRF.flatten(), axis=0).reshape(3, 1)
        # If fb_BRF was non-zero:
        # r_cb = -self.rb_vec
        # mb_BRF = np.cross(r_cb.flatten(), fb_BRF.flatten(), axis=0).reshape(3,1)

        # === 新增：计算相对速度 (New: Calculate Relative Velocity) ===
        # 获取风速 (Get wind velocity)
        # 如果使用常数风速：/ if using constant wind speed:
        V_wind_ERF = self.V_wind_erf_const
        # 如果使用函数风速：/ if using function wind speed:
        # V_wind_ERF = self.V_wind_func(t, zeta)

        # 将风速转换到体轴系 (Transform wind to Body Frame)
        V_wind_BRF = Rz.T @ V_wind_ERF.reshape(3, 1)
        V_wind_BRF_1d = V_wind_BRF.flatten()

        # 计算相对速度 (Calculate relative velocity - - relative Airspeed in BRF)
        v_airship_brf_1d = v_1d  # Body Reference Frame - BRF
        v_rel_brf_1d, u_rel, v_rel_body, w_rel = calculate_relative_velocity(v_airship_brf_1d, V_wind_BRF_1d)


        # === 计算气动力和力矩 (Calculate Aerodynamic Forces and Moments) ===
        # fa: Aerodynamic force in BRF  placeholder
        # 动压 (Dynamic Pressure  - based on relative speed)
        V_rel_mag = np.linalg.norm(v_rel_brf_1d)
        q_dyn = 0.5 * self.rho_air * V_rel_mag**2 if V_rel_mag > 1e-3 else 0  # Avoid division by zero

        # 攻角和侧滑角 (Angle of Attack & Sideslip Angle - based on relative speed) - 确保 u > 0
        alpha, beta = calculate_aoa_sideslip(u_rel, v_rel_body, w_rel, V_rel_mag)

        # --- 获取控制舵面偏转角 (Get Control Surface Deflections) ---
        # !!! 关键占位符：这些值需要由控制分配模块根据 tau[3:6] 确定 !!!
        # !!! Critical Placeholder: These values need to be determined by a Control Allocation module based on tau[3:6] !!!
        delta_RUDT = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_RUDB = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVL = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVR = np.deg2rad(0.0)  # [rad] - Placeholder

        # --- 计算气动力和力矩 (Calculate Aerodynamic Forces and Moments) ---
        fa_BRF, ma_BRF = calculate_aero_forces_moments(
            q_dyn, alpha, beta,
            params.AERO_COEFFS,
            delta_RUDT, delta_RUDB, delta_ELVL, delta_ELVR
        )

        # === 计算推力和推力矩 (Calculate Thrust and torque) ===
        print(type(tau)) # 确保 tau 是 NumPy 数组
        print(tau)
        # 使用 thrust_params_to_force_torque 将推力参数转换为推力和推力矩
        thrust_torque = thrust_params_to_force_torque(tau, self.rp_r_vec, self.rp_l_vec)
        print(thrust_torque)

        T_total = thrust_torque[0:3].reshape(3, 1)  # 推力矢量 - Thrust vector
        tau_vec = thrust_torque[3:6].reshape(3, 1)  # 推力矩矢量 - Torque vector

        # === 组合力和力矩 (Combine Forces and Torques) ===
        F_forces = fg_BRF - fb_BRF + fa_BRF + T_total
        print(F_forces.shape)
        F_torques = mg_BRF + mb_BRF + ma_BRF + tau_vec
        print(F_torques.shape)

        F_term = np.vstack((F_forces, F_torques)).flatten()  # Combine forces and torques into 6x1 vector

        # === 获取扰动 (Get Disturbance) ===
        # This is the external/unmodeled disturbance delta from the paper
        d = disturbance_func(t)

        # --- 动力学方程 (Dynamics Equation): Mx_dot + N = F + tau + d ---
        # Rearranging for x_dot: x_dot = M_inv * (F - N + tau + d)
        x_dot = self.M_inv @ (F_term - N_term + thrust_torque + d)

        # --- 组合状态导数 (Combine state derivatives) ---
        dXdt = np.concatenate((y_dot, x_dot))

        print(dXdt.shape)

        return dXdt

    def update_state(self, X_dot, dt):
        """使用欧拉积分更新状态 (Update state using Euler integration)"""
        # NOTE: Using RK4 in simulation.py is preferred for accuracy.
        # This Euler update is kept for potential direct use but not used by main loop.
        self.X = self.X + X_dot * dt
        # Normalize angles if needed (e.g., psi to [-pi, pi])
        self.X[5] = (self.X[5] + np.pi) % (2 * np.pi) - np.pi  # Normalize Psi
        self.X[4] = np.clip(self.X[4], -np.pi / 2 + 0.01, np.pi / 2 - 0.01)  # Keep Theta away from singularity

    def get_state(self):
        """获取当前状态 (Get current state)"""
        return self.X

    def get_pose(self):
        """获取当前姿态 (Get current pose)"""
        return self.X[0:6]  # zeta, gamma

    def get_velocity(self):
        """获取当前速度 (Get current velocity)"""
        return self.X[6:12]  # v, omega


class AirshipCasADiSymbolic:
    """
    气艇符号模型类
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
        # ca = __import__("casadi")  # dynamic import
        _ = t

        # === 解构状态 ===
        zeta = X[0:3]  # Position in ERF
        gamma = X[3:6]  # Attitude (Euler angles)
        v = X[6:9]  # Linear velocity in BRF
        omega = X[9:12]  # Angular velocity in BRF

        # === 运动学 (Kinematics) ===
        R = R_block(gamma)  # Combined rotation matrix
        y_dot = R @ ca.vertcat(v, omega)  # [zeta_dot, gamma_dot]



        # === 动力学 (Dynamics) ===

        #=================================================================
        #                     Coriolis and Centrifugal Effects
        #=================================================================
        # --- Calculate N term (Coriolis and centrifugal effects) ---
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
        # !!! 关键占位符：这些值需要由控制分配模块根据 tau[3:6] 确定 !!!
        # !!! Critical Placeholder: These values need to be determined by a Control Allocation module based on tau[3:6] !!!
        delta_RUDT = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_RUDB = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVL = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVR = np.deg2rad(0.0)  # [rad] - Placeholder

        # 使用提取的函数计算气动力和力矩
        fa_BRF, ma_BRF = calculate_aero_forces_moments(
            q_dyn, alpha, beta,
            self.AERO_COEFFS,
            delta_RUDT, delta_RUDB, delta_ELVL, delta_ELVR,
            use_casadi=True
        )



        #========================================================================
        #                     Thrust and torque
        #========================================================================
        # 检查 U 的维度
        print(U.shape)
        # 直接将推力参数转换为力和力矩
        U_vec = thrust_params_to_force_torque(U, self.rp_r, self.rp_l, use_casadi=True)


        print(U_vec.shape)
        T_total = U_vec[0:3]  # Thrust vector
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

        return dXdt

    def get_nmpc_model(self):
        """
        Create a CasADi function for use in NMPC.

        Returns:
            f: CasADi Function that maps (x, u) to xdot
        """
        # ca = __import__("casadi")

        # Define symbolic variables
        x = ca.SX.sym("x", 12)  # State
        u = ca.SX.sym("u", 3)  # Control (T, μ, ν)  , delta_RUDT, delta_RUDB, delta_ELVL, delta_ELVR)

        # Get dynamics
        xdot = self.rhs_symbolic(x, u)

        # Create function
        f = ca.Function("f", [x, u], [xdot], ["x", "u"], ["xdot"])

        return f

    def discrete_time_model(self, dt, integration_method="rk4"):
        """
        Get a discrete-time model using various integration methods.

        Args:
            dt: Time step
            integration_method: 'euler', 'rk4', etc.

        Returns:
            F: CasADi Function that maps (x, u) to x_next
        """
        # ca = __import__("casadi")

        # Define symbolic variables
        x = ca.SX.sym("x", 12)
        u = ca.SX.sym("u", 7)

        # Get continuous dynamics
        xdot = self.rhs_symbolic(x, u)

        # Create discrete model based on integration method
        if integration_method == "euler":
            x_next = x + dt * xdot
        elif integration_method == "rk4":
            # RK4 integration
            k1 = xdot
            k2 = self.rhs_symbolic(x + dt / 2 * k1, u)
            k3 = self.rhs_symbolic(x + dt / 2 * k2, u)
            k4 = self.rhs_symbolic(x + dt * k3, u)
            x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

        # Create a function
        F = ca.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

        return F
