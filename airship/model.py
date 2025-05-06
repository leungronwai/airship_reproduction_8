# model.py
import sys
sys.path.append(".")  # 添加上级目录到路径中 / add parent directory to path
import numpy as np
from utils import skew, R_zeta, R_y, S_omega, R_block
from config import parameters as params
import casadi as ca





class Airship:
    def __init__(self, initial_state):
        self.X = initial_state # [zeta, gamma, v, omega] (12x1)
        self.M = params.M_cfg # Combined inertia matrix from Eq. 9
        self.M_inv = params.M_inv
        self.m = params.m # Mass
        self.g = params.g # Gravity
        self.I0 = params.I0 # Inertia matrix I0
        self.M_upper_left = params.M_cfg[0:3, 0:3] # m*I + M' from Eq. 9
        self.rc_vec = params.rc # Vector CV->CG (shape (3,1))
        self.rb_vec = params.rb # Vector CV->CB (shape (3,1))
        self.rp_r_vec = params.rp_r # 右侧推力作用点向量 (Vector CV->CP Right)
        self.rp_l_vec = params.rp_l # 左侧推力作用点向量 (Vector CV->CP Left)
        self.rc_skew = skew(self.rc_vec.flatten()) # Skew-symmetric matrix for rc

        # --- 添加计算浮力和气动所需的参数 (Add parameters for buoyancy/aero) ---
        self.Vol_airship = params.Vol_airship # 体积 Volume
        self.rho_air = params.rho_air     # 空气密度 Air density
        self.S_ref = params.S_ref         # 参考面积 Reference Area
        self.L_ref = params.L_ref         # 参考长度 Reference Length


        # 添加对风速参数的引用
        self.V_wind_erf_const = params.V_WIND_ERF # 如果风速是常数
        # self.V_wind_func = params.V_WIND_FUNC # 如果风速是函数


    def rhs(self, t, X, tau, disturbance_func):
        """计算状态向量X的导数 dX/dt - Right Hand Side"""
        zeta = X[0:3]
        gamma = X[3:6]
        v = X[6:9]      # Linear velocity in BRF [u, v, w]
        omega = X[9:12] # Angular velocity in BRF [p, q, r]
        x_vec = X[6:12] # Combined velocity state [v, omega]

        # Ensure v and omega are 1D arrays for calculations
        v_1d = v.flatten()
        omega_1d = omega.flatten()
        rc_1d = self.rc_vec.flatten()   # Vector from CG to CV
        rb_1d = self.rb_vec.flatten()    # Vector from CB to CV
        u, v_body, w = v_1d[0], v_1d[1], v_1d[2]
        p, q, r = omega_1d[0], omega_1d[1], omega_1d[2]


        # --- 运动学 (Kinematics - Eq. 5 / Eq. 12 first part) ---
        # 注意：运动学方程描述的是气艇相对于地面的运动
        R = R_block(gamma) # Combined rotation matrix diag(R_zeta, R_y)
        y_dot = R @ x_vec               # [zeta_dot, gamma_dot]  (相对于地面)

        # --- 运动学 (Kinematics) ---
        # R = R_block(gamma)
        # y_dot = R @ x_vec.reshape(-1, 1)
        # y_dot = y_dot.flatten()



        # --- 动力学 (Dynamics - Eq. 8 / Eq. 12 second part) ---

        # --- 计算 N 项 (Calculate N term - Eq. 10) ---
        # N = [ N1 ; N2 ] where N1 is 3x1 (forces) and N2 is 3x1 (torques)
        v_col = v.reshape(3, 1)
        omega_col = omega.reshape(3, 1)
        rc_col = self.rc_vec # Already (3,1)  # Vector from CG to CV

        # Calculate common cross products
        omega_cross_v = np.cross(omega, v, axis=0).reshape(3, 1)   # omega x v
        omega_cross_rc = np.cross(omega, rc_col.flatten(), axis=0).reshape(3, 1)  # omega x rc
        omega_cross_omega_cross_rc = np.cross(omega, omega_cross_rc.flatten(), axis=0).reshape(3, 1)  # omega x (omega x rc)
        omega_cross_I0_omega = np.cross(omega, (self.I0 @ omega_col).flatten(), axis=0).reshape(3, 1)  # omega x (I0*omega)
        rc_cross_omega_cross_v= np.cross(rc_col.flatten(), omega_cross_v.flatten(), axis=0).reshape(3, 1)  # rc x (omega x v)

        # N1 = (m*I + M') * (omega x v) + m * omega x (omega x rc)
        # Note: M_upper_left already contains (m*I + M')
        N1 = self.M_upper_left @ omega_cross_v + self.m * omega_cross_omega_cross_rc

        # N2 = omega x (I0*omega) + m*rc x (omega x v)
        N2 = omega_cross_I0_omega + self.m * rc_cross_omega_cross_v

        N_term = np.vstack((N1, N2)).flatten() # Combine N1 and N2 into 6x1 vector


        # --- 计算 F 项 (Calculate F term - Eq. 11) ---
        # F = [ F_forces ; F_torques ]
        # F_forces = fg - fb + fa
        # F_torques = mg + mb + ma

        # fg: Gravity force in BRF
        Rz = R_zeta(gamma) # Rotation from BRF to ERF
        gravity_ERF = np.array([[0], [0], [self.m * self.g]]) # Gravity in Earth Frame
        fg_BRF = Rz.T @ gravity_ERF # Rotate gravity vector to Body Frame

        # mg: Gravity torque in BRF
        # Torque = r_cg x F_g. Since F_g acts at CG, r_cg = 0.
        mg_BRF = np.cross(rc_1d, fg_BRF.flatten()).reshape(3, 1) # Torque due to gravity acting at CG (rc is CV->CG)
        

        # fb: Buoyancy force in BRF
        # Requires displaced volume V and air density rho_air.
        F_buoyancy_ERF = np.array([[0], [0], [-self.Vol_airship * self.rho_air * self.g]]) # 向上为负Z
        fb_BRF = Rz.T @ F_buoyancy_ERF  # Rotate buoyancy vector to Body Frame

        # mb: Buoyancy torque in BRF
        # Torque = r_cb x F_b. r_cb = vector from CV to CB. Assume r_cb = -rc_vec.
        # Requires fb_BRF which is assumed zero here.
        # Torque due to buoyancy acting at CB (assumed at CV, so arm is -rb)
        mb_BRF = np.cross(-rb_1d, fb_BRF.flatten()).reshape(3, 1)
        # If fb_BRF was non-zero:
        # r_cb = -self.rb_vec
        # mb_BRF = np.cross(r_cb.flatten(), fb_BRF.flatten(), axis=0).reshape(3,1)


        # --- 新增：计算相对速度 (New: Calculate Relative Velocity) ---
        # 获取风速 (Get wind velocity)
        # 如果使用常数风速: / if using constant wind speed:
        V_wind_ERF = self.V_wind_erf_const
        # 如果使用函数风速: / if using function wind speed:
        # V_wind_ERF = self.V_wind_func(t, zeta)

        # 将风速转换到体轴系 (Transform wind to Body Frame)
        V_wind_BRF = Rz.T @ V_wind_ERF.reshape(3, 1)
        V_wind_BRF_1d = V_wind_BRF.flatten()

        # 计算相对速度 (Calculate relative velocity - - relative Airspeed in BRF)
        v_ground_brf_1d = v_1d   #Body Reference Frame - BRF
        v_rel_brf_1d = v_ground_brf_1d - V_wind_BRF_1d
        u_rel, v_rel_body, w_rel = v_rel_brf_1d[0], v_rel_brf_1d[1], v_rel_brf_1d[2]


        # fa: Aerodynamic force in BRF  placeholder
        # 动压 (Dynamic Pressure  - based on relative speed)
        V_rel_mag = np.linalg.norm(v_rel_brf_1d)
        q_dyn = 0.5 * self.rho_air * V_rel_mag**2 if V_rel_mag > 1e-3 else 0 # Avoid division by zero

        # 攻角和侧滑角 (Angle of Attack & Sideslip Angle - based on relative speed) - 确保 u > 0
        alpha = np.arctan2(w_rel, u_rel) if abs(u_rel) > 1e-3 else np.sign(w_rel)*np.pi/2
        beta = np.arcsin(v_rel_body / V_rel_mag) if V_rel_mag > 1e-3 else 0


        # --- 获取控制舵面偏转角 (Get Control Surface Deflections) ---
        # !!! 关键占位符: 这些值需要由控制分配模块根据 tau[3:6] 确定 !!!
        # !!! Critical Placeholder: These values need to be determined by a Control Allocation module based on tau[3:6] !!!
        delta_RUDT = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_RUDB = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVL = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVR = np.deg2rad(0.0)  # [rad] - Placeholder

        # --- 加载气动系数 (Load aerodynamic coefficients from parameters) ---
        # --- 占位符气动系数 (Placeholder Aero Coefficients) ---
        # airship_model.py (rhs 方法中)
        # ...
        # --- 加载气动系数 (Load aerodynamic coefficients from dictionary) ---
        # 假设 AERO_COEFFS 字典可以通过 params 或 self 访问 / Assuming AERO_COEFFS dict can be accessed via params or self
        # aero_data = params.AERO_COEFFS # or self.aero_coeffs 
        aero_data = params.AERO_COEFFS  # 如果作为全局变量或参数传入 / If AERO_COEFFS is passed as a global variable or parameter

        Cx1 = aero_data['Cx1']
        Cx2 = aero_data['Cx2']
        Cy1 = aero_data['Cy1']
        Cy2 = aero_data['Cy2']
        Cy3 = aero_data['Cy3']
        Cy4 = aero_data['Cy4']
        Cz1 = aero_data['Cz1']
        Cz2 = aero_data['Cz2']
        Cz3 = aero_data['Cz3']
        Cz4 = aero_data['Cz4']
        Cl1 = aero_data['Cl1']
        Cl2 = aero_data['Cl2']
        Cm1 = aero_data['Cm1']
        Cm2 = aero_data['Cm2']
        Cm3 = aero_data['Cm3']
        Cm4 = aero_data['Cm4']
        Cn1 = aero_data['Cn1']
        Cn2 = aero_data['Cn2']
        Cn3 = aero_data['Cn3']
        Cn4 = aero_data['Cn4']


        # --- 计算气动力 (Calculate Aerodynamic Forces - Eq. 23-25) ---
        # 预计算三角函数 (Pre-calculate trigonometric terms)
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        sin_b = np.sin(beta)
        cos_b = np.cos(beta)
        sin_abs_a = np.sin(np.abs(alpha))
        sin_abs_b = np.sin(np.abs(beta))
        sin_2a = np.sin(2 * alpha)
        sin_2b = np.sin(2 * beta)
        cos_a_half = np.cos(alpha / 2.0)
        sin_a_half = np.sin(alpha / 2.0)
        cos_b_half = np.cos(beta / 2.0)

        # X 力 (X Force - Eq. 23)
        Fax = q_dyn * (Cx1 * cos_a ** 2 * cos_b ** 2 + Cx2 * sin_2a * sin_a_half)

        # Y 力 (Y Force - Eq. 24)
        Fay = q_dyn * (Cy1 * cos_b_half * sin_2b + Cy2 * sin_2b +
                           Cy3 * sin_b * sin_abs_b + Cy4 * (delta_RUDT + delta_RUDB))

        # Z 力 (Z Force - Eq. 25)
        Faz = q_dyn * (Cz1 * cos_a_half * sin_2a + Cz2 * sin_2a +
                           Cz3 * sin_a * sin_abs_a + Cz4 * (delta_ELVL + delta_ELVR))

        fa_BRF = np.array([[Fax], [Fay], [Faz]])  # 气动力矢量  - Aerodynamic Forces in BRF

        # --- 计算气动力矩 (Calculate Aerodynamic Moments - Eq. 26-28) ---
        # L 力矩 (Roll Moment - Eq. 26)
        moment_L = q_dyn * (Cl1 * (delta_ELVL - delta_ELVR + delta_RUDB - delta_RUDT) +
                            Cl2 * sin_b * sin_abs_b)

        # M 力矩 (Pitch Moment - Eq. 27)
        moment_M = q_dyn * (Cm1 * cos_a_half * sin_2a + Cm2 * sin_2a +
                            Cm3 * sin_a * sin_abs_a + Cm4 * (delta_ELVL + delta_ELVR))

        # N 力矩 (Yaw Moment - Eq. 28)
        # !!! 警告：根据图片公式，Cn4 项依赖于升降舵，这在物理上非常可疑。通常偏航力矩应依赖于方向舵。
        # !!! Warning: According to the formula image, the Cn4 term depends on elevators, which is physically highly suspect.
        # !!! Yaw moment typically depends on rudders. Implementing as shown, but likely a typo in the source.
        moment_N = q_dyn * (Cn1 * cos_b_half * sin_2b + Cn2 * sin_2b +
                            Cn3 * sin_b * sin_abs_b + Cn4 * (delta_ELVL + delta_ELVR))  # <-- Suspect term

        ma_BRF = np.array([[moment_L], [moment_M], [moment_N]])  # 气动力矩矢量  - Aerodynamic Moments in BRF




        # --- 计算推力和推力矩 (Calculate Thrust and torque) ---
        T_total = tau[0:3].reshape(3, 1)  # 推力矢量 - Thrust vector
        tau_vec = tau[3:6].reshape(3, 1)  # 推力矩矢量 - Torque vector



        # 组合力和力矩 (Combine Forces and Torques)
        F_forces = fg_BRF - fb_BRF + fa_BRF + T_total
        print(F_forces.shape)
        F_torques = mg_BRF + mb_BRF + ma_BRF + tau_vec
        print(F_torques.shape)

        F_term = np.vstack((F_forces, F_torques)).flatten() # Combine forces and torques into 6x1 vector

        # --- 获取扰动 (Get Disturbance) ---
        # This is the external/unmodeled disturbance delta from the paper
        d = disturbance_func(t)

        # --- 动力学方程 (Dynamics Equation): Mx_dot + N = F + tau + d ---
        # Rearranging for x_dot: x_dot = M_inv * (F - N + tau + d)
        x_dot = self.M_inv @ (F_term - N_term + tau + d)

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
         self.X[5] = (self.X[5] + np.pi) % (2 * np.pi) - np.pi # Normalize Psi
         self.X[4] = np.clip(self.X[4], -np.pi/2 + 0.01, np.pi/2 - 0.01) # Keep Theta away from singularity

    def get_state(self):
        return self.X

    def get_pose(self):
        return self.X[0:6] # zeta, gamma

    def get_velocity(self):
        return self.X[6:12] # v, omega





class AirshipCasADiSymbolic:
    def __init__(self, params):
        self.params = params
        self.m = params.m
        self.g = params.g
        self.I0 = params.I0
        self.M = params.M_cfg
        self.M_inv = params.M_inv
        self.M_upper_left = self.M[0:3, 0:3]
        self.rc = params.rc.flatten()
        self.rb = params.rb.flatten()
        self.rp_r = params.rp_r.flatten()
        self.rp_l = params.rp_l.flatten()
        self.Vol_airship = params.Vol_airship
        self.rho_air = params.rho_air
        self.S_ref = params.S_ref
        self.L_ref = params.L_ref
        self.V_wind = params.V_WIND_ERF
        self.AERO_COEFFS = params.AERO_COEFFS
        
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
        ca = __import__("casadi")  # dynamic import
        
        # === 解构状态 ===
        zeta = X[0:3]   # Position in ERF
        gamma = X[3:6]  # Attitude (Euler angles)
        v = X[6:9]      # Linear velocity in BRF
        omega = X[9:12] # Angular velocity in BRF
        
        # === 运动学 (Kinematics) ===
        R = R_block(gamma)  # Combined rotation matrix
        y_dot = R @ ca.vertcat(v, omega)  # [zeta_dot, gamma_dot]
        
        # === 动力学 (Dynamics) ===
        
        # --- Calculate N term (Coriolis and centrifugal effects) ---
        omega_cross_v = ca.cross(omega, v)
        omega_cross_rc = ca.cross(omega, self.rc)
        omega_cross_omega_cross_rc = ca.cross(omega, omega_cross_rc)
        omega_cross_I0_omega = ca.cross(omega, self.I0 @ omega)
        rc_cross_omega_cross_v = ca.cross(self.rc, omega_cross_v)
        
        N1 = self.M_upper_left @ omega_cross_v + self.m * omega_cross_omega_cross_rc
        N2 = omega_cross_I0_omega + self.m * rc_cross_omega_cross_v
        N_term = ca.vertcat(N1, N2)
        
        # --- Calculate F term (forces and moments) ---
        # Gravity force and moment
        Rz = R_zeta(gamma)
        fg_earth = ca.vertcat(0, 0, self.m * self.g)  # Gravity in Earth Frame
        fg_BRF = Rz.T @ fg_earth  # Rotate gravity vector to Body Frame
        mg_BRF = ca.cross(self.rc, fg_BRF) # Torque due to gravity acting at CG (rc is CV->CG)
        
        # Buoyancy force and moment
        F_buoy_earth = ca.vertcat(0, 0, -self.rho_air * self.Vol_airship * self.g)
        fb_BRF = Rz.T @ F_buoy_earth
        mb_BRF = ca.cross(-self.rb, fb_BRF)  # Torque due to buoyancy acting at CB (assumed at CV, so arm is -rb)
        
        # --- Wind and relative velocity ---
        # Wind velocity in ERF
        V_wind_ERF = self.V_wind
        
        # Transform wind to BRF
        V_wind_BRF = Rz.T @ V_wind_ERF
        
        # Calculate relative velocity
        v_rel_brf = v - V_wind_BRF
        u_rel, v_rel_body, w_rel = v_rel_brf[0], v_rel_brf[1], v_rel_brf[2]
        
        # Dynamic pressure
        V_rel_mag = ca.norm_2(v_rel_brf)
        q_dyn = 0.5 * self.rho_air * V_rel_mag**2
        
        # Angle of attack and sideslip angle
        alpha = ca.atan2(w_rel, u_rel)
        beta = ca.asin(v_rel_body / (V_rel_mag + 1e-6))  # Add small epsilon to avoid division by zero
        

        
        # --- Aerodynamic forces and moments ---
        # Get aerodynamic coefficients
        Cx1 = self.AERO_COEFFS['Cx1']
        Cx2 = self.AERO_COEFFS['Cx2']
        Cy1 = self.AERO_COEFFS['Cy1']
        Cy2 = self.AERO_COEFFS['Cy2']
        Cy3 = self.AERO_COEFFS['Cy3']
        Cy4 = self.AERO_COEFFS['Cy4']
        Cz1 = self.AERO_COEFFS['Cz1']
        Cz2 = self.AERO_COEFFS['Cz2']
        Cz3 = self.AERO_COEFFS['Cz3']
        Cz4 = self.AERO_COEFFS['Cz4']
        Cl1 = self.AERO_COEFFS['Cl1']
        Cl2 = self.AERO_COEFFS['Cl2']
        Cm1 = self.AERO_COEFFS['Cm1']
        Cm2 = self.AERO_COEFFS['Cm2']
        Cm3 = self.AERO_COEFFS['Cm3']
        Cm4 = self.AERO_COEFFS['Cm4']
        Cn1 = self.AERO_COEFFS['Cn1']
        Cn2 = self.AERO_COEFFS['Cn2']
        Cn3 = self.AERO_COEFFS['Cn3']
        Cn4 = self.AERO_COEFFS['Cn4']
        
        # Pre-calculate trigonometric terms
        sin_a = ca.sin(alpha)
        cos_a = ca.cos(alpha)
        sin_b = ca.sin(beta)
        cos_b = ca.cos(beta)
        sin_abs_a = ca.sin(ca.fabs(alpha))
        sin_abs_b = ca.sin(ca.fabs(beta))
        sin_2a = ca.sin(2 * alpha)
        sin_2b = ca.sin(2 * beta)
        cos_a_half = ca.cos(alpha / 2.0)
        sin_a_half = ca.sin(alpha / 2.0)
        cos_b_half = ca.cos(beta / 2.0)


        # --- 获取控制舵面偏转角 (Get Control Surface Deflections) ---
        # !!! 关键占位符: 这些值需要由控制分配模块根据 tau[3:6] 确定 !!!
        # !!! Critical Placeholder: These values need to be determined by a Control Allocation module based on tau[3:6] !!!
        delta_RUDT = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_RUDB = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVL = np.deg2rad(0.0)  # [rad] - Placeholder
        delta_ELVR = np.deg2rad(0.0)  # [rad] - Placeholder

        
        # X Force - Eq. 23
        Fax = q_dyn * (Cx1 * cos_a**2 * cos_b**2 + Cx2 * sin_2a * sin_a_half)
        
        # Y Force - Eq. 24
        Fay = q_dyn * (Cy1 * cos_b_half * sin_2b + Cy2 * sin_2b +
                       Cy3 * sin_b * sin_abs_b + Cy4 * (delta_RUDT + delta_RUDB))
        
        # Z Force - Eq. 25
        Faz = q_dyn * (Cz1 * cos_a_half * sin_2a + Cz2 * sin_2a +
                       Cz3 * sin_a * sin_abs_a + Cz4 * (delta_ELVL + delta_ELVR))
        
        fa_BRF = ca.vertcat(Fax, Fay, Faz)
        
        # Roll Moment - Eq. 26
        moment_L = q_dyn * (Cl1 * (delta_ELVL - delta_ELVR + delta_RUDB - delta_RUDT) +
                           Cl2 * sin_b * sin_abs_b)
        
        # Pitch Moment - Eq. 27
        moment_M = q_dyn * (Cm1 * cos_a_half * sin_2a + Cm2 * sin_2a +
                           Cm3 * sin_a * sin_abs_a + Cm4 * (delta_ELVL + delta_ELVR))
        
        # Yaw Moment - Eq. 28
        moment_N = q_dyn * (Cn1 * cos_b_half * sin_2b + Cn2 * sin_2b +
                           Cn3 * sin_b * sin_abs_b + Cn4 * (delta_ELVL + delta_ELVR))
        
        ma_BRF = ca.vertcat(moment_L, moment_M, moment_N)


        # --- Control inputs ---
        T_total = U[0:3]  # Thrust vector
        tau_vec = U[3:6]
        
        
        # --- Combine all forces and moments ---
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
        ca = __import__("casadi")
        
        # Define symbolic variables
        x = ca.SX.sym('x', 12)  # State
        u = ca.SX.sym('u', 3)   # Control (T, μ, ν)  , delta_RUDT, delta_RUDB, delta_ELVL, delta_ELVR)
        
        # Get dynamics
        xdot = self.rhs_symbolic(x, u)
        
        # Create function
        f = ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
        
        return f
    
    def discrete_time_model(self, dt, integration_method='rk4'):
        """
        Get a discrete-time model using various integration methods.
        
        Args:
            dt: Time step
            integration_method: 'euler', 'rk4', etc.
            
        Returns:
            F: CasADi Function that maps (x, u) to x_next
        """
        ca = __import__("casadi")
        
        # Define symbolic variables
        x = ca.SX.sym('x', 12)
        u = ca.SX.sym('u', 7)
        
        # Get continuous dynamics
        xdot = self.rhs_symbolic(x, u)
        
        # Create discrete model based on integration method
        if integration_method == 'euler':
            x_next = x + dt * xdot
        elif integration_method == 'rk4':
            # RK4 integration
            k1 = xdot
            k2 = self.rhs_symbolic(x + dt/2 * k1, u)
            k3 = self.rhs_symbolic(x + dt/2 * k2, u)
            k4 = self.rhs_symbolic(x + dt * k3, u)
            x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Create a function
        F = ca.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])
        
        return F





