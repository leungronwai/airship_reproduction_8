# controller.py
import numpy as np
from airship.utils import sig, R_block, R_zeta, R_y, S_omega
from config import parameters as params


class FixedTimeBLFController:
    def __init__(self):
        # 加载参数 (Load parameters)
        self.k1 = params.k1
        self.k2 = params.k2
        self.k3 = params.k3
        self.k4 = params.k4
        self.kb_func = params.kb_func
        self.kb_dot_func = params.kb_dot_func
        self.epsilon = params.epsilon
        self.phi = params.phi
        self.rho = params.rho
        self.M = params.M_cfg
        self.M_inv = params.M_inv

        self.f_term_last = np.zeros(6) # Store f for observer

    def calculate_f(self, e1, e2, gamma, gamma_c, xc, xc_dot):
        """计算 f(e1, e2) 项 - Eq. 20 (Simplified)"""
        R = R_block(gamma)
        Rc = R_block(gamma_c)
        try:
            Rc_inv = np.linalg.inv(Rc)
        except np.linalg.LinAlgError:
             print("Warning: Rc is singular!")
             Rc_inv = np.identity(6)

        # Ṙ = R * S(omega) where S is block diag [S(w), S(w)]? No, kinematic relation.
        # Ṙζ = Rζ * S(ω); Ṙy involves derivatives of angles. Very complex.
        # Simplifying: Neglect R_dot terms, as they are often smaller or compensated by adaptation/DO.
        R_dot = np.zeros((6,6)) # Placeholder/Simplification
        Rc_dot = np.zeros((6,6)) # Placeholder/Simplification

        # F - N - M*xc_dot term is highly model dependent.
        # Assuming F-N is handled by the observer as part of delta.
        # We calculate the terms available from Eq. 20.
        term1 = R @ Rc_inv @ e2
        term2 = - R @ Rc_inv @ (R - Rc) @ xc
        term3 = (R_dot - Rc_dot) @ xc # Simplified to 0
        term4 = (R - Rc) @ xc_dot
        term5 = R @ self.M_inv @ (-self.M @ xc_dot) # RM⁻¹(-Mẋc) term? Yes, this is -R*xc_dot

        # Combine terms. Note: This 'f' is used in BOTH observer and controller.
        f = term1 + term2 + term3 + term4 - R @ xc_dot # Simplified version
        self.f_term_last = f # Store for observer use
        return f   # Return f for observer

    def calculate_control(self, t, e1, e2, delta_hat, gamma, gamma_c, xc, xc_dot):
        """计算控制输入 tau - Eq. 52"""
        kb = self.kb_func(t)
        kb_dot = self.kb_dot_func(t)

        # --- 检查约束 (Check constraints) ---
        if np.any(np.abs(e1) >= kb):
            print(f"ERROR: Constraint violated at time {t:.2f}!")
            # Implement saturation or emergency stop here
            e1 = np.clip(e1, -kb + 1e-6, kb - 1e-6) # Clip to avoid NaN

        # --- 计算BLF相关项 (Calculate BLF related terms) ---
        # Omega = diag( sec^2(...) ) = diag( kb^2 / (kb^2 - e1^2) )
        Omega_diag = kb**2 / (kb**2 - e1**2)
        # Lambda* = diag( |kb_dot/kb| + epsilon ) - See Remark 4
        lambda_star_diag = np.abs(kb_dot / kb) + self.epsilon # Use abs as in Tee 2011

        # --- 虚拟控制律 (Virtual Control - Eq. 38) ---
        # e2* = - diag(lambda*) * (k1*sig(e1, 1-phi) + k2*sig(e1, 1-phi+rho))
        term_k1 = self.k1 * sig(e1, 1 - self.phi)
        term_k2 = self.k2 * sig(e1, 1 - self.phi + self.rho)
        e2_star = - lambda_star_diag * (term_k1 + term_k2)

        # --- 计算误差 xi (Calculate error xi) ---
        xi = e2 - e2_star

        # --- 计算 f(e1, e2) ---
        # This 'f' is for the controller dynamics (Eq. 19)
        f_term = self.calculate_f(e1, e2, gamma, gamma_c, xc, xc_dot)

        # --- 计算控制律 τ (Calculate control law tau - Eq. 52 rearranged) ---
        # τ = -M * R^T * ( sig(xi, 2*phi-1)*(k3 + k4*sig(xi, phi)) + f_term ) - delta_hat
        # Need R = R_block(gamma)
        R = R_block(gamma)
        RM_inv = R @ self.M_inv

        control_dyn_term = sig(xi, 2 * self.phi - 1) * (self.k3 + self.k4 * sig(xi, self.phi))

        tau = -self.M @ R.T @ (control_dyn_term + f_term) - delta_hat

        # Apply saturation if needed based on actuator limits
        # tau = np.clip(tau, min_tau, max_tau)

        return tau  # Return control input tau

    def get_last_f(self):
        """观测器需要控制器计算的f项"""
        return self.f_term_last