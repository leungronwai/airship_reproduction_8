# controller.py
import numpy as np
from airship.utils import sig, R_block, R_zeta, R_y, S_omega
from config import parameters as params
import casadi as ca
from model import AirshipCasADiSymbolic

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




# === NMPC Controller Skeleton ===
class NMPCThrustController:
    """
    NMPC controller for airship using direct thrust allocation (T, μ, v).
    """
    def __init__(self, model, dt, N, Q, R, Qf, T_bounds, mu_bounds, nu_bounds):
        """
        :param model:    Airship instance with a casadi-compatible dynamics (rhs)
        :param dt:       sampling time
        :param N:        prediction horizon steps
        :param Q, R, Qf: weight vectors or lists for state, input, terminal cost
        :param T_bounds: (T_min, T_max)
        :param mu_bounds:(mu_min, mu_max)
        :param nu_bounds:(nu_min, nu_max)
        """
        self.model = model
        self.dt = dt
        self.N = N
        # build cost matrices
        self.Q  = ca.diag(Q)
        self.R  = ca.diag(R)
        self.Qf = ca.diag(Qf)
        # actuator limits
        self.T_min,  self.T_max  = T_bounds
        self.mu_min, self.mu_max = mu_bounds
        self.nu_min, self.nu_max = nu_bounds

        # symbolic variables
        X = ca.SX.sym('X', 12)
        U = ca.SX.sym('U', 3)

        # continuous dynamics f_cont: implement _build_continuous_dynamics to return SX 12×1
        f_cont = self._build_continuous_dynamics(X, U)

        # discrete dynamics via RK4
        h = dt
        k1 = f_cont(X, U)
        k2 = f_cont(X + 0.5*h*k1, U)
        k3 = f_cont(X + 0.5*h*k2, U)
        k4 = f_cont(X +   h*k3, U)
        X_next = X + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.f_d = ca.Function('f_d', [X, U], [X_next])

        # build NLP problem (cost, constraints)
        self._build_nlp(X, U)

    def _build_continuous_dynamics(self, X, U):
        # 用 AirshipCasADiSymbolic 构造符号表达式
        symbolic_model = AirshipCasADiSymbolic(self.model)  # self.model 是 Airship 实例
        # 生成符号函数 f_cont(X, U) -> dX/dt
        f_cont = ca.Function("f_cont", [X, U], [symbolic_model.rhs_symbolic(X, U)])
        return f_cont

    def _build_nlp(self, X_sym, U_sym):
        N = self.N
        Q = self.Q
        R = self.R
        Qf = self.Qf

        # 决策变量
        w = []
        g = []
        J = 0

        # 初始状态符号量
        Xk = ca.SX.sym("X0", 12)
        w += [Xk]

        # 参数（参考轨迹）：添加 N+1 个状态参考 + N 个控制参考
        self.p_xref = ca.SX.sym("xref", (N+1)*12)
        self.p_uref = ca.SX.sym("uref", N*3)

        for k in range(N):
            # 控制量 Uk
            Uk = ca.SX.sym(f"U_{k}", 3)
            w += [Uk]

            # 下一个状态
            Xk_next = ca.SX.sym(f"X_{k+1}", 12)
            w += [Xk_next]

            # 动力学约束：X_next - f_d(X, U) = 0
            Xk_pred = self.f_d(Xk, Uk)
            g += [Xk_next - Xk_pred]

            # 参考提取
            x_ref_k = self.p_xref[12*k:12*(k+1)]
            u_ref_k = self.p_uref[3*k:3*(k+1)]

            # e1 = pose误差，e2 = 速度误差
            e1 = Xk[0:6] - x_ref_k[0:6]
            e2 = Xk[6:12] - x_ref_k[6:12]

            # 累积代价
            J += ca.mtimes([e1.T, Q[0:6, 0:6], e1]) + \
                ca.mtimes([e2.T, Q[6:12, 6:12], e2]) + \
                ca.mtimes([(Uk - u_ref_k).T, R, (Uk - u_ref_k)])

            Xk = Xk_next  # 滚动更新

        # 终端代价
        x_ref_final = self.p_xref[12*N:12*(N+1)]
        e_terminal = Xk - x_ref_final
        J += ca.mtimes([e_terminal.T, Qf, e_terminal])

        # 创建 solver
        w_flat = ca.vertcat(*w)
        g_flat = ca.vertcat(*g)
        nlp_prob = {'f': J, 'x': w_flat, 'g': g_flat, 'p': ca.vertcat(self.p_xref, self.p_uref)}

        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # 保存变量数量，用于后续 step()
        self.num_w = w_flat.size()[0]
        self.num_g = g_flat.size()[0]

    def step(self, x0, X_ref_traj, U_ref_traj, x_init=None):
        """
        Solve the NMPC problem given current state x0 and reference trajectories.
        Return the first control input [T, mu, nu].
        """
        N = self.N

        # --- 拼接参考轨迹参数向量 ---
        p_xref = np.concatenate(X_ref_traj).reshape((12*(N+1),))
        p_uref = np.concatenate(U_ref_traj).reshape((3*N,))

        # --- 初始猜测 ---
        if x_init is not None:
            w0 = x_init
        else:
            w0 = []
            w0 += [x0]
            for k in range(N):
                w0 += [np.zeros(3)]     # Uk = [T, mu, nu]
                w0 += [X_ref_traj[k+1]] # Xk+1
            w0 = np.concatenate(w0)

        # --- 控制输入约束 ---
        lbx = []
        ubx = []

        # 对第一个状态 X0 设为当前状态
        lbx += list(x0)
        ubx += list(x0)

        for k in range(N):
            # Uk 控制变量的约束
            lbx += [self.T_min, self.mu_min, self.nu_min]
            ubx += [self.T_max, self.mu_max, self.nu_max]

            # 对状态变量 Xk+1 给一个较宽范围（如 -1e5~1e5）
            lbx += [-1e5]*12
            ubx += [ 1e5]*12

        # --- 等式约束（动力学） ---
        lbg = [0]*self.num_g
        ubg = [0]*self.num_g

        # --- 求解器调用 ---
        sol = self.solver(x0=w0, lbx=lbx, ubx=ubx,
                        lbg=lbg, ubg=ubg,
                        p=np.concatenate([p_xref, p_uref]))

        w_opt = sol['x'].full().flatten()

        # --- 提取第一个控制输入 U0 ---
        u0 = w_opt[12:15]  # 紧跟在 X0 后的就是 U0 = [T, mu, nu]
        return u0


    
