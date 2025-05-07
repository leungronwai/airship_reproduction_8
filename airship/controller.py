# controller.py
import numpy as np
from airship.utils import sig, R_block, R_zeta, R_y, S_omega
from config import parameters as params
import casadi as ca
from model import AirshipCasADiSymbolic
from numba import njit, float64


class AnyController:
    def __init__(self):
        # 原有参数 / original parameters
        self.params = params  # 确保能访问 params 参数 / Ensure access to params
        self.rp_r = params.rp_r
        self.rp_l = params.rp_l

        # 添加 PID 控制器参数 / Add PID controller parameters
        # 位置 PID 参数 / Position PID parameters
        self.Kp_pos = np.array([5.0, 5.0, 5.0])  # 位置比例增益 / Position proportional gain
        self.Ki_pos = np.array([0.1, 0.1, 0.1])  # 位置积分增益 / Position integral gain
        self.Kd_pos = np.array([2.0, 2.0, 2.0])  # 位置微分增益 / Position derivative gain

        # 姿态 PID 参数  / Attitude PID parameters
        self.Kp_att = np.array([3.0, 3.0, 3.0])  # 姿态比例增益 / Attitude proportional gain
        self.Ki_att = np.array([0.05, 0.05, 0.05])  # 姿态积分增益 / Attitude integral gain
        self.Kd_att = np.array([1.0, 1.0, 1.0])  # 姿态微分增益 / Attitude derivative gain

        # 积分误差和上一次误差（用于计算微分项） / Integral errors and last errors (for derivative calculation)
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.last_att_error = np.zeros(3)
        self.last_time = 0.0  # 上一次更新的时间 / Last update time

        # 积分限幅，防止积分饱和 /  Integral limits to prevent saturation
        self.integral_limit_pos = np.array([50.0, 50.0, 50.0])
        self.integral_limit_att = np.array([1.0, 1.0, 1.0])

    def calculate_control(self, t, x, x_ref, x_ref_dot, u_thrust=None):
        """
        基于 PID 控制计算控制输入 / Calculate control input based on PID control

        参数：
            t: 当前时间 / Current time
            x: 当前状态 [位置，姿态] / Current state [position, attitude]
            x_ref: 参考状态 [位置，姿态] / Reference state [position, attitude]
            x_ref_dot: 参考状态导数 [位置导数，姿态导数] / Reference state derivative [position derivative, attitude derivative]
            u_thrust: 初始推力参数 [T, μ, v], 如果为 None 则由控制器计算 / Initial thrust parameters [T, mu, v], if None, calculated by controller

        返回：
            tau: 6 维控制向量 [Fx, Fy, Fz, Tx, Ty, Tz] / 6D control vector [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        # 提取当前状态和参考状态 / Extract current state and reference state
        pos = x[0:3]  # 当前位置 / Current position
        att = x[3:6]  # 当前姿态 / Current attitude
        pos_ref = x_ref[0:3]  # 参考位置 / Reference position
        att_ref = x_ref[3:6]  # 参考姿态 / Reference attitude

        # 计算位置和姿态误差 / Calculate position and attitude errors
        pos_error = pos_ref - pos
        att_error = att_ref - att
        # 注意姿态误差可能需要特殊处理，如角度归一化 / Note that attitude error may need special handling, such as angle normalization
        att_error = np.array([(angle + np.pi) % (2 * np.pi) - np.pi for angle in att_error])

        # 计算时间增量 / Calculate time increment
        dt = t - self.last_time
        if dt <= 0 or self.last_time == 0:
            dt = 0.01  # 默认时间步长 / Default time step

        # 更新积分项（带限幅） / Update integral terms (with limits)
        self.pos_error_integral += pos_error * dt
        self.att_error_integral += att_error * dt

        # 积分限幅 / Integral limits
        self.pos_error_integral = np.clip(self.pos_error_integral, -self.integral_limit_pos, self.integral_limit_pos)
        self.att_error_integral = np.clip(self.att_error_integral, -self.integral_limit_att, self.integral_limit_att)

        # 计算微分项 (可以使用参考轨迹导数作为前馈项) / Calculate derivative terms (can use reference trajectory derivatives as feedforward)
        pos_error_derivative = (pos_error - self.last_pos_error) / dt
        att_error_derivative = (att_error - self.last_att_error) / dt

        # 前馈项 - 使用参考轨迹的导数 / Feedforward term - use reference trajectory derivatives
        pos_ref_dot = x_ref_dot[0:3]
        att_ref_dot = x_ref_dot[3:6]

        # 计算 PID 控制输出 (带前馈) / Calculate PID control output (with feedforward)
        F_pos = self.Kp_pos * pos_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * (pos_error_derivative + 0.5 * pos_ref_dot)

        T_att = self.Kp_att * att_error + self.Ki_att * self.att_error_integral + self.Kd_att * (att_error_derivative + 0.5 * att_ref_dot)

        # 更新上一次误差和时间 / Update last errors and time
        self.last_pos_error = pos_error.copy()
        self.last_att_error = att_error.copy()
        self.last_time = t

        # 将 PID 控制输出转换为推力参数和控制力矩 / Convert PID control output to thrust parameters and control torque
        if u_thrust is not None:
            # 使用提供的初始推力参数 / Use provided initial thrust parameters
            tau = self._thrust_params_to_tau(u_thrust)
            # 添加 PID 控制修正 / Add PID control correction
            tau[0:3] += F_pos
            tau[3:6] += T_att
        else:
            # 根据控制需求计算合适的推力参数    / Calculate suitable thrust parameters based on control needs
            T_mag = np.linalg.norm(F_pos)  # 推力大小 / Thrust magnitude
            if T_mag > 0.001:
                # 计算推力方向 / Calculate thrust direction
                mu = np.arctan2(F_pos[1], F_pos[0])  # 水平面内角度 / Angle in horizontal plane
                nu = np.arctan2(F_pos[2], np.sqrt(F_pos[0] ** 2 + F_pos[1] ** 2))  # 垂直面内角度 / Angle in vertical plane
            else:
                mu, nu = 0.0, 0.0

            thrust_params = [T_mag, mu, nu]
            tau = self._thrust_params_to_tau(thrust_params)

            # 添加姿态控制力矩 / Add attitude control torque
            tau[3:6] += T_att

        return tau

    def _thrust_params_to_tau(self, thrust_params):
        """
        将推力参数转换为力和力矩向量 / Convert thrust parameters to force and torque vector

        参数：
            thrust_params: [T, μ, v]

        返回：
            tau: [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        T_mag, mu, nu = thrust_params

        # 计算右侧推力向量 / Calculate right thrust vector
        thrust_vector_r = np.array([T_mag * np.cos(mu) * np.cos(nu), T_mag * np.sin(mu), T_mag * np.cos(mu) * np.sin(nu)])

        # 计算左侧推力向量 / Calculate left thrust vector
        thrust_vector_l = np.array([T_mag * np.cos(mu) * np.cos(nu), T_mag * np.sin(mu), T_mag * np.cos(mu) * np.sin(nu)])

        # 总推力 / Total thrust
        T_total = thrust_vector_r + thrust_vector_l

        # 获取推力作用点 / Get the thrust application points
        rp_r = self.rp_r.flatten()
        rp_l = self.rp_l.flatten()

        # 计算力矩 / Calculate torque
        tau_r = np.cross(rp_r, thrust_vector_r)
        tau_l = np.cross(rp_l, thrust_vector_l)
        tau_vec = tau_r + tau_l

        # 组合力和力矩 / Combine force and torque
        tau = np.concatenate([T_total, tau_vec])

        return tau

    def reset(self):
        """重置控制器状态 / Reset controller state"""
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.last_att_error = np.zeros(3)
        self.last_time = 0.0  # 上一次更新的时间 / Last update time


# === NMPC Controller Skeleton ===
class NMPCThrustController:
    """
    NMPC controller for airship using direct thrust allocation (T, μ, v).
    NMPC 控制器，使用直接推力分配 (T, μ, v)。
    """

    def __init__(self, model, dt, N, Q, R, Qf, T_bounds, mu_bounds, nu_bounds):
        """
        :param model:    Airship 实例，用于仿真/ Airship instance with a casadi-compatible dynamics (rhs)
        :param dt:       采样时间 sampling time
        :param N:        prediction horizon steps
        :param Q, R, Qf: weight vectors or lists for state, input, terminal cost
        :param T_bounds: (T_min, T_max)
        :param mu_bounds:(mu_min, mu_max)
        :param nu_bounds:(nu_min, nu_max)
        """
        self.model = model
        self.params = params
        self.dt = dt
        self.N = N
        # build cost matrices
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.Qf = ca.diag(Qf)
        # actuator limits
        self.T_min, self.T_max = T_bounds
        self.mu_min, self.mu_max = mu_bounds
        self.nu_min, self.nu_max = nu_bounds

        # Save propeller position parameters for thrust calculation
        self.rp_r = self.params.rp_r
        self.rp_l = self.params.rp_l

        # symbolic variables
        X = ca.SX.sym("X", 12)
        U = ca.SX.sym("U", 3)

        # 构建离散时间动力学模型 / Build discrete time dynamics model
        f_cont = self._build_continuous_dynamics(X, U)

        # discrete dynamics via RK4
        h = dt
        k1 = f_cont(X, U)
        k2 = f_cont(X + 0.5 * h * k1, U)
        k3 = f_cont(X + 0.5 * h + k2, U)
        k4 = f_cont(X + h * k3, U)
        X_next = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_d = ca.Function("f_d", [X, U], [X_next])

        # 构建非线性规划（NLP）问题，用于非线性模型预测控制（NMPC）的优化求解
        self._build_nlp(X, U)

    def _build_continuous_dynamics(self, X, U):  # 构建飞艇的连续时间动力学模型
        # 用 AirshipCasADiSymbolic 构造符号表达式
        symbolic_model = AirshipCasADiSymbolic(self.params)  # self.model 是 Airship 实例
        # 生成符号函数 f_cont(X, U) -> dX/dt
        f_cont = ca.Function("f_cont", [X, U], [symbolic_model.rhs_symbolic(X, U)])
        return f_cont  # 返回 状态导数 ( \dot{X} ) 即 ( \frac{dX}{dt} )
        # 在后续步骤中用于构建离散时间动力学模型（通过数值积分方法，如 RK4）

    def _build_nlp(self, X_sym, U_sym):
        """
        _build_nlp 的核心逻辑
           1.优化变量：
           包括预测时域内的所有状态 ( x_k ) 和控制输入 ( u_k )。
           这些变量是优化问题的决策变量。

           2.目标函数：
           最小化轨迹跟踪误差和控制输入代价。
           通过权重矩阵 ( Q, R, Q_f ) 调整不同项的重要性。

           3.约束条件：
           确保优化解满足系统动力学（通过离散时间模型 f_d）。
           确保控制输入和状态在物理限制范围内。

           4.求解器：
           使用 CasADi 的 nlpsol 创建求解器（如 IPOPT），用于求解优化问题。


        """

        N = self.N
        Q = self.Q
        R = self.R
        Qf = self.Qf

        # 决策变量  / Decision variables
        w = []  # 优化变量
        g = []  # 约束条件
        J = 0  # 代价函数

        # 初始状态符号量 / Initial state symbol
        Xk = ca.SX.sym("X0", 12)  #  初始状态 X0
        w += [Xk]  # 添加到优化变量中

        # 定义参考状态和控制输入的符号变量，用于传递参考轨迹
        self.p_xref = ca.SX.sym("xref", (N + 1) * 12)  # 参考状态
        self.p_uref = ca.SX.sym("uref", N * 3)  # 参考控制输入

        for k in range(N):  # 构建预测时域内的优化问题
            # 对于每个预测步 ( k )

            # 定义控制输入和下一个状态
            Uk = ca.SX.sym(f"U_{k}", 3)  # 控制输入
            w += [Uk]  # 添加到优化变量中

            # 下一个状态 Xk+1 / Next state Xk+1
            Xk_next = ca.SX.sym(f"X_{k+1}", 12)
            w += [Xk_next]

            # 添加动力学约束：使用离散时间动力学模型 f_d，确保优化解满足系统动力学
            Xk_pred = self.f_d(Xk, Uk)  # 预测的下一个状态
            g += [Xk_next - Xk_pred]  # 动力学约束

            # 参考提取 / Reference extraction
            x_ref_k = self.p_xref[12 * k : 12 * (k + 1)]  # 参考状态
            u_ref_k = self.p_uref[3 * k : 3 * (k + 1)]  # 参考控制输入

            # 计算误差并累积目标函数
            # 计算状态误差和控制输入误差
            e1 = Xk[0:6] - x_ref_k[0:6]
            e2 = Xk[6:12] - x_ref_k[6:12]

            # 累积代价 / Accumulate cost
            J += ca.mtimes([e1.T, Q[0:6, 0:6], e1]) + ca.mtimes([e2.T, Q[6:12, 6:12], e2]) + ca.mtimes([(Uk - u_ref_k).T, R, (Uk - u_ref_k)])

            Xk = Xk_next  # 滚动更新 / Roll update

        # 在预测时域的最后一步，添加终端状态的代价
        x_ref_final = self.p_xref[12 * N : 12 * (N + 1)]
        e_terminal = Xk - x_ref_final
        J += ca.mtimes([e_terminal.T, Qf, e_terminal])

        # 创建 NLP 求解器
        # 将优化变量、约束条件和目标函数打包成 NLP 问题
        w_flat = ca.vertcat(*w)
        g_flat = ca.vertcat(*g)
        nlp_prob = {"f": J, "x": w_flat, "g": g_flat, "p": ca.vertcat(self.p_xref, self.p_uref)}

        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)  # 使用 CasADi 的 nlpsol 创建求解器

        # 保存变量数量，用于后续 step() 方法的约束设置
        self.num_w = w_flat.size()[0]
        self.num_g = g_flat.size()[0]

    def step(self, x0, X_ref_traj, U_ref_traj, x_init=None):
        """
        用于在每个控制周期内解决非线性模型预测控制 NMPC 优化问题，计算当前时刻的最优控制输入  u_0

        1.输入：
            当前状态 ( x_0 )：飞艇的当前状态（如位置、姿态、速度等）。
            参考轨迹 ( X_{\text{ref}} )：预测时域内的参考状态序列。
            参考控制输入 ( U_{\text{ref}} )：预测时域内的参考控制输入序列。
            初始猜测 ( x_{\text{init}} )（可选）：优化变量的初始值，用于加速求解。
        2.输出：
            当前时刻的最优控制输入 ( u_0 )（如推力大小和方向）。


        方法的核心逻辑：
            1.优化变量：

            ( x_k )：预测时域内的状态。
            ( u_k )：预测时域内的控制输入。
            这些变量是优化问题的决策变量。
            2.目标函数：

            最小化轨迹跟踪误差和控制输入代价。
            通过权重矩阵 ( Q, R, Q_f ) 调整不同项的重要性。
            3.约束条件：

            动力学约束：确保优化解满足系统动力学。
            控制输入和状态的物理限制。
            4.求解器：

            使用 CasADi 的 nlpsol 求解器（如 IPOPT）解决优化问题。



        Solve the NMPC problem given current state x0 and reference trajectories.
        Return the first control input [T, mu, nu].
        """
        N = self.N

        # --- 拼接参考轨迹参数向量 /
        # 将参考状态 ( X_{\text{ref}} ) 和参考控制输入 ( U_{\text{ref}} ) 拼接成向量，作为 NLP 求解器的参数
        # 这些参数将传递给 NLP 求解器，用于计算目标函数和约束。
        p_xref = np.concatenate(X_ref_traj).reshape((12 * (N + 1),))  # 拼接参考状态向量
        p_uref = np.concatenate(U_ref_traj).reshape((3 * N,))  # 拼接参考控制输入向量

        # --- 初始猜测 / Initial guess ---
        # 如果提供了初始猜测 x_init，则使用它，否则使用当前状态 x0 和参考轨迹
        # 初始化优化变量：
        #   如果提供了初始猜测 ( x_{\text{init}} )，则直接使用。
        #   否则，初始化优化变量：
        #   ( x_0 )：当前状态。
        #   ( u_k )：控制输入，初始值为零。
        #   ( x_{k+1} )：状态，初始值为参考状态。
        if x_init is not None:
            w0 = x_init
        else:
            w0 = []
            w0 += [x0]
            for k in range(N):
                w0 += [np.zeros(3)]  # Uk = [T, mu, nu]
                w0 += [X_ref_traj[k + 1]]  # Xk+1
            w0 = np.concatenate(w0)

        # --- 控制输入约束 / Control input constraints ---
        # 设置优化变量的约束
        # 状态约束：
        #       第一个状态 ( x_0 ) 被固定为当前状态。
        #       其余状态 ( x_k ) 的范围设置为较宽的值（如 ([-1e5, 1e5])）。
        # 控制输入约束：
        #       推力大小 ( T ) 和偏转角 ( \mu, \nu ) 的范围根据物理限制设置。
        lbx = []
        ubx = []

        # 对第一个状态 X0 设为当前状态
        lbx += list(x0)
        ubx += list(x0)

        for k in range(N):
            # 控制变量 (输入) Uk 的约束
            lbx += [self.T_min, self.mu_min, self.nu_min]
            ubx += [self.T_max, self.mu_max, self.nu_max]

            # 对状态变量 Xk+1 给一个较宽范围（如 -1e5~1e5）
            lbx += [-1e5] * 12
            ubx += [1e5] * 12

        # --- 设置等式约束（动力学） / Equality constraints (dynamics) ---
        # 动力学约束：确保优化解满足离散时间动力学模型 ( x_{k+1} = f_d(x_k, u_k) )
        lbg = [0] * self.num_g  # 动力学等式约束的下限
        ubg = [0] * self.num_g  # 动力学等式约束的上限

        # --- 求解器调用 / 调用 NLP 求解器 ---
        # 使用 CasADi 的 nlpsol 求解器解决 NLP 问题，得到优化变量的最优解
        sol = self.solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=np.concatenate([p_xref, p_uref]))

        w_opt = sol["x"].full().flatten()

        # --- 提取第一个控制输入 U0 / 提取最优控制输入 ---
        # 从优化变量中提取第一个控制输入 ( u_0 )，即当前时刻的最优控制输入
        u0 = w_opt[12:15]  # 紧跟在 X0 后的就是 U0 = [T, mu, nu]
        return u0

    @njit
    def thrust_to_force_torque(self, u_thrust):
        """
        将推力控制 [T, μ, v] 转换为力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

        Args:
            u_thrust: 推力控制输入 [T, μ, v]

        Returns:
            tau: 6 维力和力矩向量
        """

        import numpy as np

        # --- Control inputs ---
        T_mag = u_thrust[0]  # 推力大小 / Thrust magnitude
        mu = u_thrust[1]  # 水平面内的推力偏转角  / Thrust deflection angle in the horizontal plane
        nu = u_thrust[2]  # 垂直面内的推力偏转角 / Thrust deflection angle in the vertical plane

        # 计算右侧和左侧推力矢量 (假设对称) / Calculate right and left thrust vectors (assuming symmetry)
        # 计算推力向量 / Calculate thrust vector
        # --- Thrust vector ---
        thrust_vector_r = ca.vertcat(T_mag * ca.cos(mu) * ca.cos(nu), T_mag * ca.sin(mu), T_mag * ca.cos(mu) * ca.sin(nu))

        thrust_vector_l = ca.vertcat(T_mag * ca.cos(mu) * ca.cos(nu), T_mag * ca.sin(mu), T_mag * ca.cos(mu) * ca.sin(nu))

        T_total = thrust_vector_r + thrust_vector_l

        # 获取推力作用点 / Get the thrust application points
        rp_r = self.params.rp_r.flatten()
        rp_l = self.params.rp_l.flatten()

        # 计算力矩 / --- Thrust torque ---
        tau_r = ca.cross(rp_r, thrust_vector_r.flatten()).reshape(3, 1)
        tau_l = ca.cross(rp_l, thrust_vector_l.flatten()).reshape(3, 1)

        # total Thrust momentsa
        tau_vec = tau_r + tau_l

        # 组合力和力矩
        tau = np.concatenate([T_total, tau_vec])  # 6D force and torque vector

        return tau  # return force and torque vector of thrust
