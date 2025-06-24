"""
controller.py - Contains implementation of AnyController and
NMPCThrustController classes for airship control.
"""
# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot nlpsol
# cspell:ignore nlpsol IPOPT traj uref xref mtimes
import numpy as np
import casadi as ca


from config import parameters as params
from airship.model import AirshipCasADiSymbolic
from airship.thrust import thrust_params_to_force_torque, calculate_thrust_direction
from airship.observer import NMPCDisturbanceObserver




class AnyController:
    """
    Base class for airship controllers.
    """
    def __init__(self):
        # 原有参数 / original parameters
        self.params = params  # 确保能访问 params 参数 / Ensure access to params
        self.rp_r = params.rp_r
        self.rp_l = params.rp_l

        # 添加 PID 控制器参数 / Add PID controller parameters
        # 位置 PID 参数 / Position PID parameters
        self.kp_pos = np.array([5.0, 5.0, 5.0])  # 位置比例增益 / Position proportional gain
        self.ki_pos = np.array([0.1, 0.1, 0.1])  # 位置积分增益 / Position integral gain
        self.kd_pos = np.array([2.0, 2.0, 2.0])  # 位置微分增益 / Position derivative gain

        # 姿态 PID 参数  / Attitude PID parameters
        self.kp_att = np.array([3.0, 3.0, 3.0])  # 姿态比例增益 / Attitude proportional gain
        self.ki_att = np.array([0.05, 0.05, 0.05])  # 姿态积分增益 / Attitude integral gain
        self.kd_att = np.array([1.0, 1.0, 1.0])  # 姿态微分增益 / Attitude derivative gain

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
            x_ref_dot: 参考状态导数 [位置导数，姿态导数]
            u_thrust: 初始推力参数 [T, μ, v], 如果为 None 则由控制器计算

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
        # 注意姿态误差可能需要特殊处理，如角度归一化
        # Note that attitude error may need special handling, such as angle normalization
        att_error = np.array([(angle + np.pi) % (2 * np.pi) - np.pi for angle in att_error])

        # 计算时间增量 / Calculate time increment
        dt = t - self.last_time
        if dt <= 0 or self.last_time == 0:
            dt = 0.01  # 默认时间步长 / Default time step

        # 更新积分项（带限幅） / Update integral terms (with limits)
        self.pos_error_integral += pos_error * dt
        self.att_error_integral += att_error * dt

        # 积分限幅 / Integral limits
        self.pos_error_integral = np.clip(
            self.pos_error_integral, -self.integral_limit_pos, self.integral_limit_pos
            )
        self.att_error_integral = np.clip(
            self.att_error_integral, -self.integral_limit_att, self.integral_limit_att
            )

        # 计算微分项 (可以使用参考轨迹导数作为前馈项)
        # Calculate derivative terms (can use reference trajectory derivatives as feedforward)
        pos_error_derivative = (pos_error - self.last_pos_error) / dt
        att_error_derivative = (att_error - self.last_att_error) / dt

        # 前馈项 - 使用参考轨迹的导数 / Feedforward term - use reference trajectory derivatives
        pos_ref_dot = x_ref_dot[0:3]
        att_ref_dot = x_ref_dot[3:6]

        # 计算 PID 控制输出 (带前馈)
        # Calculate PID control output (with feedforward)
        F_pos = (
            self.kp_pos * pos_error
            + self.ki_pos * self.pos_error_integral
            + self.kd_pos * (pos_error_derivative + 0.5 * pos_ref_dot)
            )

        T_att = (
            self.kp_att * att_error
            + self.ki_att * self.att_error_integral
            + self.kd_att * (att_error_derivative + 0.5 * att_ref_dot)
            )

        # 更新上一次误差和时间 / Update last errors and time
        self.last_pos_error = pos_error.copy()
        self.last_att_error = att_error.copy()
        self.last_time = t

        # 将 PID 控制输出转换为推力参数和控制力矩
        # Convert PID control output to thrust parameters and control torque
        if u_thrust is not None:
            # 使用提供的初始推力参数 / Use provided initial thrust parameters
            tau = thrust_params_to_force_torque(u_thrust, self.rp_r, self.rp_l)
            # 添加 PID 控制修正 / Add PID control correction
            tau[0:3] += F_pos
            tau[3:6] += T_att
        else:
            # 根据控制需求计算合适的推力参数
            # Calculate suitable thrust parameters based on control needs
            T_mag = np.linalg.norm(F_pos)  # 推力大小 / Thrust magnitude
            mu, nu = calculate_thrust_direction(F_pos)

            thrust_params = [T_mag, mu, nu]
            tau = thrust_params_to_force_torque(thrust_params, self.rp_r, self.rp_l)

            # 添加姿态控制力矩 / Add attitude control torque
            tau[3:6] += T_att

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

    def __init__(self, model, dt, N, Q, R, Qf, T_bounds, mu_bounds, nu_bounds, use_disturbance_compensation=True):
        """
        :param model:    Airship 实例，用于仿真/ Airship instance with a casadi-compatible dynamics (rhs)
        :param dt:       采样时间 sampling time
        :param N:        prediction horizon steps
        :param Q, R, Qf: weight vectors or lists for state, input, terminal cost
        :param T_bounds: (T_min, T_max)
        :param mu_bounds:(mu_min, mu_max)
        :param nu_bounds:(nu_min, nu_max)
        :param use_disturbance_compensation: 是否使用扰动补偿 / Whether to use disturbance compensation
        """
        self.model = model
        self.params = params
        self.dt = dt
        self.N = N
        # build cost matrices  将 NumPy 矩阵转换为 CasADi 矩阵 / Convert the NumPy matrix to a CasADi matrix
        self.Q = ca.DM(Q)  # 状态误差权重 / State error weight
        self.R = ca.DM(R)  # 控制输入权重 / Control input weight
        self.Qf = ca.DM(Qf) # 终端状态权重 / Terminal state weight
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

        # 构建非线性规划（NLP）问题，用于非线性模型预测控制（NMPC）的优化求解 / Build the nonlinear programming (NLP) problem, used for the optimization solution of nonlinear model predictive control (NMPC)
        self._build_nlp(X, U)

        # 添加扰动补偿选项 / Add the disturbance compensation option
        self.use_disturbance_compensation = use_disturbance_compensation
        self.disturbance_observer = NMPCDisturbanceObserver() if use_disturbance_compensation else None

        # 扰动补偿因子（控制扰动补偿强度） / Disturbance compensation factor (control the disturbance compensation strength)
        self.disturbance_compensation_factor = params.do_compensation_gain if hasattr(params, 'do_compensation_gain') else 0.9

        # 上一次的误差和控制输入（用于观测器） / The last error and control input (for the observer)
        self.prev_e1 = np.zeros(6)
        self.prev_e2 = np.zeros(6)
        self.prev_tau = np.zeros(6)
        self.prev_gamma = np.zeros(3)
        self.last_disturbance_estimate = np.zeros(6)

        # 添加存储完整控制序列的属性
        self.last_optimal_sequence = None





    def _build_continuous_dynamics(self, X, U):
        """
        构建飞艇的连续时间动力学模型 # 用 AirshipCasADiSymbolic 构造符号表达式 / Build the continuous time dynamics model of the airship using AirshipCasADiSymbolic to construct the symbolic expression
        在后续步骤中用于构建离散时间动力学模型 (通过数值积分方法，如 RK4) / Used in subsequent steps to build the discrete time dynamics model (through numerical integration methods, such as RK4)

        Args:
            X: 状态变量符号表达式 / State variable symbolic expression
            U: 控制输入符号表达式 / Control input symbolic expression

        Returns:
            f_cont: 状态导数符号表达式 / State derivative symbolic expression

            返回 状态导数 ( dX/dt ) / Return the state derivative (dX/dt)


        """
        symbolic_model = AirshipCasADiSymbolic(self.params)
        # 生成符号函数 f_cont(X, U) -> dX/dt / Generate the symbolic function f_cont(X, U) -> dX/dt
        f_cont = ca.Function("f_cont", [X, U], [symbolic_model.rhs_symbolic(X, U)])
        return f_cont

    def _build_nlp(self, X_sym, U_sym):
        """
        Args:
            X_sym: 状态变量符号表达式 / State variable symbolic expression
            U_sym: 控制输入符号表达式 / Control input symbolic expression

        Returns:
            nlp_prob: 非线性规划问题 / Nonlinear programming problem

        _build_nlp 的核心逻辑 / The core logic of _build_nlp
           1.优化变量： / Optimization variables:
           包括预测时域内的所有状态 ( x_k ) 和控制输入 ( u_k )。 / Includes all states (x_k) and control inputs (u_k) in the prediction horizon.
           这些变量是优化问题的决策变量。 / These variables are the decision variables of the optimization problem.

           2.目标函数：/ Target function:
           最小化轨迹跟踪误差和控制输入代价。 / Minimize the trajectory tracking error and control input cost.
           通过权重矩阵 ( Q, R, Q_f ) 调整不同项的重要性。 / Adjust the importance of different items through the weight matrix (Q, R, Q_f).

           3.约束条件：/ Constraints:
           确保优化解满足系统动力学（通过离散时间模型 f_d）。 / Ensure that the optimized solution satisfies the system dynamics (through the discrete time model f_d).
           确保控制输入和状态在物理限制范围内。 / Ensure that the control input and state are within the physical limits.

           4.求解器：/ Solver:
           使用 CasADi 的 nlpsol 创建求解器 (如 IPOPT),用于求解优化问题。 / Use CasADi's nlpsol to create a solver (such as IPOPT) to solve the optimization problem.


        """

        N = self.N
        Q = self.Q
        R = self.R
        Qf = self.Qf

        # 决策变量  / Decision variables
        w = []  # 优化变量，包括预测时域内的所有状态 ( x_k ) 和控制输入 ( u_k ) / Optimization variables, including all states (x_k) and control inputs (u_k) in the prediction horizon
        g = []  # 约束条件，包括动力学约束和控制输入约束 / Constraints, including dynamic constraints and control input constraints
        J = 0  # 代价函数，用于最小化轨迹跟踪误差和控制输入代价。通过权重矩阵 Q, R, Q_f 来调整不同项的重要性。 / Cost function, used to minimize the trajectory tracking error and control input cost. Adjust the importance of different items through the weight matrix Q, R, Q_f.

        # 初始状态符号量 / Initial state symbol
        Xk = ca.SX.sym("X0", 12)  #  初始状态 X0 12 维 / Initial state X0 12-dimensional
        w += [Xk]  # 添加到优化变量中 / Add to optimization variables

        # 定义参考状态和控制输入的符号变量，用于传递参考轨迹 / Define the symbolic variables of the reference state and control input, used to pass the reference trajectory
        self.p_xref = ca.SX.sym("xref", (N + 1) * 12)  # 参考状态 / Reference state
        self.p_uref = ca.SX.sym("uref", N * 3)  # 参考控制输入 / Reference control input

        for k in range(N):  # 构建预测时域内的优化问题 / Build the optimization problem in the prediction horizon
            # 对于每个预测步 ( k ) / For each prediction step (k)

            # 定义控制输入和下一个状态 / Define the control input and the next state
            Uk = ca.SX.sym(f"U_{k}", 3)  # 控制输入 / Control input
            w += [Uk]  # 添加到优化变量中 / Add to optimization variables

            # 下一个状态 Xk+1 / Next state Xk+1
            Xk_next = ca.SX.sym(f"X_{k+1}", 12)
            w += [Xk_next]

            # 添加动力学约束：使用离散时间动力学模型 f_d，确保优化解满足系统动力学 / Add dynamic constraints: use the discrete time dynamics model f_d, ensure that the optimized solution satisfies the system dynamics
            Xk_pred = self.f_d(Xk, Uk)  # 预测的下一个状态 / Predicted next state
            g += [Xk_next - Xk_pred]  # 动力学约束 / Dynamic constraints

            # 参考提取 / Reference extraction
            x_ref_k = self.p_xref[12 * k : 12 * (k + 1)]  # 参考状态 / Reference state
            u_ref_k = self.p_uref[3 * k : 3 * (k + 1)]  # 参考控制输入 / Reference control input

            # 计算误差并累积目标函数 / Calculate the error and accumulate the target function
            e1 = Xk[0:6] - x_ref_k[0:6] # 位置误差和姿态误差 / Position error and attitude error
            e2 = Xk[6:12] - x_ref_k[6:12] # 速度误差和角速度误差 / Velocity error and angular velocity error

            # 累积代价 / ca.mtimes: 用于矩阵乘法，计算误差的加权平方和 / Accumulate the cost: ca.mtimes: used for matrix multiplication, calculate the weighted square sum of the error
            J += (ca.mtimes([e1.T, Q[0:6, 0:6], e1])
                  + ca.mtimes([e2.T, Q[6:12, 6:12], e2])
                  + ca.mtimes([(Uk - u_ref_k).T, R, (Uk - u_ref_k)])
            )

            Xk = Xk_next  # 滚动更新 / Roll update

        # 在预测时域的最后一步，添加终端状态的代价 / At the last step in the prediction horizon, add the cost of the terminal state
        x_ref_final = self.p_xref[12 * N : 12 * (N + 1)]
        e_terminal = Xk - x_ref_final
        J += ca.mtimes([e_terminal.T, Qf, e_terminal])

        # 创建 NLP 求解器 / Create the NLP solver
        # 将优化变量、约束条件和目标函数打包成 NLP 问题 / Pack the optimization variables, constraints, and target function into an NLP problem
        w_flat = ca.vertcat(*w)
        g_flat = ca.vertcat(*g)
        nlp_prob = {"f": J, "x": w_flat, "g": g_flat, "p": ca.vertcat(self.p_xref, self.p_uref)}

        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)  # 使用 CasADi 的 nlpsol 创建求解器 / Use CasADi's nlpsol to create a solver

        # 保存变量数量，用于后续 step() 方法的约束设置 / Save the number of variables, used for the constraint setting in the subsequent step() method
        self.num_w = w_flat.size()[0]
        self.num_g = g_flat.size()[0]

    def step(self, x0, X_ref_traj, U_ref_traj, x_init=None, e1=None, e2=None):
        """
        用于在每个控制周期内解决非线性模型预测控制 NMPC 优化问题，计算当前时刻的最优控制输入  u_0 / Used to solve the nonlinear model predictive control NMPC optimization problem in each control cycle, calculate the optimal control input u_0 at the current time

        Args:
            x0: 当前状态 [zeta, gamma, v, omega] / Current state [zeta, gamma, v, omega]
            X_ref_traj: 参考轨迹列表 [x_ref_0, x_ref_1, ..., x_ref_N] / Reference trajectory list [x_ref_0, x_ref_1, ..., x_ref_N]
            U_ref_traj: 参考控制输入列表 [u_ref_0, u_ref_1, ..., u_ref_{N-1}] / Reference control input list [u_ref_0, u_ref_1, ..., u_ref_{N-1}]
            x_init: 初始猜测（可选） / Initial guess (optional)
            e1: 位置/姿态误差（可选，用于扰动观测器） / Position/attitude error (optional, used for the disturbance observer)
            e2: 速度误差（可选，用于扰动观测器） / Velocity error (optional, used for the disturbance observer)

        Returns:
            u0: 最优控制输入 [T, mu, nu] / Optimal control input [T, mu, nu]

        1.输入：
            当前状态 ( x_0 )：飞艇的当前状态（如位置、姿态、速度等）。 / Current state (x_0): The current state of the airship (such as position, attitude, velocity, etc.).
            参考轨迹 ( X_{\text{ref}} )：预测时域内的参考状态序列。 / Reference trajectory (X_{\text{ref}}): The reference state sequence in the prediction horizon.
            参考控制输入 ( U_{\text{ref}} )：预测时域内的参考控制输入序列。 / Reference control input (U_{\text{ref}}): The reference control input sequence in the prediction horizon.
            初始猜测 ( x_{\text{init}} )（可选）：优化变量的初始值，用于加速求解。 / Initial guess (x_{\text{init}})(optional): The initial value of the optimization variables, used to accelerate the solution.
        2.输出：
            当前时刻的最优控制输入 ( u_0 )（如推力大小和方向）。 / Optimal control input (u_0) (such as the thrust magnitude and direction) at the current time.


        方法的核心逻辑： / The core logic of the method
            1.优化变量： / Optimization variables:

            ( x_k )：预测时域内的状态。 / (x_k): The state in the prediction horizon.
            ( u_k )：预测时域内的控制输入。 / (u_k): The control input in the prediction horizon.
            这些变量是优化问题的决策变量。 / These variables are the decision variables of the optimization problem.
            2.目标函数： / Target function:

            最小化轨迹跟踪误差和控制输入代价。 / Minimize the trajectory tracking error and control input cost.
            通过权重矩阵 ( Q, R, Q_f ) 调整不同项的重要性。 / Adjust the importance of different items through the weight matrix (Q, R, Q_f).
            3.约束条件： / Constraints:

            动力学约束：确保优化解满足系统动力学。 / Dynamic constraints: ensure that the optimized solution satisfies the system dynamics.
            控制输入和状态的物理限制。 / Physical limits of control input and state.
            4.求解器： / Solver:

            使用 CasADi 的 nlpsol 求解器 如 (IPOPT) 解决优化问题。 / Use CasADi's nlpsol solver (such as IPOPT) to solve the optimization problem.



        Solve the NMPC problem given current state x0 and reference trajectories.
        Return the first control input [T, mu, nu].
        """
        N = self.N

        # 检查输入是否包含 NaN / Check if the input contains NaN
        if np.any(np.isnan(x0)):
            raise ValueError("x0 contains NaN values.")
        if np.any([np.any(np.isnan(x)) for x in X_ref_traj]):
            raise ValueError("X_ref_traj contains NaN values.")
        if np.any([np.any(np.isnan(u)) for u in U_ref_traj]):
            raise ValueError("U_ref_traj contains NaN values.")


        # --- 拼接参考轨迹参数向量 / Concatenate the reference trajectory parameter vector
        # 将参考状态 ( X_{\text{ref}} ) 和参考控制输入 ( U_{\text{ref}} ) 拼接成向量，作为 NLP 求解器的参数 / Concatenate the reference state (X_{\text{ref}}) and reference control input (U_{\text{ref}}) into a vector, as the parameter of the NLP solver
        # 这些参数将传递给 NLP 求解器，用于计算目标函数和约束。 / These parameters will be passed to the NLP solver, used to calculate the target function and constraints.
        p_xref = np.concatenate(X_ref_traj).reshape((12 * (N + 1),))  # 拼接参考状态向量 / Concatenate the reference state vector
        p_uref = np.concatenate(U_ref_traj).reshape((3 * N,))  # 拼接参考控制输入向量 / Concatenate the reference control input vector

        # --- 初始猜测 / Initial guess ---
        # 如果提供了初始猜测 x_init，则使用它，否则使用当前状态 x0 和参考轨迹 / If the initial guess x_init is provided, use it, otherwise use the current state x0 and the reference trajectory
        # 初始化优化变量： / Initialize the optimization variables:
        #   如果提供了初始猜测 ( x_{\text{init}} )，则直接使用。 / If the initial guess (x_{\text{init}}) is provided, use it directly.
        #   否则，初始化优化变量： / Otherwise, initialize the optimization variables:
        #   ( x_0 )：当前状态。 / (x_0): The current state.
        #   ( u_k )：控制输入，初始值为零。 / (u_k): The control input, initialized to zero.
        #   ( x_{k+1} )：状态，初始值为参考状态。 / (x_{k+1}): The state, initialized to the reference state.
        if x_init is not None:
            w0 = x_init
        else:
            w0 = []
            w0 += [x0]
            for k in range(N):
                w0 += [np.zeros(3)]  # Uk = [T, mu, nu]
                w0 += [X_ref_traj[k + 1]]  # Xk+1
            w0 = np.concatenate(w0)

        # 检查 w0 是否包含 NaN 或无穷大 / Check if w0 contains NaN or infinity
        if np.any(np.isnan(w0)):
            raise ValueError("w0 contains NaN values.")
        if np.any(np.isinf(w0)):
            raise ValueError("w0 contains infinite values.")



        # --- 控制输入约束 / Control input constraints ---
        # 设置优化变量的约束 / Set the constraints of the optimization variables
        # 状态约束： / State constraints:
        #       第一个状态 ( x_0 ) 被固定为当前状态。 / The first state (x_0) is fixed to the current state.
        #       其余状态 ( x_k ) 的范围设置为较宽的值（如 ([-1e5, 1e5])）。 / The range of the other states (x_k) is set to a wide value (such as ([-1e5, 1e5])).
        # 控制输入约束： / Control input constraints:
        #       推力大小 ( T ) 和偏转角 ( \mu, \nu ) 的范围根据物理限制设置。 / The range of the thrust magnitude (T) and deflection angle (\mu, \nu) is set according to the physical limits.
        lbx = []
        ubx = []

        # 对第一个状态 X0 设为当前状态 / Set the first state X0 to the current state
        lbx += list(x0) # 存储优化变量的下界（lower bounds） / Store the lower bounds of the optimization variables
        ubx += list(x0) # 存储优化变量的上界（upper bounds） / Store the upper bounds of the optimization variables

        for k in range(N):
            # 控制变量 (输入) Uk 的约束 / Constraints of the control variable (input) Uk
            lbx += [self.T_min, self.mu_min, self.nu_min]
            ubx += [self.T_max, self.mu_max, self.nu_max]

            # 对状态变量 Xk+1 给一个较宽范围（如 -1e5~1e5） / Give a wide range (such as -1e5~1e5) to the state variable Xk+1
            lbx += [-1e5] * 12
            ubx += [1e5] * 12


        # 检查 lbx 和 ubx 是否包含 NaN 或无穷大 / Check if lbx or ubx contains NaN or infinity
        if np.any(np.isnan(lbx)) or np.any(np.isnan(ubx)):
            raise ValueError("lbx or ubx contains NaN values.")
        if np.any(np.isinf(lbx)) or np.any(np.isinf(ubx)):
            raise ValueError("lbx or ubx contains infinite values.")

        # --- 设置等式约束（动力学） / Equality constraints (dynamics) ---
        # 动力学约束：确保优化解满足离散时间动力学模型 ( x_{k+1} = f_d(x_k, u_k) ) / Dynamic constraints: ensure that the optimized solution satisfies the discrete time dynamics model (x_{k+1} = f_d(x_k, u_k))
        lbg = [0] * self.num_g  # 动力学等式约束的下限 / Lower bound of the dynamic equality constraints
        ubg = [0] * self.num_g  # 动力学等式约束的上限 / Upper bound of the dynamic equality constraints

        # --- 求解器调用 / 调用 NLP 求解器 --- / Solver call / Call the NLP solver
        # 使用 CasADi 的 nlpsol 求解器解决 NLP 问题，得到优化变量的最优解 / Use CasADi's nlpsol solver to solve the NLP problem, get the optimal solution of the optimization variables
        w_opt = self.solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=np.concatenate([p_xref, p_uref]))

        # --- 提取第一个控制输入 U0 / 提取最优控制输入 --- / Extract the first control input U0 / Extract the optimal control input
        # 从优化变量中提取第一个控制输入 ( u_0 )，即当前时刻的最优控制输入 / Extract the first control input (u_0) from the optimization variables, that is, the optimal control input at the current time
        u0 = w_opt["x"].full().flatten()[12:15]  # 紧跟在 X0 后的就是 U0 = [T, mu, nu] / The first control input U0 = [T, mu, nu] follows X0

        # 将推力参数转换为力和力矩 / Convert the thrust parameters to force and torque
        tau = self.thrust_to_force_torque(u0)

        # 如果启用扰动补偿且提供了误差信息 / If the disturbance compensation is enabled and the error information is provided
        if self.use_disturbance_compensation and e1 is not None and e2 is not None:
            # 提取当前姿态 / Extract the current attitude
            gamma = x0[3:6]

            # 使用观测器更新扰动估计 / Use the observer to update the disturbance estimate
            delta_hat = self.disturbance_observer.update(
                self.dt, e1, e2, self.prev_tau, gamma,
                f_func=None  # 在 NMPC 中，f 项通常由模型内部处理 / In NMPC, the f term is usually handled internally by the model
            )

            # 保存扰动估计 / Save the disturbance estimate
            self.last_disturbance_estimate = delta_hat

            # 应用扰动补偿到控制输入 / Apply the disturbance compensation to the control input
            # tau = tau - delta_hat * self.disturbance_compensation_factor

            # 保存当前状态用于下次更新 / Save the current state for the next update
            self.prev_e1 = e1
            self.prev_e2 = e2
            self.prev_tau = tau
            self.prev_gamma = gamma

        # 保存整个最优控制序列 / Save the entire optimal control sequence
        u_sequence = []
        for i in range(N):
            u_i = w_opt["x"].full().flatten()[(i+1)*12+i*3:(i+1)*12+(i+1)*3]
            u_sequence.append(u_i)
        self.last_optimal_sequence = u_sequence

        return u0


    def get_current_disturbance_estimate(self):
        """获取当前的扰动估计 / Get the current disturbance estimate
         启用了扰动补偿，则返回的是最近一次通过扰动观测器估计的扰动值 / If the disturbance compensation is enabled, the disturbance value estimated by the disturbance observer is returned
        Args:
            None

        Returns:
            disturbance_estimate: 当前的扰动估计
        """
        if self.use_disturbance_compensation:
            return np.array(self.last_disturbance_estimate).flatten()
        else:
            return np.zeros(6)


    def thrust_to_force_torque(self, u_thrust):
        """
        将推力控制 [T, μ, v] 转换为力和力矩 [Fx, Fy, Fz, Mx, My, Mz] / Convert the thrust control [T, μ, v] to force and torque [Fx, Fy, Fz, Mx, My, Mz]

        Args:
            u_thrust: 推力控制输入 [T, μ, v] / Thrust control input [T, μ, v]

        Returns:
            tau: 6 维力和力矩向量 / 6-dimensional force and torque vector
        """
        # --- Control inputs ---

        # 确保返回的是 6 维向量 / Ensure that the return is a 6-dimensional vector
        thrust_force_torque = thrust_params_to_force_torque(u_thrust, self.rp_r, self.rp_l, use_casadi=True)
        # 检查 tau 的维度 / Check the dimension of tau
        if isinstance(thrust_force_torque, np.ndarray) and thrust_force_torque.size != 6:
            print(f"Warning: Tau has a dimension of {thrust_force_torque.size}, expected to be 6")
            # 如果不是 6 维，补充为 6 维 / If it is not 6-dimensional, supplement it to 6-dimensional
            if thrust_force_torque.size < 6:
                tau_corrected = np.zeros(6)
                tau_corrected[:thrust_force_torque.size] = thrust_force_torque
                return tau_corrected

        return thrust_force_torque
