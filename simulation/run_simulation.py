# simulation/run_simulation.py
"""
Main simulation script for airship trajectory tracking and controller evaluation.
"""

# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot nlpsol xlabel ylabel zlabel


# === 标准库 ===
import os
import sys
import time as timer  # Use timer to avoid conflict with time variable t
import logging

# === 第三方库 ===
import numpy as np
import matplotlib.pyplot as plt

# === 本地模块 ===
from config import parameters as params
from airship.utils import R_block, rk4_step
from airship.model import Airship
from airship.trajectory import Trajectory
from airship.controller import AnyController, NMPCThrustController


# === 添加项目根目录到路径（如需从根目录运行） ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))





# 全局 logger（在这里做一次 basicConfig） / Global logger (basicConfig done here)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_simulation(trajectory_type="default"):
    """
    1.初始化：创建气艇模型、轨迹生成器、控制器等对象




    """

    # --- 初始化 / Initialize ---
    airship = Airship(params.X0)
    trajectory = Trajectory()
    controller = AnyController()

    # --- 数据记录 / Data logging ---
    sim_time = np.arange(0, params.T_SPAN, params.DT)
    n_steps = len(sim_time)
    state_history = np.zeros((12, n_steps))
    error_history = np.zeros((6, n_steps))  # e1 = y - yc
    error2_history = np.zeros((6, n_steps))  # e2 = y_dot - yc_dot
    control_history = np.zeros((6, n_steps))
    disturbance_history = np.zeros((6, n_steps))
    estimate_history = np.zeros((6, n_steps))
    kb_history = np.zeros((6, n_steps))
    yc_history = np.zeros((6, n_steps))
    y_history = np.zeros((6, n_steps))

    # --- 仿真循环 / Simulation loop ---
    for i, t in enumerate(sim_time):
        # 2.仿真循环：在**每个时间步**:
        #     a. 获取参考轨迹
        #     b. 调用 NMPC 控制器计算控制输入
        #     c. 更新气艇状态


        # 获取当前状态 (Get current state)
        X = airship.get_state()

        # 基于轨迹类型获取期望状态 (Get desired state based on trajectory type)
        if trajectory_type == "default":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_linear_trajectory(t)
        elif trajectory_type == "spiral":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_spiral_trajectory(t)
            # 转换螺旋轨迹为所需格式...
            # 这里需要额外转换代码 - 需要实现
        elif trajectory_type == "figure8":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_figure8_trajectory(t)
        elif trajectory_type == "lemniscate":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_lemniscate_trajectory(t)
        elif trajectory_type == "linear":
            # 直线轨迹，可以自定义起点和终点
            start_point = np.array([0.0, 0.0, -19000.0])
            end_point = np.array([5000.0, 5000.0, -19000.0])
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_linear_trajectory(
                t, start_point=start_point, end_point=end_point, speed=15.0, hover_at_end=True  # 飞行速度 15 m/s  # 到达终点后悬停
            )
        else:
            raise ValueError(f"未知的轨迹类型：{trajectory_type}")

        zeta_d, gamma_d = yc[0:3], yc[3:6]  # Desired position/attitude vector
        zeta_d_dot, gamma_d_dot = yc_dot[0:3], yc_dot[3:6]  # Desired velocity vector

        logger.debug(f"[{i}] Desired zeta_d={zeta_d}, gamma_d={gamma_d}")

        # 计算误差 (Calculate errors)
        R = R_block(gamma)
        y = np.concatenate((zeta, gamma))  # Current position/attitude vector
        y_dot = R @ x_vec  # Current velocity vector in yc space

        e1 = y - yc
        e2 = y_dot - yc_dot

        # 更新扰动观测器 (Update disturbance observer)
        # Observer needs f calculated by controller based on CURRENT e1, e2
        # Calculate f for the observer first (it uses last state's effect)
        f_for_observer = controller.get_last_f()  # Get f from previous step calculation
        delta_hat = observer.update(params.DT, e1, e2, tau, gamma, lambda e1_arg, e2_arg: f_for_observer)

        # 计算控制输入 (Calculate control input)
        # Controller calculates its own f based on current state/errors
        tau = controller.calculate_control(t, e1, e2, delta_hat, gamma, gamma_d, xc, xc_dot)

        if i % (n_steps // 10) == 0:  # Print every 10% of the simulation 每 10 步打印一次 e1、tau 的范数
            logger.debug(f"[{i}] t={t:.2f}s | e1 norm={np.linalg.norm(e1):.4f} | tau norm={np.linalg.norm(tau):.4f}")

        # 获取实际扰动 (Get actual disturbance)
        actual_delta = params.disturbance_delta(t)

        # 积分气艇模型 (Integrate airship model - using RK4)
        # Define the ODE function for the solver/RK4
        def airship_ode(t_rk, X_rk):
            # Need control 'tau' and disturbance 'actual_delta' at time t_rk
            # Assume they are constant over the small step dt for simplicity
            # A better RK4 would re-evaluate control/disturbance within the step
            return airship.rhs(t_rk, X_rk, tau, lambda t_ignore: actual_delta)

        # RK4 Step
        X_next = rk4_step(airship_ode, t, X, params.DT)

        # 更新状态 (Update state)
        airship.X = X_next
        airship.X[3:6] = (airship.X[3:6] + np.pi) % (2 * np.pi) - np.pi  # Normalize angles

        # 记录数据 (Log data)
        state_history[:, i] = X
        error_history[:, i] = e1
        error2_history[:, i] = e2
        control_history[:, i] = tau
        disturbance_history[:, i] = actual_delta
        estimate_history[:, i] = delta_hat
        kb_history[:, i] = params.kb_func(t)
        yc_history[:, i] = yc
        y_history[:, i] = y

        # 打印进度 (Print progress)
        if i % (n_steps // 10) == 0:
            # print(f"Simulation progress: {i / n_steps * 100:.0f}%")
            logger.info(f"Simulation progress: {i / n_steps * 100:.0f}%")

    end_time = timer.time()
    # print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    logger.info(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # --- 结果绘图 / Plotting results ---
    plt.style.use("seaborn-v0_8-whitegrid")

    # 图 1: 三维轨迹跟踪 (3D Trajectory Tracking)
    fig1 = plt.figure("3D Trajectory")
    ax3d = fig1.add_subplot(111, projection="3d")
    ax3d.plot(state_history[0, :], state_history[1, :], state_history[2, :], label="Airship Trajectory (Actual)")
    ax3d.plot(yc_history[0, :], yc_history[1, :], yc_history[2, :], "--", label="Desired Trajectory")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Trajectory Tracking")
    ax3d.legend()
    # Equal aspect ratio might be needed depending on scale
    min_lim = np.min(yc_history[0:3, :])
    max_lim = np.max(yc_history[0:3, :])
    # ax3d.set_xlim([min_lim, max_lim])
    # ax3d.set_ylim([min_lim, max_lim])
    # ax3d.set_zlim([np.min(state_history[2,:]), np.max(state_history[2,:])])

    # 图 2: 位置跟踪误差 (Position Tracking Error e1 vs Constraints)
    fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig2.suptitle("Position Tracking Error (e1) vs Constraints")
    pos_labels = ["e_x", "e_y", "e_z"]
    for i in range(3):
        axs2[i].plot(sim_time, error_history[i, :], label=f"{pos_labels[i]} (Actual Error)")
        axs2[i].plot(sim_time, kb_history[i, :], "r--", label="Constraint kb")
        axs2[i].plot(sim_time, -kb_history[i, :], "r--")
        axs2[i].set_ylabel(f"{pos_labels[i]} [m]")
        axs2[i].legend()
        axs2[i].grid(True)
    axs2[2].set_xlabel("Time [s]")

    # 图 3: 姿态跟踪误差 (Attitude Tracking Error e1 vs Constraints)
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig3.suptitle("Attitude Tracking Error (e1) vs Constraints")
    att_labels = ["e_phi", "e_theta", "e_psi"]
    for i in range(3):
        axs3[i].plot(sim_time, error_history[i + 3, :], label=f"{att_labels[i]} (Actual Error)")
        axs3[i].plot(sim_time, kb_history[i + 3, :], "r--", label="Constraint kb")
        axs3[i].plot(sim_time, -kb_history[i + 3, :], "r--")
        axs3[i].set_ylabel(f"{att_labels[i]} [rad]")
        axs3[i].legend()
        axs3[i].grid(True)
    axs3[2].set_xlabel("Time [s]")

    # 图 4: 控制输入 (Control Inputs)
    fig4, axs4 = plt.subplots(6, 1, sharex=True, figsize=(10, 12))
    fig4.suptitle("Control Input Tau")
    control_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for i in range(6):
        axs4[i].plot(sim_time, control_history[i, :], label=f"{control_labels[i]}")
        axs4[i].set_ylabel(f"{control_labels[i]} [N or Nm]")
        axs4[i].legend()
        axs4[i].grid(True)
    axs4[5].set_xlabel("Time [s]")

    # 图 5: 扰动估计 (Disturbance Estimation)
    fig5, axs5 = plt.subplots(6, 1, sharex=True, figsize=(10, 12))
    fig5.suptitle("Disturbance Estimation")
    dist_labels = ["d1", "d2", "d3", "d4", "d5", "d6"]
    for i in range(6):
        axs5[i].plot(sim_time, disturbance_history[i, :], "k-", label=f"Actual {dist_labels[i]}")
        axs5[i].plot(sim_time, estimate_history[i, :], "r--", label=f"Estimated {dist_labels[i]}")
        axs5[i].set_ylabel(f"{dist_labels[i]}")
        axs5[i].legend()
        axs5[i].grid(True)
    axs5[5].set_xlabel("Time [s]")

    # 图 6: 速度跟踪误差 e2 (Velocity Tracking Error e2)
    fig6, axs6 = plt.subplots(6, 1, sharex=True, figsize=(10, 12))
    fig6.suptitle("Velocity Tracking Error (e2)")
    vel_err_labels = ["e_zeta_dot_x", "e_zeta_dot_y", "e_zeta_dot_z", "e_gamma_dot_phi", "e_gamma_dot_theta", "e_gamma_dot_psi"]
    for i in range(6):
        axs6[i].plot(sim_time, error2_history[i, :], label=f"{vel_err_labels[i]}")
        axs6[i].set_ylabel(f"{vel_err_labels[i]}")
        axs6[i].legend()
        axs6[i].grid(True)
    axs6[5].set_xlabel("Time [s]")

    plt.show()


def run_nmpc_simulation(use_disturbance_compensation=True):
    """
    Run NMPC simulation.
    """

    airship = Airship(params.X0)
    trajectory = Trajectory()


    controller = NMPCThrustController(
        model=airship,
        dt=params.DT,
        N=params.N_HORIZON,
        Q=params.Q,
        R=params.R,
        Qf=params.Qf,
        T_bounds=(params.T_MIN, params.T_MAX),
        mu_bounds=(params.MU_MIN, params.MU_MAX),
        nu_bounds=(params.NU_MIN, params.NU_MAX),
        use_disturbance_compensation=use_disturbance_compensation  # 启用扰动补偿
    )

    # 仿真时间和步数
    sim_time = np.arange(0, params.T_SPAN, params.DT)
    n_steps = len(sim_time)

    # 状态历史记录
    state_history = np.zeros((12, n_steps))
    control_history = np.zeros((3, n_steps))  # [T, mu, nu]
    yc_history = np.zeros((6, n_steps))  # zeta + gamma 位置和姿态
    disturbance_history = np.zeros((6, n_steps))  # 实际扰动
    disturbance_estimate_history = np.zeros((6, n_steps))  # 扰动估计
    error_history = np.zeros((6, n_steps))  # e1 = y - yc
    error2_history = np.zeros((6, n_steps))  # e2 = y_dot - yc_dot

    # 初始状态
    X = airship.get_state()


    # 仿真循环
    for i, t in enumerate(sim_time):
        # 获取当前位置和姿态
        zeta = X[0:3]
        gamma = X[3:6]
        v = X[6:9]
        omega = X[9:12]
        x_vec = X[6:12]

        # === 获取参考轨迹 ===
        X_ref = [] # 用于存储预测时域内的参考状态轨迹
        U_ref = [] # 用于存储预测时域内的参考控制输入
        # 基本参考轨迹（当前时刻）
        yc, yc_dot, _, xc, _ = trajectory.get_linear_trajectory(t)

        # 生成预测时域内的参考轨迹
        for j in range(params.N_HORIZON + 1):
            t_future = t + j * params.DT
            yc_j, yc_dot_j, _, _, _ = trajectory.get_linear_trajectory(t_future)
            X_ref.append(np.concatenate([yc_j, yc_dot_j]))

        # 生成参考控制输入（简单初始猜测）
        for j in range(params.N_HORIZON):
            U_ref.append(np.array([8.0, 0.0, 0.0]))  # 可替换为更智能的猜测

        # 计算误差（用于扰动观测器）
        R = R_block(gamma)
        y = np.concatenate((zeta, gamma))
        y_dot = R @ x_vec

        e1 = y - yc
        e2 = y_dot - yc_dot

        # 获取实际扰动
        actual_delta = params.disturbance_delta(t)

        # NMPC 控制器计算最优控制输入
        u_cmd = controller.step(X, X_ref, U_ref, e1=e1, e2=e2)

        # 获取当前扰动估计
        delta_hat = controller.get_current_disturbance_estimate()

        # 将推力参数转换为力和力矩
        tau = controller.thrust_to_force_torque(u_cmd)

        # 补偿扰动（扰动已经在控制器内部处理）
        tau_compensated = tau - delta_hat * controller.disturbance_compensation_factor

        # 积分系统（RK4）- 考虑实际扰动
        def f(t_rk, x_rk):
            return airship.rhs(t_rk, x_rk, tau_compensated, lambda _: actual_delta)

        # RK4 步进
        X_next = rk4_step(f, t, X, params.DT)

        # 状态更新
        airship.X = X_next
        X = X_next

        # 记录
        state_history[:, i] = X
        control_history[:, i] = u_cmd
        yc_history[:, i] = yc
        disturbance_history[:, i] = actual_delta
        disturbance_estimate_history[:, i] = delta_hat
        error_history[:, i] = e1
        error2_history[:, i] = e2

        # 显示进度
        if i % (n_steps // 10) == 0:
            logger.info("NMPC Simulation progress: %d%%", int(i / n_steps * 100))

    # 绘图
    plt.style.use("seaborn-v0_8-whitegrid")

    # 三维轨迹图
    fig = plt.figure("NMPC 3D Trajectory")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(state_history[0, :], state_history[1, :], state_history[2, :], label="Airship")
    ax.plot(yc_history[0, :], yc_history[1, :], yc_history[2, :], "--", label="Reference")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # 扰动估计图
    fig2, axs2 = plt.subplots(6, 1, sharex=True, figsize=(10, 12))
    fig2.suptitle("Disturbance Estimation")
    dist_labels = ["d1", "d2", "d3", "d4", "d5", "d6"]
    for i in range(6):
        axs2[i].plot(sim_time, disturbance_history[i, :], "k-", label=f"Actual {dist_labels[i]}")
        axs2[i].plot(sim_time, disturbance_estimate_history[i, :], "r--", label=f"Estimated {dist_labels[i]}")
        axs2[i].set_ylabel(f"{dist_labels[i]}")
        axs2[i].legend()
        axs2[i].grid(True)
    axs2[5].set_xlabel("Time [s]")

    # 轨迹误差图
    fig3, axs3 = plt.subplots(6, 1, sharex=True, figsize=(10, 12))
    fig3.suptitle("Tracking Error")
    err_labels = ["e_x", "e_y", "e_z", "e_phi", "e_theta", "e_psi"]
    for i in range(6):
        axs3[i].plot(sim_time, error_history[i, :], label=f"{err_labels[i]}")
        axs3[i].set_ylabel(f"{err_labels[i]}")
        axs3[i].legend()
        axs3[i].grid(True)
    axs3[5].set_xlabel("Time [s]")

    plt.show()






