# simulation/run_simulation.py


"""
Main simulation script for airship trajectory tracking and controller evaluation.
"""

# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot nlpsol xlabel ylabel zlabel
# cspell: ignore whitegrid figsize sharex suptitle


# === Standard libraries ===
import os
import sys
import time as timer  # Use timer to avoid conflict with time variable t
import logging

# === Third-party libraries ===
import numpy as np
import matplotlib.pyplot as plt

# === Local modules ===
from config import parameters as params
from airship.utils import R_block, rk4_step
from airship.model import Airship
from airship.trajectory import Trajectory
from airship.controller import AnyController, NMPCThrustController


# === Add project root directory to path (if running from root directory) ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))





# Global logger (basicConfig done here)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_simulation(trajectory_type="default"):
    """
    Run simulation with trajectory tracking.
    
    1. Initialize: Create airship model, trajectory generator, controller and other objects
    2. Simulation loop: At each time step:
        a. Get reference trajectory
        b. Call NMPC controller to calculate control input
        c. Update airship state
    
    Parameters:
    ----------
    trajectory_type : str, optional
        Type of trajectory to follow ("default", "spiral", "figure8", "lemniscate", "linear")
    """

    # --- Initialize ---
    airship = Airship(params.X0)
    trajectory = Trajectory()
    controller = AnyController()

    # --- Data logging ---
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

    # --- Simulation loop ---
    for i, t in enumerate(sim_time):
        # 2. Simulation loop: At each time step:
        #     a. Get reference trajectory
        #     b. Call NMPC controller to calculate control input
        #     c. Update airship state


        # Get current state
        X = airship.get_state()

        # Get desired state based on trajectory type
        if trajectory_type == "default":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_linear_trajectory(t)
        elif trajectory_type == "spiral":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_spiral_trajectory(t)
            # Convert spiral trajectory to required format...
            # Additional conversion code needed here - to be implemented
        elif trajectory_type == "figure8":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_figure8_trajectory(t)
        elif trajectory_type == "lemniscate":
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_lemniscate_trajectory(t)
        elif trajectory_type == "linear":
            # Linear trajectory, can customize start and end points
            start_point = np.array([0.0, 0.0, -19000.0])
            end_point = np.array([5000.0, 5000.0, -19000.0])
            yc, yc_dot, yc_ddot, xc, xc_dot = trajectory.get_linear_trajectory(
                t, start_point=start_point, end_point=end_point, speed=15.0, hover_at_end=True  # Flight speed 15 m/s  # Hover at end point
            )
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

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


def run_nmpc_simulation(use_disturbance_compensation=True, trajectory_type="linear"):
    """
    Run NMPC simulation.
    """

    airship = Airship(params.X0)
    trajectory = Trajectory()


    controller = NMPCThrustController(
        model=airship,
        dt=params.DT, # 时间步长
        N=params.N_HORIZON, # 预测时域长度
        Q=params.Q, # 状态误差权重
        R=params.R, # 控制输入权重
        Qf=params.Qf, # 终端状态误差权重
        T_bounds=(params.T_MIN, params.T_MAX), # 推力约束
        mu_bounds=(params.MU_MIN, params.MU_MAX), # 俯仰角约束
        nu_bounds=(params.NU_MIN, params.NU_MAX), # 偏航角约束
        use_disturbance_compensation=use_disturbance_compensation  # 启用扰动补偿
    )




    # 仿真时间和步数
    sim_time = np.arange(0, params.T_SPAN, params.DT)
    n_steps = len(sim_time)

    # 状态历史数据记录
    state_history = np.zeros((12, n_steps))
    control_history = np.zeros((3, n_steps))  # [T, mu, nu]
    yc_history = np.zeros((6, n_steps))  # zeta + gamma 位置和姿态
    disturbance_history = np.zeros((6, n_steps))  # 实际扰动
    disturbance_estimate_history = np.zeros((6, n_steps))  # 扰动估计
    error_history = np.zeros((6, n_steps))  # e1 = y - yc
    error2_history = np.zeros((6, n_steps))  # e2 = y_dot - yc_dot

    # 获取气艇的当前状态，包括位置、姿态、速度、角速度
    X = airship.get_state()

    yc = None
    yc_dot = None
    xc = None
    xc_current = None
    xc_next = None


    # 仿真循环
    for i, t in enumerate(sim_time):
        # '''
        # 在每个时间步：
        #     1.获取当前状态。
        #     2.获取参考轨迹。
        #     3.计算误差。
        #     4.调用 NMPC 控制器计算控制输入。
        #     5.更新气艇状态。
        #     6.记录数据。
        #

        # 获取当前位置和姿态
        zeta = X[0:3]  # 位置
        gamma = X[3:6]  # 姿态
        v = X[6:9]  # 速度
        omega = X[9:12]  # 角速度
        x_vec = X[6:12]  # 状态向量


        # 基本参考轨迹（当前时刻）
        # yc：参考位置和姿态 [zeta_d, gamma_d]。
        # yc_dot：参考速度和角速度 [zeta_d_dot, gamma_d_dot]。
        # xc：参考状态 [zeta_d, gamma_d, v_d, omega_d]。
        if trajectory_type == "linear":
            yc, yc_dot, _, xc, _ = trajectory.get_linear_trajectory(t)
        elif trajectory_type == "spiral":
            yc, yc_dot, _, xc, _ = trajectory.get_spiral_trajectory(t)
        elif trajectory_type == "figure8":
            yc, yc_dot, _, xc, _ = trajectory.get_figure8_trajectory(t)
        elif trajectory_type == "lemniscate":
            yc, yc_dot, _, xc, _ = trajectory.get_lemniscate_trajectory(t)






        # === 通过循环生成预测时域内的参考状态轨迹 X_ref 和参考控制输入 U_ref ===
        X_ref = [] # 用于存储预测时域内的参考状态轨迹
        U_ref = [] # 用于存储预测时域内的参考控制输入

        # 生成预测时域内的参考轨迹
        for j in range(params.N_HORIZON + 1):
            t_future = t + j * params.DT
            if trajectory_type == "spiral":
                yc_j, yc_dot_j, _, _, _ = trajectory.get_spiral_trajectory(t_future)
            elif trajectory_type == "figure8":
                yc_j, yc_dot_j, _, _, _ = trajectory.get_figure8_trajectory(t_future)
            elif trajectory_type == "lemniscate":
                yc_j, yc_dot_j, _, _, _ = trajectory.get_lemniscate_trajectory(t_future)
            else:  # default to linear
                yc_j, yc_dot_j, _, _, _ = trajectory.get_linear_trajectory(t_future)
            X_ref.append(np.concatenate([yc_j, yc_dot_j]))

        # 生成参考控制输入
        for j in range(params.N_HORIZON):
            t_future = t + j * params.DT

            # 获取当前和下一时刻的参考位置和速度
            if trajectory_type == "spiral":
                _, _, _, xc_current, _ = trajectory.get_spiral_trajectory(t_future)
                _, _, _, xc_next, _ = trajectory.get_spiral_trajectory(t_future + params.DT)
            elif trajectory_type == "figure8":
                _, _, _, xc_current, _ = trajectory.get_figure8_trajectory(t_future)
                _, _, _, xc_next, _ = trajectory.get_figure8_trajectory(t_future + params.DT)
            elif trajectory_type == "lemniscate":
                _, _, _, xc_current, _ = trajectory.get_lemniscate_trajectory(t_future)
                _, _, _, xc_next, _ = trajectory.get_lemniscate_trajectory(t_future + params.DT)
            else:  # linear
                _, _, _, xc_current, _ = trajectory.get_linear_trajectory(t_future)
                _, _, _, xc_next, _ = trajectory.get_linear_trajectory(t_future + params.DT)

            # === 计算期望推力大小和方向 ===
            velocity = xc_current[0:3] # 包含线速度
            velocity_mag = np.linalg.norm(velocity) # 线速度的大小

            # 基于速度估计推力大小
            T_est = min(0.8 * velocity_mag, params.T_MAX)
            T_est = max(T_est, 0.5)  # 至少有一些推力

            # 基于航向估计推力方向
            if velocity_mag > 0.5:
                mu_est = 0.0  # 简化处理
                nu_est = 0.0  # 简化处理
            else:
                mu_est = 0.0
                nu_est = 0.0

            U_ref.append(np.array([T_est, mu_est, nu_est]))

        # 然后在 run_nmpc_simulation 中使用（若采用 U_ref 方案 2，直接把注释取消）
        # if controller.last_optimal_sequence is not None:
        #     # 使用前一时刻的最优控制序列后移一步
        #     for j in range(params.N_HORIZON-1):
        #         U_ref.append(controller.last_optimal_sequence[j+1])
        #     # 对最后一步重复使用最后的控制
        #     U_ref.append(controller.last_optimal_sequence[-1])
        # else:
        #     # 首次运行时使用简单猜测
        #     for j in range(params.N_HORIZON):
        #         U_ref.append(np.array([8.0, 0.0, 0.0]))




        # 计算当前状态与参考状态之间的误差（用于扰动观测器）
        R = R_block(gamma)  # 姿态旋转矩阵
        y = np.concatenate((zeta, gamma))  # 当前位置和姿态
        y_dot = R @ x_vec  # 当前速度和角速度

        e1 = y - yc #位置和姿态误差
        e2 = y_dot - yc_dot #速度和角速度误差

        # 获取实际扰动
        actual_delta = params.disturbance_delta(t)

        # NMPC 控制器计算当前时刻的最优控制输入（推力大小和方向角）
        u_cmd = controller.step(X, X_ref, U_ref, e1=e1, e2=e2) # 控制输入 [T, mu, nu]

        # 获取当前扰动估计值
        delta_hat = controller.get_current_disturbance_estimate()

        # 将推力参数转换为力和力矩
        tau = controller.thrust_to_force_torque(u_cmd) # 气艇的推力和力矩 [Fx, Fy, Fz, Tx, Ty, Tz]

        # 补偿扰动（扰动已经在控制器内部处理）
        tau_compensated = tau - delta_hat * controller.disturbance_compensation_factor

        # 使用 RK4 方法对气艇的动力学方程进行积分，更新气艇状态 - 考虑实际扰动
        def f(t_rk, x_rk, tau_comp=tau_compensated):
            return airship.rhs(t_rk, x_rk, tau_comp, lambda _: actual_delta)

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
        disturbance_estimate_history[:, i] = delta_hat.flatten()
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






