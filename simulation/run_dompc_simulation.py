"""
使用 do-mpc 的气艇仿真脚本
展示如何使用基于 do-mpc 的 NMPC 控制器进行气艇轨迹跟踪
"""


# pylint: disable=invalid-name
# cspell:ignore linalg
# cspell: ignore dompc levelname figsize


# === 标准库 ===
import os
import sys
import time as timer
import logging

import numpy as np
import matplotlib.pyplot as plt



from config import parameters as params
from airship.utils import rk4_step, R_block
from airship.model import Airship
from airship.trajectory import Trajectory
from airship.controller_dompc import DoMPCAirshipController, convert_trajectory_format
from airship.thrust import thrust_params_to_force_torque


# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_dompc_simulation(trajectory_type="linear", use_disturbance_compensation=True):
    """
    运行基于 do-mpc 的 NMPC 仿真

    Args:
        trajectory_type: 轨迹类型 ("linear", "spiral", "figure8", "lemniscate")
        use_disturbance_compensation: 是否使用扰动补偿
    """
    logger.info("开始运行 do-mpc NMPC 仿真 - 轨迹类型：%s", trajectory_type)
    start_time = timer.time()

    # === 初始化组件 ===
    airship = Airship(params.X0)
    trajectory = Trajectory()

    # 创建基于 do-mpc 的控制器
    controller = DoMPCAirshipController(use_disturbance_compensation=use_disturbance_compensation)

    # === 仿真设置 ===
    sim_time = np.arange(0, params.T_SPAN, params.DT)
    n_steps = len(sim_time)

    # 数据记录数组
    state_history = np.zeros((12, n_steps))
    control_history = np.zeros((3, n_steps))  # [T, mu, nu]
    yc_history = np.zeros((6, n_steps))  # 参考位置和姿态
    yc_dot_history = np.zeros((6, n_steps))  # 参考速度和角速度
    disturbance_history = np.zeros((6, n_steps))
    disturbance_estimate_history = np.zeros((6, n_steps))
    error_history = np.zeros((6, n_steps))  # 位置和姿态误差
    error2_history = np.zeros((6, n_steps))  # 速度和角速度误差
    force_torque_history = np.zeros((6, n_steps))  # 实际力和力矩

    # === 主仿真循环 ===
    for i, t in enumerate(sim_time):
        # 获取当前状态
        X = airship.get_state()

        # 获取参考轨迹
        if trajectory_type == "linear":
            yc, yc_dot, _, _, _ = trajectory.get_linear_trajectory(t)
        elif trajectory_type == "spiral":
            yc, yc_dot, _, _, _ = trajectory.get_spiral_trajectory(t)
        elif trajectory_type == "figure8":
            yc, yc_dot, _, _, _ = trajectory.get_figure8_trajectory(t)
        elif trajectory_type == "lemniscate":
            yc, yc_dot, _, _, _ = trajectory.get_lemniscate_trajectory(t)
        else:
            raise ValueError(f"未知轨迹类型：{trajectory_type}")

        # 转换参考轨迹格式
        reference_trajectory = convert_trajectory_format(yc, yc_dot)

        # 计算误差（用于记录）
        zeta = X[0:3]  # 当前位置
        gamma = X[3:6]  # 当前姿态
        v = X[6:9]  # 当前速度
        omega = X[9:12]  # 当前角速度
        x_vec = X[6:12]

        R = R_block(gamma)
        y = np.concatenate((zeta, gamma))
        y_dot = R @ x_vec

        e1 = y - yc  # 位置和姿态误差
        e2 = y_dot - yc_dot  # 速度和角速度误差

        # 使用 do-mpc 控制器计算控制输入
        u_cmd = controller.step(X, reference_trajectory, t)

        # 获取扰动估计
        delta_hat = controller.get_current_disturbance_estimate()

        # 将推力参数转换为力和力矩
        tau = thrust_params_to_force_torque(u_cmd, params.rp_r, params.rp_l)

        # 应用扰动补偿
        if use_disturbance_compensation:
            tau_compensated = tau - delta_hat * controller.disturbance_compensation_factor
        else:
            tau_compensated = tau

        # 获取实际扰动
        actual_delta = params.disturbance_delta(t)

        # 使用 RK4 积分更新气艇状态
        def airship_dynamics(t_rk, X_rk):
            return airship.rhs(t_rk, X_rk, tau_compensated, lambda _: actual_delta)

        X_next = rk4_step(airship_dynamics, t, X, params.DT)

        # 更新气艇状态
        airship.X = X_next

        # 角度标准化
        airship.X[3:6] = (airship.X[3:6] + np.pi) % (2 * np.pi) - np.pi

        # 记录数据
        state_history[:, i] = X
        control_history[:, i] = u_cmd
        yc_history[:, i] = yc
        yc_dot_history[:, i] = yc_dot
        disturbance_history[:, i] = actual_delta
        disturbance_estimate_history[:, i] = delta_hat
        error_history[:, i] = e1
        error2_history[:, i] = e2
        force_torque_history[:, i] = tau_compensated

        # 显示进度
        if i % (n_steps // 10) == 0:
            logger.info("仿真进度：%d / n_steps ≈ %.1f%%", i, 100 * i / n_steps)
            logger.debug("位置误差范数：%.4f", np.linalg.norm(e1))
            logger.debug("姿态误差范数：%.4f", np.linalg.norm(e1[3:6]))

    end_time = timer.time()
    logger.info("仿真完成，耗时：%.2f 秒", end_time - start_time)

    # === 结果分析和绘图 ===
    _plot_simulation_results(
        sim_time, state_history, control_history,
        yc_history, yc_dot_history, error_history, error2_history,
        disturbance_history, disturbance_estimate_history,
        force_torque_history, trajectory_type
    )

    # === 性能评估 ===
    _evaluate_performance(error_history, error2_history, control_history, sim_time)

    return {
        'time': sim_time,
        'states': state_history,
        'controls': control_history,
        'references': yc_history,
        'errors': error_history,
        'disturbances': disturbance_history,
        'estimates': disturbance_estimate_history
    }


def _plot_simulation_results(sim_time, state_history, control_history,
                           yc_history, yc_dot_history, error_history, error2_history,
                           disturbance_history, disturbance_estimate_history,
                           force_torque_history, trajectory_type):
    """绘制仿真结果"""
    plt.style.use("seaborn-v0_8-whitegrid")

    # 图 1: 三维轨迹跟踪
    fig1 = plt.figure(f"do-mpc NMPC 3D 轨迹跟踪 - {trajectory_type}", figsize=(12, 8))
    ax3d = fig1.add_subplot(111, projection="3d")

    # 绘制轨迹
    ax3d.plot(state_history[0, :], state_history[1, :], state_history[2, :],
              'b-', linewidth=2, label="实际轨迹")
    ax3d.plot(yc_history[0, :], yc_history[1, :], yc_history[2, :],
              'r--', linewidth=2, label="参考轨迹")

    # 标记起点和终点
    ax3d.scatter(state_history[0, 0], state_history[1, 0], state_history[2, 0],
                c='green', s=100, marker='o', label="起点")
    ax3d.scatter(state_history[0, -1], state_history[1, -1], state_history[2, -1],
                c='red', s=100, marker='s', label="终点")

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"do-mpc NMPC 3D 轨迹跟踪 - {trajectory_type}")
    ax3d.legend()

    # 图 2: 位置跟踪误差
    fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig2.suptitle("位置跟踪误差")

    pos_labels = ["X 误差", "Y 误差", "Z 误差"]
    for i in range(3):
        axs2[i].plot(sim_time, error_history[i, :], 'b-', linewidth=2, label=f"{pos_labels[i]}")
        axs2[i].set_ylabel(f"{pos_labels[i]} [m]")
        axs2[i].legend()
        axs2[i].grid(True, alpha=0.3)
    axs2[2].set_xlabel("时间 [s]")

    # 图 3: 姿态跟踪误差
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig3.suptitle("姿态跟踪误差")

    att_labels = ["横滚角误差", "俯仰角误差", "航向角误差"]
    for i in range(3):
        axs3[i].plot(sim_time, np.rad2deg(error_history[i + 3, :]), 'r-',
                    linewidth=2, label=f"{att_labels[i]}")
        axs3[i].set_ylabel(f"{att_labels[i]} [度]")
        axs3[i].legend()
        axs3[i].grid(True, alpha=0.3)
    axs3[2].set_xlabel("时间 [s]")

    # 图 4: 控制输入
    fig4, axs4 = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig4.suptitle("控制输入")

    control_labels = ["推力大小 T [N]", "水平偏转角 μ [度]", "垂直偏转角 ν [度]"]
    control_data = [
        control_history[0, :],
        np.rad2deg(control_history[1, :]),
        np.rad2deg(control_history[2, :])
    ]

    for i in range(3):
        axs4[i].plot(sim_time, control_data[i], 'g-', linewidth=2, label=control_labels[i])
        axs4[i].set_ylabel(control_labels[i])
        axs4[i].legend()
        axs4[i].grid(True, alpha=0.3)
    axs4[2].set_xlabel("时间 [s]")

    # 图 5: 扰动估计
    fig5, axs5 = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    fig5.suptitle("扰动估计 vs 实际扰动")

    dist_labels = ["力扰动 Fx", "力扰动 Fy", "力扰动 Fz",
                   "力矩扰动 Mx", "力矩扰动 My", "力矩扰动 Mz"]

    for i in range(6):
        axs5[i].plot(sim_time, disturbance_history[i, :], 'k-',
                    linewidth=2, label=f"实际 {dist_labels[i]}")
        axs5[i].plot(sim_time, disturbance_estimate_history[i, :], 'r--',
                    linewidth=2, label=f"估计 {dist_labels[i]}")
        axs5[i].set_ylabel(f"{dist_labels[i]}")
        axs5[i].legend()
        axs5[i].grid(True, alpha=0.3)
    axs5[5].set_xlabel("时间 [s]")

    plt.tight_layout()
    plt.show()


def _evaluate_performance(error_history, error2_history, control_history, sim_time):
    """评估控制性能"""
    logger.info("=== 控制性能评估 ===")

    # 位置误差统计
    pos_errors = error_history[0:3, :]
    pos_rmse = np.sqrt(np.mean(pos_errors**2, axis=1))
    pos_max = np.max(np.abs(pos_errors), axis=1)

    logger.info(f"位置 RMSE: X={pos_rmse[0]:.3f}m, Y={pos_rmse[1]:.3f}m, Z={pos_rmse[2]:.3f}m")
    logger.info(f"位置最大误差：X={pos_max[0]:.3f}m, Y={pos_max[1]:.3f}m, Z={pos_max[2]:.3f}m")

    # 姿态误差统计
    att_errors = error_history[3:6, :]
    att_rmse = np.sqrt(np.mean(att_errors**2, axis=1))
    att_max = np.max(np.abs(att_errors), axis=1)

    logger.info(f"姿态 RMSE: φ={np.rad2deg(att_rmse[0]):.3f}°, θ={np.rad2deg(att_rmse[1]):.3f}°, ψ={np.rad2deg(att_rmse[2]):.3f}°")
    logger.info(f"姿态最大误差：φ={np.rad2deg(att_max[0]):.3f}°, θ={np.rad2deg(att_max[1]):.3f}°, ψ={np.rad2deg(att_max[2]):.3f}°")

    # 控制输入统计
    control_mean = np.mean(control_history, axis=1)
    control_std = np.std(control_history, axis=1)

    logger.info(f"控制输入均值：T={control_mean[0]:.3f}N, μ={np.rad2deg(control_mean[1]):.3f}°, ν={np.rad2deg(control_mean[2]):.3f}°")
    logger.info(f"控制输入标准差：T={control_std[0]:.3f}N, μ={np.rad2deg(control_std[1]):.3f}°, ν={np.rad2deg(control_std[2]):.3f}°")

    # 稳态误差分析（最后 10% 的数据）
    steady_start = int(0.9 * len(sim_time))
    pos_steady = np.mean(np.abs(pos_errors[:, steady_start:]), axis=1)
    att_steady = np.mean(np.abs(att_errors[:, steady_start:]), axis=1)

    logger.info(f"稳态位置误差：X={pos_steady[0]:.3f}m, Y={pos_steady[1]:.3f}m, Z={pos_steady[2]:.3f}m")
    logger.info(f"稳态姿态误差：φ={np.rad2deg(att_steady[0]):.3f}°, θ={np.rad2deg(att_steady[1]):.3f}°, ψ={np.rad2deg(att_steady[2]):.3f}°")


if __name__ == "__main__":
    # 主程序入口

    print("=== do-mpc NMPC 气艇仿真 ===")

    # 可以选择不同的轨迹类型进行测试
    trajectory_types = ["linear", "spiral", "figure8", "lemniscate"]

    # 选择轨迹类型
    selected_trajectory = "linear"  # 可以改为其他轨迹类型

    # 运行仿真
    results = run_dompc_simulation(
        trajectory_type=selected_trajectory,
        use_disturbance_compensation=True
    )

    print(f"仿真完成！轨迹类型：{selected_trajectory}")
