"""
使用 do-mpc Simulator 的气艇仿真脚本
展示如何使用 do-mpc 的完整生态系统进行气艇轨迹跟踪
"""

# pylint: disable=invalid-name
# cspell:ignore linalg suptitle sharex sharey whitegrid
# cspell: ignore dompc levelname figsize set_xlabel set_ylabel set_zlabel

import os
import sys
import time as timer
import logging

import numpy as np
import matplotlib.pyplot as plt

from config import parameters as params
from airship.trajectory import Trajectory
from airship.controller_dompc import DoMPCAirshipController, convert_trajectory_format
from airship.utils import rk4_step
from airship.model import AirshipCasADiSymbolic

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_dompc_simulation(trajectory_type="linear", use_disturbance_compensation=True, use_simulator=True):
    """
    运行基于 do-mpc 的 NMPC 仿真

    Args:
        trajectory_type: 轨迹类型
        use_disturbance_compensation: 是否使用扰动补偿
        use_simulator: 是否使用 do-mpc Simulator（True）或仅使用 MPC 控制器（False）

    Returns:
        dict: 仿真结果数据
    """
    logger.info("开始运行 do-mpc Simulator NMPC 仿真 - 轨迹类型：%s", trajectory_type)
    start_time = timer.time()

    # === 初始化组件 ===
    trajectory = Trajectory()

    # 创建基于 do-mpc 的控制器（包含 Simulator）
    controller = DoMPCAirshipController(
        use_disturbance_compensation=use_disturbance_compensation,
        create_simulator=use_simulator
    )

    # === 仿真设置 ===
    sim_time = np.arange(0, params.T_SPAN, params.DT)
    n_steps = len(sim_time)

    # 数据记录数组
    state_history = np.zeros((12, n_steps))
    control_history = np.zeros((3, n_steps))
    reference_history = np.zeros((12, n_steps))
    disturbance_history = np.zeros((6, n_steps))
    disturbance_estimate_history = np.zeros((6, n_steps))
    error_history = np.zeros((12, n_steps))
    prediction_history = []

    # 设置初始状态
    current_state = params.X0.copy()
    controller.simulator.x0 = np.concatenate([
        current_state[0:3].reshape(-1, 1),
        current_state[3:6].reshape(-1, 1),
        current_state[6:9].reshape(-1, 1),
        current_state[9:12].reshape(-1, 1)
    ])

    # === 主仿真循环 ===
    for i, t in enumerate(sim_time):
        # 获取参考轨迹
        yc, yc_dot = _get_reference_trajectory(trajectory, trajectory_type, t)
        reference_trajectory = convert_trajectory_format(yc, yc_dot)

        # 记录参考轨迹
        reference_state = np.concatenate([yc, yc_dot])
        reference_history[:, i] = reference_state

        # 计算误差
        error = current_state - reference_state
        error_history[:, i] = error

        # 使用 MPC 控制器计算控制输入
        u_cmd = controller.step(current_state, reference_trajectory, t)

        # 记录控制输入
        control_history[:, i] = u_cmd

        # 获取扰动估计
        delta_hat = controller.get_current_disturbance_estimate()
        disturbance_estimate_history[:, i] = delta_hat

        # 获取实际扰动
        actual_delta = params.disturbance_delta(t)
        disturbance_history[:, i] = actual_delta

        # 记录当前状态
        state_history[:, i] = current_state

        # 根据模式选择状态更新方法
        if use_simulator and controller.simulator is not None:
            # 使用 do-mpc Simulator
            try:
                x_next = controller.simulate_step(u_cmd.reshape(-1, 1))
                if hasattr(x_next, 'full'):
                    current_state = x_next.full().flatten()
                else:
                    current_state = np.array(x_next).flatten()
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error("Simulator 步骤失败：%s", e)
                current_state = _fallback_integration(current_state, u_cmd, actual_delta, params.DT)
        else:
            # 使用传统的数值积分
            current_state = _fallback_integration(current_state, u_cmd, actual_delta, params.DT)

        # 获取 MPC 预测（可选）
        if i % 5 == 0:  # 每 5 步记录一次预测，减少存储
            prediction = controller.get_prediction()
            if prediction['states'] is not None:
                prediction_history.append({
                    'time': t,
                    'prediction': prediction
                })

        # 显示进度
        if i % (n_steps // 10) == 0:
            logger.info("仿真进度：%d / %d ≈ %.1f%%", i, n_steps, 100 * i / n_steps)
            logger.debug("位置误差范数：%.4f", np.linalg.norm(error[0:3]))
            logger.debug("姿态误差范数：%.4f", np.linalg.norm(error[3:6]))

    end_time = timer.time()
    logger.info("仿真完成，耗时：%.2f 秒", end_time - start_time)

    # === 结果分析和绘图 ===
    results = {
        'time': sim_time,
        'states': state_history,
        'controls': control_history,
        'references': reference_history,
        'errors': error_history,
        'disturbances': disturbance_history,
        'estimates': disturbance_estimate_history,
        'predictions': prediction_history
    }

    _plot_simulation_results(results, trajectory_type)
    _evaluate_performance(results)

    return results


def _get_reference_trajectory(trajectory, trajectory_type, t):
    """获取参考轨迹"""
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

    return yc, yc_dot


def _fallback_integration(current_state, u_cmd, disturbance, dt):
    """
    改进的回退积分方法
    使用简化但更稳定的动力学模型
    """


    # 使用简化的符号模型进行积分
    try:
        symbolic_model = AirshipCasADiSymbolic(params)

        def dynamics_func(t, x):
            return symbolic_model.rhs_symbolic(x, u_cmd, external_disturbance=disturbance)

        # 使用 RK4 积分
        next_state = rk4_step(dynamics_func, 0, current_state, dt)

        # 角度归一化
        next_state[3:6] = (next_state[3:6] + np.pi) % (2 * np.pi) - np.pi

        return next_state

    except Exception as e: # pylint: disable=broad-exception-caught
        logger.warning("回退积分失败，使用最简积分：%s", e)

        # 最简单的积分作为最后回退
        next_state = current_state.copy()
        next_state[0:3] += current_state[6:9] * dt  # 位置更新
        return next_state


def _plot_simulation_results(results, trajectory_type):
    """绘制仿真结果"""
    sim_time = results['time']
    state_history = results['states']
    control_history = results['controls']
    reference_history = results['references']
    error_history = results['errors']
    disturbance_history = results['disturbances']
    disturbance_estimate_history = results['estimates']

    plt.style.use("seaborn-v0_8-whitegrid")

    # 图 1: 三维轨迹跟踪
    fig1 = plt.figure(f"do-mpc Simulator 3D 轨迹跟踪 - {trajectory_type}", figsize=(12, 8))
    ax3d = fig1.add_subplot(111, projection="3d")

    ax3d.plot(state_history[0, :], state_history[1, :], state_history[2, :],
              'b-', linewidth=2, label="实际轨迹")
    ax3d.plot(reference_history[0, :], reference_history[1, :], reference_history[2, :],
              'r--', linewidth=2, label="参考轨迹")

    ax3d.scatter(state_history[0, 0], state_history[1, 0], state_history[2, 0],
                c='green', s=100, marker='o', label="起点")
    ax3d.scatter(state_history[0, -1], state_history[1, -1], state_history[2, -1],
                c='red', s=100, marker='s', label="终点")

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"do-mpc Simulator 3D 轨迹跟踪 - {trajectory_type}")
    ax3d.legend()

    # 图 2: 状态跟踪误差
    fig2, axs2 = plt.subplots(4, 3, sharex=True, figsize=(15, 12))
    fig2.suptitle("状态跟踪误差")

    state_labels = [
        ["位置 X", "位置 Y", "位置 Z"],
        ["姿态 φ", "姿态 θ", "姿态 ψ"],
        ["速度 u", "速度 v", "速度 w"],
        ["角速度 p", "角速度 q", "角速度 r"]
    ]

    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            if i == 1:  # 姿态角度转换为度
                axs2[i, j].plot(sim_time, np.rad2deg(error_history[idx, :]), 'b-', linewidth=2)
                axs2[i, j].set_ylabel(f"{state_labels[i][j]} [度]")
            else:
                axs2[i, j].plot(sim_time, error_history[idx, :], 'b-', linewidth=2)
                unit = "[m]" if i == 0 else ("[m/s]" if i == 2 else "[rad/s]")
                axs2[i, j].set_ylabel(f"{state_labels[i][j]} {unit}")
            
            axs2[i, j].set_title(state_labels[i][j])
            axs2[i, j].grid(True, alpha=0.3)

    axs2[3, 1].set_xlabel("时间 [s]")

    # 图 3: 控制输入
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig3.suptitle("控制输入")

    control_labels = ["推力大小 T [N]", "水平偏转角 μ [度]", "垂直偏转角 ν [度]"]
    control_data = [
        control_history[0, :],
        np.rad2deg(control_history[1, :]),
        np.rad2deg(control_history[2, :])
    ]

    for i in range(3):
        axs3[i].plot(sim_time, control_data[i], 'g-', linewidth=2)
        axs3[i].set_ylabel(control_labels[i])
        axs3[i].set_title(control_labels[i])
        axs3[i].grid(True, alpha=0.3)

    axs3[2].set_xlabel("时间 [s]")

    # 图 4: 扰动估计对比
    fig4, axs4 = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    fig4.suptitle("扰动估计 vs 实际扰动")

    dist_labels = ["力扰动 Fx", "力扰动 Fy", "力扰动 Fz",
                   "力矩扰动 Mx", "力矩扰动 My", "力矩扰动 Mz"]

    for i in range(6):
        axs4[i].plot(sim_time, disturbance_history[i, :], 'k-',
                    linewidth=2, label=f"实际 {dist_labels[i]}")
        axs4[i].plot(sim_time, disturbance_estimate_history[i, :], 'r--',
                    linewidth=2, label=f"估计 {dist_labels[i]}")
        axs4[i].set_ylabel(f"{dist_labels[i]}")
        axs4[i].set_title(dist_labels[i])
        axs4[i].legend()
        axs4[i].grid(True, alpha=0.3)

    axs4[5].set_xlabel("时间 [s]")

    plt.tight_layout()
    plt.show()


def _evaluate_performance(results):
    """评估控制性能"""
    logger.info("=== do-mpc Simulator 控制性能评估 ===")
    
    error_history = results['errors']
    control_history = results['controls']
    sim_time = results['time']

    # 位置误差统计
    pos_errors = error_history[0:3, :]
    pos_rmse = np.sqrt(np.mean(pos_errors**2, axis=1))
    pos_max = np.max(np.abs(pos_errors), axis=1)

    logger.info("位置 RMSE: X=%.3fm, Y=%.3fm, Z=%.3fm", pos_rmse[0], pos_rmse[1], pos_rmse[2])
    logger.info("位置最大误差：X=%.3fm, Y=%.3fm, Z=%.3fm", pos_max[0], pos_max[1], pos_max[2])

    # 姿态误差统计
    att_errors = error_history[3:6, :]
    att_rmse = np.sqrt(np.mean(att_errors**2, axis=1))
    att_max = np.max(np.abs(att_errors), axis=1)

    logger.info("姿态 RMSE: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                np.rad2deg(att_rmse[0]), np.rad2deg(att_rmse[1]), np.rad2deg(att_rmse[2]))
    logger.info("姿态最大误差：φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                np.rad2deg(att_max[0]), np.rad2deg(att_max[1]), np.rad2deg(att_max[2]))

    # 控制输入统计
    control_mean = np.mean(control_history, axis=1)
    control_std = np.std(control_history, axis=1)

    logger.info("控制输入均值：T=%.3fN, μ=%.3f°, ν=%.3f°",
                control_mean[0], np.rad2deg(control_mean[1]), np.rad2deg(control_mean[2]))
    logger.info("控制输入标准差：T=%.3fN, μ=%.3f°, ν=%.3f°",
                control_std[0], np.rad2deg(control_std[1]), np.rad2deg(control_std[2]))

    # 稳态误差分析
    steady_start = int(0.8 * len(sim_time))
    pos_steady = np.mean(np.abs(pos_errors[:, steady_start:]), axis=1)
    att_steady = np.mean(np.abs(att_errors[:, steady_start:]), axis=1)

    logger.info("稳态位置误差：X=%.3fm, Y=%.3fm, Z=%.3fm",
                pos_steady[0], pos_steady[1], pos_steady[2])
    logger.info("稳态姿态误差：φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                np.rad2deg(att_steady[0]), np.rad2deg(att_steady[1]), np.rad2deg(att_steady[2]))


if __name__ == "__main__":
    print("=== do-mpc Simulator 气艇仿真 ===")

    # 可以选择不同的轨迹类型进行测试
    trajectory_types = ["linear", "spiral", "figure8", "lemniscate"]

    # 选择轨迹类型
    selected_trajectory = "linear"

    # 运行仿真
    simulation_results = run_dompc_simulation(
        trajectory_type=selected_trajectory,
        use_disturbance_compensation=True
    )

    print(f"仿真完成！轨迹类型：{selected_trajectory}")
    print("结果已保存并显示图表")
