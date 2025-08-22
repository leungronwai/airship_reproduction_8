"""
# main.py
Airship Trajectory Tracking Simulation
"""
# pylint: disable=invalid-name
# cspell:disable symvar_type


import do_mpc
import matplotlib.pyplot as plt
from casadi.tools import *
from do_mpc.tools import Timer
from matplotlib.ticker import ScalarFormatter

from src.system.controller_dompc import DoMpcConfig


def run_simulation():
    """
    Run the simulation

    """
    # 清空终端
    import os
    if os.name == 'nt':   # Windows
        os.system('cls')
    else:                 # Mac / Linux
        os.system('clear')

    plt.close('all')

    # user settings
    show_animations = False  # Set to True to show animations
    store_results = False

    # setting up the model
    airship_mpc = DoMpcConfig()

    model = airship_mpc.model

    # setting up a mpc controller, given the model
    # mpc = airship_mpc.create_mpc_controller(model)
    mpc = airship_mpc.create_mpc_controller(silence_solver=True)

    # setting up a simulator, given the model
    simulator = airship_mpc.create_simulator(model)

    # setting up an estimator, given the model
    estimator = do_mpc.estimator.StateFeedback(model)


    # 初始化 mpc and simulator 状态设置
    # 设置初始状态，起始点在原点
    initial_position = np.array([0.0, 0.0, -20000.0])  # 起始点在原点
    initial_attitude = np.array([0.0, 0.0, np.pi/12])  # 偏航角 90 度，面向 Y 轴
    initial_velocity = np.array([0.0, 16.0, 0.0])  # 初始切向速度（向上，与圆形轨迹相切）
    initial_angular_velocity = np.array([0.0, 0.0, 0.0])  # 初始无角速度

    X0 = np.concatenate([initial_position, initial_attitude, initial_velocity, initial_angular_velocity])
    x0 = X0.copy().reshape(-1, 1)

    # pushing initial condition to mpc and the simulator
    mpc.x0 = x0
    simulator.x0 = x0

    # setting up initial guesses
    mpc.set_initial_guess()
    simulator.set_initial_guess()

    # simulation of the plant
    timer = Timer()
    optimal_control = []
    optimal_states = []
    optimal_states.append(x0)
    reference_trajectory = []  # 用于存储参考轨迹信息

    # 仿真参数
    DT = 1  # 仿真步长 (s)
    T_SPAN = 1000  # 仿真总时间 (s)

    for i in range(int(T_SPAN / DT)):
        current_time = i * DT

        # 获取当前参考轨迹信息
        yc_ref, yc_dot_ref = airship_mpc.trajectory.get_helix_trajectory(current_time)
        ref_vel_x, ref_vel_y, ref_vel_z = yc_dot_ref[0:3]  # 提取参考速度分量
        ref_vel_magnitude = np.linalg.norm(yc_dot_ref[0:3])  # 计算参考速度大小

        reference_trajectory.append(yc_ref[:3])  # 只存储位置部分

        # for the current state x0, mpc computes the optimal control action u0
        # print(f"Time: {current_time:.2f}s, Ramp factor: {ramp_factor:.2f}")
        timer.tic()
        u0 = mpc.make_step(x0)  # 这里计算出 [T, μ, v]
        # 打印控制输入参数
        print(f"\n=== Time: {current_time:.1f}s ===")
        print(f"Control input: [T={float(u0[0]):.2f}, mu={float(u0[1]):.2f}, nu={float(u0[2]):.2f}]")
        timer.toc()

        # for the current state u0, computes the next state y_next
        y_next = simulator.make_step(u0)

        # for the current state y_next, estimates the next state x0
        x0 = estimator.make_step(y_next)


        vel_x = float(x0[6].item())  # 使用 .item() 方法
        vel_y = float(x0[7].item())
        vel_z = float(x0[8].item())
        vel_magnitude = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)  # 计算实际速度大小

        # 打印参考速度和实际速度
        print(f"\n=== Time: {current_time:.1f}s ===")
        print(f"Reference velocity: [{ref_vel_x:.2f}, {ref_vel_y:.2f}, {ref_vel_z:.2f}] m/s")
        print(f"Actual velocity: [{vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f}] m/s")

        # store the optimal control and state
        optimal_control.append(u0)
        optimal_states.append(x0)

    # reference trajectory plot
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    reference_trajectory = np.array(reference_trajectory)
    optimal_states = np.array(optimal_states)

    print("\n=== Trajectory Summary ===")
    print(f"Reference Trajectory Start: {reference_trajectory[0]} (meters)")
    print(f"Actual Trajectory Start: {optimal_states[0, :3]} (meters)")
    print(f"Reference Trajectory End: {reference_trajectory[-1]} (meters)")
    print(f"Actual Trajectory End: {optimal_states[-1, :3]} (meters)")

    # 添加详细的起始点验证
    print(f"\n=== 起始点详细信息 ===")
    print(
        f"参考轨迹第一个点 (图中绿色点): x={reference_trajectory[0, 0]:.6f}, y={reference_trajectory[0, 1]:.6f}, z={reference_trajectory[0, 2]:.6f}")
    print(
        f"实际轨迹第一个点：x={optimal_states[0, 0].item():.6f}, y={optimal_states[0, 1].item():.6f}, z={optimal_states[0, 2].item():.6f}")

    # 验证 t=0 时的轨迹
    yc_t0, yc_dot_t0 = airship_mpc.trajectory.get_helix_trajectory(0.0)
    print(f"直接调用 get_helix_trajectory(0.0): x={yc_t0[0]:.6f}, y={yc_t0[1]:.6f}, z={yc_t0[2]:.6f}")

    # 验证初始状态设置
    print(
        f"main.py 中设置的初始位置：x={initial_position[0]:.6f}, y={initial_position[1]:.6f}, z={initial_position[2]:.6f}")
    print(f"仿真循环第一次调用时间：current_time = {0 * DT:.6f}")

    # 验证第一个仿真步骤 - 这里也要修复函数名和返回值数量
    first_time = 0 * DT
    yc_first, yc_dot_first = airship_mpc.trajectory.get_helix_trajectory(first_time)
    print(f"仿真第一步 t={first_time}: x={yc_first[0]:.6f}, y={yc_first[1]:.6f}, z={yc_first[2]:.6f}")

    # 绘制轨迹：直接使用参考系的 z 值（向下为正），以 km 为单位显示并保持负号
    ax.plot3D(reference_trajectory[:, 0], reference_trajectory[:, 1], reference_trajectory[:, 2] / 1000,
              label='Reference Trajectory', color='blue')
    ax.plot3D(optimal_states[:, 0], optimal_states[:, 1], optimal_states[:, 2] / 1000,
              label='Actual Trajectory', color='orange')

    # 起点和终点（显示为负数 km）
    ax.scatter(reference_trajectory[0, 0], reference_trajectory[0, 1], reference_trajectory[0, 2] / 1000,
               color="green", s=100, label="Start")
    ax.text(reference_trajectory[0, 0], reference_trajectory[0, 1], reference_trajectory[0, 2] / 1000 + 0.5,
            "Start Point", color="green", fontsize=12, weight='bold')
    ax.scatter(reference_trajectory[-1, 0], reference_trajectory[-1, 1], reference_trajectory[-1, 2] / 1000,
               color="red", s=100, label="End")
    ax.text(reference_trajectory[-1, 0], reference_trajectory[-1, 1],
            reference_trajectory[-1, 2] / 1000 + 0.5,
            "Target Point", color="red", fontsize=12, weight='bold')

    # Adjust axis labels and range
    ax.set_xlabel('X Pos (m)')
    ax.set_ylabel('Y Pos (m)')
    ax.set_zlabel('Altitude (km)')
    ax.set_title('3D Circular Trajectory Comparison')

    # Adjust X and Y axis range to accommodate the new circular trajectory
    ax.set_xlim(-4000, 3000)  # 调整 X 轴范围以适应新的圆形轨迹
    ax.set_ylim(-3000, 3000)  # Y 轴范围

    # Set X axis ticks
    x_ticks = np.arange(-4000, 3001, 1000)  # 与 X 轴范围匹配
    ax.set_xticks(x_ticks)

    # Set Y axis ticks
    y_ticks = np.arange(-3000, 3001, 1000)  # 与 Y 轴范围匹配
    ax.set_yticks(y_ticks)

    # 设置 z 轴刻度为负数范围 (km)
    z_ticks = np.arange(-24.0, -17.5, 0.5)  # 根据数据范围调整
    ax.set_zticks(z_ticks)
    ax.set_zlim(z_ticks[-1], z_ticks[0])

    # 设置所有轴的格式器，强制显示原始值而不是科学计数法
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)

    ax.xaxis.set_major_formatter(formatter)  # X 轴格式器
    ax.yaxis.set_major_formatter(formatter)  # Y 轴格式器
    ax.zaxis.set_major_formatter(formatter)  # Z 轴格式器

    plt.grid(True)
    plt.show()

    # make plots
    optimal_control = np.array(optimal_control)
    plt.figure(figsize=(10, 6))
    plt.plot(optimal_control[:, 0], label='T (Thrust)')
    plt.plot(optimal_control[:, 1], label='μ (Horizontal Deflection)')
    plt.plot(optimal_control[:, 2], label='ν (Vertical Deflection)')
    plt.xlabel('Time Step')
    plt.ylabel('Control Values')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True)
    plt.show()

    optimal_states = np.array(optimal_states)

    # Wrap yaw angle (index 5) to [-π, π] range to avoid sawtooth in plots
    optimal_states[:, 5] = np.arctan2(np.sin(optimal_states[:, 5]), np.cos(optimal_states[:, 5]))

    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(optimal_states[:, 0], label='X Position')
    plt.plot(optimal_states[:, 1], label='Y Position')
    plt.plot(optimal_states[:, 2], label='Z Position')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.title('Position')
    plt.legend()
    plt.grid(True)

    # Attitude plot
    plt.subplot(2, 2, 2)
    plt.plot(optimal_states[:, 3], label='φ (Roll)')
    plt.plot(optimal_states[:, 4], label='θ (Pitch)')
    plt.plot(optimal_states[:, 5], label='ψ (Yaw)')
    plt.xlabel('Time Step')
    plt.ylabel('Attitude (rad)')
    plt.title('Attitude')
    plt.legend()
    plt.grid(True)

    # Velocity plot
    plt.subplot(2, 2, 3)
    plt.plot(optimal_states[:, 6], label='u (X Velocity)')
    plt.plot(optimal_states[:, 7], label='v (Y Velocity)')
    plt.plot(optimal_states[:, 8], label='w (Z Velocity)')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity (m/s)')
    plt.title('Linear Velocity')
    plt.legend()
    plt.grid(True)

    # Angular velocity plot
    plt.subplot(2, 2, 4)
    plt.plot(optimal_states[:, 9], label='p (Roll Rate)')
    plt.plot(optimal_states[:, 10], label='q (Pitch Rate)')
    plt.plot(optimal_states[:, 11], label='r (Yaw Rate)')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 3D trajectory plot
    plt.figure(figsize=(10, 8))
    plt.plot(optimal_states[:, 0], optimal_states[:, 1], label='XY Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Actual Trajectory (XY Plane)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    run_simulation()
