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

from src.system.controller_dompc import DoMpcConfig


def run_simulation():
    """
    Run the simulation

    """

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

    # Set the initial state of mpc and simulator
    # 初始化状态设置
    initial_position = np.array([1500.0, 0.0, 0.0])  # 起始位置
    initial_attitude = np.array([0.0, 0.0, np.pi / 2])  # 起始姿态（面朝 Y 方向）
    initial_velocity = np.array([0, 0.0, 0.0])  # 起始线速度
    initial_angular_velocity = np.array([0.0, 0.0, 0.00])  # 起始角速度

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
    T_SPAN = 100  # 仿真总时间 (s)

    for i in range(int(T_SPAN / DT)):
        current_time = i * DT

        # 获取当前参考轨迹信息
        yc_ref, yc_dot_ref, _, _, _ = airship_mpc.trajectory.get_spiral_trajectory(current_time)
        ref_vel_x, ref_vel_y, ref_vel_z = yc_dot_ref[0:3]  # 提取参考速度分量
        ref_vel_magnitude = np.linalg.norm(yc_dot_ref[0:3])  # 计算参考速度大小

        reference_trajectory.append(yc_ref[:3])  # 只存储位置部分

        # for the current state x0, mpc computes the optimal control action u0
        # print(f"Time: {current_time:.2f}s, Ramp factor: {ramp_factor:.2f}")
        timer.tic()
        u0 = mpc.make_step(x0)
        # 打印控制输入
        print(f"\n=== Time: {current_time:.1f}s ===")
        print(f"Control input: [T={float(u0[0]):.2f}, μ={float(u0[1]):.2f}, ν={float(u0[2]):.2f}]")
        timer.toc()

        # for the current state u0, computes the next state y_next
        y_next = simulator.make_step(u0)

        # for the current state y_next, estimates the next state x0
        x0 = estimator.make_step(y_next)

        # 修复 NumPy 弃用警告
        vel_x = float(x0[6].item())  # 使用 .item() 方法
        vel_y = float(x0[7].item())
        vel_z = float(x0[8].item())
        vel_magnitude = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)  # 计算实际速度大小

        # 打印参考速度和实际速度
        print(f"\n=== Time: {current_time:.1f}s ===")
        print(f"Reference velocity: [{ref_vel_x:.2f}, {ref_vel_y:.2f}, {ref_vel_z:.2f}] m/s")
        print(f"Reference |v|: {ref_vel_magnitude:.2f} m/s")
        print(f"Actual velocity: [{vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f}] m/s")
        print(f"Actual |v|: {vel_magnitude:.2f} m/s")

        # # 强制限制实际速度
        # MAX_VEL = 20.0
        # if vel_magnitude > MAX_VEL:
        #     print(f"强制限制速度：{vel_magnitude:.2f} → {vel_magnitude} m/s")
        #     scale_factor = MAX_VEL / vel_magnitude
        #     x0[6:9] = x0[6:9] * scale_factor  # 直接赋值而不是 *=

        # store the optimal control and state
        optimal_control.append(u0)
        optimal_states.append(x0)

    # reference trajectory plot
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    reference_trajectory = np.array(reference_trajectory)
    optimal_states = np.array(optimal_states)
    ax.plot3D(reference_trajectory[:, 0], reference_trajectory[:, 1], reference_trajectory[:, 2],
              label='Reference Trajectory', color='blue')
    ax.plot3D(optimal_states[:, 0], optimal_states[:, 1], optimal_states[:, 2],
              label='Actual Trajectory', color='orange')
    ax.scatter(reference_trajectory[0, 0], reference_trajectory[0, 1], reference_trajectory[0, 2],
               color="green", s=100, label="Start")
    ax.text(reference_trajectory[0, 0], reference_trajectory[0, 1], reference_trajectory[0, 2] + 150,
            "Start Point", color="green", fontsize=12, weight='bold')
    ax.scatter(reference_trajectory[-1, 0], reference_trajectory[-1, 1], reference_trajectory[-1, 2],
               color="red", s=100, label="End")
    ax.text(reference_trajectory[-1, 0], reference_trajectory[-1, 1], reference_trajectory[-1, 2] + 150,
            "Target Point", color="red", fontsize=12, weight='bold')
    ax.set_xlabel('X Pos (m)')
    ax.set_ylabel('Y Pos (m)')
    ax.set_zlabel('Z Pos (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
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
    plt.title('2D Trajectory (XY Plane)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    run_simulation()
