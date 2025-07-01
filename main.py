"""
# main.py
Airship Trajectory Tracking Simulation
"""
# pylint: disable=invalid-name
# cspell:disable symvar_type


import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import do_mpc

from do_mpc.tools import Timer



from src.config import parameters as params

from src.system.controller_dompc import DoMpcConfig



def run_simulation():
    """
    Run the simulation

    """

    # user settings
    show_animations = False  # Set to True to show animations
    store_results = False

    print("Generating reference trajectory...")
    subprocess.run([sys.executable, "-m", "src.tests.test_desired_trajectory", "--type", "spiral"])

    # setting up the model
    airship_mpc = DoMpcConfig()

    model = airship_mpc.model

    # setting up a mpc controller, given the model
    # mpc = airship_mpc.create_mpc_controller(model)
    mpc = airship_mpc.create_mpc_controller(silence_solver=True)


    # setting up a simulator, given the model
    simulator = airship_mpc.create_simulator()


    # setting up an estimator, given the model
    estimator = do_mpc.estimator.StateFeedback(model)


    # Set the initial state of mpc and simulator
    x0 = params.X0.copy().reshape(-1, 1)

    # pushing initial condition to mpc and the simulator
    mpc.x0 = x0
    simulator.x0 = x0


    # setting up initial guesses
    mpc.set_initial_guess()



    # simulation of the plant
    timer = Timer()
    optimal_control = []
    optimal_states = []
    optimal_states.append(x0)

    # 添加软启动参数
    # ramp_up_time = 5.0  # 软启动时间（秒）


    for i in range(int(params.T_SPAN / params.DT)):
        current_time = i * params.DT

        # 获取当前参考轨迹信息
        yc_ref, yc_dot_ref, _, _, _ = airship_mpc.trajectory.get_spiral_trajectory(current_time)
        ref_vel_x, ref_vel_y, ref_vel_z = yc_dot_ref[0:3]  # 提取参考速度分量
        ref_vel_magnitude = np.linalg.norm(yc_dot_ref[0:3])  # 计算参考速度大小

        # 软启动因子
        # if current_time < ramp_up_time:
        #     ramp_factor = current_time / ramp_up_time
        # else:
        #     ramp_factor = 1.0


        # for the current state x0, mpc computes the optimal control action u0
        # print(f"Time: {current_time:.2f}s, Ramp factor: {ramp_factor:.2f}")
        timer.tic()
        u0 = mpc.make_step(x0)
        timer.toc()

        # 应用软启动到控制输入
        # if current_time < ramp_up_time:
        #     # 在软启动阶段逐渐增加控制力度
        #     u0_smooth = u0.copy()
        #     u0_smooth[0] = u0[0] * ramp_factor + params.T_HOVER * (1 - ramp_factor)
        #     u0_smooth[1] *= ramp_factor
        #     u0_smooth[2] *= ramp_factor
        #     u0 = u0_smooth

        # for the current state u0, computes the next state y_next
        y_next = simulator.make_step(u0)

        # for the current state y_next, estimates the next state x0
        x0 = estimator.make_step(y_next)

        # 修复 NumPy 弃用警告
        vel_x = float(x0[6].item())  # 使用 .item() 方法
        vel_y = float(x0[7].item())
        vel_z = float(x0[8].item())
        vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)  # 计算实际速度大小

        # 打印参考速度和实际速度
        print(f"\n=== Time: {current_time:.1f}s ===")
        print(f"Reference velocity: [{ref_vel_x:.2f}, {ref_vel_y:.2f}, {ref_vel_z:.2f}] m/s")
        print(f"Reference |v|: {ref_vel_magnitude:.2f} m/s")
        print(f"Actual velocity: [{vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f}] m/s")
        print(f"Actual |v|: {vel_magnitude:.2f} m/s")

        # 强制限制实际速度
        MAX_VEL = 20.0
        if vel_magnitude > MAX_VEL:
            print(f"强制限制速度：{vel_magnitude:.2f} → {MAX_VEL} m/s")
            scale_factor = MAX_VEL / vel_magnitude
            x0[6:9] = x0[6:9] * scale_factor  # 直接赋值而不是 *=

        # store the optimal control and state
        optimal_control.append(u0)
        optimal_states.append(x0)


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









