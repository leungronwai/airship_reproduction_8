"""
Airship simulation script using do-mpc Simulator
Demonstrates how to use the complete do-mpc ecosystem for airship trajectory tracking
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

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_dompc_simulation(trajectory_type="linear", use_disturbance_compensation=True, use_simulator=True):
    """
    Run do-mpc based NMPC simulation

    Args:
        trajectory_type: Trajectory type
        use_disturbance_compensation: Whether to use disturbance compensation
        use_simulator: Whether to use do-mpc Simulator (True) or only MPC controller (False)

    Returns:
        dict: Simulation result data
    """
    logger.info("Starting do-mpc Simulator NMPC simulation - Trajectory type: %s", trajectory_type)
    start_time = timer.time()

    # === Initialize components ===
    trajectory = Trajectory()

    # Create do-mpc based controller (including Simulator)
    controller = DoMPCAirshipController(
        use_disturbance_compensation=use_disturbance_compensation,
        create_simulator=use_simulator
    )

    # === Simulation setup ===
    sim_time = np.arange(0, params.T_SPAN, params.DT)
    n_steps = len(sim_time)

    # Data recording arrays
    state_history = np.zeros((12, n_steps))
    control_history = np.zeros((3, n_steps))
    reference_history = np.zeros((12, n_steps))
    disturbance_history = np.zeros((6, n_steps))
    disturbance_estimate_history = np.zeros((6, n_steps))
    error_history = np.zeros((12, n_steps))
    prediction_history = []

    # Set initial state
    current_state = params.X0.copy()

    # Check if simulator was created successfully
    if use_simulator and controller.simulator is not None:
        try:
            # Set do-mpc Simulator initial state
            x0_dict = controller.mpc.x0
            x0_dict['pos'] = current_state[0:3].reshape(-1, 1)
            x0_dict['att'] = current_state[3:6].reshape(-1, 1)
            x0_dict['vel'] = current_state[6:9].reshape(-1, 1)
            x0_dict['omega'] = current_state[9:12].reshape(-1, 1)

            # Use correct method to set simulator initial state
            controller.simulator.x0 = controller.mpc.x0
            logger.info("Using do-mpc Simulator for simulation")
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to set Simulator initial state: %s, using numerical integration", e)
            use_simulator = False
    else:
        logger.info("Using numerical integration for simulation")
        use_simulator = False

    # === Main simulation loop ===
    for i, t in enumerate(sim_time):
        # Get reference trajectory
        yc, yc_dot = _get_reference_trajectory(trajectory, trajectory_type, t)
        reference_trajectory = convert_trajectory_format(yc, yc_dot)

        # Record reference trajectory
        reference_state = np.concatenate([yc, yc_dot])
        reference_history[:, i] = reference_state

        # Calculate error
        error = current_state - reference_state
        error_history[:, i] = error

        # Use MPC controller to calculate control input
        u_cmd = controller.step(current_state, reference_trajectory, t)

        # Record control input
        control_history[:, i] = u_cmd

        # Get disturbance estimate
        delta_hat = controller.get_current_disturbance_estimate()
        disturbance_estimate_history[:, i] = delta_hat

        # Get actual disturbance
        actual_delta = params.disturbance_delta(t)
        disturbance_history[:, i] = actual_delta

        # Record current state
        state_history[:, i] = current_state

        # Choose state update method based on mode
        if use_simulator and controller.simulator is not None:
            # Use do-mpc Simulator - direct call without wrapper function
            try:
                x_next = controller.simulator.make_step(u_cmd.reshape(-1, 1))
                if hasattr(x_next, 'full'):
                    current_state = x_next.full().flatten()
                else:
                    current_state = np.array(x_next).flatten()
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error("Simulator step failed: %s", e)
                current_state = _fallback_integration(current_state, u_cmd, actual_delta, params.DT)
        elif use_simulator:
            # Error handling for Simulator not properly initialized
            logger.error("Simulator requested but not properly initialized, falling back to numerical integration")
            current_state = _fallback_integration(current_state, u_cmd, actual_delta, params.DT)
        else:
            # Use traditional numerical integration
            current_state = _fallback_integration(current_state, u_cmd, actual_delta, params.DT)

        # Get MPC prediction (optional)
        if i % 5 == 0:  # Record prediction every 5 steps to reduce storage
            prediction = controller.get_prediction()
            if prediction['states'] is not None:
                prediction_history.append({
                    'time': t,
                    'prediction': prediction
                })

        # Display progress
        if i % (n_steps // 10) == 0:
            logger.info("Simulation progress: %d / %d ≈ %.1f%%", i, n_steps, 100 * i / n_steps)
            logger.debug("Position error norm: %.4f", np.linalg.norm(error[0:3]))
            logger.debug("Attitude error norm: %.4f", np.linalg.norm(error[3:6]))

    end_time = timer.time()
    logger.info("Simulation completed, elapsed time: %.2f seconds", end_time - start_time)

    # === Result analysis and plotting ===
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
    """Get reference trajectory"""
    if trajectory_type == "linear":
        yc, yc_dot, _, _, _ = trajectory.get_linear_trajectory(t)
    elif trajectory_type == "spiral":
        yc, yc_dot, _, _, _ = trajectory.get_spiral_trajectory(t)
    elif trajectory_type == "figure8":
        yc, yc_dot, _, _, _ = trajectory.get_figure8_trajectory(t)
    elif trajectory_type == "lemniscate":
        yc, yc_dot, _, _, _ = trajectory.get_lemniscate_trajectory(t)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    return yc, yc_dot


def _fallback_integration(current_state, u_cmd, disturbance, dt):
    """
    Improved fallback integration method
    Uses simplified but more stable dynamics model
    """


    # Use simplified symbolic model for integration
    try:
        symbolic_model = AirshipCasADiSymbolic(params)

        def dynamics_func(t, x):
            _ = t
            return symbolic_model.rhs_symbolic(x, u_cmd, external_disturbance=disturbance)

        # Use RK4 integration
        next_state = rk4_step(dynamics_func, 0, current_state, dt)

        # Angle normalization
        next_state[3:6] = (next_state[3:6] + np.pi) % (2 * np.pi) - np.pi

        return next_state

    except Exception as e: # pylint: disable=broad-exception-caught
        logger.warning("Fallback integration failed, using simplest integration: %s", e)

        # Simplest integration as final fallback
        next_state = current_state.copy()
        next_state[0:3] += current_state[6:9] * dt  # Position update
        return next_state


def _plot_simulation_results(results, trajectory_type):
    """Plot simulation results"""
    sim_time = results['time']
    state_history = results['states']
    control_history = results['controls']
    reference_history = results['references']
    error_history = results['errors']
    disturbance_history = results['disturbances']
    disturbance_estimate_history = results['estimates']

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: 3D trajectory tracking
    fig1 = plt.figure(f"do-mpc Simulator 3D Trajectory Tracking - {trajectory_type}", figsize=(12, 8))
    ax3d = fig1.add_subplot(111, projection="3d")

    ax3d.plot(state_history[0, :], state_history[1, :], state_history[2, :],
              'b-', linewidth=2, label="Actual trajectory")
    ax3d.plot(reference_history[0, :], reference_history[1, :], reference_history[2, :],
              'r--', linewidth=2, label="Reference trajectory")

    ax3d.scatter(state_history[0, 0], state_history[1, 0], state_history[2, 0],
                c='green', s=100, marker='o', label="Start point")
    ax3d.scatter(state_history[0, -1], state_history[1, -1], state_history[2, -1],
                c='red', s=100, marker='s', label="End point")

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"do-mpc Simulator 3D Trajectory Tracking - {trajectory_type}")
    ax3d.legend()

    # Figure 2: State tracking errors
    fig2, axs2 = plt.subplots(4, 3, sharex=True, figsize=(15, 12))
    fig2.suptitle("State Tracking Errors")

    state_labels = [
        ["Position X", "Position Y", "Position Z"],
        ["Attitude φ", "Attitude θ", "Attitude ψ"],
        ["Velocity u", "Velocity v", "Velocity w"],
        ["Angular velocity p", "Angular velocity q", "Angular velocity r"]
    ]

    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            if i == 1:  # Convert attitude angles to degrees
                axs2[i, j].plot(sim_time, np.rad2deg(error_history[idx, :]), 'b-', linewidth=2)
                axs2[i, j].set_ylabel(f"{state_labels[i][j]} [deg]")
            else:
                axs2[i, j].plot(sim_time, error_history[idx, :], 'b-', linewidth=2)
                unit = "[m]" if i == 0 else ("[m/s]" if i == 2 else "[rad/s]")
                axs2[i, j].set_ylabel(f"{state_labels[i][j]} {unit}")

            axs2[i, j].set_title(state_labels[i][j])
            axs2[i, j].grid(True, alpha=0.3)

    axs2[3, 1].set_xlabel("Time [s]")

    # Figure 3: Control inputs
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig3.suptitle("Control Inputs")

    control_labels = ["Thrust magnitude T [N]", "Horizontal deflection μ [deg]", "Vertical deflection ν [deg]"]
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

    axs3[2].set_xlabel("Time [s]")

    # Figure 4: Disturbance estimation comparison
    fig4, axs4 = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    fig4.suptitle("Disturbance Estimation vs Actual Disturbance")

    dist_labels = ["Force disturbance Fx", "Force disturbance Fy", "Force disturbance Fz",
                   "Moment disturbance Mx", "Moment disturbance My", "Moment disturbance Mz"]

    for i in range(6):
        axs4[i].plot(sim_time, disturbance_history[i, :], 'k-',
                    linewidth=2, label=f"Actual {dist_labels[i]}")
        axs4[i].plot(sim_time, disturbance_estimate_history[i, :], 'r--',
                    linewidth=2, label=f"Estimated {dist_labels[i]}")
        axs4[i].set_ylabel(f"{dist_labels[i]}")
        axs4[i].set_title(dist_labels[i])
        axs4[i].legend()
        axs4[i].grid(True, alpha=0.3)

    axs4[5].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


def _evaluate_performance(results):
    """Evaluate control performance"""
    logger.info("=== do-mpc Simulator Control Performance Evaluation ===")

    error_history = results['errors']
    control_history = results['controls']
    sim_time = results['time']

    # Position error statistics
    pos_errors = error_history[0:3, :]
    pos_rmse = np.sqrt(np.mean(pos_errors**2, axis=1))
    pos_max = np.max(np.abs(pos_errors), axis=1)

    logger.info("Position RMSE: X=%.3fm, Y=%.3fm, Z=%.3fm", pos_rmse[0], pos_rmse[1], pos_rmse[2])
    logger.info("Position max error: X=%.3fm, Y=%.3fm, Z=%.3fm", pos_max[0], pos_max[1], pos_max[2])

    # Attitude error statistics
    att_errors = error_history[3:6, :]
    att_rmse = np.sqrt(np.mean(att_errors**2, axis=1))
    att_max = np.max(np.abs(att_errors), axis=1)

    logger.info("Attitude RMSE: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                np.rad2deg(att_rmse[0]), np.rad2deg(att_rmse[1]), np.rad2deg(att_rmse[2]))
    logger.info("Attitude max error: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                np.rad2deg(att_max[0]), np.rad2deg(att_max[1]), np.rad2deg(att_max[2]))

    # Control input statistics
    control_mean = np.mean(control_history, axis=1)
    control_std = np.std(control_history, axis=1)

    logger.info("Control input mean: T=%.3fN, μ=%.3f°, ν=%.3f°",
                control_mean[0], np.rad2deg(control_mean[1]), np.rad2deg(control_mean[2]))
    logger.info("Control input standard deviation: T=%.3fN, μ=%.3f°, ν=%.3f°",
                control_std[0], np.rad2deg(control_std[1]), np.rad2deg(control_std[2]))

    # Steady-state error analysis
    steady_start = int(0.8 * len(sim_time))
    pos_steady = np.mean(np.abs(pos_errors[:, steady_start:]), axis=1)
    att_steady = np.mean(np.abs(att_errors[:, steady_start:]), axis=1)

    logger.info("Steady-state position error: X=%.3fm, Y=%.3fm, Z=%.3fm",
                pos_steady[0], pos_steady[1], pos_steady[2])
    logger.info("Steady-state attitude error: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                np.rad2deg(att_steady[0]), np.rad2deg(att_steady[1]), np.rad2deg(att_steady[2]))


if __name__ == "__main__":
    print("=== do-mpc Simulator Airship Simulation ===")

    # Can choose different trajectory types for testing
    trajectory_types = ["linear", "spiral", "figure8", "lemniscate"]

    # Select trajectory type
    selected_trajectory = "linear"

    # Run simulation
    simulation_results = run_dompc_simulation(
        trajectory_type=selected_trajectory,
        use_disturbance_compensation=True
    )

    print(f"Simulation completed! Trajectory type: {selected_trajectory}")
    print("Results saved and charts displayed")
