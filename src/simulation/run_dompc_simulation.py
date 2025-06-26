"""
Airship simulation script using do-mpc Simulator
Demonstrates how to use the complete do-mpc ecosystem for airship trajectory tracking
"""

# pylint: disable=invalid-name
# cspell:ignore linalg suptitle sharex sharey whitegrid
# cspell: ignore dompc levelname figsize set_xlabel set_ylabel set_zlabel


import time as timer
import logging

import numpy as np




from config import parameters as params
from AirshipModeling.trajectory_ref import Trajectory
from AirshipModeling.controller_dompc import do_mpc_controller
from visualization.plot_results import plot_simulation_results
from analysis.performance_evaluator import evaluate_performance





# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_dompc_simulation(trajectory_type="spiral", use_disturbance_compensation=True, use_simulator=True):
    """
    Args:
        trajectory_type: spiral
        use_disturbance_compensation: Whether to use disturbance compensation
        use_simulator: Must be True for this implementation

    Returns:
        dict: Simulation result data
    """
    if not use_simulator:
        raise ValueError("This implementation requires use_simulator=True")

    logger.info("Starting do-mpc Simulator NMPC simulation - Trajectory type: %s", trajectory_type)
    start_time = timer.time()

    # === Initialize components ===
    trajectory = Trajectory()

    # Create do-mpc based controller
    controller = do_mpc_controller(
        use_disturbance_compensation=use_disturbance_compensation,
        create_simulator=True
    )

    # Verify simulator was created
    if controller.simulator is None:
        raise RuntimeError("Failed to create do-mpc Simulator")

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

    # Initialize do-mpc Simulator with proper initial state
    try:
        # Set do-mpc Simulator initial state
        x0_dict = controller.mpc.x0
        x0_dict['pos'] = current_state[0:3].reshape(-1, 1)
        x0_dict['att'] = current_state[3:6].reshape(-1, 1)
        x0_dict['vel'] = current_state[6:9].reshape(-1, 1)
        x0_dict['omega'] = current_state[9:12].reshape(-1, 1)

        # Set simulator initial state
        controller.simulator.x0 = controller.mpc.x0
        logger.info("do-mpc Simulator initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize do-mpc Simulator: {e}") # pylint: disable=raise-missing-from

    # === Main simulation loop ===
    for i, t in enumerate(sim_time):
        # Get reference trajectory
        # yc: position (zeta_d): [x, y, z] + attitude (gamma_d): [phi, theta, psi]
        # yc_dot: velocity (zeta_d_dot): [vx, vy, vz] + angular velocity (gamma_d_dot): [p, q, r]
        yc, yc_dot = _get_reference_trajectory(trajectory, trajectory_type, t)
        reference_trajectory = _convert_trajectory_format(yc, yc_dot) # 12x1

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

        # === Use do-mpc Simulator for state update ===
        try:
            # Update simulator parameters for current time step
            _update_simulator_parameters(controller.simulator, reference_trajectory, t)

            # Execute simulator step
            x_next = controller.simulator.make_step(u_cmd.reshape(-1, 1))

            # Extract next state
            if hasattr(x_next, 'full'):
                current_state = x_next.full().flatten()
            else:
                current_state = np.array(x_next).flatten()

        except Exception as e:
            logger.error("do-mpc Simulator step failed: %s", e)
            raise RuntimeError(f"Simulator failed at time {t:.2f}s: {e}") # pylint: disable=raise-missing-from

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

    # Plot results and evaluate performance
    plot_simulation_results(results, trajectory_type, show_plots=True, save_plots=True)
    evaluate_performance(results)

    return results


def _update_simulator_parameters(simulator, reference_trajectory, t):
    """
    Update simulator parameters for current time step

    Args:
        simulator: do-mpc simulator instance
        reference_trajectory: Current reference trajectory
        t: Current time
    """
    try:
        # Get parameter template
        p_template = simulator.get_p_template()

        # Set reference trajectory parameters
        p_template['pos_ref'] = reference_trajectory['position'].reshape(-1, 1)
        p_template['att_ref'] = reference_trajectory['attitude'].reshape(-1, 1)
        p_template['vel_ref'] = reference_trajectory['velocity'].reshape(-1, 1)
        p_template['omega_ref'] = reference_trajectory['angular_velocity'].reshape(-1, 1)

        # Set disturbance
        disturbance = params.disturbance_delta(t)
        p_template['disturbance'] = disturbance.reshape(-1, 1)

        # Update simulator parameters
        simulator.set_p_fun(lambda t_now: p_template)

    except Exception as e: # pylint: disable=broad-exception-caught
        logger.warning("Failed to update simulator parameters: %s", e)




def _get_reference_trajectory(trajectory, trajectory_type, t):
    """Get reference trajectory"""
    if trajectory_type == "spiral":
        yc, yc_dot, _, _, _ = trajectory.get_spiral_trajectory(t)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    return yc, yc_dot

def _convert_trajectory_format(yc, yc_dot):
    """
    Convert trajectory format to do-mpc controller required format

    Args:
        yc: Reference state [position (3) + attitude (3)]
        yc_dot: Reference state derivatives [position derivatives (3) + attitude derivatives (3)]

    Returns:
        dict: Formatted reference trajectory
    """
    return {
        'position': yc[0:3],
        'attitude': yc[3:6],
        'velocity': yc_dot[0:3],
        'angular_velocity': yc_dot[3:6]
    }





if __name__ == "__main__":



    selected_trajectory = "spiral"

    # Run simulation
    simulation_results = run_dompc_simulation(
        trajectory_type=selected_trajectory,
        use_disturbance_compensation=True
    )

    print(f"Simulation completed! Trajectory type: {selected_trajectory}")
    print("Results saved and charts displayed")
