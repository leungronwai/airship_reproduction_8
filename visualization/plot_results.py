"""
Simulation results visualization module
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class SimulationVisualizer:
    """
    Simulation results visualization class
    """

    def __init__(self, style="seaborn-v0_8-whitegrid"):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        self.figure_size_3d = (12, 8)
        self.figure_size_multi = (15, 12)
        self.figure_size_control = (12, 8)
        self.figure_size_disturbance = (12, 12)

    def plot_3d_trajectory(self, results, trajectory_type="spiral"):
        """
        Plot 3D trajectory tracking comparison

        Args:
            results: Simulation results dictionary
            trajectory_type: Type of trajectory
        """
        state_history = results['states']
        reference_history = results['references']

        fig = plt.figure(f"3D Trajectory Tracking - {trajectory_type}", figsize=self.figure_size_3d)
        ax3d = fig.add_subplot(111, projection="3d")

        # Plot trajectories
        ax3d.plot(state_history[0, :], state_history[1, :], state_history[2, :],
                  'b-', linewidth=2, label="Actual trajectory")
        ax3d.plot(reference_history[0, :], reference_history[1, :], reference_history[2, :],
                  'r--', linewidth=2, label="Reference trajectory")

        # Mark start and end points
        ax3d.scatter(state_history[0, 0], state_history[1, 0], state_history[2, 0],
                    c='green', s=100, marker='o', label="Start point")
        ax3d.scatter(state_history[0, -1], state_history[1, -1], state_history[2, -1],
                    c='red', s=100, marker='s', label="End point")

        # Labels and formatting
        ax3d.set_xlabel("X [m]")
        ax3d.set_ylabel("Y [m]")
        ax3d.set_zlabel("Z [m]")
        ax3d.set_title(f"3D Trajectory Tracking - {trajectory_type}")
        ax3d.legend()

        return fig

    def plot_tracking_errors(self, results):
        """
        Plot state tracking errors

        Args:
            results: Simulation results dictionary
        """
        sim_time = results['time']
        error_history = results['errors']

        fig, axs = plt.subplots(4, 3, sharex=True, figsize=self.figure_size_multi)
        fig.suptitle("State Tracking Errors")

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
                    axs[i, j].plot(sim_time, np.rad2deg(error_history[idx, :]), 'b-', linewidth=2)
                    axs[i, j].set_ylabel(f"{state_labels[i][j]} [deg]")
                else:
                    axs[i, j].plot(sim_time, error_history[idx, :], 'b-', linewidth=2)
                    unit = "[m]" if i == 0 else ("[m/s]" if i == 2 else "[rad/s]")
                    axs[i, j].set_ylabel(f"{state_labels[i][j]} {unit}")
                
                axs[i, j].set_title(state_labels[i][j])
                axs[i, j].grid(True, alpha=0.3)
        
        axs[3, 1].set_xlabel("Time [s]")
        
        return fig
    
    def plot_control_inputs(self, results):
        """
        Plot control inputs
        
        Args:
            results: Simulation results dictionary
        """
        sim_time = results['time']
        control_history = results['controls']
        
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figure_size_control)
        fig.suptitle("Control Inputs")
        
        control_labels = ["Thrust magnitude T [N]", "Horizontal deflection μ [deg]", "Vertical deflection ν [deg]"]
        control_data = [
            control_history[0, :],
            np.rad2deg(control_history[1, :]),
            np.rad2deg(control_history[2, :])
        ]
        
        for i in range(3):
            axs[i].plot(sim_time, control_data[i], 'g-', linewidth=2)
            axs[i].set_ylabel(control_labels[i])
            axs[i].set_title(control_labels[i])
            axs[i].grid(True, alpha=0.3)
        
        axs[2].set_xlabel("Time [s]")
        
        return fig
    
    def plot_disturbance_comparison(self, results):
        """
        Plot disturbance estimation vs actual disturbance
        
        Args:
            results: Simulation results dictionary
        """
        sim_time = results['time']
        disturbance_history = results['disturbances']
        disturbance_estimate_history = results['estimates']
        
        fig, axs = plt.subplots(6, 1, sharex=True, figsize=self.figure_size_disturbance)
        fig.suptitle("Disturbance Estimation vs Actual Disturbance")
        
        dist_labels = ["Force disturbance Fx", "Force disturbance Fy", "Force disturbance Fz",
                       "Moment disturbance Mx", "Moment disturbance My", "Moment disturbance Mz"]
        
        for i in range(6):
            axs[i].plot(sim_time, disturbance_history[i, :], 'k-',
                        linewidth=2, label=f"Actual {dist_labels[i]}")
            axs[i].plot(sim_time, disturbance_estimate_history[i, :], 'r--',
                        linewidth=2, label=f"Estimated {dist_labels[i]}")
            axs[i].set_ylabel(f"{dist_labels[i]}")
            axs[i].set_title(dist_labels[i])
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
        
        axs[5].set_xlabel("Time [s]")
        
        return fig
    
    def plot_all_results(self, results, trajectory_type="spiral", show_plots=True, save_plots=False, save_dir="plots"):
        """
        Plot all simulation results
        
        Args:
            results: Simulation results dictionary
            trajectory_type: Type of trajectory
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
            save_dir: Directory to save plots
        """
        logger.info("Generating simulation result plots...")
        
        figures = {}
        
        try:
            # Generate all plots
            figures['3d_trajectory'] = self.plot_3d_trajectory(results, trajectory_type)
            figures['tracking_errors'] = self.plot_tracking_errors(results)
            figures['control_inputs'] = self.plot_control_inputs(results)
            figures['disturbance_comparison'] = self.plot_disturbance_comparison(results)
            
            # Save plots if requested
            if save_plots:
                import os
                os.makedirs(save_dir, exist_ok=True)
                
                for name, fig in figures.items():
                    filename = f"{save_dir}/{name}_{trajectory_type}.png"
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    logger.info(f"Plot saved: {filename}")
            
            # Show plots if requested
            if show_plots:
                plt.tight_layout()
                plt.show()
            
            logger.info("Plot generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
        
        return figures


def plot_simulation_results(results, trajectory_type="spiral", **kwargs):
    """
    Convenient function to plot simulation results
    
    Args:
        results: Simulation results dictionary
        trajectory_type: Type of trajectory
        **kwargs: Additional arguments for SimulationVisualizer.plot_all_results()
    """
    visualizer = SimulationVisualizer()
    return visualizer.plot_all_results(results, trajectory_type, **kwargs)