"""
# main.py

"""
# pylint: disable=invalid-name
# cspell:ignore dompc levelname figsize traj

import sys
import os
import logging

from datetime import datetime
from simulation.run_simulation import run_simulation
from simulation.run_dompc_simulation import run_dompc_simulation



# Add project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)




def setup_logger():
    """
    Global logging configuration: Configure once here, and other modules in the project
    can retrieve the same logger.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"simulation_{timestamp}.log")

    # Configure log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            # logging.FileHandler(log_filename, encoding='utf-8'),  # File handler - save to file
            logging.StreamHandler(sys.stdout)  # Console handler
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info("Log file: %s", log_filename)
    return logger


def main():
    """Main program entry point"""
    logger = setup_logger()

    try:
        logger.info("=== Airship Trajectory Tracking Simulation Started ===")


        print("\n=== Airship Simulation Configuration ===")

        # ========== Modify your choices here ==========
        choice = "2"        # "1" = PID Controller, "2" = do-mpc NMPC Controller
        traj_choice = "2"   # "1" = Linear, "2" = Spiral, "3" = Figure-8, "4" = Lemniscate
        # =====================================

        print(f"Selected controller: {'PID Controller' if choice == '1' else 'do-mpc NMPC Controller'}")

        # Trajectory mapping
        trajectory_map = {
            "1": "linear",
            "2": "spiral",
            "3": "figure8",
            "4": "lemniscate"
        }

        trajectory_type = trajectory_map.get(traj_choice, "linear")
        print(f"Selected trajectory type: {trajectory_type}")
        logger.info("Selected trajectory type: %s", trajectory_type)

        # Run corresponding simulation based on selection
        if choice == "1":
            logger.info("Running traditional PID controller simulation")
            run_simulation(trajectory_type=trajectory_type)

        elif choice == "2":
            logger.info("Running do-mpc NMPC controller simulation")
            run_dompc_simulation(
                trajectory_type=trajectory_type,
                use_disturbance_compensation=True,
                use_simulator=False  # Only use MPC controller
            )

        else:
            logger.warning("Invalid selection, running default do-mpc NMPC controller")
            run_dompc_simulation(
                trajectory_type="linear",
                use_disturbance_compensation=True,
                use_simulator=False
            )

    except KeyboardInterrupt:
        logger.info("User interrupted the program")
    finally:
        logger.info("=== Simulation program ended ===")


def _run_comparison_simulation(trajectory_type):
    """Run controller comparison simulation"""
    logger = logging.getLogger(__name__)

    print(f"\n=== Starting comparison simulation (trajectory: {trajectory_type}) ===")

    results = {}

    # 1. Run traditional PID controller
    try:
        print("1/2 Running traditional PID controller...")
        logger.info("Starting PID controller simulation")
        run_simulation(trajectory_type=trajectory_type)
        results['PID'] = "Success"
        print("PID controller simulation completed")
    except KeyboardInterrupt:
        logger.info("User interrupted the program")
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Program execution error: %s", e)
    except Exception as e:     # pylint: disable=broad-except
        logger.exception("Unhandled exception: %s", e)

    # 2. Run do-mpc NMPC controller
    try:
        print("2/2 Running do-mpc NMPC controller...")
        logger.info("Starting do-mpc NMPC controller simulation")
        run_dompc_simulation(
            trajectory_type=trajectory_type,
            use_disturbance_compensation=True
        )
        results['do-mpc NMPC'] = "Success"
        print("do-mpc NMPC controller simulation completed")
    except Exception as e:      # pylint: disable=broad-except
        logger.error("do-mpc NMPC controller simulation failed: %s", e)
        results['do-mpc NMPC'] = f"Failed: {e}"
        print("do-mpc NMPC controller simulation failed")

    # Summary of results
    print("\n=== Comparison simulation results summary ===")
    for controller, result in results.items():
        print(f"{controller}: {result}")

    logger.info("Comparison simulation completed, results: %s", results)


if __name__ == "__main__":
    main()
