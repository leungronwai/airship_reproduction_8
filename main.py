"""
# main.py

"""
# pylint: disable=invalid-name
# cspell:ignore dompc levelname figsize traj

import sys
import os
import logging

from datetime import datetime

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
        print("Using NMPC Controller with Spiral Trajectory")

        #
        trajectory_type = "spiral"
        logger.info("Selected trajectory type: %s", trajectory_type)

        logger.info("Running do-mpc NMPC controller simulation")
        run_dompc_simulation(
            trajectory_type=trajectory_type,
            use_disturbance_compensation=True,
            use_simulator=True
        )

    except KeyboardInterrupt:
        logger.info("User interrupted the program")
    finally:
        logger.info("=== Simulation program ended ===")





if __name__ == "__main__":
    main()
