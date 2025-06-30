"""
# main.py
Airship Trajectory Tracking Simulation
"""
# pylint: disable=invalid-name
# cspell:disable symvar_type


import os
import sys
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

    Args:
        None

    Returns:
        None
    """
    # user settings
    show_animations = False  # Set to True to show animations
    store_results = False

    # setting up the model
    airship_mpc = DoMpcConfig()

    model = airship_mpc.model

    # setting up a mpc controller, given the model
    mpc = airship_mpc.create_mpc_controller(model)


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


    for i in range(int(params.T_SPAN / params.DT)):

        # for the current state x0, mpc computes the optimal control action u0
        print(f"Time: {i * params.DT:.2f}s")
        timer.tic()
        u0 = mpc.make_step(x0)
        timer.toc()


        # for the current state u0, computes the next state y_next
        y_next = simulator.make_step(u0)

        # for the current state y_next, estimates the next state x0
        x0 = estimator.make_step(y_next)

        # store the optimal control and state
        optimal_control.append(u0)
        optimal_states.append(x0)


    # make plots
    optimal_control = np.array(optimal_control)
    plt.plot(optimal_control[:, 0], label='Delta')
    plt.plot(optimal_control[:, 1], label='Acc')
    plt.legend()
    plt.show()

    optimal_states = np.array(optimal_states)
    plt.plot(optimal_states[:, 0], label='X_p')
    plt.plot(optimal_states[:, 1], label='Y_p')
    plt.plot(optimal_states[:, 2], label='Psi')
    plt.plot(optimal_states[:, 3], label='V')

    plt.legend()
    plt.show()

    plt.plot(optimal_states[:, 0], optimal_states[:, 1])
    plt.show()



if __name__ == "__main__":
    run_simulation()









