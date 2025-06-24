'''
this file is used to test the do_mpc package

'''
# pylint: disable=invalid-name
# cspell:ignore 
# cspell:ignore arctan RUDT RUDB ELVL ELVR unmodeled
#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip

rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

#%%
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

#%%
# Certain parameters



#%%
# Uncertain parameters:



#%%
# States struct (optimization variables):




#%%
# Input struct (optimization variables):



#%%
# algebraic equations



#%%
# Differential equations



#%%
# Build the model
model.setup()



#%%
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 50.0/3600.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)



#%%
# Objective



#%%
# Constraints



#%%
# Initial conditions

#%%
#Scaling



#%%
#Uncertain values




#%%
mpc.setup()

#%%
# Estimator

estimator = do_mpc.estimator.StateFeedback(model)

#%%
#Simulator
simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 50.0/3600.0
}

simulator.set_param(**params_simulator)