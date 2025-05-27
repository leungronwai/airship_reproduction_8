"""
测试控制器
"""
# pylint: disable=invalid-name
# cspell:ignore traj
# cspell:ignore arctan

import numpy as np
from airship.controller import NMPCThrustController
from airship.model import Airship
from config import parameters as params

def test_controller_step():
    """测试控制器的 step 方法"""
    airship = Airship(params.X0)
    controller = NMPCThrustController(
        model=airship,
        dt=params.DT,
        N=params.N_HORIZON,
        Q=params.Q,
        R=params.R,
        Qf=params.Qf,
        T_bounds=(params.T_MIN, params.T_MAX),
        mu_bounds=(params.MU_MIN, params.MU_MAX),
        nu_bounds=(params.NU_MIN, params.NU_MAX),
    )
    x0 = params.X0
    X_ref_traj = [x0] * (params.N_HORIZON + 1)
    U_ref_traj = [np.zeros(3)] * params.N_HORIZON
    u0 = controller.step(x0, X_ref_traj, U_ref_traj)
    assert len(u0) == 3