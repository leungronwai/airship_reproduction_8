"""
测试推力模块
"""
# pylint: disable=invalid-name
# cspell:ignore thrust_to_force_torque F_desired

import numpy as np
from airship.thrust import thrust_params_to_force_torque, calculate_thrust_direction

def test_thrust_params_to_force_torque():
    """测试推力参数到力和力矩的转换"""
    thrust_params = [10, np.pi / 6, np.pi / 4]
    rp_r = np.array([1, 0, 0])
    rp_l = np.array([-1, 0, 0])
    thrust_to_force_torque = thrust_params_to_force_torque(thrust_params, rp_r, rp_l, use_casadi=False)
    assert thrust_to_force_torque.shape == (6,)

def test_calculate_thrust_direction():
    """测试推力方向计算"""
    F_desired = np.array([1, 1, 1])
    mu, nu = calculate_thrust_direction(F_desired)
    assert -np.pi / 2 <= mu <= np.pi / 2
    assert -np.pi / 2 <= nu <= np.pi / 2
