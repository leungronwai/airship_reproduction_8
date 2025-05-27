"""
    测试气艇模型的初始化

    Returns:
        _type_: _description_
"""
# tests/test_model.py
# pylint: disable=invalid-name
# cspell:ignore ndarray xdot allclose
# pylint: disable=unnecessary-lambda

import numpy as np
import pytest
from airship.model import Airship, AirshipCasADiSymbolic
from config import parameters as params

class TestAirshipModel:
    """
    测试气艇模型的初始化
    """
    @pytest.fixture
    def airship(self):
        """创建一个气艇模型实例用于测试"""
        initial_state = np.zeros(12)  # [zeta, gamma, v, omega]
        return Airship(initial_state)

    def test_initialization(self, airship):
        """测试气艇模型的初始化"""
        # 检查状态向量初始化
        assert airship.X.shape == (12,)
        assert np.all(airship.X == 0)

        # 检查关键参数是否正确加载
        assert airship.m == params.m
        assert airship.g == params.g
        assert np.array_equal(airship.I0, params.I0)
        assert np.array_equal(airship.M, params.M_cfg)

    def test_rhs_dimensions(self, airship):
        """测试动力学方程右端的维度"""
        t = 0.0
        X = np.zeros(12)
        tau_test = np.array([0.0, 0.0, 0.0])
        def disturbance_test(t):
            return np.zeros(6)

        dXdt = airship.rhs(t, X, tau_test, disturbance_test)
        assert dXdt.shape == (12,)

    def test_gravity_force(self, airship):
        """测试重力计算"""
        t = 0.0
        X = np.zeros(12)  # 水平姿态
        tau = np.array([0.0, 0.0, 0.0])
        disturbance = lambda t: np.zeros(6)

        dXdt = airship.rhs(t, X, tau, disturbance)
        # 在水平姿态下，重力应该沿着体坐标系的 z 轴向下
        assert dXdt[8] > 0  # z 方向加速度应该为正（向下）

    def test_buoyancy_force(self, airship):
        """测试浮力计算"""
        t = 0.0
        X = np.zeros(12)  # 水平姿态
        tau = np.array([0.0, 0.0, 0.0])
        disturbance = lambda t: np.zeros(6)

        dXdt = airship.rhs(t, X, tau, disturbance)
        # 浮力应该部分抵消重力
        assert abs(dXdt[8]) < airship.m * airship.g  # z 方向的净加速度应该小于重力加速度

    def test_thrust_input(self, airship):
        """测试推力输入的影响"""
        t = 0.0
        X = np.zeros(12)
        tau = np.array([10.0, 0.0, 0.0])
        disturbance = lambda t: np.zeros(6)

        dXdt = airship.rhs(t, X, tau, disturbance)
        assert abs(dXdt[6]) > 0  # x 方向速度应该增加

    def test_wind_effect(self, airship):
        """测试风的影响"""
        t = 0.0
        X = np.zeros(12)
        tau = np.array([0.0, 0.0, 0.0])
        disturbance = lambda t: np.zeros(6)

        # 设置一个非零的风速
        original_wind = airship.V_wind_erf_const.copy()
        airship.V_wind_erf_const = np.array([5.0, 0.0, 0.0])  # 5m/s的东风

        dXdt_with_wind = airship.rhs(t, X, tau, disturbance)

        # 恢复原始风速
        airship.V_wind_erf_const = original_wind
        dXdt_no_wind = airship.rhs(t, X, tau, disturbance)

        # 检查风是否影响了气动力
        assert not np.array_equal(dXdt_with_wind, dXdt_no_wind)

    def test_state_update(self, airship):
        """测试状态更新"""
        dt = 0.01
        X_dot = np.ones(12)
        initial_state = airship.X.copy()

        airship.update_state(X_dot, dt)
        assert np.allclose(airship.X, initial_state + X_dot * dt, rtol=1e-10)

    def test_get_methods(self, airship):
        """测试获取状态的方法"""
        assert np.array_equal(airship.get_state(), airship.X)
        assert np.array_equal(airship.get_pose(), airship.X[0:6])
        assert np.array_equal(airship.get_velocity(), airship.X[6:12])

class TestAirshipCasADiSymbolic:
    """
    测试符号模型
    """
    @pytest.fixture
    def symbolic_model(self):
        """创建一个符号模型实例"""
        return AirshipCasADiSymbolic(params)

    def test_symbolic_model_initialization(self, symbolic_model):
        """测试符号模型的初始化"""
        assert symbolic_model.m == params.m
        assert symbolic_model.g == params.g
        assert np.array_equal(symbolic_model.I0, params.I0)

    def test_get_nmpc_model(self, symbolic_model):
        """测试 NMPC 模型生成"""
        f = symbolic_model.get_nmpc_model()
        assert f.name() == "f"
        assert f.n_in() == 2  # x 和 u 两个输入
        assert f.n_out() == 1  # xdot 一个输出

    def test_discrete_time_model(self, symbolic_model):
        """测试离散时间模型"""
        dt = 0.01
        F = symbolic_model.discrete_time_model(dt, integration_method="rk4")
        assert F.name() == "F"
        assert F.n_in() == 2  # x 和 u 两个输入
        assert F.n_out() == 1  # x_next 一个输出

# pytest tests/test_model.py -v
