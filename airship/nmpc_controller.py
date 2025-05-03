# nmpc_controller.py
import numpy as np
import casadi as ca
from config import parameters as params
from airship.utils import R_block, R_zeta, R_y_inv

class AirshipNMPC:
    def __init__(self):
        # 控制器参数
        self.N = 10  # 预测时域长度
        self.dt = 0.1  # 控制周期
        
        # 权重矩阵
        self.Q_pos = np.diag([10.0, 10.0, 10.0])  # 位置误差权重
        self.Q_att = np.diag([5.0, 5.0, 5.0])     # 姿态误差权重
        self.Q_vel = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 速度误差权重
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])      # 控制输入权重
        
        # 控制输入约束
        self.tau_max = np.array([1000, 1000, 1000, 500, 500, 500])  # 最大控制输入
        self.tau_min = -self.tau_max                                # 最小控制输入
        
        # 初始化CasADi优化器
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """设置CasADi优化器"""
        # 状态变量
        self.x = ca.SX.sym('x', 12)  # 完整状态 [zeta, gamma, v, omega]
        
        # 控制输入
        self.u = ca.SX.sym('u', 6)   # 控制输入 tau
        
        # 系统动力学
        self.f = self._get_dynamics_model()
        
        # 设置优化问题
        self._setup_optimization_problem()
        
    def _get_dynamics_model(self):
        """获取系统动力学模型（简化版）"""
        x = self.x
        u = self.u
        
        # 提取状态分量
        zeta = x[0:3]
        gamma = x[3:6]
        v_omega = x[6:12]
        
        # 简化的动力学模型（实际应用中需要完整实现）
        # 这里我们使用简化版，实际应用中应该使用完整的气艇动力学模型
        R = self._casadi_R_block(gamma)
        
        # 运动学方程
        y_dot = R @ v_omega
        
        # 动力学方程（简化版）
        # 实际应用中，应该实现完整的动力学方程，包括重力、浮力、气动力等
        M_inv = ca.DM(params.M_inv)
        x_dot = M_inv @ u  # 简化版，实际中需要考虑 F_term, N_term 等
        
        # 组合状态导数
        dxdt = ca.vertcat(y_dot, x_dot)
        
        # 返回连续时间动力学模型
        return dxdt
    
    def _casadi_R_block(self, gamma):
        """CasADi版本的R_block函数"""
        phi, theta, psi = gamma[0], gamma[1], gamma[2]
        
        # 计算R_zeta (简化版)
        cphi, sphi = ca.cos(phi), ca.sin(phi)
        cth, sth = ca.cos(theta), ca.sin(theta)
        cpsi, spsi = ca.cos(psi), ca.sin(psi)
        
        R_zeta = ca.SX(3, 3)
        R_zeta[0, 0] = cth * cpsi
        R_zeta[0, 1] = sphi * sth * cpsi - cphi * spsi
        R_zeta[0, 2] = cphi * sth * cpsi + sphi * spsi
        R_zeta[1, 0] = cth * spsi
        R_zeta[1, 1] = sphi * sth * spsi + cphi * cpsi
        R_zeta[1, 2] = cphi * sth * spsi - sphi * cpsi
        R_zeta[2, 0] = -sth
        R_zeta[2, 1] = sphi * cth
        R_zeta[2, 2] = cphi * cth
        
        # 计算R_y (简化版)
        R_y = ca.SX(3, 3)
        R_y[0, 0] = 1
        R_y[0, 1] = sphi * sth / cth
        R_y[0, 2] = cphi * sth / cth
        R_y[1, 0] = 0
        R_y[1, 1] = cphi
        R_y[1, 2] = -sphi
        R_y[2, 0] = 0
        R_y[2, 1] = sphi / cth
        R_y[2, 2] = cphi / cth
        
        # 构建块对角矩阵
        R_block = ca.blockcat([[R_zeta, ca.SX.zeros(3, 3)], 
                              [ca.SX.zeros(3, 3), R_y]])
        
        return R_block
    
    def _setup_optimization_problem(self):
        """设置优化问题"""
        # 状态变量
        X = ca.SX.sym('X', 12, self.N+1)
        
        # 控制变量
        U = ca.SX.sym('U', 6, self.N)
        
        # 参考轨迹
        Y_ref = ca.SX.sym('Y_ref', 6, self.N+1)  # 位置和姿态参考
        Y_dot_ref = ca.SX.sym('Y_dot_ref', 6, self.N+1)  # 速度参考
        
        # 初始状态
        X0 = ca.SX.sym('X0', 12)
        
        # 代价函数
        obj = 0
        g = []  # 约束
        
        # 初始状态约束
        g.append(X[:, 0] - X0)
        
        # 动力学约束和代价函数
        for k in range(self.N):
            # 当前状态和控制
            x_k = X[:, k]
            u_k = U[:, k]
            
            # 下一个状态
            x_k1 = X[:, k+1]
            
            # 使用RK4积分
            k1 = self.f(x_k, u_k)
            k2 = self.f(x_k + self.dt/2 * k1, u_k)
            k3 = self.f(x_k + self.dt/2 * k2, u_k)
            k4 = self.f(x_k + self.dt * k3, u_k)
            x_next = x_k + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # 添加动力学约束
            g.append(x_k1 - x_next)
            
            # 提取当前状态的位置、姿态和速度
            zeta_k = x_k[0:3]
            gamma_k = x_k[3:6]
            v_omega_k = x_k[6:12]
            
            # 提取参考轨迹
            y_ref_k = Y_ref[:, k]
            y_dot_ref_k = Y_dot_ref[:, k]
            
            # 计算当前输出
            y_k = ca.vertcat(zeta_k, gamma_k)
            R_k = self._casadi_R_block(gamma_k)
            y_dot_k = R_k @ v_omega_k
            
            # 计算误差
            e1 = y_k - y_ref_k
            e2 = y_dot_k - y_dot_ref_k
            
            # 添加代价函数
            obj += e1[0:3].T @ self.Q_pos @ e1[0:3]  # 位置误差
            obj += e1[3:6].T @ self.Q_att @ e1[3:6]  # 姿态误差
            obj += e2.T @ self.Q_vel @ e2            # 速度误差
            obj += u_k.T @ self.R @ u_k              # 控制输入
        
        # 最终状态代价
        zeta_N = X[0:3, self.N]
        gamma_N = X[3:6, self.N]
        v_omega_N = X[6:12, self.N]
        
        y_N = ca.vertcat(zeta_N, gamma_N)
        R_N = self._casadi_R_block(gamma_N)
        y_dot_N = R_N @ v_omega_N
        
        y_ref_N = Y_ref[:, self.N]
        y_dot_ref_N = Y_dot_ref[:, self.N]
        
        e1_N = y_N - y_ref_N
        e2_N = y_dot_N - y_dot_ref_N
        
        obj += 10 * e1_N[0:3].T @ self.Q_pos @ e1_N[0:3]  # 终端位置误差（权重更大）
        obj += 10 * e1_N[3:6].T @ self.Q_att @ e1_N[3:6]  # 终端姿态误差（权重更大）
        
        # 控制输入约束
        for k in range(self.N):
            g.append(U[:, k] - self.tau_max)
            g.append(-U[:, k] - self.tau_min)
        
        # 创建优化问题
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        # 问题参数
        p = ca.vertcat(X0, ca.reshape(Y_ref, -1, 1), ca.reshape(Y_dot_ref, -1, 1))
        
        # 设置NLP问题
        nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': p}
        
        # 求解器选项
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        
        # 创建求解器
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # 保存问题维度
        self.nx = 12
        self.nu = 6
        self.n_states = self.nx * (self.N + 1)
        self.n_controls = self.nu * self.N
        
    def calculate_control(self, t, current_state, trajectory):
        """计算控制输入"""
        # 获取当前状态
        x0 = current_state
        
        # 获取参考轨迹
        y_ref = np.zeros((6, self.N+1))
        y_dot_ref = np.zeros((6, self.N+1))
        x_ref = np.zeros((6, self.N+1))
        x_dot_ref = np.zeros((6, self.N+1))
        
        for i in range(self.N+1):
            t_pred = t + i * self.dt
            yc, yc_dot, _, xc, xc_dot = trajectory.get_desired_state(t_pred)
            y_ref[:, i] = yc
            y_dot_ref[:, i] = yc_dot
            x_ref[:, i] = xc
            x_dot_ref[:, i] = xc_dot
        
        # 设置初始猜测
        x_init = np.zeros(self.n_states)
        for i in range(self.N+1):
            x_init[i*self.nx:(i+1)*self.nx] = x0
        
        u_init = np.zeros(self.n_controls)
        
        # 设置约束上下界
        lbg = np.zeros(self.nx + 2*self.nu*self.N)
        ubg = np.zeros(self.nx + 2*self.nu*self.N)
        
        # 初始状态约束
        lbg[0:self.nx] = 0
        ubg[0:self.nx] = 0
        
        # 控制输入约束
        for i in range(self.N):
            lbg[self.nx+2*i*self.nu:self.nx+(2*i+1)*self.nu] = -np.inf
            ubg[self.nx+2*i*self.nu:self.nx+(2*i+1)*self.nu] = 0
            
            lbg[self.nx+(2*i+1)*self.nu:self.nx+(2*i+2)*self.nu] = -np.inf
            ubg[self.nx+(2*i+1)*self.nu:self.nx+(2*i+2)*self.nu] = 0
        
        # 设置参数
        p = np.concatenate([x0, y_ref.flatten(), y_dot_ref.flatten()])
        
        # 求解NLP
        sol = self.solver(
            x0=np.concatenate([x_init, u_init]),
            lbg=lbg,
            ubg=ubg,
            p=p
        )
        
        # 提取最优控制输入
        u_opt = sol['x'][self.n_states:self.n_states+self.nu]
        
        return u_opt