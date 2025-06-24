"""
controller.py - Contains implementation of AnyController and
NMPCThrustController classes for airship control.
"""
# pylint: disable=invalid-name
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff ndarray linalg vertcat xdot nlpsol
# cspell:ignore nlpsol IPOPT traj uref xref mtimes
import numpy as np
import casadi as ca


from config import parameters as params
from airship.model import AirshipCasADiSymbolic
from airship.thrust import thrust_params_to_force_torque, calculate_thrust_direction
from airship.observer import NMPCDisturbanceObserver




class AnyController:
    """
    Base class for airship controllers.
    """
    def __init__(self):
        # Original parameters
        self.params = params  # Ensure access to params parameters
        self.rp_r = params.rp_r
        self.rp_l = params.rp_l

        # Add PID controller parameters
        # Position PID parameters
        self.kp_pos = np.array([5.0, 5.0, 5.0])  # Position proportional gain
        self.ki_pos = np.array([0.1, 0.1, 0.1])  # Position integral gain
        self.kd_pos = np.array([2.0, 2.0, 2.0])  # Position derivative gain

        # Attitude PID parameters
        self.kp_att = np.array([3.0, 3.0, 3.0])  # Attitude proportional gain
        self.ki_att = np.array([0.05, 0.05, 0.05])  # Attitude integral gain
        self.kd_att = np.array([1.0, 1.0, 1.0])  # Attitude derivative gain

        # Integral errors and last errors (for derivative calculation)
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.last_att_error = np.zeros(3)
        self.last_time = 0.0  # Last update time

        # Integral limits to prevent saturation
        self.integral_limit_pos = np.array([50.0, 50.0, 50.0])
        self.integral_limit_att = np.array([1.0, 1.0, 1.0])

    def calculate_control(self, t, x, x_ref, x_ref_dot, u_thrust=None):
        """
        Calculate control input based on PID control

        Parameters:
            t: Current time
            x: Current state [position, attitude]
            x_ref: Reference state [position, attitude]
            x_ref_dot: Reference state derivative [position derivative, attitude derivative]
            u_thrust: Initial thrust parameters [T, μ, v], if None, calculated by controller

        Returns:
            tau: 6D control vector [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        # Extract current state and reference state
        pos = x[0:3]  # Current position
        att = x[3:6]  # Current attitude
        pos_ref = x_ref[0:3]  # Reference position
        att_ref = x_ref[3:6]  # Reference attitude

        # Calculate position and attitude errors
        pos_error = pos_ref - pos
        att_error = att_ref - att
        # Note that attitude error may need special handling, such as angle normalization
        att_error = np.array([(angle + np.pi) % (2 * np.pi) - np.pi for angle in att_error])

        # Calculate time increment
        dt = t - self.last_time
        if dt <= 0 or self.last_time == 0:
            dt = 0.01  # Default time step

        # Update integral terms (with limits)
        self.pos_error_integral += pos_error * dt
        self.att_error_integral += att_error * dt

        # Integral limits
        self.pos_error_integral = np.clip(
            self.pos_error_integral, -self.integral_limit_pos, self.integral_limit_pos
            )
        self.att_error_integral = np.clip(
            self.att_error_integral, -self.integral_limit_att, self.integral_limit_att
            )

        # Calculate derivative terms (can use reference trajectory derivatives as feedforward)
        pos_error_derivative = (pos_error - self.last_pos_error) / dt
        att_error_derivative = (att_error - self.last_att_error) / dt

        # Feedforward term - use reference trajectory derivatives
        pos_ref_dot = x_ref_dot[0:3]
        att_ref_dot = x_ref_dot[3:6]

        # Calculate PID control output (with feedforward)
        F_pos = (
            self.kp_pos * pos_error
            + self.ki_pos * self.pos_error_integral
            + self.kd_pos * (pos_error_derivative + 0.5 * pos_ref_dot)
            )

        T_att = (
            self.kp_att * att_error
            + self.ki_att * self.att_error_integral
            + self.kd_att * (att_error_derivative + 0.5 * att_ref_dot)
            )

        # Update last errors and time
        self.last_pos_error = pos_error.copy()
        self.last_att_error = att_error.copy()
        self.last_time = t

        # Convert PID control output to thrust parameters and control torque
        if u_thrust is not None:
            # Use provided initial thrust parameters
            tau = thrust_params_to_force_torque(u_thrust, self.rp_r, self.rp_l)
            # Add PID control correction
            tau[0:3] += F_pos
            tau[3:6] += T_att
        else:
            # Calculate suitable thrust parameters based on control needs
            T_mag = np.linalg.norm(F_pos)  # Thrust magnitude
            mu, nu = calculate_thrust_direction(F_pos)

            thrust_params = [T_mag, mu, nu]
            tau = thrust_params_to_force_torque(thrust_params, self.rp_r, self.rp_l)

            # Add attitude control torque
            tau[3:6] += T_att

        return tau

    def reset(self):
        """Reset controller state"""
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.last_att_error = np.zeros(3)
        self.last_time = 0.0  # Last update time


# === NMPC Controller Skeleton ===
class NMPCThrustController:
    """
    NMPC controller for airship using direct thrust allocation (T, μ, v).
    """

    def __init__(self, model, dt, N, Q, R, Qf, T_bounds, mu_bounds, nu_bounds, use_disturbance_compensation=True):
        """
        :param model:    Airship instance with a casadi-compatible dynamics (rhs)
        :param dt:       sampling time
        :param N:        prediction horizon steps
        :param Q, R, Qf: weight vectors or lists for state, input, terminal cost
        :param T_bounds: (T_min, T_max)
        :param mu_bounds:(mu_min, mu_max)
        :param nu_bounds:(nu_min, nu_max)
        :param use_disturbance_compensation: Whether to use disturbance compensation
        """
        self.model = model
        self.params = params
        self.dt = dt
        self.N = N
        # build cost matrices - Convert the NumPy matrix to a CasADi matrix
        self.Q = ca.DM(Q)  # State error weight
        self.R = ca.DM(R)  # Control input weight
        self.Qf = ca.DM(Qf) # Terminal state weight
        # actuator limits
        self.T_min, self.T_max = T_bounds
        self.mu_min, self.mu_max = mu_bounds
        self.nu_min, self.nu_max = nu_bounds

        # Save propeller position parameters for thrust calculation
        self.rp_r = self.params.rp_r
        self.rp_l = self.params.rp_l

        # symbolic variables
        X = ca.SX.sym("X", 12)
        U = ca.SX.sym("U", 3)

        # Build discrete time dynamics model
        f_cont = self._build_continuous_dynamics(X, U)

        # discrete dynamics via RK4
        h = dt
        k1 = f_cont(X, U)
        k2 = f_cont(X + 0.5 * h * k1, U)
        k3 = f_cont(X + 0.5 * h + k2, U)
        k4 = f_cont(X + h * k3, U)
        X_next = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_d = ca.Function("f_d", [X, U], [X_next])

        # Build the nonlinear programming (NLP) problem, used for the optimization solution of nonlinear model predictive control (NMPC)
        self._build_nlp(X, U)

        # Add the disturbance compensation option
        self.use_disturbance_compensation = use_disturbance_compensation
        self.disturbance_observer = NMPCDisturbanceObserver() if use_disturbance_compensation else None

        # Disturbance compensation factor (controls the disturbance compensation strength)
        self.disturbance_compensation_factor = params.do_compensation_gain if hasattr(params, 'do_compensation_gain') else 0.9

        # The last error and control input (for the observer)
        self.prev_e1 = np.zeros(6)
        self.prev_e2 = np.zeros(6)
        self.prev_tau = np.zeros(6)
        self.prev_gamma = np.zeros(3)
        self.last_disturbance_estimate = np.zeros(6)

        # Add attributes to store the complete control sequence
        self.last_optimal_sequence = None





    def _build_continuous_dynamics(self, X, U):
        """
        Build the continuous time dynamics model of the airship using AirshipCasADiSymbolic to construct the symbolic expression
        Used in subsequent steps to build the discrete time dynamics model (through numerical integration methods, such as RK4)

        Args:
            X: State variable symbolic expression
            U: Control input symbolic expression

        Returns:
            f_cont: State derivative symbolic expression

            Return the state derivative (dX/dt)

        """
        symbolic_model = AirshipCasADiSymbolic(self.params)
        # Generate the symbolic function f_cont(X, U) -> dX/dt
        f_cont = ca.Function("f_cont", [X, U], [symbolic_model.rhs_symbolic(X, U)])
        return f_cont

    def _build_nlp(self, X_sym, U_sym):
        """
        Args:
            X_sym: State variable symbolic expression
            U_sym: Control input symbolic expression

        Returns:
            nlp_prob: Nonlinear programming problem

        The core logic of _build_nlp:
           1. Optimization variables:
           Includes all states (x_k) and control inputs (u_k) in the prediction horizon.
           These variables are the decision variables of the optimization problem.

           2. Target function:
           Minimize the trajectory tracking error and control input cost.
           Adjust the importance of different items through the weight matrix (Q, R, Q_f).

           3. Constraints:
           Ensure that the optimized solution satisfies the system dynamics (through the discrete time model f_d).
           Ensure that the control input and state are within the physical limits.

           4. Solver:
           Use CasADi's nlpsol to create a solver (such as IPOPT) to solve the optimization problem.

        """

        N = self.N
        Q = self.Q
        R = self.R
        Qf = self.Qf

        # Decision variables
        w = []  # Optimization variables, including all states (x_k) and control inputs (u_k) in the prediction horizon
        g = []  # Constraints, including dynamic constraints and control input constraints
        J = 0  # Cost function, used to minimize the trajectory tracking error and control input cost. Adjust the importance of different items through the weight matrix Q, R, Q_f.

        # Initial state symbol
        Xk = ca.SX.sym("X0", 12)  # Initial state X0 12-dimensional
        w += [Xk]  # Add to optimization variables

        # Define the symbolic variables of the reference state and control input, used to pass the reference trajectory
        self.p_xref = ca.SX.sym("xref", (N + 1) * 12)  # Reference state
        self.p_uref = ca.SX.sym("uref", N * 3)  # Reference control input

        for k in range(N):  # Build the optimization problem in the prediction horizon
            # For each prediction step (k)

            # Define the control input and the next state
            Uk = ca.SX.sym(f"U_{k}", 3)  # Control input
            w += [Uk]  # Add to optimization variables

            # Next state Xk+1
            Xk_next = ca.SX.sym(f"X_{k+1}", 12)
            w += [Xk_next]

            # Add dynamic constraints: use the discrete time dynamics model f_d, ensure that the optimized solution satisfies the system dynamics
            Xk_pred = self.f_d(Xk, Uk)  # Predicted next state
            g += [Xk_next - Xk_pred]  # Dynamic constraints

            # Reference extraction
            x_ref_k = self.p_xref[12 * k : 12 * (k + 1)]  # Reference state
            u_ref_k = self.p_uref[3 * k : 3 * (k + 1)]  # Reference control input

            # Calculate the error and accumulate the target function
            e1 = Xk[0:6] - x_ref_k[0:6] # Position error and attitude error
            e2 = Xk[6:12] - x_ref_k[6:12] # Velocity error and angular velocity error

            # Accumulate the cost: ca.mtimes: used for matrix multiplication, calculate the weighted square sum of the error
            J += (ca.mtimes([e1.T, Q[0:6, 0:6], e1])
                  + ca.mtimes([e2.T, Q[6:12, 6:12], e2])
                  + ca.mtimes([(Uk - u_ref_k).T, R, (Uk - u_ref_k)])
            )

            Xk = Xk_next  # Roll update

        # At the last step in the prediction horizon, add the cost of the terminal state
        x_ref_final = self.p_xref[12 * N : 12 * (N + 1)]
        e_terminal = Xk - x_ref_final
        J += ca.mtimes([e_terminal.T, Qf, e_terminal])

        # Create the NLP solver
        # Pack the optimization variables, constraints, and target function into an NLP problem
        w_flat = ca.vertcat(*w)
        g_flat = ca.vertcat(*g)
        nlp_prob = {"f": J, "x": w_flat, "g": g_flat, "p": ca.vertcat(self.p_xref, self.p_uref)}

        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)  # Use CasADi's nlpsol to create a solver

        # Save the number of variables, used for the constraint setting in the subsequent step() method
        self.num_w = w_flat.size()[0]
        self.num_g = g_flat.size()[0]

    def step(self, x0, X_ref_traj, U_ref_traj, x_init=None, e1=None, e2=None):
        """
        Used to solve the nonlinear model predictive control NMPC optimization problem in each control cycle, calculate the optimal control input u_0 at the current time

        Args:
            x0: Current state [zeta, gamma, v, omega]
            X_ref_traj: Reference trajectory list [x_ref_0, x_ref_1, ..., x_ref_N]
            U_ref_traj: Reference control input list [u_ref_0, u_ref_1, ..., u_ref_{N-1}]
            x_init: Initial guess (optional)
            e1: Position/attitude error (optional, used for the disturbance observer)
            e2: Velocity error (optional, used for the disturbance observer)

        Returns:
            u0: Optimal control input [T, mu, nu]

        1. Input:
            Current state (x_0): The current state of the airship (such as position, attitude, velocity, etc.).
            Reference trajectory (X_ref): The reference state sequence in the prediction horizon.
            Reference control input (U_ref): The reference control input sequence in the prediction horizon.
            Initial guess (x_init) (optional): The initial value of the optimization variables, used to accelerate the solution.
        2. Output:
            Optimal control input (u_0) (such as the thrust magnitude and direction) at the current time.

        The core logic of the method:
            1. Optimization variables:
            (x_k): The state in the prediction horizon.
            (u_k): The control input in the prediction horizon.
            These variables are the decision variables of the optimization problem.

            2. Target function:
            Minimize the trajectory tracking error and control input cost.
            Adjust the importance of different items through the weight matrix (Q, R, Q_f).

            3. Constraints:
            Dynamic constraints: ensure that the optimized solution satisfies the system dynamics.
            Physical limits of control input and state.

            4. Solver:
            Use CasADi's nlpsol solver (such as IPOPT) to solve the optimization problem.

        Solve the NMPC problem given current state x0 and reference trajectories.
        Return the first control input [T, mu, nu].
        """
        N = self.N

        # Check if the input contains NaN
        if np.any(np.isnan(x0)):
            raise ValueError("x0 contains NaN values.")
        if np.any([np.any(np.isnan(x)) for x in X_ref_traj]):
            raise ValueError("X_ref_traj contains NaN values.")
        if np.any([np.any(np.isnan(u)) for u in U_ref_traj]):
            raise ValueError("U_ref_traj contains NaN values.")


        # --- Concatenate the reference trajectory parameter vector
        # Concatenate the reference state (X_ref) and reference control input (U_ref) into a vector, as the parameter of the NLP solver
        # These parameters will be passed to the NLP solver, used to calculate the target function and constraints.
        p_xref = np.concatenate(X_ref_traj).reshape((12 * (N + 1),))  # Concatenate the reference state vector
        p_uref = np.concatenate(U_ref_traj).reshape((3 * N,))  # Concatenate the reference control input vector

        # --- Initial guess ---
        # If the initial guess x_init is provided, use it, otherwise use the current state x0 and the reference trajectory
        # Initialize the optimization variables:
        #   If the initial guess (x_init) is provided, use it directly.
        #   Otherwise, initialize the optimization variables:
        #   (x_0): The current state.
        #   (u_k): The control input, initialized to zero.
        #   (x_{k+1}): The state, initialized to the reference state.
        if x_init is not None:
            w0 = x_init
        else:
            w0 = []
            w0 += [x0]
            for k in range(N):
                w0 += [np.zeros(3)]  # Uk = [T, mu, nu]
                w0 += [X_ref_traj[k + 1]]  # Xk+1
            w0 = np.concatenate(w0)

        # Check if w0 contains NaN or infinity
        if np.any(np.isnan(w0)):
            raise ValueError("w0 contains NaN values.")
        if np.any(np.isinf(w0)):
            raise ValueError("w0 contains infinite values.")



        # --- Control input constraints ---
        # Set the constraints of the optimization variables
        # State constraints:
        #       The first state (x_0) is fixed to the current state.
        #       The range of the other states (x_k) is set to a wide value (such as ([-1e5, 1e5])).
        # Control input constraints:
        #       The range of the thrust magnitude (T) and deflection angle (\mu, \nu) is set according to the physical limits.
        lbx = []
        ubx = []

        # Set the first state X0 to the current state
        lbx += list(x0) # Store the lower bounds of the optimization variables
        ubx += list(x0) # Store the upper bounds of the optimization variables

        for k in range(N):
            # Constraints of the control variable (input) Uk
            lbx += [self.T_min, self.mu_min, self.nu_min]
            ubx += [self.T_max, self.mu_max, self.nu_max]

            # Give a wide range (such as -1e5~1e5) to the state variable Xk+1
            lbx += [-1e5] * 12
            ubx += [1e5] * 12


        # Check if lbx or ubx contains NaN or infinity
        if np.any(np.isnan(lbx)) or np.any(np.isnan(ubx)):
            raise ValueError("lbx or ubx contains NaN values.")
        if np.any(np.isinf(lbx)) or np.any(np.isinf(ubx)):
            raise ValueError("lbx or ubx contains infinite values.")

        # --- Equality constraints (dynamics) ---
        # Dynamic constraints: ensure that the optimized solution satisfies the discrete time dynamics model (x_{k+1} = f_d(x_k, u_k))
        lbg = [0] * self.num_g  # Lower bound of the dynamic equality constraints
        ubg = [0] * self.num_g  # Upper bound of the dynamic equality constraints

        # --- Solver call / Call the NLP solver ---
        # Use CasADi's nlpsol solver to solve the NLP problem, get the optimal solution of the optimization variables
        w_opt = self.solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=np.concatenate([p_xref, p_uref]))

        # --- Extract the first control input U0 / Extract the optimal control input ---
        # Extract the first control input (u_0) from the optimization variables, that is, the optimal control input at the current time
        u0 = w_opt["x"].full().flatten()[12:15]  # The first control input U0 = [T, mu, nu] follows X0

        # Convert the thrust parameters to force and torque
        tau = self.thrust_to_force_torque(u0)

        # If the disturbance compensation is enabled and the error information is provided
        if self.use_disturbance_compensation and e1 is not None and e2 is not None:
            # Extract the current attitude
            gamma = x0[3:6]

            # Use the observer to update the disturbance estimate
            delta_hat = self.disturbance_observer.update(
                self.dt, e1, e2, self.prev_tau, gamma,
                f_func=None  # In NMPC, the f term is usually handled internally by the model
            )

            # Save the disturbance estimate
            self.last_disturbance_estimate = delta_hat

            # Apply the disturbance compensation to the control input
            # tau = tau - delta_hat * self.disturbance_compensation_factor

            # Save the current state for the next update
            self.prev_e1 = e1
            self.prev_e2 = e2
            self.prev_tau = tau
            self.prev_gamma = gamma

        # Save the entire optimal control sequence
        u_sequence = []
        for i in range(N):
            u_i = w_opt["x"].full().flatten()[(i+1)*12+i*3:(i+1)*12+(i+1)*3]
            u_sequence.append(u_i)
        self.last_optimal_sequence = u_sequence

        return u0


    def get_current_disturbance_estimate(self):
        """Get the current disturbance estimate
         If disturbance compensation is enabled, returns the disturbance value estimated by the disturbance observer
        Args:
            None

        Returns:
            disturbance_estimate: Current disturbance estimate
        """
        if self.use_disturbance_compensation:
            return np.array(self.last_disturbance_estimate).flatten()
        else:
            return np.zeros(6)


    def thrust_to_force_torque(self, u_thrust):
        """
        Convert thrust control [T, μ, v] to force and torque [Fx, Fy, Fz, Mx, My, Mz]

        Args:
            u_thrust: Thrust control input [T, μ, v]

        Returns:
            tau: 6-dimensional force and torque vector
        """
        # --- Control inputs ---

        # Ensure that the return is a 6-dimensional vector
        thrust_force_torque = thrust_params_to_force_torque(u_thrust, self.rp_r, self.rp_l, use_casadi=True)
        # Check the dimension of tau
        if isinstance(thrust_force_torque, np.ndarray) and thrust_force_torque.size != 6:
            print(f"Warning: Tau has a dimension of {thrust_force_torque.size}, expected to be 6")
            # If it is not 6-dimensional, supplement it to 6-dimensional
            if thrust_force_torque.size < 6:
                tau_corrected = np.zeros(6)
                tau_corrected[:thrust_force_torque.size] = thrust_force_torque
                return tau_corrected

        return thrust_force_torque
