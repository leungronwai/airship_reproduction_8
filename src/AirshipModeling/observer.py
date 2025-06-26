"""
observer.py
Fixed-time Disturbance Observer (DO) for Airship
Disturbance observer module, supporting disturbance compensation under the NMPC control strategy.
"""


# cspell:ignore R_block coeff casadi NMPC
# pylint: disable=invalid-name




import numpy as np
import casadi as ca
from config import parameters as params
from AirshipModeling.rotation_matrices import R_block



# Modified DisturbanceObserver class in airship/observer.py

class DisturbanceObserver:
    """
    Estimate the disturbance: by observing the state error of the airship (position error, velocity error, etc.), estimate the magnitude and direction of the external disturbance.
    Compensate the disturbance: feed the estimated disturbance value back to the controller, used to compensate the influence of the external disturbance on the airship trajectory tracking.
    Support NMPC controller: provide a symbolic disturbance observer equation for the prediction model of the NMPC controller.
    A disturbance observer specifically designed for the NMPC controller, using CasADi symbolic calculation
    """
    def __init__(self):
        # Basic observer parameters (obtained from the params module or using default values)
        self.l1 = params.l1 if hasattr(params, 'l1') else 2.0
        self.l2 = params.l2 if hasattr(params, 'l2') else 1.0
        self.l3 = params.l3 if hasattr(params, 'l3') else 1.5
        self.l4 = params.l4 if hasattr(params, 'l4') else 2.0
        self.l5 = params.l5 if hasattr(params, 'l5') else 1.0
        self.beta1 = params.beta1 if hasattr(params, 'beta1') else 0.5
        self.beta2 = params.beta2 if hasattr(params, 'beta2') else 1.5
        self.M = params.M_cfg
        self.M_inv = params.M_inv

        # Initialize the observer state
        self.z1_hat = np.zeros(6) # Estimation of position and velocity errors
        self.e2_hat = np.zeros(6) # Estimation of velocity errors
        self.delta_hat = np.zeros(6) # Estimation of disturbance

        # Parameters for disturbance filtering
        self.filter_coeff = params.do_filter_coeff if hasattr(params, 'do_filter_coeff') else 0.7
        self.prev_delta_hat = np.zeros(6)

        # Disturbance compensation gain
        self.compensation_gain = params.do_compensation_gain if hasattr(params, 'do_compensation_gain') else 0.9

        # History
        self.history = []

        # Create a CasADi symbolic function version, used for NMPC prediction
        self._create_symbolic_observer()

    def _create_symbolic_observer(self):
        """Create a CasADi symbolic version of the observer equation, used for the prediction model of the NMPC controller
        Note: This function does not return a value because it constructs a CasADi symbolic function
        observer_update_func, and binds it to the class variable self.observer_update_func,
        for other methods (such as update()) to call.
        
        Args:
            None
        Returns:
            None
        """
        # Define symbolic variables
        # __import__() is Python's underlying function for dynamic module import, equivalent to import casadi as ca
        # ca = __import__("casadi") # Import casadi library
        e1_sym = ca.SX.sym("e1", 6) # Position/attitude error vector
        e2_sym = ca.SX.sym("e2", 6) # Velocity and angular velocity error
        tau_sym = ca.SX.sym("tau", 6) # Control input (force and torque)
        gamma_sym = ca.SX.sym("gamma", 3) # Attitude angle (Euler angle)
        dt_sym = ca.SX.sym("dt", 1) # Time step
        z1_hat_sym = ca.SX.sym("z1_hat", 6) # Internal state of the observer
        e2_hat_sym = ca.SX.sym("e2_hat", 6) # Internal state of the observer

        # Build the observer equation
        R_sym = R_block(gamma_sym)  # Assume R_block already supports CasADi symbolic
        RM_inv_sym = R_sym @ self.M_inv # Combination of rotation matrix and mass matrix, used to convert the control input to the rate of change of velocity error

        # e2_hat update
        e2_hat_dot_sym = -self.l1 * e2_hat_sym + RM_inv_sym @ tau_sym
        e2_hat_next_sym = e2_hat_sym + e2_hat_dot_sym * dt_sym

        # z1 and z2 calculation
        z1_sym = e2_sym - e2_hat_sym
        z2_sym = self.l2 * z1_sym

        # z1_hat update (full version, including nonlinear terms)
        # Define the symbolic version of the sig function
        def sig_sym(x, alpha):
            """
            Symbolic version of the sig function, used to replace np.sign(x) * (np.abs(x) ** alpha)
            sig(x, alpha) = sign(x) * |x|^alpha
            Compute signed power function, used to construct nonlinear terms (such as sliding mode control, etc.)
            
            Args:
                x: Input variable
                alpha: Power exponent

            Returns:
                Signed power result
            """
            return ca.sign(x) * (ca.fabs(x) ** alpha)  # np.pow(ca.fabs(x), alpha)

        # z1_hat update
        z1_hat_dot_sym = (-self.l1 * z1_hat_sym + z1_sym + self.l3 * z2_sym +
                          self.l4 * sig_sym(z1_hat_sym, self.beta1) +
                          self.l5 * sig_sym(z1_hat_sym, self.beta2))
        z1_hat_next_sym = z1_hat_sym + z1_hat_dot_sym * dt_sym

        # Calculate the disturbance estimate
        delta_star_hat_sym = (z2_sym + self.l1 * self.l2 * z1_hat_sym) / self.l2
        # f_term is usually handled internally by the controller in the NMPC model
        f_term_sym = ca.SX.zeros(6)  # Simplified processing, provided by the controller

        # Calculate the disturbance estimate
        delta_hat_raw_sym = self.M @ ca.transpose(R_sym) @ (delta_star_hat_sym - self.l1 * e2_sym - f_term_sym)

        # Create the update function, generate the symbolic function
        self.observer_update_func = ca.Function(
            'observer_update',
            [e1_sym, e2_sym, tau_sym, gamma_sym, dt_sym, z1_hat_sym, e2_hat_sym],
            [z1_hat_next_sym, e2_hat_next_sym, delta_hat_raw_sym],
            ['e1', 'e2', 'tau', 'gamma', 'dt', 'z1_hat', 'e2_hat'],
            ['z1_hat_next', 'e2_hat_next', 'delta_hat_raw']
        )

    def update(self, dt, e1, e2, tau, gamma, f_func=None):
        """
        Update the disturbance estimate

        Parameters:
            dt: Time step
            e1: Position/attitude error vector
            e2: Velocity/angular velocity error vector
            tau: Control input (force and torque)
            gamma: Attitude angle (Euler angle)
            f_func: Function to calculate f(e1,e2) (optional)

        Returns:
            delta_hat_compensated: Disturbance vector that can be directly used for compensation
        """
        # Update the internal state
        z1_hat_next, e2_hat_next, delta_hat_raw = self.observer_update_func(
            e1, e2, tau, gamma, dt, self.z1_hat, self.e2_hat
        )

        self.z1_hat = z1_hat_next
        self.e2_hat = e2_hat_next

        # If f_func is provided, use it to calculate f_term
        if f_func is not None:
            # Calculate the rotation matrix
            R = R_block(gamma)
            z1 = e2 - self.e2_hat
            z2 = self.l2 * z1
            delta_star_hat = (z2 + self.l1 * self.l2 * self.z1_hat) / self.l2

            # Calculate the f(e1, e2) term
            f_term = f_func(e1, e2)

            # Recalculate the disturbance estimate (considering f_term)
            delta_hat_raw = self.M @ R.T @ (delta_star_hat - self.l1 * e2 - f_term)

        # Apply low-pass filter
        self.delta_hat = self.filter_coeff * self.prev_delta_hat + (1 - self.filter_coeff) * delta_hat_raw
        self.prev_delta_hat = self.delta_hat

        # Apply compensation gain / Disturbance estimate that can be directly used for compensation
        delta_hat_compensated = self.delta_hat * self.compensation_gain

        # Record history
        self.history.append({
            'delta_hat_raw': np.array(delta_hat_raw).flatten(),
            'delta_hat_filtered': self.delta_hat.full().flatten(),
            'delta_hat_compensated': delta_hat_compensated.full().flatten()
        })

        return delta_hat_compensated

    def get_current_disturbance_estimate(self):
        """Return the current disturbance estimate

        Returns:
            Current disturbance estimate
        """
        return self.delta_hat

    def get_compensated_estimate(self):
        """Return the disturbance estimate considering the compensation gain
        """
        return self.delta_hat * self.compensation_gain

    def reset(self):
        """Reset the observer state
        Used to reset the state variables and history of the observer
        """
        self.z1_hat = np.zeros(6)
        self.e2_hat = np.zeros(6)
        self.delta_hat = np.zeros(6)
        self.prev_delta_hat = np.zeros(6)
        self.history = []
