"""
Trajectory generation module (trajectory.py)
"""
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# cspell:ignore R_zeta R_y_inv Rc_z Rc_y_inv ddot arctan2 linalg xdot phiddot phidot    psiddot
# cspell:ignore phidot phiddot psidot psiddot thetaddot ydot

import numpy as np
from airship.utils import R_zeta, R_y_inv


class Trajectory:
    """
    Trajectory generation module


        Parameters:
            t: Current time
            start_point: Starting point coordinates [x, y, z], default is origin
            end_point: End point coordinates [x, y, z], default is [5000, 5000, -19000]
            speed: Flight speed in m/s
            hover_at_end: Whether to hover after reaching the end point, otherwise continue flying straight

        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot: Desired states and derivatives


        Description:
            - yc:

                Represents the desired state vector, containing the desired position and attitude of the airship.
                Specifically includes:
                    Position: [x, y, z], the desired position of the airship in space.
                    Attitude: [φ, θ, ψ], the desired attitude angles of the airship (roll, pitch, yaw).
            - yc_dot:

                Represents the first derivative of the desired state, i.e., the desired velocity vector.
                Specifically includes:
                    Linear velocity: [vx, vy, vz], the desired linear velocity of the airship in space.
                    Angular velocity: [ωφ, ωθ, ωψ], the desired angular velocity of the airship.
            - yc_ddot:

                Represents the second derivative of the desired state, i.e., the desired acceleration vector.
                Specifically includes:
                    Linear acceleration: [ax, ay, az], the desired linear acceleration of the airship in space.
                    Angular acceleration: [αφ, αθ, αψ], the desired angular acceleration of the airship.
            - xc:

                Represents the control command vector, containing desired linear and angular velocities.
                Specifically includes:
                    Linear velocity command: [vx, vy, vz], the desired linear velocity of the airship.
                    Angular velocity command: [ωφ, ωθ, ωψ], the desired angular velocity of the airship.
            - xc_dot:

                Represents the first derivative of the control command, i.e., the rate of change of control commands.
                Specifically includes:
                    Linear velocity rate: [dvx/dt, dvy/dt, dvz/dt], the time rate of change of linear velocity.
                    Angular velocity rate: [dωφ/dt, dωθ/dt, dωψ/dt], the time rate of change of angular velocity.
    """
    def __init__(self):
        pass  # No specific initialization needed for this trajectory

    # ┌─────────────────────────────────────────────────────┐
    # │          Spiral trajectory function                  │
    # └─────────────────────────────────────────────────────┘

    def get_spiral_trajectory(self, t):

        """
        Generate a spiral trajectory with altitude variation
        
        Args:
            t: Current time
        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot
        """

        dt_small = 1e-4

        # --- Trajectory parameters ---
        omega = 0.07  # Angular velocity
        r = 1500  # Radius
        h_max = 2000  # Maximum height

        # Print starting point information at initial time
        if abs(t) < 1e-3:  # When t approaches 0
            start_x = r * np.cos(0)  # = r = 1500
            start_y = r * np.sin(0)  # = 0
            start_z = h_max * (1 - np.exp(0))  # = 0
            print(f"[Spiral Trajectory] Starting point position: [{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (meters)")
            print(f"[Spiral Trajectory] Trajectory parameters: radius={r}m, max height={h_max}m, angular velocity={omega}rad/s")

        # --- Directly calculate position, velocity and acceleration ---
        theta = omega * t
        # Position
        xd = r * np.cos(theta)
        yd = r * np.sin(theta)
        zd = h_max * (1 - np.exp(-theta / 10))
        pos = np.array([xd, yd, zd])

        # Velocity
        xd_dot = -r * omega * np.sin(theta)
        yd_dot = r * omega * np.cos(theta)
        zd_dot = h_max * (1 / 10) * np.exp(-theta / 10) * omega
        vel = np.array([xd_dot, yd_dot, zd_dot])

        # Acceleration
        xd_ddot = -r * omega**2 * np.cos(theta)
        yd_ddot = -r * omega**2 * np.sin(theta)
        zd_ddot = -h_max * (1 / 10) * omega**2 * np.exp(-theta / 10)
        acc = np.array([xd_ddot, yd_ddot, zd_ddot])

        # Construct position and velocity vectors
        zeta_d = pos
        zeta_d_dot = vel
        zeta_d_ddot = acc

        # Calculate attitude
        phi_d = 0.0  # Maintain zero roll
        theta_d = np.arctan2(-vel[2], np.sqrt(vel[0] ** 2 + vel[1] ** 2))  # Pitch angle
        psi_d = np.arctan2(vel[1], vel[0])  # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # Use numerical differentiation to get attitude derivatives
        _, gamma_d_plus = self.get_spiral_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_spiral_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # Combine yc, yc_dot
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # Velocity commands vc, wc
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc))

        # xc_dot simplified approximation through symbolic derivatives
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # Simplified processing, assuming small angular velocity change rate
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # yc_ddot simplified processing
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_spiral_pos_att(self, t):
        """
        Calculate the position and attitude of the spiral trajectory at time t, used for derivative calculation
        Avoid recursive calls to define_spiral_trajectory
        """
        # --- Trajectory parameters ---
        omega = 0.07  # Angular velocity (rad/s)
        r = 1500  # Basic radius (m)
        h_max = 2000  # Maximum height (m)

        # --- Position calculation ---
        theta = omega * t
        xd = r * np.cos(theta)
        yd = r * np.sin(theta)
        zd = h_max * (1 - np.exp(-theta / 10))
        zeta_d = np.array([xd, yd, zd])

        # --- Velocity calculation (for attitude determination) ---
        xd_dot = -r * omega * np.sin(theta)
        yd_dot = r * omega * np.cos(theta)
        zd_dot = h_max * (1 / 10) * np.exp(-theta / 10) * omega

        # --- Attitude calculation ---
        phi_d = 0.0  # Maintain zero roll
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))  # Pitch angle
        psi_d = np.arctan2(yd_dot, xd_dot)  # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ********************* Figure-8 trajectory function *********************

    def get_figure8_trajectory(self, t):
        """
        Generate a horizontal figure-8 trajectory with smooth altitude variation
        Return the desired state and derivatives of the figure-8 trajectory        
        Parameters:
            t: Current time

        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot: Same output format as get_desired_state
        """
        dt_small = 1e-4

        # --- Trajectory parameters ---
        a = 3000  # Width of figure-8
        b = 2000  # Height of figure-8
        omega = 0.003  # Angular velocity, controls the speed of movement along the trajectory
        h_center = -19000  # Center altitude
        h_amp = 500  # Altitude oscillation amplitude
        omega_h = 0.002  # Angular velocity of altitude variation

        # Print starting point information at initial time
        if abs(t) < 1e-3:  # When t approaches 0
            start_x = a * np.sin(0)  # = 0
            start_y = b * np.sin(0) * np.cos(0)  # = 0
            start_z = h_center + h_amp * np.sin(0)  # = h_center = -19000
            print(f"[Figure-8 Trajectory] Starting point position: [{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (meters)")
            print(f"[Figure-8 Trajectory] Trajectory parameters: width={a}m, height={b}m, center altitude={h_center}m, angular velocity={omega}rad/s")

        # --- Position calculation ---
        # Parametric equations for figure-8
        xd = a * np.sin(omega * t)
        yd = b * np.sin(omega * t) * np.cos(omega * t)
        zd = h_center + h_amp * np.sin(omega_h * t)
        zeta_d = np.array([xd, yd, zd])

        # --- Velocity calculation ---
        xd_dot = a * omega * np.cos(omega * t)
        yd_dot = b * omega * (np.cos(omega * t) * np.cos(omega * t)
                              - np.sin(omega * t) * np.sin(omega * t)
                              )
        zd_dot = h_amp * omega_h * np.cos(omega_h * t)
        zeta_d_dot = np.array([xd_dot, yd_dot, zd_dot])

        # --- Acceleration calculation (using numerical differentiation) ---
        # Calculate velocity at t+dt time
        xd_dot_plus = a * omega * np.cos(omega * (t + dt_small))
        yd_dot_plus = (
            b * omega * (np.cos(omega * (t + dt_small)) * np.cos(omega * (t + dt_small))
                         - np.sin(omega * (t + dt_small)) * np.sin(omega * (t + dt_small))
                         )
        )
        zd_dot_plus = h_amp * omega_h * np.cos(omega_h * (t + dt_small))

        # Calculate velocity at t-dt time
        xd_dot_minus = a * omega * np.cos(omega * (t - dt_small))
        yd_dot_minus = (
            b * omega * (np.cos(omega * (t - dt_small)) * np.cos(omega * (t - dt_small))
               - np.sin(omega * (t - dt_small)) * np.sin(omega * (t - dt_small)))
        )
        zd_dot_minus = h_amp * omega_h * np.cos(omega_h * (t - dt_small))

        # Calculate acceleration using central difference
        zeta_d_ddot = np.array(
            [
                (xd_dot_plus - xd_dot_minus) / (2 * dt_small),
                (yd_dot_plus - yd_dot_minus) / (2 * dt_small),
                (zd_dot_plus - zd_dot_minus) / (2 * dt_small),
            ]
        )

        # --- Attitude calculation ---
        # Calculate desired heading angle (tangent direction)
        phi_d = 0.0  # Maintain zero roll
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))  # Pitch angle
        psi_d = np.arctan2(yd_dot, xd_dot)  # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- Attitude derivative calculation ---
        # Use numerical differentiation to obtain attitude derivatives
        _, gamma_d_plus = self.get_figure8_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_figure8_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # --- Combine yc, yc_dot ---
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # --- Velocity commands vc, wc ---
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc))

        # --- xc_dot simplified approximation using symbolic derivatives ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # Simplified processing, assuming small angular velocity rate changes
        xc_dot = np.concatenate((vc_dot, wc_dot))

        # --- yc_ddot also simplified processing ---
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_figure8_pos_att(self, t):
        """
        Only calculate the position and attitude of the figure-8 trajectory at time t for derivative calculation
        Avoid recursive calls to get_figure8_trajectory
        """
        # --- Trajectory parameters ---
        a = 3000  # Width of the figure-8
        b = 2000  # Height of the figure-8
        omega = 0.003  # Angular velocity, controls airship movement speed on trajectory
        h_center = -19000  # Center altitude
        h_amp = 500  # Altitude oscillation amplitude
        omega_h = 0.002  # Angular velocity for altitude variation

        # --- Position ---
        xd = a * np.sin(omega * t)
        yd = b * np.sin(omega * t) * np.cos(omega * t)
        zd = h_center + h_amp * np.sin(omega_h * t)
        zeta_d = np.array([xd, yd, zd])

        # --- Velocity (for attitude calculation) ---
        xd_dot = a * omega * np.cos(omega * t)
        yd_dot = b * omega * (np.cos(omega * t) * np.cos(omega * t)
                              - np.sin(omega * t) * np.sin(omega * t)
                              )
        zd_dot = h_amp * omega_h * np.cos(omega_h * t)

        # --- Attitude ---
        phi_d = 0.0
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))
        psi_d = np.arctan2(yd_dot, xd_dot)
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ********************* Lemniscate trajectory function *********************

    def get_lemniscate_trajectory(self, t):
        """
        Generate Lemniscate trajectory, resembling infinity symbol, with altitude variation

        Parameters:
            t: Current time

        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot: Desired states and their derivatives
        """
        dt_small = 1e-4

        # --- Trajectory parameters ---
        a = 2500  # Curve scale parameter
        omega = 0.004  # Angular velocity
        h_center = -19000  # Center altitude
        h_amp = 800  # Altitude variation amplitude
        h_freq = 0.001  # Altitude variation frequency

        # Print starting point information at initial time
        if abs(t) < 1e-3:  # When t is close to 0
            theta = 0
            denom = 1 + np.sin(theta) ** 2  # = 1
            start_x = a * np.cos(theta) / denom  # = a = 2500
            start_y = a * np.sin(theta) * np.cos(theta) / denom  # = 0
            start_z = h_center + h_amp * np.sin(0)  # = h_center = -19000
            print(f"[Lemniscate Trajectory] Starting point position: [{start_x:.1f}, {start_y:.1f}, {start_z:.1f}] (meters)")
            print(f"[Lemniscate Trajectory] Trajectory parameters: scale={a}m, center altitude={h_center}m, angular velocity={omega}rad/s")

        # --- Parametric curve parameters ---
        theta = omega * t
        # Lemniscate parametric equations
        denom = 1 + np.sin(theta) ** 2
        xd = a * np.cos(theta) / denom
        yd = a * np.sin(theta) * np.cos(theta) / denom
        zd = h_center + h_amp * np.sin(h_freq * t)
        zeta_d = np.array([xd, yd, zd])

        # --- Velocity calculation (analytical derivatives) ---
        xd_dot_num = (
            -a * np.sin(theta) * denom
            - a * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        yd_dot_num = (
            a * (np.cos(theta) ** 2 - np.sin(theta) ** 2) * denom
            - a * np.sin(theta) * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        xd_dot = (xd_dot_num / denom**2) * omega
        yd_dot = (yd_dot_num / denom**2) * omega
        zd_dot = h_amp * h_freq * np.cos(h_freq * t)
        zeta_d_dot = np.array([xd_dot, yd_dot, zd_dot])

        # --- Calculate acceleration using numerical differentiation ---
        # Calculate position and velocity at t+dt time
        theta_plus = omega * (t + dt_small)
        denom_plus = 1 + np.sin(theta_plus) ** 2

        xd_dot_num_plus = (
            -a * np.sin(theta_plus) * denom_plus
            - a * np.cos(theta_plus) * 2 * np.sin(theta_plus) * np.cos(theta_plus)
        )
        yd_dot_num_plus = (
            a * (np.cos(theta_plus) ** 2 - np.sin(theta_plus) ** 2) * denom_plus
            - a * np.sin(theta_plus)
            * np.cos(theta_plus)
            * 2
            * np.sin(theta_plus)
            * np.cos(theta_plus)
        )

        xd_dot_plus = (xd_dot_num_plus / denom_plus**2) * omega
        yd_dot_plus = (yd_dot_num_plus / denom_plus**2) * omega
        zd_dot_plus = h_amp * h_freq * np.cos(h_freq * (t + dt_small))

        # Calculate position and velocity at t-dt time
        theta_minus = omega * (t - dt_small)
        denom_minus = 1 + np.sin(theta_minus) ** 2

        xd_dot_num_minus = (
            -a * np.sin(theta_minus) * denom_minus
            - a * np.cos(theta_minus) * 2 * np.sin(theta_minus) * np.cos(theta_minus)
        )
        yd_dot_num_minus = (
            a * (np.cos(theta_minus) ** 2
            - np.sin(theta_minus) ** 2) * denom_minus
            - (a * np.sin(theta_minus) * np.cos(theta_minus)
                * 2 * np.sin(theta_minus) * np.cos(theta_minus))
        )
        xd_dot_minus = (xd_dot_num_minus / denom_minus**2) * omega
        yd_dot_minus = (yd_dot_num_minus / denom_minus**2) * omega
        zd_dot_minus = h_amp * h_freq * np.cos(h_freq * (t - dt_small))

        # Calculate acceleration using central difference
        zeta_d_ddot = np.array(
            [
                (xd_dot_plus - xd_dot_minus) / (2 * dt_small),
                (yd_dot_plus - yd_dot_minus) / (2 * dt_small),
                (zd_dot_plus - zd_dot_minus) / (2 * dt_small),
            ]
        )

        # --- Attitude calculation ---
        phi_d = 0.0  # Maintain zero roll
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))  # Pitch angle
        psi_d = np.arctan2(yd_dot, xd_dot)  # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- Attitude derivative calculation (using auxiliary function) ---
        _, gamma_d_plus = self.get_lemniscate_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_lemniscate_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # --- Combine yc, yc_dot ---
        yc = np.concatenate((zeta_d, gamma_d))
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot))

        # --- Velocity commands vc, wc ---
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc))

        # --- xc_dot and yc_ddot ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  # Simplified processing
        xc_dot = np.concatenate((vc_dot, wc_dot))

        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot))

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_lemniscate_pos_att(self, t):
        """Calculate lemniscate position and attitude at time t for derivative computation"""
        # --- Trajectory parameters ---
        a = 2500
        omega = 0.004
        h_center = -19000
        h_amp = 800
        h_freq = 0.001

        # --- Parametric curve parameters ---
        theta = omega * t
        # Lemniscate parametric equations
        denom = 1 + np.sin(theta) ** 2
        xd = a * np.cos(theta) / denom
        yd = a * np.sin(theta) * np.cos(theta) / denom
        zd = h_center + h_amp * np.sin(h_freq * t)
        zeta_d = np.array([xd, yd, zd])

        # --- Velocity calculation ---
        xd_dot_num = (
            -a * np.sin(theta) * denom
            - a * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        yd_dot_num = (
            a * (np.cos(theta) ** 2 - np.sin(theta) ** 2) * denom
            - a * np.sin(theta) * np.cos(theta) * 2 * np.sin(theta) * np.cos(theta)
        )
        xd_dot = (xd_dot_num / denom**2) * omega
        yd_dot = (yd_dot_num / denom**2) * omega
        zd_dot = h_amp * h_freq * np.cos(h_freq * t)

        # --- Attitude ---
        phi_d = 0.0  # Maintain zero roll
        theta_d = np.arctan2(-zd_dot, np.sqrt(xd_dot**2 + yd_dot**2))
        psi_d = np.arctan2(yd_dot, xd_dot)
        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d

    # ********************* Straight line trajectory function *********************
    def get_linear_trajectory(self, t, start_point=None, end_point=None, speed=10.0, hover_at_end=True):
        """
        Generate a straight line trajectory from start point to end point

        Parameters:
            t: Current time
            start_point: Starting coordinates [x, y, z], defaults to origin
            end_point: Ending coordinates [x, y, z], defaults to [5000, 5000, -19000]
            speed: Flight speed in m/s
            hover_at_end: Whether to hover at end point, otherwise continue flying straight

        Returns:
            yc, yc_dot, yc_ddot, xc, xc_dot: Desired states and derivatives        
        Notes:
            - yc:
                Represents the desired state vector, containing airship's desired position and attitude.
                Specifically includes:
                    Position: [x, y, z], airship's desired position in space.
                    Attitude: [φ, θ, ψ], airship's desired attitude angles (roll, pitch, yaw).
            - yc_dot:
                Represents the first derivative of desired state, i.e., desired velocity vector.
                Specifically includes:
                    Linear velocity: [vx, vy, vz], airship's desired linear velocity in space.
                    Angular velocity: [ωφ, ωθ, ωψ], airship's desired angular velocity.
            - yc_ddot:
                Represents the second derivative of desired state, i.e., desired acceleration vector.
                Specifically includes:
                    Linear acceleration: [ax, ay, az], airship's desired linear acceleration in space.
                    Angular acceleration: [αφ, αθ, αψ], airship's desired angular acceleration.
            - xc:
                Represents the control command vector, containing desired linear and angular velocities.
                Specifically includes:
                    Linear velocity command: [vx, vy, vz], airship's desired linear velocity.
                    Angular velocity command: [ωφ, ωθ, ωψ], airship's desired angular velocity.
            - xc_dot:
                Represents the first derivative of control commands, i.e., rate of change of control commands.
                Specifically includes:
                    Linear velocity rate: [dvx/dt, dvy/dt, dvz/dt], time rate of change of linear velocity.
                    Angular velocity rate: [dωφ/dt, dωθ/dt, dωψ/dt], time rate of change of angular velocity.
        """
        dt_small = 1e-4

        # --- Trajectory parameters ---
        if start_point is None:
            start_point = np.array([0.0, 0.0, -19000.0])  # Default start point
        if end_point is None:
            end_point = np.array([5000.0, 5000.0, -19000.0])  # Default end point

        # Calculate direction vector and distance
        direction = end_point - start_point
        distance = np.linalg.norm(direction)
        unit_direction = direction / max(distance, 1e-10)  # Avoid division by zero

        # Calculate total flight time
        total_time = distance / speed

        # --- Position calculation ---
        if t < total_time or not hover_at_end:
            # Still flying, or no need to hover
            effective_t = t if hover_at_end else t % total_time
            progress = min(effective_t / total_time, 1.0) if hover_at_end else effective_t / total_time

            # Linear interpolation to calculate current position
            zeta_d = start_point + progress * direction
        else:
            # Reached end point and need to hover
            zeta_d = end_point

        # --- Velocity calculation ---
        if t < total_time or not hover_at_end:
            # Constant velocity during flight
            if not hover_at_end or t < total_time:
                zeta_d_dot = unit_direction * speed
            else:
                # Zero velocity after reaching end point
                zeta_d_dot = np.zeros(3)
        else:
            # Hovering state, zero velocity
            zeta_d_dot = np.zeros(3)

        # --- Acceleration calculation (theoretically zero, but kept for compatibility) ---
        zeta_d_ddot = np.zeros(3)

        # --- Attitude calculation ---
        # Calculate desired heading angle (towards forward direction)
        if np.linalg.norm(zeta_d_dot) > 1e-6:  # If there is velocity
            phi_d = 0.0  # Maintain zero roll
            theta_d = np.arctan2(-zeta_d_dot[2], np.sqrt(zeta_d_dot[0] ** 2 + zeta_d_dot[1] ** 2))  # Pitch angle
            psi_d = np.arctan2(zeta_d_dot[1], zeta_d_dot[0])  # Yaw angle
        else:
            # Hovering state maintains last attitude
            phi_d, theta_d, psi_d = self.get_linear_pos_att(max(0, t - dt_small), start_point, end_point, speed, hover_at_end)[1]

        gamma_d = np.array([phi_d, theta_d, psi_d])

        # --- Attitude derivative calculation ---
        # Use numerical differentiation to obtain attitude derivatives
        _, gamma_d_plus = self.get_linear_pos_att(t + dt_small, start_point, end_point, speed, hover_at_end)
        _, gamma_d_minus = self.get_linear_pos_att(t - dt_small, start_point, end_point, speed, hover_at_end)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # --- Combine yc, yc_dot ---
        yc = np.concatenate((zeta_d, gamma_d)) # Position and attitude
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot)) # Linear velocity and attitude angular velocity

        # --- Velocity commands vc, wc ---
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.flatten()
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.flatten()
        xc = np.concatenate((vc, wc)) # Contains desired linear and angular velocities (control command vector)

        # --- xc_dot simplified approximation using symbolic derivatives ---
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.flatten()
        wc_dot = np.zeros(3)  #
        xc_dot = np.concatenate((vc_dot, wc_dot)) # First derivative of control commands, i.e., rate of change of control commands, linear velocity rate and angular velocity rate

        # --- yc_ddot also simplified processing ---
        gamma_d_ddot = np.zeros(3)
        yc_ddot = np.concatenate((zeta_d_ddot, gamma_d_ddot)) # Second derivative of desired state, i.e., desired linear acceleration and angular acceleration vector.

        return yc, yc_dot, yc_ddot, xc, xc_dot

    def get_linear_pos_att(self, t, start_point, end_point, speed, hover_at_end):
        """
        Calculate linear trajectory position and attitude at time t for derivative computation
        Avoid recursive calls to get_linear_trajectory
        args:
            t: Current time
            start_point: Starting coordinates [x, y, z]
            end_point: Ending coordinates [x, y, z]
            speed: Flight speed in m/s
            hover_at_end: Whether to hover at end point, otherwise continue flying straight
        return:
            zeta_d: Desired position
            gamma_d: Desired attitude
        """
        # Calculate direction vector and distance
        direction = end_point - start_point
        distance = np.linalg.norm(direction)
        unit_direction = direction / max(distance, 1e-10)  # Avoid division by zero

        # Calculate total flight time
        total_time = distance / speed

        # --- Position calculation ---
        if t < total_time or not hover_at_end:
            effective_t = t if hover_at_end else t % total_time
            progress = min(effective_t / total_time, 1.0) if hover_at_end else effective_t / total_time

            # Linear interpolation to calculate current position
            zeta_d = start_point + progress * direction
        else:
            # Reached end point and need to hover
            zeta_d = end_point

        # --- Velocity calculation (for attitude calculation) ---
        if t < total_time or not hover_at_end:
            if not hover_at_end or t < total_time:
                zeta_d_dot = unit_direction * speed
            else:
                zeta_d_dot = np.zeros(3)
        else:
            zeta_d_dot = np.zeros(3)

        # --- Attitude calculation ---
        if np.linalg.norm(zeta_d_dot) > 1e-6:  # If there is velocity
            phi_d = 0.0  # Maintain zero roll
            theta_d = np.arctan2(-zeta_d_dot[2],
                                 np.sqrt(zeta_d_dot[0] ** 2 + zeta_d_dot[1] ** 2))  # Pitch angle
            psi_d = np.arctan2(zeta_d_dot[1], zeta_d_dot[0])  # Yaw angle
        else:
            # Hovering state maintains last attitude, simplified here as default attitude
            phi_d = 0.0
            theta_d = 0.0
            # Use direction vector to calculate default heading angle
            if np.linalg.norm(direction[:2]) > 1e-6:
                psi_d = np.arctan2(direction[1], direction[0])
            else:
                psi_d = 0.0  # Default heading angle

        gamma_d = np.array([phi_d, theta_d, psi_d])

        return zeta_d, gamma_d
