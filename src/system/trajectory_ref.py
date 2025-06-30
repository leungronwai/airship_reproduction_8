"""
Trajectory generation module (trajectory.py)
"""
# pylint: disable=invalid-name
# pylint: disable=line-too-long


import numpy as np
from src.system.rotation_matrices import R_zeta, R_y_inv

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
        self.omega = 0.04  # Angular velocity (rad/s)
        self.r = 2500  # Radius (m)
        self.h_max = 2000  # Maximum height (m)

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
        omega = self.omega  # Angular velocity
        r = self.r # Radius
        h_max = self.h_max  # Maximum height

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
        theta_d = float(np.arctan2(-vel[2], np.sqrt(vel[0] ** 2 + vel[1] ** 2)) ) # Pitch angle
        psi_d = float(np.arctan2(vel[1], vel[0]) ) # Yaw angle
        gamma_d = np.array([phi_d, theta_d, psi_d])

        # Use numerical differentiation to get attitude derivatives
        _, gamma_d_plus = self.get_spiral_pos_att(t + dt_small)
        _, gamma_d_minus = self.get_spiral_pos_att(t - dt_small)
        gamma_d_dot = (gamma_d_plus - gamma_d_minus) / (2 * dt_small)

        # Combine yc, yc_dot
        yc = np.concatenate((zeta_d, gamma_d)) # !!!!!!!! position (zeta_d): [x, y, z] + attitude (gamma_d): [phi, theta, psi]
        yc_dot = np.concatenate((zeta_d_dot, gamma_d_dot)) #!!!!!! velocity (zeta_d_dot): [vx, vy, vz] + angular velocity (gamma_d_dot): [p, q, r]

        # Velocity commands vc, wc
        Rc_z = R_zeta(gamma_d)
        Rc_y_inv = R_y_inv(gamma_d)
        vc = Rc_z.T @ zeta_d_dot.reshape(-1, 1)
        vc = vc.reshape((-1,1))
        wc = Rc_y_inv @ gamma_d_dot.reshape(-1, 1)
        wc = wc.reshape((-1,1))
        xc = np.concatenate((vc, wc))

        # xc_dot simplified approximation through symbolic derivatives
        vc_dot = Rc_z.T @ zeta_d_ddot.reshape(-1, 1)
        vc_dot = vc_dot.reshape((-1,1))
        wc_dot = np.zeros(3).reshape(-1, 1)  # Simplified processing, assuming small angular velocity change rate
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
        omega = self.omega  # Angular velocity (rad/s)
        r = self.r # Basic radius (m)
        h_max = self.h_max  # Maximum height (m)

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
