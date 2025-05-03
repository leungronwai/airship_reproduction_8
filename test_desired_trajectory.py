import numpy as np
import matplotlib.pyplot as plt
from airship.trajectory import Trajectory

def plot_spiral_trajectory():
    """
    Plot the spiral trajectory defined in trajectory.py.
    """
    # Time range for simulation
    t_values = np.linspace(0, 200, 1000)  # Simulate for 200 seconds with 1000 points

    # Initialize arrays to store trajectory data
    x_values, y_values, z_values = [], [], []

    # Get the trajectory function
    trajectory_function = Trajectory.define_spiral_trajectory(t_values)

    # Calculate position for each time step
    for t in t_values:
        pos, _, _ = trajectory_function(t)  # Get position only
        x_values.append(pos[0])
        y_values.append(pos[1])
        z_values.append(pos[2])

    # Plot 2D trajectory (X-Y plane)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="Spiral Trajectory (X-Y)", color="blue")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Spiral Trajectory in X-Y Plane")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot 3D trajectory (X-Y-Z space)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_values, y_values, z_values, label="Spiral Trajectory (X-Y-Z)", color="green")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Spiral Trajectory in 3D Space")
    ax.legend()
    plt.show()

# Call the function to plot the trajectory
plot_spiral_trajectory()