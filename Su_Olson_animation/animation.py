# animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the data from the .npy file
data = np.load('Su_Olson_animation/simulation_data.npy')

# Set up the plot
fig, ax = plt.subplots()

# Create a plot element to animate. For example, we'll plot the radiation energy density
line, = ax.plot([], [], lw=0.6, markersize=5, marker='x', color='r')

# Define initialization function
# : this will be called at the start
def init():
    ax.set_xlim(0.0, 5.0)  # Set x limits based on cell positions
    ax.set_ylim(0.0, 2.5)  # Set y limits based on the energy
    ax.set_xlabel('x')
    ax.set_ylabel('Radiation Energy Density')
    return line,

# Define the update function for each frame
def update(frame):
    # Extract the data for the current frame
    x = data[frame, :, 0]  # Cell positions at this time step
    y = data[frame, :, 1]  # Radiation energy density at this time step
    
    # Update the plot with new data
    line.set_data(x, y)

    # Update the title with the current timestep
    ax.set_title(f"IMC Radiation Energy Density Over Time")
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(data.shape[0]), init_func=init, blit=True)

# Show the animation
# plt.show()

# Optionally, save the animation to a file
ani.save('imc_radiation_animation.mp4', writer='ffmpeg', fps=60, dpi=600)