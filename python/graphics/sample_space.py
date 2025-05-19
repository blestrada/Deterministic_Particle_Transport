import numpy as np
import matplotlib.pyplot as plt

Nx = 15
Nmu = 15

# Monte Carlo
num_particles = Nx * Nmu
mc_x_positions = np.zeros(num_particles, dtype=np.float64)
mc_angles = np.zeros(num_particles, dtype=np.float64)
for i in range(num_particles):
    mc_x_positions[i] = np.random.uniform()
    mc_angles[i] = np.random.uniform(-1.0,1.0)

# DPT
dpt_x_positions = (np.arange(Nx) + 0.5) / Nx
dpt_angles =  -1.0 + (np.arange(Nmu) + 0.5) * 2 / Nmu
dpt_particles = np.array([(x, mu) for x in dpt_x_positions for mu in dpt_angles])

        
# Extract x and mu values for plotting
x_values = dpt_particles[:, 0]
mu_values = dpt_particles[:, 1]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Monte Carlo Sampling Plot
axes[0].set_title('Monte Carlo Sampling in 2D')
axes[0].set_xlabel('x')
axes[0].set_ylabel(r'$\mu$')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(-1, 1)
axes[0].plot(mc_x_positions, mc_angles, '.', label='MC', color='r')

# Deterministic Sampling Plot
axes[1].set_title('Deterministic Sampling in 2D')
axes[1].set_xlabel('x')
axes[1].set_ylabel(r'$\mu$')
axes[1].set_xlim(0, 1.0)
axes[1].set_ylim(-1.0, 1.0)
axes[1].plot(x_values, mu_values, '.', label='DPT', color='b')

# Adjust layout and show the figure
plt.tight_layout()
plt.savefig('MC vs DPT Sampling in 2D.png',dpi=900)
plt.close()
