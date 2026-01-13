import numpy as np
import matplotlib.pyplot as plt

Nx, Ny, Nmu = 6, 6, 4
num_particles = Nx * Ny * Nmu

# --- Monte Carlo ---
mc_x = np.random.uniform(0, 1, num_particles)
mc_y = np.random.uniform(0, 1, num_particles)
mc_angles = np.random.uniform(0, 2*np.pi, num_particles)

# Calculate MC vector components
mc_u = np.cos(mc_angles)
mc_v = np.sin(mc_angles)

# --- DPT (Deterministic) ---
dpt_x_coords = (np.arange(Nx) + 0.5) / Nx
dpt_y_coords = (np.arange(Ny) + 0.5) / Ny
dpt_angle_coords = (np.arange(Nmu) + 0.5) * 2*np.pi / Nmu

# Creating the full list of particles from the grid
dpt_particles = np.array([(x, y, mu) for x in dpt_x_coords for y in dpt_y_coords for mu in dpt_angle_coords])

dpt_x = dpt_particles[:, 0]
dpt_y = dpt_particles[:, 1]
dpt_angles = dpt_particles[:, 2]

# Calculate DPT vector components
dpt_u = np.cos(dpt_angles)
dpt_v = np.sin(dpt_angles)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Monte Carlo Plot
# 'pivot=middle' ensures the arrow rotates around the (x,y) point
axes[0].quiver(mc_x, mc_y, mc_u, mc_v, color='red', alpha=0.5, pivot='tail', scale=30)
axes[0].set_title(f'Random Sampling')

# Deterministic Plot
axes[1].quiver(dpt_x, dpt_y, dpt_u, dpt_v, color='blue', alpha=0.5, pivot='tail', scale=30)
axes[1].set_title(f'Deterministic Sampling')

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.savefig('MC_vs_DPT_Quiver.png', dpi=900)
plt.show()