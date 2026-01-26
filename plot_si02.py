import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load your CSV
df_imc = pd.read_csv("si02_imc_0.4sh.csv")

# Define the 10 points of interest
epsilon = 5e-5
r_points = np.array([epsilon, 0.08 - epsilon])
z_points = np.array([epsilon, .025 + epsilon, .05 + epsilon, .075 + epsilon, .1 - epsilon])

z_edges = np.array([0., 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03, 0.0305, 0.031, 0.0315, 0.032, 0.0325, 0.033, 0.0335, 0.034, 0.0345, 0.035, 0.0355, 0.036, 0.0365, 0.037, 0.0375, 0.038, 0.0385, 0.039, 0.0395, 0.04, 0.0405, 0.041, 0.0415, 0.042, 0.0425, 0.043, 0.0435, 0.044, 0.0445, 0.045, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485, 0.049, 0.0495, 0.05, 0.0505, 0.051, 0.0515, 0.052, 0.0525, 0.053, 0.0535, 0.054, 0.0545, 0.055, 0.0555, 0.056, 0.0565, 0.057, 0.0575, 0.058, 0.0585, 0.059, 0.0595, 0.06, 0.0605, 0.061, 0.0615, 0.062, 0.0625, 0.063, 0.0635, 0.064, 0.0645, 0.065, 0.0655, 0.066, 0.0665, 0.067, 0.0675, 0.068, 0.0685, 0.069, 0.0695, 0.07, 0.0705, 0.071, 0.0715, 0.072, 0.0725, 0.073, 0.0735, 0.074, 0.0745, 0.075, 0.0755, 0.076, 0.0765, 0.077, 0.0775, 0.078, 0.0785, 0.079, 0.0795, 0.08, 0.0805, 0.081, 0.0815, 0.082, 0.0825, 0.083, 0.0835, 0.084, 0.0845, 0.085, 0.0855, 0.086, 0.0865, 0.087, 0.0875, 0.088, 0.0885, 0.089, 0.0895, 0.09, 0.0905, 0.091, 0.0915, 0.092, 0.0925, 0.093, 0.0935, 0.094, 0.0945, 0.095, 0.0955, 0.096, 0.0965, 0.097, 0.0975, 0.098, 0.0985, 0.099, 0.0995, 0.1])
r_edges = np.array([0., 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03, 0.0305, 0.031, 0.0315, 0.032, 0.0325, 0.033, 0.0335, 0.034, 0.0345, 0.035, 0.0355, 0.036, 0.0365, 0.037, 0.0375, 0.038, 0.0385, 0.039, 0.0395, 0.04, 0.0405, 0.041, 0.0415, 0.042, 0.0425, 0.043, 0.0435, 0.044, 0.0445, 0.045, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485, 0.049, 0.0495, 0.05, 0.0505, 0.051, 0.0515, 0.052, 0.0525, 0.053, 0.0535, 0.054, 0.0545, 0.055, 0.0555, 0.056, 0.0565, 0.057, 0.0575, 0.058, 0.0585, 0.059, 0.0595, 0.06, 0.0605, 0.061, 0.0615, 0.062, 0.0625, 0.063, 0.0635, 0.064, 0.0645, 0.065, 0.0655, 0.066, 0.0665, 0.067, 0.0675, 0.068, 0.0685, 0.069, 0.0695, 0.07, 0.0705, 0.071, 0.0715, 0.072, 0.0725, 0.073, 0.0735, 0.074, 0.0745, 0.075, 0.0755, 0.076, 0.0765, 0.077, 0.0775, 0.078, 0.0785, 0.079, 0.0795, 0.08])

# Get the indices of the 10 points
r_indices = np.searchsorted(r_edges, r_points) - 1
z_indices = np.searchsorted(z_edges, z_points) - 1

# 2. Generate the 10 pairs of (z_idx, r_idx)
# This creates a list of indices for every combination of your points
point_indices = []
for z_idx in z_indices:
    for r_idx in r_indices:
        point_indices.append((z_idx, r_idx))

# 3. Output results
print("Z-indices:", z_indices)
print("R-indices:", r_indices)
print("10 Point Index Pairs (z_idx, r_idx):")
for i, pair in enumerate(point_indices):
    print(f"Point {i+1}: {pair}")

# Get the material and rad temp at end of simulation
# User-specified time (pick the closest one available)
available_times = df_imc['time'].unique()
target_time = available_times[-501] # -1 for final, -501 for 0.35 sh

# 1. Filter the dataframe for that specific time
df_step = df_imc[df_imc['time'] == target_time]

# 2. Reshape the data back into a 2D grid
# We use pivot to organize x_idx as rows and y_idx as columns
temp_grid = df_step.pivot(index='x_idx', columns='y_idx', values='temp').values
radtemp_grid = df_step.pivot(index='x_idx', columns='y_idx', values='radtemp').values


full_r_edges = np.concatenate([-r_edges[::-1], r_edges[1:]])

# 2. Prepare the data
# Flip the radiation grid vertically so the "center" (index 0) stays at R=0
# and the "outer edge" moves toward R = -0.08
rad_half = np.flip(radtemp_grid, axis=1) 
mat_half = temp_grid

# 3. Stack them: Radiation on bottom (negative R), Material on top (positive R)
# Stack along the radial axis (axis 1)
combined_data = np.hstack([rad_half, mat_half])

plt.figure()
pc = plt.pcolormesh(z_edges, 
                    full_r_edges, 
                    combined_data.T, 
                    cmap="jet", 
                    shading='flat=')
plt.colorbar(pc, label="Temperature [keV]")
plt.clim(vmin=2.419e-05, vmax=0.139)
plt.xlabel("Z (cm)")
plt.ylabel("R (cm)")
plt.xlim(z_edges[0], z_edges[-1])
plt.ylim(full_r_edges[0], full_r_edges[-1])
plt.axis('scaled')
plt.title(f"Temperature at t={target_time}")

plt.savefig('si02_temp.png',dpi=900)
plt.show()

# Plot Fiducial Points

axis_pairs = [(zi, r_indices[0]) for zi in z_indices]
edge_pairs = [(zi, r_indices[1]) for zi in z_indices]

plt.figure()
for zi, ri in axis_pairs:
    # Filter dataframe for this specific spatial cell
    subset = df_imc[(df_imc['x_idx'] == zi) & (df_imc['y_idx'] == ri)]
    
    # Plot Material Temp (solid line) and Rad Temp (dashed)
    z_val = z_edges[zi]
    p = plt.plot(subset['time'], subset['temp'], label=f"Tmat z={z_val:.3f}")
    plt.plot(subset['time'], subset['radtemp'], color=p[0].get_color(), linestyle='--')

plt.xlabel("Time (sh)")
plt.ylabel("T (keV)")
plt.xlim(-0.01, 0.4)
plt.ylim(0.0, 0.15)
plt.legend(fontsize='small', ncol=2)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.title("Temperature on Axis (R=0)")
plt.tight_layout()
plt.savefig('sio2_axis.png', dpi=900)
plt.show()

plt.figure()
for zi, ri in edge_pairs:
    subset = df_imc[(df_imc['x_idx'] == zi) & (df_imc['y_idx'] == ri)]
    
    z_val = z_edges[zi]
    p = plt.plot(subset['time'], subset['temp'], label=f"Tmat z={z_val:.3f}")
    plt.plot(subset['time'], subset['radtemp'], color=p[0].get_color(), linestyle='--')

plt.xlabel("Time (sh)")
plt.ylabel("T (keV)")
plt.xlim(-0.01, 0.4)
plt.ylim(0.0, 0.13) # Note the lower limit requested
plt.legend(fontsize='small', ncol=2)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.title("Temperature at Pipe Edge (R=0.08)")
plt.tight_layout()
plt.savefig('sio2_edge.png', dpi=900)
plt.show()

