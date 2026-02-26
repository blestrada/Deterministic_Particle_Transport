import matplotlib.pyplot as plt
import numpy as np

"""Benchmark Solutions"""
x_bench = [0.01000, 0.10000, 0.17783, 0.31623, 0.45000, 0.50000, 0.56234, 0.75000, 1.00000, 1.33352, 1.77828, 3.16228, 5.62341, 10.00000, 17.78279]

rad_benchtwo = [[0.09757, 0.09757, 0.09758, 0.09756, 0.09033, 0.04878, 0.00383, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], 
                [0.29363, 0.29365, 0.29364, 0.28024, 0.21573, 0.14681, 0.06783, 0.00292, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.72799, 0.71888, 0.69974, 0.63203, 0.50315, 0.40769, 0.29612, 0.13756, 0.04396, 0.00324, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [1.28138, 1.26929, 1.24193, 1.15018, 0.98599, 0.87477, 0.74142, 0.51563, 0.33319, 0.18673, 0.08229, 0.00160, 0.00000, 0.00000, 0.00000],
                [2.26474, 2.24858, 2.21291, 2.09496, 1.89259, 1.76429, 1.60822, 1.30947, 1.02559, 0.74721, 0.48739, 0.11641, 0.00554, 0.00000, 0.00000],
                [0.68703, 0.68656, 0.68556, 0.68235, 0.67761, 0.67550, 0.67252, 0.66146, 0.64239, 0.61024, 0.55789, 0.36631, 0.11177, 0.00491, 0.00000],
                [0.35675, 0.35668, 0.35654, 0.35618, 0.35552, 0.35527, 0.35491, 0.35346, 0.35092, 0.34646, 0.33868, 0.30281, 0.21323, 0.07236, 0.00296]
               ] # ca = 0.5, cs = 0.5

mat_benchtwo = [[0.00242, 0.00242, 0.00242, 0.00242, 0.00235, 0.00121, 0.00003, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.02255, 0.02253, 0.02256, 0.02223, 0.01826, 0.01128, 0.00350, 0.00003, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.17609, 0.17420, 0.17035, 0.15520, 0.12164, 0.09194, 0.05765, 0.01954, 0.00390, 0.00009, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.77654, 0.76878, 0.75108, 0.69082, 0.57895, 0.49902, 0.40399, 0.25610, 0.14829, 0.07161, 0.02519, 0.00018, 0.00000, 0.00000, 0.00000],
                [2.00183, 1.98657, 1.95286, 1.84104, 1.64778, 1.52383, 1.37351, 1.09216, 0.83248, 0.58640, 0.36629, 0.07658, 0.00290, 0.00000, 0.00000],
                [0.71860, 0.71805, 0.71687, 0.71312, 0.70755, 0.70499, 0.70144, 0.68851, 0.66637, 0.62937, 0.57001, 0.36066, 0.10181, 0.00385, 0.00000],
                [0.36067, 0.36065, 0.36047, 0.36005, 0.35945, 0.35917, 0.35876, 0.35727, 0.35465, 0.35004, 0.34200, 0.30553, 0.21308, 0.07077, 0.00273]
               ] # ca = 0.5, cs = 0.5


import numpy as np
import matplotlib.pyplot as plt

# 1. Load your simulation data
data = np.load("SuOlson_RZ_results.npz")
time_hist = data['time']
# Shape is (steps, Z, R) -> we take [:, :, 0] because it's a 1D equivalent
rad_data = data['radnrgdens'][:, :, 0] 
mat_data = data['matnrgdens'][:, :, 0]
z_edges = data['z_edges']
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

# 2. Define the benchmark times you want to compare
# Benchmark indices: 0: t=0.1, 1: t=0.316, 2: t=1.0, 3: t=3.16, 4: t=10.0, 5: t=31.6, 6: t=100.0
target_times = [0.1, 1.0, 10.0]
bench_indices = [0, 2, 4] # Mapping targets to rad_benchtwo rows

plt.figure(figsize=(10, 8))
    
# Plot Simulation
plt.plot(z_centers, rad_data[10], label=f'DPT t = 0.1', marker='o', markersize=2, alpha=0.6)
plt.plot(z_centers, rad_data[100], label=f'DPT t = 1.0', marker='o', markersize=2, alpha=0.6)
plt.plot(z_centers, rad_data[1000], label=f'DPT t = 10.0', marker='o', markersize=2, alpha=0.6)
    
# Plot Benchmark 
plt.plot(x_bench, rad_benchtwo[0], 'x', 
            markersize=4, alpha=0.9, label=f'Analytic t = 0.1')
plt.plot(x_bench, rad_benchtwo[2], 'x', 
            markersize=4, alpha=0.9, label=f'Analytic t = 1.0')
plt.plot(x_bench, rad_benchtwo[4], 'x', 
            markersize=4, alpha=0.9, label=f'Analytic t = 10.0')

plt.xlim(0, 5)
plt.ylim(0.0, 2.5)
plt.ylabel('Radiation Energy Density')
plt.xlabel('Position (z)')
plt.title('Su-Olson RZ')
plt.legend(ncol=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('radnrgdensity_RZ_SuOlson.png', dpi=900)
plt.show()
plt.close()

# Subplot 2: Material Energy Density
plt.figure(figsize=(10, 8))

# Plot simulation
plt.plot(z_centers, mat_data[10], label=f'DPT t = 0.1', marker='o', markersize=2, alpha=0.6)
plt.plot(z_centers, mat_data[100], label=f'DPT t = 1.0', marker='o', markersize=2, alpha=0.6)
plt.plot(z_centers, mat_data[1000], label=f'DPT t = 10.0', marker='o', markersize=2, alpha=0.6)
    

# Plot Benchmark
plt.plot(x_bench, mat_benchtwo[0], 'x', 
            markersize=4, alpha=0.9, label=f'Analytic t = 0.1')
plt.plot(x_bench, mat_benchtwo[2], 'x', 
            markersize=4, alpha=0.9, label=f'Analytic t = 1.0')
plt.plot(x_bench, mat_benchtwo[4], 'x', 
            markersize=4, alpha=0.9, label=f'Analytic t = 10.0')
plt.title('Su-Olson RZ')
plt.yscale('log')
plt.xlim(0, 5)
plt.ylim(1e-4, 5)
plt.xlabel('Position (z)')
plt.ylabel('Material Energy Density')
plt.legend(ncol=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('matnrgdens_RZ_SuOlson.png', dpi=900)
plt.show()