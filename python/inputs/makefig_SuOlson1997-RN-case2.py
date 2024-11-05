import pickle
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

# Create figure for radiation energy density plot
plt.figure(figsize=(7, 6))

# Open the output file
fname = open("SuOlson1997-RN-case2.out", "rb")

# Times corresponding to data in the output file
times = [r"$\tau$ = 0.1", r"$\tau$ = 1.0", r"$\tau$ = 10", r"$\tau$ = 100"]

# Loop to read and plot data from the output file
colors = ['blue', 'crimson', 'purple', 'black']  # List of colors for each iteration

for i in range(3):
    time_line = fname.readline().decode().strip()  # Read the time line
    
    xdata = pickle.load(fname)      # cellpos
    matnrgdens = pickle.load(fname) # material energy density
    radnrgdens = pickle.load(fname) # radiation energy density

    # Plot the radiation energy density data
    plt.plot(xdata, radnrgdens, color=colors[i], marker='o', label=f"Radiation at {times[i]} (IMC)", linewidth=2, markersize='5')

# Now plot benchmark data 
plt.plot(x_bench, rad_benchtwo[0], 'x--', label=r'Radiation at $\tau$ = 0.1 (Benchmark)', linewidth=1, color ='orange')  
plt.plot(x_bench, rad_benchtwo[2], 'x--', label=r'Radiation at $\tau$ = 1.0 (Benchmark)', linewidth=1, color='teal')
plt.plot(x_bench, rad_benchtwo[4], 'x--', label=r'Radiation at $\tau$ = 10 (Benchmark)', linewidth=1, color='forestgreen')

plt.xlim(0.0, 5.0)
plt.ylim(0.0, 2.5)
plt.xlabel("x")
plt.ylabel("Radiation Energy Density")

# Add legend
plt.legend(loc='upper right', numpoints=1, frameon=True)

# Save figure for radiation energy density
plt.savefig("SuOlson1997-RN-case2-rad.png", bbox_inches="tight", dpi=900)
plt.show()

# Close the file
fname.close()

# Create figure for material energy density plot
plt.figure(figsize=(7, 6))

# Open the output file again to read the data for material energy density
fname = open("SuOlson1997-RN-case2.out", "rb")

# Loop to read and plot data from the output file for material
for i in range(3):
    time_line = fname.readline().decode().strip()  # Read the time line again
    
    xdata = pickle.load(fname)      # cellpos
    matnrgdens = pickle.load(fname) # material energy density
    radnrgdens = pickle.load(fname) # radiation energy density

    # Plot the material energy density data
    plt.plot(xdata, matnrgdens, color=colors[i], marker='o', label=f"Material at {times[i]} (IMC)", linewidth=2, markersize='5')

# Now plot benchmark data for material
# Uncomment if you want to add material benchmark data
plt.plot(x_bench, mat_benchtwo[0], 'x--', label=r'Material at $\tau$ = 0.1 (Benchmark)', linewidth=1, color='orange')  
plt.plot(x_bench, mat_benchtwo[2], 'x--', label=r'Material at $\tau$ = 1.0 (Benchmark)', linewidth=1, color='teal')  
plt.plot(x_bench, mat_benchtwo[4], 'x--', label=r'Material at $\tau$ = 10 (Benchmark)', linewidth=1, color='forestgreen')  

plt.xlim(0.01, 8.0)
plt.ylim(1e-3, 1e2)
plt.xlabel("x")
plt.ylabel("Material Energy Density")
plt.yscale('log')
plt.xscale('log')

# Add legend
plt.legend(loc='upper right', numpoints=1, frameon=True)

# Save figure for material energy density
plt.savefig("SuOlson1997-RN-case2-mat.png", bbox_inches="tight", dpi=900)
plt.show()

# Close the file
fname.close()
