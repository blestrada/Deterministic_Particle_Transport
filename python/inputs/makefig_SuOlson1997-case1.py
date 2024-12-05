import pickle
import matplotlib.pyplot as plt
import numpy as np

"""Benchmark Solutions"""
x_bench = [0.01000, 0.10000, 0.17783, 0.31623, 0.45000, 0.50000, 0.56234, 0.75000, 1.00000, 1.33352, 1.77828, 3.16228, 5.62341, 10.00000, 17.78279]

t_bench = [0.10000, 0.31623, 1.00000, 3.16228, 10.00000, 31.6228, 100.000]

rad_benchone = [[0.09531, 0.09531, 0.09532, 0.09529, 0.08823, 0.04765, 0.00375, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], 
             [0.27526, 0.27526, 0.27527, 0.26262, 0.20312, 0.13762, 0.06277, 0.00280, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
             [0.64308, 0.63585, 0.61958, 0.56187, 0.44711, 0.35801, 0.25374, 0.11430, 0.03648, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
             [1.20052, 1.18869, 1.16190, 1.07175, 0.90951, 0.79902, 0.66678, 0.44675, .027540, 0.14531, 0.05968, 0.00123, 0.00000, 0.00000, 0.00000],
             [2.23575, 2.21944, 2.18344, 2.06448, 1.86072, 1.73178, 1.57496, 1.27398, 0.98782, 0.70822, 0.45016, 0.09673, 0.00375, 0.00000, 0.00000],
             [0.69020, 0.68974, 0.68878, 0.68569, 0.68111, 0.67908, 0.67619, 0.66548, 0.64691, 0.61538, 0.56353, 0.36965, 0.10830, 0.00390, 0.00000],
             [0.35720, 0.35714, 0.35702, 0.35664, 0.35599, 0.35574, 0.35538, 0.35393, 0.35141, 0.34697, 0.33924, 0.30346, 0.21382, 0.07200, 0.00272]
            ] # ca = 1, cs = 0 

mat_benchone = [[0.00468, 0.00468, 0.00468, 0.00468, 0.00455, 0.00234, 0.00005, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], 
                [0.04093, 0.04093, 0.04093, 0.04032, 0.03314, 0.02046, 0.00635, 0.00005, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.27126, 0.26839, 0.26261, 0.23978, 0.18826, 0.14187, 0.08838, 0.03014, 0.00625, 0.00017, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [0.94670, 0.93712, 0.91525, 0.84082, 0.70286, 0.60492, 0.48843, 0.30656, 0.17519, 0.08352, 0.02935, 0.00025, 0.00000, 0.00000, 0.00000],
                [2.11186, 2.09585, 2.06052, 1.94365, 1.74291, 1.61536, 1.46027, 1.16591, 0.88992, 0.62521, 0.38688, 0.07642, 0.00253, 0.00000, 0.00000],
                [0.70499, 0.70452, 0.70348, 0.70020, 0.69532, 0.69308, 0.68994, 0.67850, 0.65868, 0.62507, 0.57003, 0.36727, 0.10312, 0.00342, 0.00000],
                [0.35914, 0.35908, 0.35895, 0.35854, 0.35793, 0.35766, 0.35728, 0.35581, 0.35326, 0.34875, 0.34086, 0.30517, 0.21377, 0.07122, 0.00261]
               ] # ca = 1, cs = 0

# Create figure and axis for combined plot
plt.figure(figsize=(7, 6))

# Open the output file
fname = open("SuOlson1997-case1-test2.out", "rb")

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
    plt.plot(xdata, radnrgdens, color=colors[i], marker='o', label=f"Radiation at {times[i]} (DPT)", linewidth=2, markersize='5')

# Now plot benchmark data 
plt.plot(x_bench, rad_benchone[0], 'x--', label=r'Radiation at $\tau$ = 0.1 (Benchmark)', linewidth=1, color='crimson')  
plt.plot(x_bench, rad_benchone[2], 'x--', label=r'Radiation at $\tau$ = 1.0 (Benchmark)', linewidth=1, color='purple')
plt.plot(x_bench, rad_benchone[4], 'x--', label=r'Radiation at $\tau$ = 10 (Benchmark)', linewidth=1, color='teal')

plt.xlim(0.0, 5.0)
plt.ylim(0.0, 2.5)
plt.xlabel("x")
plt.ylabel("Radiation Energy Density")

# Add legend
plt.legend(loc='upper right', numpoints=1, frameon=True)

# Save figure
plt.savefig("SuOlson1997-case1-test-rad.png", bbox_inches="tight", dpi=900)
plt.show()

# Close the file
fname.close()

# Create figure for material energy density plot
plt.figure(figsize=(7, 6))

# Open the output file again to read the data for material energy density
fname = open("SuOlson1997-case1-test2.out", "rb")

# Loop to read and plot data from the output file for material
for i in range(3):
    time_line = fname.readline().decode().strip()  # Read the time line again
    
    xdata = pickle.load(fname)      # cellpos
    matnrgdens = pickle.load(fname) # material energy density
    radnrgdens = pickle.load(fname) # radiation energy density

    # Plot the material energy density data
    plt.plot(xdata, matnrgdens, color=colors[i], marker='o', label=f"Material at {times[i]} (DPT)", linewidth=2, markersize=5)

# Now plot benchmark data for material
plt.plot(x_bench, mat_benchone[0], 'x-', label=r'Material at $\tau$ = 0.1 (Benchmark)', linewidth=1, color='crimson')  
plt.plot(x_bench, mat_benchone[2], 'x-', label=r'Material at $\tau$ = 1.0 (Benchmark)', linewidth=1, color='purple')  
plt.plot(x_bench, mat_benchone[4], 'x-', label=r'Material at $\tau$ = 10 (Benchmark)', linewidth=1, color='teal')  

plt.xlim(0.01, 8.0)
plt.ylim(1e-3, 1e2)
plt.xlabel("x")
plt.ylabel("Material Energy Density")
plt.yscale('log')
plt.xscale('log')

# Add legend
plt.legend(loc='upper right', numpoints=1, frameon=True)

# Save figure for material energy density
plt.savefig("SuOlson1997-case1-test-mat.png", bbox_inches="tight", dpi=900)
plt.show()

# Close the file
fname.close()
