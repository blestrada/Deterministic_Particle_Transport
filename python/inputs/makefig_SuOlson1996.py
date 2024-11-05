import pickle
import matplotlib.pyplot as plt
import numpy as np

# Su-Olson Surface Source Numerical Results (1996)
# radiation_at_100 = [0.90895, 0.90107, 0.88926, 0.86965, 0.85011, 0.83067, 0.71657, 0.54080, 0.38982, 0.26789, 0.10906, 0.03624]
radiation_at_10 = [0.73611, 0.71338, 0.67978, 0.62523, 0.57274, 0.52255, 0.27705, 0.07075, 0.01271, 0.00167, 0.00002]
radiation_at_1 = [0.46599, 0.42133, 0.36020, 0.27323, 0.20332, 0.14837, 0.01441, 0.00005, 0.00001]
material_at_100 = [0.90849, 0.90057, 0.88871, 0.86900, 0.84937, 0.82983, 0.71521, 0.53877, 0.38745, 0.26551, 0.10732, 0.03534]
material_at_10 = [0.72328, 0.69946, 0.66432, 0.60749, 0.55308, 0.50134, 0.25413, 0.05936, 0.00968, 0.00115, 0.00001]
material_at_1 = [0.24762, 0.21614, 0.17530, 0.12182, 0.08306, 0.05556, 0.00324, 0.00001]

x_values = [0, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 15, 20]

# Create figure and axis for combined plot
plt.figure(figsize=(12, 6))

# Analytical results (dashed lines)
# plt.plot(x_values, radiation_at_100, 'r--', label=r'Radiation at $\tau$ = 100 (Analytical)')
# plt.plot(x_values, material_at_100, 'b--', label=r'Material at $\tau$ = 100 (Analytical)')
plt.plot(x_values[:len(radiation_at_10)], radiation_at_10, 'r--', label=r'Radiation at $\tau$ = 10 (Analytical)')
plt.plot(x_values[:len(material_at_10)], material_at_10, 'b--', label=r'Material at $\tau$ = 10 (Analytical)')
plt.plot(x_values[:len(radiation_at_1)], radiation_at_1, 'r--', label=r'Radiation at $\tau$ = 1 (Analytical)')
plt.plot(x_values[:len(material_at_1)], material_at_1, 'b--', label=r'Material at $\tau$ = 1 (Analytical)')

# Open the output file
fname = open("SuOlson1996-1.out", "rb")

# Times corresponding to data in the output file
times = [r"$\tau$ = 1", r"$\tau$ = 10", r"$\tau$ = 100"]

# Loop to read and plot data from the output file
for i in range(3):
    time_line = fname.readline().decode().strip()  # Skip the time line and decode it properly
    
    xdata = pickle.load(fname)      # cellpos
    # temp = pickle.load(fname)       # material temp
    # radtemp = pickle.load(fname)    # radiation temp
    matnrgdens = pickle.load(fname) # material energy density
    radnrgdens = pickle.load(fname) # radiation energy density

    # Plot the file data (solid lines)
    plt.plot(xdata * np.sqrt(3), matnrgdens , 'b-', label=f"Material at {times[i]}", linewidth=2)
    plt.plot(xdata * np.sqrt(3), radnrgdens, 'r-', label=f"Radiation at {times[i]}", linewidth=2)

# Set axis labels and limits
plt.xlim(0, 15.0)
plt.ylim(0, 2.0)
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel("x - cm")
plt.ylabel("Energy Density")

# Add legend
plt.legend(loc='upper right', numpoints=1, frameon=False)

# Save figure
plt.savefig("SuOlson1996-1.png", bbox_inches="tight", dpi=900)

# Optionally show the plot
plt.show()

# Close the file
fname.close()
