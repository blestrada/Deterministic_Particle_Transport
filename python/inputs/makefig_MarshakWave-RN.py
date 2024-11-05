import pickle
import matplotlib.pyplot as plt
import numpy as np


# Create figure for Temperature plot
plt.figure(figsize=(7, 6))

# Open the output file
fname = open("marshak_wave-rn.out", "rb")

# Times corresponding to data in the output file
times = [r"$t$ = 0.01 sh", r"$t$ = 0.02 sh", r"$t$ = 0.03 sh"]

# Loop to read and plot data from the output file
colors = ['red', 'blue', 'green', 'black']  # List of colors for each iteration

for i in range(3):
    time_line = fname.readline().decode().strip()  # Read the time line
    
    xdata = pickle.load(fname)      # cellpos
    temp = pickle.load(fname)       # temperature

    # Plot the temperature data
    plt.plot(xdata, temp, color=colors[i], marker='o', label=f"Temperature at {times[i]} (IMC)", linewidth=2, alpha=.5)

plt.xlabel("x - cm")
plt.ylabel("T - keV")

# Add legend
plt.legend(loc='upper right', numpoints=1, frameon=False)

# Save figure for radiation energy density
plt.savefig("marshak_wave.png", bbox_inches="tight", dpi=900)
plt.show()

# Close the file
fname.close()
