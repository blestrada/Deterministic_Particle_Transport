import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

# Get data from the gnuplot file.
# Load the data file
file_path = 'marshak_wave_imc'

# Read the data, skipping the first row if it's a header
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)

# Assign column names based on the data header or assume generic names
data.columns = ['x', 'Te', 'Tr', 'Column4', 'Column5', 'Column6']  # Adjust based on actual data columns

# Create figure for IMC material temperature plot
plt.figure(figsize=(7, 6))
# Plot columns 1 and 2 as in Gnuplot
plt.plot(data['x'], data['Te'], color='blue', label=r'Material at $t$ = 0.3 sh (IMC)',linewidth=1)
plt.xlabel('x - cm')
plt.ylabel('Tm - keV')
plt.ylim(0.0, 1.1)
plt.legend(loc='upper right', numpoints=1, frameon=True)
plt.savefig("marshak_wave-mat-temp-IMC.png", bbox_inches="tight", dpi=900)
plt.close

# Create figure for IMC radiation energy density plot
plt.figure(figsize=(7, 6))
# Calculate radiation energy density from radiation temperature
radnrgdens_imc = (data['Tr'] / 0.013720169037741436) ** (1/4)
plt.plot(data['x'], radnrgdens_imc, color='red', label=r'Radiation at $t$ = 0.3 sh (IMC)', linewidth=1)
plt.xlabel('x - cm')
plt.ylabel(r"Radiation Energy Density - $\frac{jrk}{cm^3}$")
plt.legend(loc='upper right', numpoints=1, frameon=True)
plt.savefig("marshak_wave-radnrgdens-IMC.png", bbox_inches="tight", dpi=900)
plt.close()



# Open the output file
fname = open("marshak_wave-20.out", "rb")

# Times corresponding to data in the output file
times = [r"$t$ = 0.3 sh"]

time_line = fname.readline().decode().strip()  # Read the time line
xdata = pickle.load(fname)      # cellpos
temp = pickle.load(fname)       # temperature
#radnrgdens_dpt = pickle.load(fname) # radnrgdens

# Create figure for the material temperature for DPT
plt.figure(figsize=(7, 6))
plt.plot(xdata, temp, '-' , color='blue', label=r"Material at $t$ = 0.3 sh (DPT)", linewidth=1, alpha=1)
plt.xlabel("x - cm")
plt.ylabel("Tm - keV")
plt.ylim(0.0, 1.1)
plt.legend(loc='upper right', numpoints=1, frameon=True)
plt.savefig("marshak_wave-mat-temp-DPT-case1.png", bbox_inches="tight", dpi=900)
plt.close()

# Create figure for the radiation energy density for DPT
plt.figure(figsize=(7, 6))
plt.plot(xdata, (temp / 0.013720169037741436) ** (1/4), '-' , color='red', label=r"Radiation at $t$ = 0.3 sh (DPT)", linewidth=1, alpha=1, markersize=3)
plt.xlabel("x - cm")
plt.ylabel(r"Radiation Energy Density - $\frac{jrk}{cm^3}$")
plt.legend(loc='upper right', numpoints=1, frameon=True)
plt.savefig("marshak_wave-radnrgdens-DPT-case1.png", bbox_inches="tight", dpi=900)
plt.close()
# Close the file
fname.close()
