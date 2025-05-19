import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set up frequency group structure
# 50 groups, logarithmically spaced between 3.0 × 10−3 keV and 30.0 keV
# Define the energy range
E_min = 3.0e-3  # keV
E_max = 30.0  # keV
Ng = 50  # Number of frequency groups

# Generate logarithmically spaced edges
edges = np.logspace(np.log10(E_min), np.log10(E_max), Ng + 1)

# Compute the group center points (geometric mean of adjacent edges)
centers = np.sqrt(edges[:-1] * edges[1:])

radnrgdens_analytic = np.array([6.56128e06, 9.37744e06, 1.33706e07, 1.90091e07, 2.69304e07,
                                3.79893e07, 5.33093e07, 7.43291e07, 1.02827e08, 1.40887e08,
                                1.90765e08, 2.54559e08, 3.33604e08, 4.27461e08, 5.32466e08,
                                6.40012e08, 7.35205e08, 7.97306e08, 8.05117e08, 7.49915e08,
                                6.06784e08, 4.32675e08, 2.59951e08, 1.26955e08, 4.83345e07, 
                                1.36588e07, 2.70428e06, 3.50392e05, 2.75095e04, 2.85002e05, 
                                3.91809e07, 9.85459e08, 6.33419e09, 1.64600e10, 6.22834e09, 
                                1.00872e09, 2.28836e09, 1.80809e09, 7.26508e08, 1.64067e08, 
                                2.01749e07, 1.34876e06, 4.44994e04, 6.33976e02, 3.49212e00, 
                                6.12484e-03, 2.66266e-06, 2.44078e-10, 3.13420e-15, 3.80253e-21])
radnrgdens_imc = np.array([0.00000e00, 0.00000e00, 6.08837e06, 2.21914e07, 0.00000e00,
                           3.39862e07, 9.77224e07, 5.35097e07, 1.28744e08, 9.27517e07,
                           1.57907e08, 2.83336e08, 3.82734e08, 5.08110e08, 4.12233e08,
                           4.44628e08, 9.17212e08, 8.23536e08, 6.79309e08, 9.75612e08,
                           8.02509e08, 3.78322e08, 2.77227e08, 1.14425e08, 5.79367e07,
                           1.37617e07, 2.32763e06, 0.00000e00, 0.00000e00, 0.00000e00,
                           3.22636e07, 9.34736e08, 6.30907e09, 1.64366e10, 6.11433e09,
                           9.91868e08, 2.36965e09, 1.77814e09, 7.47867e08, 1.37272e08,
                           1.77538e07, 7.71867e05, 0.00000e00, 0.00000e00, 0.00000e00,
                           0.00000e00, 0.00000e00, 0.00000e00, 0.00000e00, 0.00000e00])



# Open the output file
fname = open("graziani_slab.out", "rb")

# Times corresponding to data in the output file

time_line = fname.readline().decode().strip()  # Read the time line
radnrgdens_dpt = pickle.load(fname)
radnrgdens_dpt = radnrgdens_dpt[9, :]
print(f'{radnrgdens_dpt}')
# Create figure for the material temperature for DPT

plt.figure()
plt.plot(centers, radnrgdens_analytic, marker='o', label='Analytic', fillstyle='none', linewidth=0.6, color='b')
plt.plot(centers, radnrgdens_imc, marker='x', label='IMC', linewidth=0.6, color='r')
plt.legend()
plt.xscale('log')
plt.yscale('log')
# plt.xlim(1e-3, 10)
plt.ylim(1e4,1e11)
plt.xlabel("Frequency (keV)")
plt.ylabel("Radiation Energy Density (erg/cm³-keV)")
plt.title("Slab spectrum")
plt.savefig('graziani_slab_imc.png', dpi=600)
plt.close()

plt.figure()
plt.plot(centers, radnrgdens_analytic, marker='o', label='Analytic', fillstyle='none', linewidth=0.6, color='b')
plt.plot(centers, radnrgdens_dpt, marker='+', label='DPT', linewidth=0.6, color='r')
plt.legend()
plt.xscale('log')
plt.yscale('log')
# plt.xlim(1e-3, 10)
plt.ylim(1e4,1e11)
plt.xlabel("Frequency (keV)")
plt.ylabel("Radiation Energy Density (erg/cm³-keV)")
plt.title("Slab spectrum")
plt.savefig('graziani_slab_dpt.png', dpi=600)
plt.close()

print(f'sum of imc radnrgdens = {np.sum(radnrgdens_imc)}')
print(f'sum of dpt radnrgdens = {np.sum(radnrgdens_dpt)}')
print(f'sum of analytic radnrgdens = {np.sum(radnrgdens_analytic)}')
# Close the file
fname.close()