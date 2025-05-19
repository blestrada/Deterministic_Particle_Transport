"""Make plot of opacity data for Graziani Slab"""

import numpy as np
import matplotlib.pyplot as plt


# Set up frequency group structure
# 50 groups, logarithmically spaced between 3.0 × 10−3 keV and 30.0 keV
# Define the energy range
E_min = 3.0e-3  # keV
E_max = 30.0  # keV
Ng = 50  # Number of frequency groups

# Generate logarithmically spaced edges
edges = np.logspace(np.log10(E_min), np.log10(E_max), Ng + 1)
print(f'edges = {edges}')

# Compute the group center points (geometric mean of adjacent edges)
centers = np.sqrt(edges[:-1] * edges[1:])
print(f'centers = {centers}')

# Group Opacities (given)
sigma_g = np.array([9.16000e04, 9.06781e04, 6.08939e04, 4.08607e04, 2.72149e04, 
                    1.86425e04, 1.24389e04, 8.19288e03, 5.79710e03, 5.14390e03, 
                    5.20350e03, 8.69569e03, 6.67314e03, 4.15912e03, 2.62038e03, 
                    1.64328e03, 1.01613e03, 6.19069e02, 3.75748e02, 2.97349e02, 
                    8.21172e02, 4.01655e03, 4.54828e03, 3.50487e03, 3.02359e03, 
                    4.34203e03, 2.98594e03, 1.55364e03, 9.42213e02, 5.76390e02, 
                    3.52953e02, 2.09882e02, 1.26546e02, 7.80087e01, 9.97421e01, 
                    1.48848e02, 8.22907e01, 4.86915e01, 2.91258e01, 1.68133e01, 
                    9.92194e00, 5.18722e00, 2.24699e00, 1.29604e00, 7.46975e-01, 
                    8.43058e-01, 2.43746e00, 1.50509e00, 9.01762e-01, 5.38182e-01])

plt.figure()
plt.plot(centers, sigma_g, marker='+', label='CHBr Opacity', color='k', linewidth=0.7)
plt.ylabel(r'Opacity ($\frac{1}{\text{cm}}$)')
plt.xlabel(r'Frequency (keV)')
plt.xlim(0.001, 100)
plt.ylim(0.1, 100000)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig('graziani_opacity.png', dpi=900)
plt.close()
