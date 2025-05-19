import numpy as np
import matplotlib.pyplot as plt

Nx = 4
Nmu = 4

# DPT
dpt_x_positions = (np.arange(Nx) + 0.5) / Nx
dpt_angles =  -1.0 + (np.arange(Nmu) + 0.5) * 2 / Nmu
dpt_particles = np.array([(x, mu) for x in dpt_x_positions for mu in dpt_angles])

        
# Extract x and mu values for plotting
x_values = dpt_particles[:, 0]
mu_values = dpt_particles[:, 1]

plt.figure()
plt.title('Population Control Routine')
plt.xlabel('x')
plt.ylabel(r'$\mu$')
plt.xlim(0,1.0)
plt.ylim(-1.0, 1.0)
plt.plot(x_values, mu_values,'bo')
plt.plot(0.2, -0.75, 'r.')
plt.plot(0.55,-0.1, 'r.')
plt.show()
