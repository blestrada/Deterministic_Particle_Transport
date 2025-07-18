import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Initial Parameters
sigma = 1.0
T_m = 0.001
T_r = 3.0
a = 1.0
c = 1.0
c_v = 50.0
U0 = c_v * T_m
E0 = a * T_r ** 4

# To summarize, given the material heat capacity Cv and
# initial energy densities E0 and U0, the constants p, q, x,
# y, r1, and r2 can be calculated (in that order). The equi-
# librium material temperature, T∞, can then be calculated
# from Eq. (9). After calculating additional constants z4 and
# A1 through A4, the time t at which T(t) (bounded by T0
# and T∞) occurs can be calculated according to Eq. (12).

p = c_v / a
q = (E0 + U0) / a
x = ((p/4) ** 2 + np.sqrt((p/4)**4 + (q/3)**3)) ** (1/3)
y = x - q/(3*x)
r1 = np.sqrt(y/2)
r2 = np.sqrt(y**2 + q)

z1 = r1 + np.sqrt(r2 + r1**2) *1j
z2 = r1 - np.sqrt(r2 + r1**2) *1j
z3 = -r1 + np.sqrt(r2 - r1**2)
z4 = -r1 - np.sqrt(r2 - r1**2)

A1 = -r1 / (8 * r1**4 + r2**2)
A2 = (4 * r1**2 - r2) / (16 * r1**4 + 2 * r2**2)
A3 = 1 / (4 * z3**3 + 4 * r1 * r2)
A4 = 1 / (-4 * z3**3 + 16 * r1**3 - 20 * r1 * r2)

T_eq = z3
def integral_T(T):
    term1 = r1 * A1
    term2 = A2 * np.sqrt(r1**2 + r2) * np.arctan((T - r1) / np.sqrt(r1**2 + r2))
    term3 = 0.5 * A1 * np.log(np.abs(T**2 - 2*r1*T + 2*r1**2 + r2))
    term4 = A3 * np.log(np.abs(T - z3))
    term5 = A4 * np.log(np.abs(T - z4))
    return term1 + term2 + term3 + term4 + term5

#Solve for T at time t ---
def T_of_t(t):
    rhs = -a * c * sigma / c_v * t
    def f(T):
        return integral_T(T) - integral_T(T_m) - rhs
    epsilon = 1e-10  # small offset to prevent singularities
    sol = root_scalar(f, bracket=(T_m, z3 - epsilon), method='brentq')
    return sol.root

times = np.arange(0.5, 15.01, 0.5)
print(f'times = {times}')
# Allocate array for analytic temperatures
analytic_material_temps = np.zeros_like(times)

# Compute T(t) for each time
for i, t in enumerate(times):
    T = T_of_t(t)
    analytic_material_temps[i] = T
    print(f"T({t:.5f}) = {T:.6f}")

# Save the data to a file
np.savez("Mosher_analytic.npz", times=times, analytic_material_temps=analytic_material_temps)
