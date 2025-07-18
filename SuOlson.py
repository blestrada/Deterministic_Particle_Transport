import numpy as np
import scipy.integrate as integrate
import time
from numba import njit


# Problem parameters (set defaults, override as needed)
epsilon = 1.0
c_a = 1.0
c_s = 1.0 - c_a
tau_0 = 10.0
x_0 = 0.5

# Radiation Energy Density
@njit
def fW(k, w, x, tau):
    P = (np.arctan(k + epsilon * w) + np.arctan(k - epsilon * w)) / (2 * k)
    Q = np.arctanh((2 * epsilon * k * w) / (1 + (epsilon * w)**2 + k**2)) / (2 * k)
    R = (c_a**2 + (c_s * w)**2) * (P**2 + Q**2) + 2 * (c_a**2) * w * Q - 2 * (c_a**2 + c_s * w**2) * P + w**2 + c_a**2
    Q_1 = 1 if x_0 == 0 else np.sin(k * x_0) / (k * x_0)
    h = Q_1 * ((c_a**2 + w**2) * P - (c_a**2 + c_s * w**2) * (P**2 + Q**2)) / R
    g = Q_1 * ((c_a**2 + w**2) * Q + (c_a**2) * w * (P**2 + Q**2)) / R

    if tau <= tau_0:
        return np.cos(k * x) * (h * np.sin(w * tau) + g * (1 - np.cos(w * tau))) / (w * np.pi**2)
    else:
        return np.cos(k * x) * (
            h * (np.sin(w * tau) - np.sin(w * (tau - tau_0))) +
            g * (np.cos(w * (tau - tau_0)) - np.cos(w * tau))
        ) / (w * np.pi**2)

# Material Energy Density
@njit
def fV(k, w, x, tau):
    P = (np.arctan(k + epsilon * w) + np.arctan(k - epsilon * w)) / (2 * k)
    Q = np.arctanh((2 * epsilon * k * w) / (1 + (epsilon * w)**2 + k**2)) / (2 * k)
    R = (c_a**2 + (c_s * w)**2) * (P**2 + Q**2) + 2 * (c_a**2) * w * Q - 2 * (c_a**2 + c_s * w**2) * P + w**2 + c_a**2
    Q_1 = 1 if x_0 == 0 else np.sin(k * x_0) / (k * x_0)
    h = Q_1 * ((c_a**2 + w**2) * P - (c_a**2 + c_s * w**2) * (P**2 + Q**2)) / R
    g = Q_1 * ((c_a**2 + w**2) * Q + (c_a**2) * w * (P**2 + Q**2)) / R

    if tau <= tau_0:
        return np.cos(k * x) * (
            (w * h + c_a * g) * np.sin(w * tau) +
            (c_a * h - w * g) * (np.cos(w * tau) - np.exp(-c_a * tau))
        ) / ((c_a**2 + w**2) * np.pi**2)
    else:
        return np.cos(k * x) * (
            (w * h + c_a * g) * (np.sin(w * tau) - np.sin(w * (tau - tau_0))) +
            (c_a * h - w * g) * (np.cos(w * tau) - np.cos(w * (tau - tau_0)) +
                                 np.exp(-c_a * (tau - tau_0)) - np.exp(-c_a * tau))
        ) / ((c_a**2 + w**2) * np.pi**2)

# Compute and save W and V
def run_custom_grid(x_values, tau_values, epsilon_in, c_a_in, x_0_in, tau_0_in, output_filename="benchmark_results.npz"):
    global epsilon, c_a, c_s, x_0, tau_0
    epsilon = epsilon_in
    c_a = c_a_in
    c_s = 1.0 - c_a
    x_0 = x_0_in
    tau_0 = tau_0_in

    W_array = np.zeros((len(x_values), len(tau_values)))
    V_array = np.zeros((len(x_values), len(tau_values)))


    print(f"Running benchmark for {len(x_values)} x-values and {len(tau_values)} tau-values...")
    tic = time.perf_counter()

    for i, x in enumerate(x_values):
        for j, tau in enumerate(tau_values):
            try:
                W = integrate.dblquad(lambda k, w: fW(k, w, x, tau), 0, np.inf, lambda k: 0, lambda k: np.inf)[0]
                V = W - integrate.dblquad(lambda k, w: fV(k, w, x, tau), 0, np.inf, lambda k: 0, lambda k: np.inf)[0]
                W_array[i, j] = W
                V_array[i, j] = V
            except ZeroDivisionError:
                print(f"Skipped (x={x:.4f}, tau={tau:.4f}) due to division by zero")
                W_array[i, j] = np.nan
                V_array[i, j] = np.nan
            print(f"W(x={x:.7f}, τ={tau:.5f}) = {W:.5f}, V = {V:.5f}")
    toc = time.perf_counter()
    print(f"Finished calculations in {toc - tic:.2f} seconds")

    np.savez(output_filename, x_values=x_values, tau_values=tau_values, W=W_array, V=V_array)
    print(f"Results saved to {output_filename}")

# ----------------------------------------------------------------------------------

# Define the spatial and time points where you want to evaluate W and V
# Define x_values as 
x_values = np.linspace(0.025, 5.00, 10)
#x_values = np.array([0.01, 0.1, 0.17783, 0.31623])
tau_values = np.linspace(0.01, 10.0, 10)
# tau_values = np.array([10.0])

# Parameters (can override the defaults)
epsilon = 1.0
c_a = 0.5
x_0 = 0.5
tau_0 = 10.0

# Run the benchmark calculation and save results
run_custom_grid(x_values, tau_values, epsilon, c_a, x_0, tau_0, output_filename="benchmark_results.npz")

