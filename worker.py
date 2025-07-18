# worker.py

import numpy as np
from scipy.integrate import dblquad
from numba import njit
epsilon = 1.0
c_a = 0.5
c_s = 0.5
x_0 = 0.5
tau_0 = 10.0
@njit
def fW_numba(k, w, x, tau):
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

@njit
def fV_numba(k, w, x, tau):
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

# ------------------------------
# Step 2: Python wrappers for dblquad
# ------------------------------

def fW_wrapper(k, w, x, tau):
    return fW_numba(k, w, x, tau)

def fV_wrapper(k, w, x, tau):
    return fV_numba(k, w, x, tau)

# ------------------------------
# Step 3: Function to compute W and V
# ------------------------------

def compute_WV_dblquad(args):
    x, tau = args  # k_min, k_max, etc. no longer needed
    W, _ = dblquad(fW_wrapper, 0, np.inf, lambda k: 0, lambda k: np.inf, args=(x, tau))
    V_integral, _ = dblquad(fV_wrapper, 0, np.inf, lambda k: 0, lambda k: np.inf, args=(x, tau))
    V = W - V_integral
    return (x, tau, W, V)
