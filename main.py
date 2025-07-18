# main.py

import numpy as np
import multiprocessing as mp
from worker import compute_WV_dblquad

total_tasks = 0  # global variable

def compute_with_progress(idx_task):
    idx, args = idx_task
    x, tau = args
    try:
        result = compute_WV_dblquad(args)
        W, V = result[2], result[3]
    except ZeroDivisionError:
        print(f"ZeroDivisionError at x={x}, tau={tau} — setting W and V to NaN.")
        W, V = np.nan, np.nan
    print(f"Finished {idx + 1} / {total_tasks}: x={x:.4f}, tau={tau:.4f}")
    return x, tau, W, V

def run_parallel_dblquad(x_values, tau_values):
    global total_tasks  # allow access in compute_with_progress
    tasks = [(x, tau) for x in x_values for tau in tau_values]
    total_tasks = len(tasks)
    indexed_tasks = list(enumerate(tasks))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(compute_with_progress, indexed_tasks)

    W_array = np.zeros((len(x_values), len(tau_values)))
    V_array = np.zeros((len(x_values), len(tau_values)))

    for x, tau, W, V in results:
        i = np.where(np.isclose(x_values, x))[0][0]
        j = np.where(np.isclose(tau_values, tau))[0][0]
        W_array[i, j] = W
        V_array[i, j] = V

    return W_array, V_array


if __name__ == "__main__":
    x_vals = 0.025 + 0.05 * np.arange(100)
    tau_vals = np.linspace(0.01, 10.0, 1000)

    print("Running parallel integration...")
    W, V = run_parallel_dblquad(x_vals, tau_vals)
    print("Done.")

    np.savez("WV_output.npz", x_vals=x_vals, tau_vals=tau_vals, W=W, V=V)
    print("Results saved to 'WV_output.npz'")