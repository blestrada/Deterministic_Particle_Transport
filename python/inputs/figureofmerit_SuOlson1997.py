import matplotlib.pyplot as plt
import numpy as np

num_points = 100
# Load benchmark
benchmark = np.load("WV_output.npz")
x_vals = benchmark['x_vals'][:num_points]
tau_vals = benchmark['tau_vals']
W = benchmark['W'].T[:, :num_points]  # Transpose to (Nt, Nx)

# Load IMC and DPT simulation outputs
imc = np.load("Su_Olson_output_IMC.npz")
dpt = np.load("Su_Olson_output_DPT.npz")

x_imc = imc['mesh_nodepos'][:num_points]
times_imc = imc['time']
rad_imc = imc['radnrgdens'][:, :num_points]
runtime_imc = imc['runtimes']

x_dpt = dpt['mesh_nodepos'][:num_points]
times_dpt = dpt['time']
rad_dpt = dpt['radnrgdens'][:, :num_points]
runtime_dpt = dpt['runtimes']

# Sanity checks
assert np.allclose(tau_vals, times_imc)
assert np.allclose(tau_vals, times_dpt)
assert rad_imc.shape[0] == len(runtime_imc)
assert rad_dpt.shape[0] == len(runtime_dpt)

Nt = len(tau_vals)

# Initialize error and FOM arrays
l2_error_imc = np.zeros(Nt)
l2_error_dpt = np.zeros(Nt)

fom_l2_imc = np.zeros(Nt)
fom_l2_dpt = np.zeros(Nt)

def l2_relative_error(sim, ref):
    mask = ~np.isnan(ref)
    return np.linalg.norm(sim[mask] - ref[mask]) / np.linalg.norm(ref[mask])

# Compute errors and FOMs
for t in range(Nt):
    W_ref = W[t, :]
    W_imc = rad_imc[t, :]
    W_dpt = rad_dpt[t, :]

    l2_error_imc[t] = l2_relative_error(W_imc, W_ref) ** 2
    l2_error_dpt[t] = l2_relative_error(W_dpt, W_ref) ** 2

    # Avoid division by zero
    if l2_error_imc[t] > 0 and runtime_imc[t] > 0:
        fom_l2_imc[t] = 1.0 / (l2_error_imc[t] * runtime_imc[t])
    if l2_error_dpt[t] > 0 and runtime_dpt[t] > 0:
        fom_l2_dpt[t] = 1.0 / (l2_error_dpt[t] * runtime_dpt[t])


# Plot L2 Error over Time
plt.figure()
plt.plot(tau_vals, l2_error_imc, marker='x', label='IMC', color='r')
plt.plot(tau_vals, l2_error_dpt, marker='+', label='DPT', color='b')
plt.xlabel('Time (τ)')
plt.ylabel(r'Relative $L^2$ Norm of Radiation Energy Density')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig('SuOlson_Error_over_time.png', dpi=900)
plt.show()


# Plot FOM (L2-based)
plt.figure()
plt.plot(tau_vals, fom_l2_imc, 'x-', label='IMC', color='r')
plt.plot(tau_vals, fom_l2_dpt, '+-', label='DPT', color='b')
plt.xlabel('Time (τ)')
plt.ylabel('FOM')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('SuOlson_FOM_over_time.png', dpi=900)
plt.show()

# Subsample every 50 time values
step = 50
tau_sub = tau_vals[::step]
l2_error_imc_sub = l2_error_imc[::step]
l2_error_dpt_sub = l2_error_dpt[::step]
fom_l2_imc_sub = fom_l2_imc[::step]
fom_l2_dpt_sub = fom_l2_dpt[::step]

# Plot L2 Error over Time
plt.figure()
plt.plot(tau_sub, l2_error_imc_sub, marker='x', label='IMC', color='r')
plt.plot(tau_sub, l2_error_dpt_sub, marker='+', label='DPT', color='b')
plt.xlabel('Time (τ)')
plt.ylabel(r'Relative $L^2$ Norm of Radiation Energy Density')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig('SuOlson_Error_over_time_subsampled.png', dpi=900)
plt.show()

# Plot FOM (L2-based)
plt.figure()
plt.plot(tau_sub, fom_l2_imc_sub, 'x-', label='IMC', color='r')
plt.plot(tau_sub, fom_l2_dpt_sub, '+-', label='DPT', color='b')
plt.xlabel('Time (τ)')
plt.ylabel('FOM')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('SuOlson_FOM_over_time_subsampled.png', dpi=900)
plt.show()