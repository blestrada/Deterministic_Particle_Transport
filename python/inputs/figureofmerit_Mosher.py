import matplotlib.pyplot as plt
import numpy as np

# Load Mosher Analytic
benchmark = np.load("Mosher_analytic.npz")
times = benchmark["times"]
analytic_mat_temp = benchmark["analytic_material_temps"]

# Load IMC and DPT sim data
imc = np.load("Mosher_output_IMC.npz")
dpt = np.load("Mosher_output_DPT.npz")

temps_imc = imc["mat_temps"]
runtimes_imc = imc["runtimes"]

temps_dpt = dpt["mat_temps"]
runtimes_dpt = dpt["runtimes"]

Nt = len(times)

# Initialize error and FOM arrays
l2_error_imc = np.zeros(Nt)
l2_error_dpt = np.zeros(Nt)

fom_l2_imc = np.zeros(Nt)
fom_l2_dpt = np.zeros(Nt)

# Error metric functions
def rms_relative_error(sim, ref):
    mask = ~np.isnan(ref)
    return np.sqrt(np.mean(((sim[mask] - ref[mask]) / ref[mask]) ** 2))

def l2_relative_error(sim, ref):
    mask = ~np.isnan(ref)
    return np.linalg.norm(sim[mask] - ref[mask]) / np.linalg.norm(ref[mask])

# Compute errors and FOMs
for t in range(Nt):
    T_analytic = analytic_mat_temp[t]
    T_imc = temps_imc[t]
    T_dpt = temps_dpt[t]
    # print(f'T_analytic = {T_analytic}')
    l2_error_imc[t] = l2_relative_error(T_imc, T_analytic) ** 2
    l2_error_dpt[t] = l2_relative_error(T_dpt, T_analytic) ** 2

    # Avoid division by zero
    if l2_error_imc[t] > 0 and runtimes_imc[t] > 0:
        fom_l2_imc[t] = 1.0 / (l2_error_imc[t] * runtimes_imc[t])
    if l2_error_dpt[t] > 0 and runtimes_dpt[t] > 0:
        fom_l2_dpt[t] = 1.0 / (l2_error_dpt[t] * runtimes_dpt[t])


# Plot L2 Error over Time
plt.figure()
plt.plot(times, l2_error_imc, marker='x', label='IMC', color='r')
plt.plot(times, l2_error_dpt, marker='+', label='DPT', color='b')
plt.xlabel('Time (s)')
plt.xlim((0.0, 15.5))
plt.ylabel(r'Relative $L^2$ Norm of Material Temperature')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig('mosher_Error_over_time.png', dpi=900)
plt.show()


# Plot FOM (L2-based)
plt.figure()
plt.plot(times, fom_l2_imc, 'x-', label='IMC', color='r')
plt.plot(times, fom_l2_dpt, '+-', label='DPT', color='b')
plt.xlabel('Time (s)')
plt.xlim((0.0, 15.5))
plt.ylabel('FOM')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mosher_FOM_over_time.png', dpi=900)
plt.show()
