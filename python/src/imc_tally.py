"""Tally end-of-timestep quantities"""

import numpy as np

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_part_data as part

def run():
    """Tally end-of-timestep quantities."""
    print("\n" + "-" * 79)
    print("Tally step ({:4d})".format(time.step))
    print("-" * 79)

    # Start-of-step radiation energy density
    radnrgdens = np.zeros(mesh.ncells)
    radnrgdens[:] = phys.a * mesh.radtemp[:] ** 4

    # Temperature increase
    nrg_emitted = phys.c * mesh.sigma_a * phys.a *  mesh.temp ** 4 * time.dt * mesh.dx  
    #print(f' nrg emitted = {nrg_emitted}')
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = mesh.nrgdep[:] - nrg_emitted[:]
    #print(f' nrginc = {nrg_inc}')

    # Update radiation temperature
    actual_energy_remaining = np.zeros(mesh.ncells)
    for particle in part.particle_prop:
        cell_index = particle[2]
        actual_energy_remaining[cell_index] += particle[6]
    mesh.temp[:] += mat.b ** -1 * nrg_inc[:]
    mesh.radtemp[:] = (actual_energy_remaining / phys.a / mesh.dx) ** (1 / 4)

    print("\nMaterial temperature:")
    print(mesh.temp)
    print("\nRadiation temperature:")
    print(mesh.radtemp)


