"""Tally end-of-timestep quantities"""

import numpy as np

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_part_data as part



def run():
    """Tally end-of-timestep quantities."""

    # Calculate the energy emitted by the body during the time-step
    nrg_emitted = mesh.sigma_a * mesh.fleck * phys.c * mesh.dx * time.dt * mesh.temp ** 4
    print(f' nrg emitted = {nrg_emitted[:10]}')

    # Calculate the energy increase in the matter
    nrg_inc = mesh.nrgdep / mesh.dx - mesh.fleck * mesh.sigma_a * mesh.radnrgdens * phys.c * time.dt
    print(f' nrg_inc = {nrg_inc[:10]}')

    # Calculate the material energy density
    mesh.matnrgdens = mesh.matnrgdens + nrg_inc

    # Make sure no matnrgdens value is less than 0.
    for j in range(len(mesh.matnrgdens)):
        if mesh.matnrgdens[j] < 0:
            mesh.matnrgdens[j] = 0.0
    
    # Update Material temperature
    mesh.temp = mesh.matnrgdens ** (1/4)

    # Calculate the radiation energy density
    mesh.radnrgdens = np.zeros(mesh.ncells)
    for particle in part.particle_prop:
        cell_index = particle[2]
        mesh.radnrgdens[cell_index] += particle[5] / mesh.dx
    
    # Update Radiation Temperature
    mesh.radtemp = mesh.radnrgdens ** (1/4)

    print(f'mesh.matnrgdens = {mesh.matnrgdens[:10]}')
    print(f'mesh.radnrgdens = {mesh.radnrgdens[:10]}')

    print(f"Material temperature: {mesh.temp[:10]}")
    print(f"Radiation temperature: {mesh.radtemp[:10]}")

