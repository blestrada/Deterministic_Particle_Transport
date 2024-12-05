"""Tally end-of-timestep quantities"""

import numpy as np
from numba import njit, objmode

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_part_data as part
import imc_global_volsource_data as vol

@njit
def SuOlson_tally(nrgdep, n_particles, particle_prop, matnrgdens, temp):
    """Tally end of timestep quantities """
    source_cells = int((np.ceil(vol.x_0 / mesh.dx)))
    # start-of-step radiation energy density
    radenergydens = np.zeros(mesh.ncells)
    radenergydens[:] = phys.a * temp[:] ** 4 # keV/cm^3


    sourceenergy = np.zeros(mesh.ncells)
    sourceenergy[0:source_cells] = 1.0
    
    # Temperature increase
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = (nrgdep[:] / mesh.dx) - (mesh.sigma_a[:] * mesh.fleck[:] * radenergydens[:] * phys.c * time.dt) 
    matnrgdens[:] = matnrgdens[:] + nrg_inc[:]

    for i in range(mesh.ncells):
        if matnrgdens[i] < 0:
            matnrgdens[i] = 0
        
    # Calculate new temperature
    temp = np.zeros(mesh.ncells, dtype=np.float64)
    temp[:] = matnrgdens[:] ** (1/4)

    # Calculate end-of-step radiation energy density
    radnrgdens = np.zeros(mesh.ncells)

    for i in range(n_particles[0]):
        particle = particle_prop[i]  
        nrg = particle[5]
        if nrg >= 0.0:
            cell_index = int(particle[2])  # The cell index where the particle resides
            radnrgdens[cell_index] += nrg / mesh.dx  # Update the energy density in the corresponding cell

    with objmode:
        print(f'Material Energy Density = {matnrgdens[:10]}')
        print(f'Radiation Energy Density = {radnrgdens[:10]}')
        print(f'Temperature = {temp[:10]}')
    return matnrgdens, radnrgdens, temp
    


def marshak_wave_tally():
    """Tally end of timestep quantities """

    print(f'The total energy deposited this time-step = {np.sum(mesh.nrgdep)}')


    # start-of-step material energy
    matnrg = np.zeros(mesh.ncells)
    matnrg[:] = mat.rho * mat.b[:] * mesh.temp[:]
    print(f'start of step material energy = {matnrg[:10]}')

    # Radiation energy density
    radenergydens = np.zeros(mesh.ncells)
    radenergydens[:] = phys.a * mesh.temp[:] ** 4 # keV/cm^3
    print(f'start of step radiation energy = {radenergydens[:10]}')


    # Energy increase
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = (mesh.nrgdep[:] / mesh.dx) - (mesh.sigma_a[:] * mesh.fleck[:] * radenergydens[:] * phys.c * time.dt) 
  
    mesh.matnrgdens[:] = mesh.matnrgdens[:] + nrg_inc[:]
    print(f'end of step matnrgdens = {mesh.matnrgdens[:10]}')

    # Material temperature update
    mesh.temp[:] = mesh.temp[:] + nrg_inc[:] / mat.b[:]

    print(f'mesh.temp = {mesh.temp[:10]}')
    print(f'mesh.temp last 10 = {mesh.temp[-10:]}')
    # Save radiation energy
    mesh.radnrgdens = np.zeros(mesh.ncells)
    for particle in part.particle_prop:
        cell_index = particle[2]
        mesh.radnrgdens[cell_index] += particle[5] / mesh.dx

    mesh.radtemp = (mesh.radnrgdens / phys.a) ** (1/4)

    print(f'end of step mesh.radnrgdens = {mesh.radnrgdens[:10]}')
    print(f'mesh.radtemp = {mesh.radtemp[:10]}')