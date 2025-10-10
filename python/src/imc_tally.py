"""Tally end-of-timestep quantities"""

import numpy as np
from numba import njit, objmode

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_part_data as part

@njit
def SuOlson_tally(nrgdep, n_particles, particle_prop, matnrgdens, temp):
    """Tally end of timestep quantities """
    # start-of-step radiation energy density
    radenergydens = np.zeros(mesh.ncells)
    radenergydens[:] = phys.a * temp[:] ** 4 # keV/cm^3
    
    # Temperature increase
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = (nrgdep[:] / mesh.dx) - (mesh.sigma_a[:] * mesh.fleck[:] * radenergydens[:] * phys.c * time.dt) 
    # print(f'nrg_inc = {nrg_inc}')
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
        nrg = particle[6]
        if nrg >= 0.0:
            cell_index = int(particle[2])  # The cell index where the particle resides
            radnrgdens[cell_index] += nrg / mesh.dx  # Update the energy density in the corresponding cell

    with objmode:
        print(f'Material Energy Density = {matnrgdens[:10]}')
        print(f'Radiation Energy Density = {radnrgdens[:10]}')
        print(f'Temperature = {temp[:10]}')
    return matnrgdens, radnrgdens, temp
    

@njit
def marshak_wave_tally(nrgdep, n_particles, particle_prop, matnrgdens, temp, sigma_a, fleck, dt):
    """Tally end of timestep quantities """

    # start-of-step material energy
    matnrg = np.zeros(mesh.ncells)
    matnrg[:] = mat.rho * mat.b[:] * temp[:]
    # with objmode:
    #     print(f'matnrg = {matnrg[:10]}')

    # Radiation energy density
    radenergydens = np.zeros(mesh.ncells)
    radenergydens[:] = phys.a * temp[:] ** 4 # keV/cm^3
    # with objmode:
    #     print(f'Start of step radiation energy density = {radenergydens[:10]}')

    # Energy increase
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = (nrgdep[:] / mesh.dx) - (sigma_a[:] * fleck[:] * radenergydens[:] * phys.c * dt)
    # with objmode:
    #     print(f'Energy increase = {nrg_inc[:10]}') 
    matnrgdens[:] = matnrgdens[:] + nrg_inc[:]

    # Material temperature update
    temp[:] = temp[:] + nrg_inc[:] / mat.b[:]

    # Calculate end-of-step radiation energy density
    radnrgdens = np.zeros(mesh.ncells)

    for i in range(n_particles):
        particle = particle_prop[i]  
        nrg = particle[6]
        if nrg >= 0.0:
            cell_index = int(particle[2])  # The cell index where the particle resides
            radnrgdens[cell_index] += nrg / mesh.dx  # Update the energy density in the corresponding cell

    with objmode:
        print(f'Material Energy Density = {matnrgdens[:10]}')
        print(f'Radiation Energy Density = {radnrgdens[:10]}')
        print(f'Temperature = {temp[:10]}')
        # print(f'Temperature last 10 = {temp[-10:]}')
    return matnrgdens, radnrgdens, temp

@njit
def graziani_slab_tally(n_particles, particle_prop):
    """Tally end-of-step quantities"""

    # # Sum energy deposition over all frequency groups in each cell
    # nrg_inc = np.zeros(mesh.ncells)
    # nrg_inc[:] = (np.sum(nrgdep, axis=1) / mesh.dx) - (sigma_p * phys.a * temp ** 4 * phys.c * dt)
    # with objmode:
    #     print(f'nrginc = {nrg_inc}')
    #     print(f'old temp = {temp}')
    # # Material temperature update
    
    # temp[:] = temp[:] + nrg_inc[:] / mat.b[:]
    # with objmode:
    #     print(f'new temp = {temp}')
    # 50 groups, logarithmically spaced between 3.0 × 10−3 keV and 30.0 keV
    # Define the energy range
    E_min = 3.0e-3  # keV
    E_max = 30.0  # keV
    # Generate logarithmically spaced edges
    edges = np.logspace(np.log10(E_min), np.log10(E_max), 50 + 1)
    # Calculate end-of-step radiation energy density
    radnrgdens = np.zeros((mesh.ncells, part.Ng), dtype=np.float64)

    for i in range(n_particles[0]):
        particle = particle_prop[i]  
        nrg = particle[6]
        if nrg >= 0.0:
            cell_index = int(particle[2])
            frq_index = int(particle[5])
            radnrgdens[cell_index, frq_index] += nrg #/ mesh.dx / (edges[frq_index + 1]-edges[frq_index])

    return radnrgdens


def general_tally(nrgdep, n_particles, particle_prop, temp):
    # Start-of-step radiation energy density
    radenergydens = np.zeros(mesh.ncells)
    radenergydens[:] = phys.a * mesh.temp[:] ** 4 

    # Energy increase
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = (nrgdep / mesh.dx) - (mesh.sigma_a[:] * mesh.fleck[:] * radenergydens[:] * phys.c * time.dt)

    # Material temperature update
    temp[:] = temp[:] + nrg_inc[:] / mat.b[:]
    
    # Radiation temperature
    radnrgdens = np.zeros(mesh.ncells)
    for i in range(n_particles[0]):
        particle = particle_prop[i]  
        nrg = particle[6]
        if nrg >= 0.0:
            cell_index = int(particle[2])  # The cell index where the particle resides
            radnrgdens[cell_index] += nrg / mesh.dx  # Update the energy density in the corresponding cell

    radtemp = (radnrgdens / phys.a) ** (1/4)

    return temp, radtemp


def crooked_pipe_tally(nrgdep, mesh_dx, mesh_dy, n_particles, particle_prop, temp, mesh_sigma_a, mesh_fleck, dt):
    num_x_cells = len(mesh_dx)
    num_y_cells = len(mesh_dy)

    # 2D area array
    area = np.outer(mesh_dx, mesh_dy)  # shape: (num_x_cells, num_y_cells)

    # Start of step radiation energy density
    radenergydens = phys.a * temp ** 4  # keV/cm^3
    if np.any(nrgdep < 0.0):
        raise RuntimeError(f"Negative energy deposited detected! min(nrgdep) = {np.min(nrgdep)}")
    
    # Energy increase per cell
    nrg_inc = (nrgdep / area) - (mesh_sigma_a * mesh_fleck * radenergydens * phys.c * dt)

    # Material temperature update
    temp = temp + nrg_inc / mat.b
    if np.any(temp < 0.0):
        raise RuntimeError(f"Negative temperature detected! min(temp) = {np.min(temp)}")
    
    # Calculate the radiation temperature
    radnrgdens = np.zeros((num_x_cells, num_y_cells))
    for i in range(n_particles):
        particle = particle_prop[i]
        nrg = particle[7]
        if nrg >= 0.0:
            x_cell_index = int(particle[1])
            y_cell_index = int(particle[2])
            radnrgdens[x_cell_index, y_cell_index] += nrg / (mesh_dx[x_cell_index] * mesh_dy[y_cell_index])  # Update the energy density in the corresponding cell

    radtemp = (radnrgdens / phys.a) ** (1/4)
    
    return temp, radtemp