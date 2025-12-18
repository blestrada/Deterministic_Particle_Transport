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


def crooked_pipe_tally(nrgdep, 
                       mesh_z_edges, mesh_r_edges, 
                       n_particles, particle_prop, 
                       temp, mesh_sigma_a, mesh_fleck, dt):
    # z corresponds to your x-axis, r corresponds to your y-axis
    num_z_cells = len(mesh_z_edges) - 1
    num_r_cells = len(mesh_r_edges) - 1

    # 1. Calculate RZ Volumes: V = pi * (r_outer^2 - r_inner^2) * delta_z
    dz = np.diff(mesh_z_edges) # shape (num_z_cells,)
    r_inner = mesh_r_edges[:-1]
    r_outer = mesh_r_edges[1:]
    dr2 = r_outer**2 - r_inner**2 # shape (num_r_cells,)
    
    # Create 2D volume array using outer product
    # volumes[z_idx, r_idx]
    volumes = np.pi * np.outer(dz, dr2)

    # 2. Start of step radiation energy density
    radenergydens_old = phys.a * temp ** 4 
    
    if np.any(nrgdep < 0.0):
        raise RuntimeError(f"Negative energy deposited detected! min(nrgdep) = {np.min(nrgdep)}")
    
    # 3. Energy increase per cell (using Volumetric Tally)
    # Energy per unit volume = (Total Energy) / Volume
    # nrg_inc is d(InternalEnergy)/dt integrated over dt
    nrg_inc = (nrgdep / volumes) - (mesh_sigma_a * mesh_fleck * radenergydens_old * phys.c * dt)

    # 4. Material temperature update
    # Note: Ensure mat.b (Cv) is in units of keV / (cm^3 * keV_temp)
    temp = temp + (nrg_inc / mat.b)
    
    # 5. Calculate the Census Radiation Temperature
    # We sum the energies of particles currently in flight (census)
    radnrgdens_census = np.zeros((num_z_cells, num_r_cells))
    
    for i in range(n_particles):
        p = particle_prop[i]
        nrg = p[8]
        if nrg > 0.0:
            z_idx = int(p[1])
            r_idx = int(p[2])
            # Check bounds to prevent indexing errors at census
            if 0 <= z_idx < num_z_cells and 0 <= r_idx < num_r_cells:
                radnrgdens_census[z_idx, r_idx] += nrg

    # Convert total census energy to density
    radnrgdens_census /= volumes

    # Avoid zero/negative before power of 1/4
    radtemp = (np.maximum(radnrgdens_census, 0.0) / phys.a) ** 0.25
    
    return temp, radtemp