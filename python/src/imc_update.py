"""Update at start of time-step"""

from numba import njit, objmode
import matplotlib.pyplot as plt

import imc_global_phys_data as phys
import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_time_data as time
import imc_global_bcon_data as bcon
import imc_global_part_data as part
import numpy as np

@njit
def SuOlson_update(temp):
    """Update temperature-dependent quantities at start of time-step"""

    # Calculate new heat capacity
    b = np.zeros(mesh.ncells, dtype=np.float64)
    b[:] = mat.alpha * temp[:] ** 3
    with objmode:
        print(f'Heat Capacity = {b[:10]}')
    return b

@njit
def marshak_wave_update(temp, dt):
    """Update temperature-dependent quantities at start of time-step"""
    # Calculate beta
    beta = np.zeros(mesh.ncells)
    beta[:] = 4 * phys.a * temp[:] ** 3 / (1.0 * mat.b[:]) # rho = 1.0
    # print(f'mesh.beta = {beta[:10]}')
    
    # Calculate new opacity
    sigma_a = np.zeros(mesh.ncells)
    sigma_a[:] = 100.0 / (temp[:] ** 3)
    
    with objmode:
        print(f'mesh.sigma_a = {sigma_a[:10]}')
    # print(f'last 10 mesh.sigma_a = {sigma_a[-10:]}')

    # Calculate new fleck factor
    fleck = np.zeros(mesh.ncells)
    fleck[:] = 1.0 / (1.0 + beta[:] * sigma_a[:] * phys.c * dt)
    with objmode:
        print(f'mesh.fleck[0] = {fleck[0]}')

    # Calculate total opacity
    sigma_t = np.copy(sigma_a)
    # mesh.sigma_t[:] = mesh.sigma_a[:] * mesh.fleck[:] + (1.0 - mesh.fleck[:]) * mesh.sigma_a[:] + mesh.sigma_s[:]
    # print(f'mesh.sigma_t = {mesh.sigma_t[:10]}')
    return beta, sigma_a, sigma_t, fleck

@njit
def population_control(n_particles, particle_prop, current_time, Nmu):
    """Reduces the number of particles and consolidates energy in the census grid."""
    with objmode:
        print(f'Particle count before pop control = {n_particles[0]}')
        print(f'current time in the simulation = {current_time}')

    # Generate the grid of census particles (cell, position, angle, energy)
    census_particles = np.zeros((part.max_array_size, 8), dtype=np.float64)
    n_census_ptcls = 0

    for icell in range(mesh.ncells):
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1 + (np.arange(Nmu[icell]) + 0.5) * 2 / Nmu[icell]
        origin = icell

        for xpos in x_positions:
            for mu in angles:
                census_particles[n_census_ptcls] = [origin, current_time, icell, xpos, mu, 0, 0, 0]
                n_census_ptcls += 1

    # Compute energy before population control
    nrgprepopctrl = np.sum(particle_prop[:n_particles[0], 6])
    with objmode:
        print(f'Energy in the particles pre population control = {nrgprepopctrl}')

    # Iterate **backwards** to safely delete particles
    i = n_particles[0] - 1
    while i >= 0:
        icell = int(particle_prop[i, 2])
        xpos = particle_prop[i, 3]
        mu = particle_prop[i, 4]
        nrg = particle_prop[i, 6]
        startnrg = particle_prop[i, 7]

        if nrg < 0.01 * startnrg:
            # Transfer energy to census particles
            position_fraction = (xpos - mesh.nodepos[icell]) / mesh.dx
            ix = round(position_fraction * part.Nx)
            ix = min(max(ix, 0), part.Nx - 1)

            angle_fraction = (mu + 1) / 2
            imu = round(angle_fraction * Nmu[icell])
            imu = min(max(imu, 0), Nmu[icell] - 1)

            # Compute linear index for census_particles
            linear_index = icell * part.Nx * Nmu[icell] + ix * Nmu[icell] + imu
            census_particles[linear_index, 6] += nrg

            # Overwrite deleted particle with the last active one
            particle_prop[i] = particle_prop[n_particles[0] - 1]
            n_particles[0] -= 1  # Reduce count
        i -= 1  # Move to the next particle

    # Move nonzero-energy census particles to particle_prop
    for i in range(n_census_ptcls):
        origin, ttt, icell, xpos, mu, frq, nrg, startnrg = census_particles[i]
        if nrg > 0:
            particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, nrg]
            n_particles[0] += 1

    # Print diagnostics
    with objmode:
        print(f'Particle count after population control: {n_particles[0]}')
    nrgpostpopctrl = np.sum(particle_prop[:n_particles[0], 6])
    with objmode:
        print(f'Energy in the particles post population control = {nrgpostpopctrl}')
        print('Population control applied...')

    return n_particles, particle_prop
