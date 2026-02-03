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


def crooked_pipe_update(mesh_sigma_a, temp, dt):
    # Calculate beta
    beta = 4 * phys.a * temp ** 3 / (mat.b) # unitless

    # Update the Fleck factor
    fleck = 1.0 / (1.0 + beta * mesh_sigma_a * phys.c * dt)
    
    # Check that Fleck is physically valid
    if np.any(fleck < 0.0) or np.any(fleck > 1.0):
        raise RuntimeError(
            f"Invalid Fleck factor detected! "
            f"min = {np.min(fleck)}, max = {np.max(fleck)}"
        )
    print(f'fleck = {fleck}')
    return fleck


@njit
def si02_update(temp, rho, dt, phys_a, phys_c):
    # Calculate heat capacity
    c_v = 0.1448 * 1.1 * (temp ** 0.1) 
    c_v *= rho 
    mat_b = c_v
    
    # Calculate opacity
    sigma_a = 2.70773 * (rho ** 0.75) * (temp ** -3.53) * rho
    
    # opacity ceiling 1e9
    sigma_a = np.maximum(sigma_a, 1e9) if 1e9 < 0 else np.minimum(sigma_a, 1e9)
    
    sigma_s = np.zeros_like(sigma_a)
    sigma_t = sigma_a.copy()

    # Calculate beta
    beta = 4 * phys_a * (temp ** 3) / c_v 

    # Update the Fleck factor
    fleck = 1.0 / (1.0 + beta * sigma_a * phys_c * dt)
    # fleck = np.ones_like(beta)
    # Validation
    # Note: Numba's raise doesn't support formatted strings well
    for val in fleck.flat:
        if val < 0.0 or val > 1.0:
            raise RuntimeError("Invalid Fleck factor detected!")

    return sigma_a, sigma_s, sigma_t, fleck, mat_b


@njit
def ramping_time_step(current_time, dt, dt_max, dt_rampfactor, t_final, step):
    """
    Updates the simulation time and scales the time-step.
    Numba-compatible version.
    """
    # 1. Update time and step count
    # Note: np.round is used as it is supported by Numba
    new_time = np.round(current_time + dt, 8)
    new_step = step + 1

    # 2. Scale the time-step (Ramping logic)
    new_dt = dt
    if new_dt < dt_max:
        new_dt = new_dt * dt_rampfactor
        if new_dt > dt_max:
            new_dt = dt_max

    # 3. Check for final time-step truncation
    if new_time + new_dt > t_final:
        new_dt = t_final - new_time
        
    return new_time, new_dt, new_step

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
