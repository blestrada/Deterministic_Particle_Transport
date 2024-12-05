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


def marshak_wave_update():
    """Update temperature-dependent quantities at start of time-step"""
    # Calculate beta
    mesh.beta = np.zeros(mesh.ncells)
    mesh.beta[:] = 4 * phys.a * mesh.temp[:] ** 3 / (1.0 * mat.b[:]) # rho = 1.0
    # print(f'mesh.beta = {mesh.beta[:10]}')
    
    # Calculate new opacity
    mesh.sigma_a = np.zeros(mesh.ncells)
    mesh.sigma_a[:] = 1000.0 / (mesh.temp[:] ** 3)
    # print(f'mesh.sigma_a = {mesh.sigma_a[:10]}')
    # print(f'last 10 mesh.sigma_a = {mesh.sigma_a[-10:]}')

    # Calculate new fleck factor
    mesh.fleck = np.zeros(mesh.ncells)
    mesh.fleck[:] = 1.0 / (1.0 + mesh.beta[:] * mesh.sigma_a[:] * phys.c * time.dt)
    # print(f'mesh.fleck = {mesh.fleck[:10]}')

    # Calculate total opacity
    mesh.sigma_t = np.copy(mesh.sigma_a)
    # mesh.sigma_t[:] = mesh.sigma_a[:] * mesh.fleck[:] + (1.0 - mesh.fleck[:]) * mesh.sigma_a[:] + mesh.sigma_s[:]
    # print(f'mesh.sigma_t = {mesh.sigma_t[:10]}')

#@njit
def population_control(n_particles, particle_prop, current_time):
    """Reduces the number of particles and consolidates energy in the census grid."""
    # intialize array for census particles
    num_census_ptcls = mesh.ncells * part.Nx * part.Nmu
    census_ptcls = np.zeros((num_census_ptcls, 7), dtype=np.float64)
    idx = 0

    # Create the grid of points to emit from
    for icell in range(mesh.ncells):
        # Define position and angles
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1.0 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
        nrg = 0
        startnrg = 0
        origin = icell
        ttt = current_time
        for xpos in x_positions:
            for mu in angles:
                # Fill the array entries
                census_ptcls[idx, :] = [origin, ttt, icell, xpos, mu, nrg, startnrg]
                idx += 1
    
    # Calculate the energy in the particles before population control
    nrgprepopctrl = np.sum(particle_prop[:n_particles[0], 5])
    print(f'Energy in the particles pre population control = {nrgprepopctrl}')
    # # plot position vs energy
    # # Extract active particles
    # active_particles = particle_prop[:n_particles[0]]

    # # Extract position and energy columns
    # positions = active_particles[:, 3]  # xpos is the 4th column (index 3)
    # energies = active_particles[:, 5]   # nrg is the 6th column (index 5)

    # # Create the plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(positions, energies, s=5, c='blue', alpha=0.7, label='Particles')
    # plt.xlabel('Position')
    # plt.ylabel('Energy')
    # plt.title('Position vs Energy for All Particles')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # Iterate through the active particles
    for i in range(n_particles[0]):
        # Extract particle properties
        icell = int(particle_prop[i, 2])
        xpos = particle_prop[i, 3]
        mu = particle_prop[i, 4]
        nrg = particle_prop[i, 5]

        # Calculate ix (position index) and imu (angle index)
        position_fraction = (xpos - mesh.nodepos[icell]) / mesh.dx
        # print(f'position fraction = {position_fraction}')
        ix = round(position_fraction * part.Nx)
        ix = min(max(ix, 0), part.Nx - 1)
        # print(f' ix = {ix}')

        angle_fraction = (mu + 1) / 2
        # print(f'angle fraction = {angle_fraction}')
        imu = round(angle_fraction * part.Nmu)
        imu = min(max(imu, 0), part.Nmu - 1)
        # print(f'imu = {imu}')
        # Calculate the linear index for the census_grid
        linear_index = icell * part.Nx * part.Nmu + ix * part.Nmu + imu
        # print(f'linear index = {linear_index}')
        census_ptcls[linear_index][5] += nrg

    # Print the particle count before control
    particle_count_before = n_particles[0]
    print(f'Particle count before population control: {particle_count_before}')

    # Reset particle_prop and n_particles for new population control particles
    n_particles[0] = 0

    # Move particles from census_ptcls to particle_prop
    # Only keep entries with non-zero energy
    valid_indices = census_ptcls[:, 5] > 0  # Select particles with positive energy
    new_particles = census_ptcls[valid_indices]

    # Update particle_prop with the new particles
    num_new_particles = len(new_particles)
    particle_prop[:num_new_particles] = new_particles
    n_particles[0] = num_new_particles

    # # Extract active particles
    # active_particles = particle_prop[:n_particles[0]]

    # # Extract position and energy columns
    # positions = active_particles[:, 3]  # xpos is the 4th column (index 3)
    # energies = active_particles[:, 5]   # nrg is the 6th column (index 5)

    # # Create the plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(positions, energies, s=5, c='blue', alpha=0.7, label='Particles')
    # plt.xlabel('Position')
    # plt.ylabel('Energy')
    # plt.title('Position vs Energy for All Particles')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Print the particle count after population control
    particle_count_after = n_particles[0]
    print(f'Particle count after population control: {particle_count_after}')

    # Calculate the energy in the particles after population control
    nrgpostpopctrl = np.sum(particle_prop[:n_particles[0], 5])
    print(f'Energy in the particles post population control = {nrgpostpopctrl}')
    print('Population control applied...')
    return n_particles, particle_prop

