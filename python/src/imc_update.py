"""Update at start of time-step"""

import imc_global_phys_data as phys
import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_time_data as time
import imc_global_bcon_data as bcon
import imc_global_part_data as part
import numpy as np

def SuOlson_update():
    """Update temperature-dependent quantities at start of time-step"""

    # Calculate new heat capacity
    mat.b = mat.alpha * mesh.temp ** 3
    print(f'heat capacity = {mat.b[:10]}')


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


def population_control():
    """Reduces number of particles"""
    # The energy in the particles before population control
    nrgprepopctrl = sum(item[5] for item in part.particle_prop)
    print(f'Energy in the particles pre population control = {nrgprepopctrl}')
    # Make a copy of the census grid
    census_grid_popctrl = part.census_grid.copy()

    # Reset the nrg term to zero for all energy entries in census_grid_popctrl
    for entry in census_grid_popctrl:
        entry[3] = 0

    for particle in part.particle_prop:
        icell, xpos, mu, nrg = particle[2], particle[3], particle[4], particle[5], 

        # Calculate ix and imu for the particle
        position_fraction = (xpos - mesh.nodepos[icell]) / mesh.dx
        ix = int(position_fraction * part.Nx)
        ix = min(max(ix, 0), part.Nx - 1)

        angle_fraction = (mu + 1) / 2
        imu = int(angle_fraction * part.Nmu)
        imu = min(max(imu, 0), part.Nmu - 1)

        # Calculate the linear index for census_grid
        linear_index = icell * part.Nx * part.Nmu + ix * part.Nmu + imu 
        census_grid_popctrl[linear_index][3] += nrg  # Accumulate energy in the nearest census particle

    particle_count_before = len(part.particle_prop)
    print(f'Particle count before population control: {particle_count_before}')

    # Remove old particles from particle_prop
    part.particle_prop = []

    # Add new particles from census_grid_popctrl
    for entry in census_grid_popctrl:
        icell = entry[0]
        xpos = entry[1]
        mu = entry[2]
        nrg = entry[3]

        # Insert new parameters into each particle entry
        origin = entry[0]  # icell from census grid
        ttt = time.time  # current time in the simulation
        startnrg = nrg  # energy from census grid

        # Append updated particle entry to particle_prop
        part.particle_prop.append([origin, ttt, icell, xpos, mu, nrg, startnrg])

    particle_count_after = len(part.particle_prop)
    print(f'Particle count after population control: {particle_count_after}')
    # energy in the particles post population control
    nrgpostpopctrl = sum(item[5] for item in part.particle_prop)
    print(f'Energy in the particles post population control = {nrgpostpopctrl}')
    print(f'Population control applied...')
