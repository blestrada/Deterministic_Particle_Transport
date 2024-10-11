"""Source IMC particles"""

import numpy as np

import imc_global_part_data as part
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_bcon_data as bcon
import imc_global_time_data as time
import imc_global_volsource_data as vol


def create_census_grid():
    for icell in range(mesh.ncells):
        # Create position, angle, and scattering arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu

        census_grid = [[icell, xpos, mu, 0] 
               for xpos in x_positions 
               for mu in angles]
        part.census_grid.extend(census_grid)


def create_census_particles():
    """Creates census particles for the first time-step"""
    for icell in range(mesh.ncells):
        # Create position, angle, and scattering arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu

        # Assign energy-weights
        n_census_ptcls = part.Nx * part.Nmu
        nrg = phys.a * (mesh.radtemp[icell] ** 4) * mesh.dx / n_census_ptcls
        startnrg = nrg

        # Assign origin and time of emission
        ttt = time.time
        origin = icell

        # Create particles and add them to the global list
        particles = [[origin, ttt, icell, xpos, mu, nrg, startnrg] 
             for xpos in x_positions 
             for mu in angles]
        part.particle_prop.extend(particles)

        census_grid = [[icell, xpos, mu, 0] 
               for xpos in x_positions 
               for mu in angles]
        part.census_grid.extend(census_grid)


def create_body_source_particles():
    """Creates source particles for the mesh"""
    e_total_body = 0.0
    for icell in range(mesh.ncells):
        # Create position, angle, and time arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1.0 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
        emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt

        # Assign energy-weights
        n_source_ptcls = part.Nx * part.Nmu * part.Nt
        nrg = phys.c * mesh.fleck[icell] * mesh.sigma_a[icell] * phys.a * (mesh.temp[icell] ** 4) * time.dt * mesh.dx / n_source_ptcls
        e_total_body += phys.c * mesh.fleck[icell] * mesh.sigma_a[icell] * phys.a * (mesh.temp[icell] ** 4) * time.dt * mesh.dx
        startnrg = nrg
        # Create particles and add them to global list
        origin = icell
        
        for xpos in x_positions:
            for mu in angles:
                for ttt in emission_times:
                    part.particle_prop.append([origin, ttt, icell, xpos, mu, nrg, startnrg])

    print(f'e_total_body = {e_total_body}')


def create_surface_source_particles():
    """Creates source particles for the boundary condition."""
    e_surf = phys.sb * bcon.T0 ** 4 * time.dt  # transport 
    e_surf2 = phys.a * phys.c / 3 * bcon.T0 ** 4 * time.dt # diffusion

    # Create source particles for the surface
    xpos = 1e-9
    angles = -1 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
    emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt

    # Create energy-weights
    n_source_ptcls = len(angles) * len(emission_times)
    nrg = e_surf2 / n_source_ptcls
    startnrg = nrg
    icell = 0  # starts in leftmost cell
    origin = icell
    # Create particles and add them to global list
    particles = [[origin, ttt, icell, xpos, mu, nrg, startnrg]
        for mu in angles
        for ttt in emission_times]
    part.particle_prop.extend(particles)


def create_volume_source_particles():
    """ Creates source particles for the volume source."""
    # Calculate the numbers of cells the source will span
    source_cells = int((np.ceil(vol.x_0/mesh.dx)))

    # Create zeros vector spanning mesh.ncells
    source = np.zeros(source_cells)

    # The source spans from 0 to x_0
    source[0:source_cells] = 1.0 * phys.a * phys.c * mesh.sigma_t[0:source_cells]

    # Formula for radiation source
    e_source = source[:] * time.dt * mesh.dx
    e_total_vol = np.sum(e_source)
    print(f'e_total_vol = {e_total_vol}')

    # Make particles from volume source
    for icell in range(source_cells): # For each cell that has volume source energy
        # Create position, angle, and time arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1.0 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
        emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt

        # Assign energy-weights
        n_source_ptcls = part.Nx * part.Nmu * part.Nt
        nrg = e_source[icell] / n_source_ptcls
        startnrg = nrg
        origin = icell

        # Create particles and add them to the global list
        for xpos in x_positions:
            for mu in angles:
                for ttt in emission_times:
                    part.particle_prop.append([origin, ttt, icell, xpos, mu, nrg, startnrg])





"""These functions below use random numbers."""


def create_census_particles_random():

    rng = np.random.default_rng()
    for icell in range(mesh.ncells):

        # Create position, angle and scattering arrays
        x_positions = np.random.uniform(mesh.nodepos[icell], mesh.nodepos[icell + 1], size=part.Nx)
        angles = -1 + 2 * rng.random(size=part.Nmu)
        

        # Assign energy-weights
        n_census_ptcls = part.Nx * part.Nmu * part.Nt
        nrg = phys.a * (mesh.radtemp[icell] ** 4) * mesh.dx / n_census_ptcls
        startnrg = nrg

        # Assign origin and time of emission
        ttt = time.time
        origin = icell

        # Assign xi variable - not used, but done to keep the same array structure.
        xi = 0

        # Create particles and add them to the global list
        particles = [[origin, ttt, icell, xpos, mu, xi, nrg, startnrg] 
             for xpos in x_positions 
             for mu in angles]
        part.particle_prop.extend(particles)


def create_body_source_particles_random():
    """Creates source particles for the mesh using random numbers."""
    
    for icell in range(mesh.ncells):
        # Create position, angle, time, and scattering arrays
        x_positions = np.random.uniform(mesh.nodepos[icell], mesh.nodepos[icell + 1], size=part.Nx)
        angles = -1 + 2 * np.random.uniform(size=part.Nmu)
        emission_times = np.random.uniform(time.time, time.time + time.dt, size=part.Nt)

        # Assign energy-weights
        n_source_ptcls = part.Nx * part.Nmu * part.Nt
        nrg = phys.c * mesh.sigma_a[icell] * phys.a * (mesh.temp[icell] ** 4) * time.dt * mesh.dx / n_source_ptcls
        startnrg = nrg

        # Create particles and add them to global list
        origin = icell

        # Assign xi variable - not used, but done to keep the same array structure
        xi = 0
        
        particles = [[origin, ttt, icell, xpos, mu, xi, nrg, startnrg] 
             for xpos in x_positions 
             for mu in angles
             for ttt in emission_times]
        part.particle_prop.extend(particles)


def create_surface_source_particles_random():
    """Creates source particles for the boundary condition using random numbers."""

    e_surf = phys.sb * bcon.T0 ** 4 * time.dt
    e_surf2 = phys.a * phys.c / 3 * bcon.T0 ** 4 * time.dt
    # Create source particles for the surface
    xpos = 1e-9
    angles = -1 + 2 * np.random.uniform(size=part.Nmu)
    emission_times = np.random.uniform(time.time, time.time + time.dt, size=part.Nt)

    # Create energy-weights
    n_source_ptcls = part.Nx * part.Nmu * part.Nt
    nrg = e_surf2 / n_source_ptcls
    startnrg = nrg
    icell = 0  # starts in leftmost cell
    origin = icell

    # Assign xi variable - not used, but done to keep the same array structure
    xi = 0

    # Create particles and add them to global list
    particles = [[origin, ttt, icell, xpos, mu, xi, nrg, startnrg]
        for mu in angles
        for ttt in emission_times]
    part.particle_prop.extend(particles)


def volume_sourcing_random():
    # Calculate the number of cells the volume source spans
    source_cells = int((np.ceil(vol.x_0 / mesh.dx)))

    sources = np.zeros(mesh.ncells)
    sources[0:source_cells] = 1.0

    if time.time <= vol.tau_0 / phys.c:
        e_cell = (mesh.sigma_a * mesh.fleck * phys.c * mesh.dx * time.dt * phys.a * (mesh.temp ** 4)) + (sources[:] * time.dt * mesh.dx)
    else:
        e_cell = (mesh.sigma_a * mesh.fleck * phys.c * mesh.dx * time.dt * phys.a * (mesh.temp ** 4))

    e_total = sum(e_cell[:])

    probablity = e_cell[:] / e_total

    n_census = part.n_census
    n_input = part.n_input
    n_max = part.n_max

    n_source = n_input
    if (n_source + n_census) > n_max:
        n_source = n_max - n_census - mesh.ncells - 1

    # Start by allocating 1 particle per cell
    n_particles_per_cell = np.ones(mesh.ncells, dtype=np.uint64)

    # Reduce the number of particles to distribute since 1 particle per cell is already assigned
    remaining_particles = n_source - mesh.ncells

    if remaining_particles > 0:
        # Distribute the remaining particles based on the energy probabilities
        additional_particles = np.floor(probablity * remaining_particles).astype(np.uint64)

        total_assigned = np.sum(additional_particles)
        unassigned_particles = remaining_particles - total_assigned

        n_particles_per_cell += additional_particles

        # Randomly assign any unassigned particles to cells based on probabilities
        if unassigned_particles > 0:
            selected_cells = np.random.choice(mesh.ncells, unassigned_particles, p=probablity, replace=True)
            for cell in selected_cells:
                n_particles_per_cell[cell] += 1

    print(f'Number of particles to emit in each cell: {n_particles_per_cell}')
    print(f'Total number of particles: {np.sum(n_particles_per_cell)}')

    """Create particles"""
    # Create the body-source particles
    for icell in range(mesh.ncells):
        if n_particles_per_cell[icell] <= 0:
            continue
        nrg = e_cell[icell] / float(n_particles_per_cell[icell])
        startnrg = nrg
        for _ in range(n_particles_per_cell[icell]):
            origin = icell
            xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
            mu = 1.0 - 2.0 * np.random.uniform()
            ttt = time.time + np.random.uniform() * time.dt
            # Add this ptcl to the global list
            part.particle_prop.append([origin, ttt, icell, xpos, mu, nrg, startnrg])