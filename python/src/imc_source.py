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

    # print(f'e_total_body = {e_total_body}')
    e_surf = phys.a * phys.c / 4 * (bcon.T0 ** 4) * time.dt
    print(f'Body-source energy sourced this time-step = {np.sum(e_total_body)}')


def create_body_source_particles_2():

    nx_max = part.Nx
    nmu_max = part.Nmu
    nt_max = part.Nt

    nx_min = 1
    nmu_min = 2
    nt_min = 1


    max_particles_per_cell = nx_max * nmu_max * nt_max
    print(f' max particles per cell = {max_particles_per_cell}')

    # Calculate the energy per cell

    e_cell = mesh.sigma_a * mesh.fleck * phys.c * mesh.dx * time.dt * phys.a * (mesh.temp ** 4)

    e_total = np.sum(e_cell)

    probability = e_cell[:] / e_total

    # Calculate the number of particles per cell based on energy proportion
    n_particles_per_cell = np.round(probability * max_particles_per_cell).astype(np.uint64)

    # Ensure at least 2 particles are assigned to each cell
    n_particles_per_cell = np.maximum(n_particles_per_cell, 2)

    # Function to find the nearest even integer for Nmu
    def nearest_even(value):
        return int(2 * np.round(value / 2))

    # Initialize arrays to store Nx, Nmu, Nt values per cell
    Nx_per_cell = np.zeros_like(n_particles_per_cell, dtype=np.uint64)
    Nmu_per_cell = np.zeros_like(n_particles_per_cell, dtype=np.uint64)
    Nt_per_cell = np.zeros_like(n_particles_per_cell, dtype=np.uint64)

    # Assign Nx, Nmu, Nt for each cell based on the cube root of the number of particles
    for i in range(len(n_particles_per_cell)):
        # Take the cube root of the number of particles
        cube_root = np.cbrt(n_particles_per_cell[i])

        # Find the nearest even integer for Nmu
        Nmu = nearest_even(cube_root)

        # Set Nx = Nt = Nmu
        Nx = Nmu
        Nt = Nmu

        # Store the values in the corresponding arrays
        Nx_per_cell[i] = Nx
        Nmu_per_cell[i] = Nmu
        Nt_per_cell[i] = Nt

    # Print the resulting Nx, Nmu, Nt for each cell
    print(f'N_particles_per_cell = {n_particles_per_cell}')
    print(f'Nx_per_cell = {Nx_per_cell}')
    print(f'Nmu_per_cell = {Nmu_per_cell}')
    print(f'Nt_per_cell = {Nt_per_cell}')
    e_total_body = 0.0
    # Create source particles
    for icell in range(mesh.ncells):
        # Create position, angle, and time arrays
        x_positions = mesh.nodepos[icell] + (np.arange(Nx_per_cell[icell]) + 0.5) * mesh.dx / Nx_per_cell[icell]
        angles = -1.0 + (np.arange(Nmu_per_cell[icell]) + 0.5) * 2 / Nmu_per_cell[icell]
        emission_times = time.time + (np.arange(Nt_per_cell[icell]) + 0.5) * time.dt / Nt_per_cell[icell]

        # Assign energy-weights
        n_source_ptcls = Nx_per_cell[icell] * Nmu_per_cell[icell] * Nt_per_cell[icell]
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
    e_surf = phys.a * phys.c / 4 * (bcon.T0 ** 4) * time.dt 
    print(f'Energy emitted by the surface = {e_surf}')

    # Create source particles for the surface
    xpos = 0.0
    angles = (np.arange(part.Nmu / 2) + 0.5) / (part.Nmu / 2)
    emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt

    # Create energy-weights
    n_source_ptcls = len(angles) * len(emission_times)
    print(f'Number of surface source particles = {n_source_ptcls}')
    
    nrg = e_surf / n_source_ptcls
    startnrg = nrg
    icell = 0  # starts in leftmost cell
    origin = -1
    # Create particles and add them to global list
    particles = [[origin, ttt, icell, xpos, mu, 2 * mu * nrg, 2 * mu * startnrg]
        for mu in angles
        for ttt in emission_times]
    part.particle_prop.extend(particles)
    

def create_surface_source_particles_diffusion():
    """Creates source particles for the boundary condition using diffusion (SuOlson1996)."""
    e_surf = phys.a * phys.c / 4 * (bcon.T0 ** 4) * time.dt 
    print(f'Energy emitted by the surface = {e_surf}')

    # Create source particles for the surface
    xpos = 0.0
    angles = (np.arange(part.Nmu / 2) + 0.5) / (part.Nmu / 2)
    emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt

    # Create energy-weights
    n_source_ptcls = len(angles) * len(emission_times)
    print(f'Number of surface source particles = {n_source_ptcls}')
    
    nrg = e_surf / n_source_ptcls
    startnrg = nrg
    icell = 0  # starts in leftmost cell
    origin = -1
    # Create particles and add them to global list
    particles = [[origin, ttt, icell, xpos, mu, 2 * mu * nrg, 2 * mu * startnrg]
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
    source[0:source_cells] = 1.0 * phys.a * phys.c

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
        
        particles = [[origin, ttt, icell, xpos, mu, nrg, startnrg] 
             for xpos in x_positions 
             for mu in angles
             for ttt in emission_times]
        part.particle_prop.extend(particles)


def create_surface_source_particles_random():
    """Creates source particles for the boundary condition using random numbers."""

    e_surf = phys.sb * bcon.T0 ** 4 * time.dt
    # Create source particles for the surface
    xpos = 1e-9
    angles = -1 + 2 * np.random.uniform(size=part.Nmu)
    emission_times = np.random.uniform(time.time, time.time + time.dt, size=part.Nt)

    # Create energy-weights
    n_source_ptcls = part.Nx * part.Nmu * part.Nt
    nrg = e_surf / n_source_ptcls
    startnrg = nrg
    icell = 0  # starts in leftmost cell
    origin = icell


    # Create particles and add them to global list
    particles = [[origin, ttt, icell, xpos, mu, nrg, startnrg]
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

"""IMC sourcing routine from Fleck and Cummings"""


def imc_get_energy_sources():
    """Get energy source terms for surface and mesh."""
    # Left-hand boundary is a black-body emitter at constant temperature T0
    # (Energy radiatied per unit area = sigma.T0**4 (sigma = S-B constant)
    e_surf = phys.sb * bcon.T0 ** 4 * time.dt

    # Emission source term
    e_body = np.zeros(mesh.ncells)  # Energy emitted per cell per time-step
    e_body[:] = (
        mesh.fleck[:]
        * mesh.sigma_a[:]
        * phys.a
        * phys.c
        * mesh.temp[:] ** 4
        * mesh.dx
        * time.dt
    )

    # Total energy emitted
    e_total = e_surf + sum(e_body[:])

    print("\nEnergy radiated in timestep:")
    print("\nIn total:")
    print("{:24.16E}".format(e_total))

    return e_surf, e_body, e_total


def imc_get_emission_probabilities(e_surf, e_body, e_total):
    """Convert energy source terms to particle emission probabilities."""
    p_surf = e_surf / e_total
    # Probability of each cell _given that the particle is from the mesh not the
    # surface_
    p_body = np.zeros(mesh.ncells)
    p_body[:] = np.cumsum(e_body[:]) / sum(e_body[:])

    return p_surf, p_body


def imc_get_source_particle_numbers(p_surf, p_body):
    """Calculate number of source particles to create at surface / throughout mesh."""
    n_census = part.n_census
    n_input = part.n_input
    n_max = part.n_max

    # Determine total number of particles to source this time-step
    n_source = n_input
    if (n_source + n_census) > n_max:
        n_source = n_max - n_census - mesh.ncells - 1

    print("\nSourcing {:8d} particles this timestep".format(n_source))
    print("(User requested {:8d} per timestep)".format(n_input))

    n_surf = 0
    n_body = np.zeros(mesh.ncells, dtype=np.uint64)

    # Calculate the number of particles emitted by the surface source and each mesh cell
    for _ in range(n_source):
        if np.random.uniform() <= p_surf:
            n_surf += 1
        else:
            eta = np.random.uniform()
            for icell in range(mesh.ncells):
                if eta <= p_body[icell]:
                    n_body[icell] += 1
                    break

    print("\nBody source")
    print(n_body)

    return n_surf, n_body


def imc_source_particles(e_surf, n_surf, e_body, n_body):
    """For known energy distribution (surface and mesh), create source particles."""
    # Create the surface-source particles
    nrg = e_surf / float(n_surf)
    startnrg = nrg
    for _ in range(n_surf):
        origin = -1
        xpos = 0.0
        muu = np.sqrt(np.random.uniform())  # Corresponds to f(mu) = 2mu
        ttt = time.time + np.random.uniform() * time.dt
        part.particle_prop.append(
            [origin, ttt, 0, xpos, muu, nrg, startnrg]
        )  # Add this ptcl to the global list

    # Create the body-source particles
    for icell in range(mesh.ncells):
        if n_body[icell] <= 0:
            continue
        nrg = e_body[icell] / float(n_body[icell])
        startnrg = nrg
        for _ in range(n_body[icell]):
            origin = icell
            xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
            muu = 1.0 - 2.0 * np.random.uniform()
            ttt = time.time + np.random.uniform() * time.dt
            # Add this ptcl to the global list
            part.particle_prop.append([origin, ttt, icell, xpos, muu, nrg, startnrg])


def run():
    """
    Source new IMC particles.

    This routine calculates the energy sources for
    the (a) left-hand boundary (which is currently held at a constant
    temperature, T0), and (b) the computational cells, as well as the overall
    total for the time-step. These are then converted into particle emission
    probabilities. The number of particles to source in this time-step is
    determined (ensuring that the total number in the system does not exceed
    some pre-defined maximum), and then these are attributed either to the
    boundary or to one of the mesh cells, according to the probabilities
    calculated earlier. The particles are then created.
    """
    print("\n" + "-" * 79)
    print("Source step ({:4d})".format(time.step))
    print("-" * 79)

    # Determine probability of particles belonging to sources
    # -------------------------------------------------------

    # Get the energy source terms
    (e_surf, e_body, e_total) = imc_get_energy_sources()

    # Emission probabilities
    (p_surf, p_body) = imc_get_emission_probabilities(e_surf, e_body, e_total)

    # Number of source particles
    (n_surf, n_body) = imc_get_source_particle_numbers(p_surf, p_body)

    # Create the particles
    imc_source_particles(e_surf, n_surf, e_body, n_body)

    # Particle count
    print("Number of particles in the system = {:12d}".format(len(part.particle_prop)))