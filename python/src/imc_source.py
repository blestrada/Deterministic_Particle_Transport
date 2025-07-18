"""Source IMC particles"""

import numpy as np
from numba import njit, objmode
import matplotlib.pyplot as plt

import imc_global_part_data as part
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_bcon_data as bcon
import imc_global_time_data as time
import imc_global_volsource_data as vol
import imc_utilities as imc_util


def calculate_even_angles(ef, ef_ref=1/3, ef_max=1.0, n_min=2, n_max=16):
    # Calculate normalized deviation
    deviation = np.abs(ef - ef_ref) / (ef_max - ef_ref)
    # Calculate number of angles
    n_angles = n_min + (n_max - n_min) * deviation
    # Round to nearest even integer
    n_angles = int(np.round(n_angles / 2) * 2)
    # Ensure bounds are respected
    return max(n_min, min(n_max, n_angles))


def create_census_particles(n_particles, particle_prop, radtemp):
    """Creates census particles for the first time-step"""
    for icell in range(mesh.ncells):
        # Create position, angle, and scattering arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1 + (np.arange(part.Nmu[icell]) + 0.5) * 2 / part.Nmu[icell]

        # Assign energy-weights
        n_census_ptcls = part.Nx * part.Nmu[icell]
        nrg = phys.a * (radtemp[icell] ** 4) * mesh.dx / n_census_ptcls
        startnrg = nrg

        # Assign origin and time of emission
        ttt = time.time
        origin = icell

        # Create particles and add them to the global list
        for xpos in x_positions:
            for mu in angles:
                if n_particles[0] < part.max_array_size:
                    # Fill the preallocated array with particle properties
                    # 0 is for frequency which we are ignoring here.
                    particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                    n_particles[0] += 1
                else:
                    print("Warning: Maximum number of particles reached!")
    
    return n_particles, particle_prop


def create_body_source_particles(n_particles, particle_prop, temp, current_time, dt, sigma_a, fleck):
    """Creates source particles for the mesh"""
    for icell in range(mesh.ncells):
        # Define positions, angles, and emission times
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1.0 + (np.arange(part.Nmu[icell]) + 0.5) * 2 / part.Nmu[icell]
        emission_times = current_time + (np.arange(part.Nt) + 0.5) * dt / part.Nt

        # Calculate energy weight
        n_source_ptcls = part.Nx * part.Nmu[icell] * part.Nt
        nrg = phys.c * fleck[icell] * sigma_a[icell] * phys.a * (temp[icell] ** 4) * dt * mesh.dx / n_source_ptcls
        startnrg = nrg
        origin = icell

        # Loop to create particles and add them to the preallocated array
        for xpos in x_positions:
            for mu in angles:
                for ttt in emission_times:
                    if n_particles[0] < part.max_array_size:
                        # Fill the preallocated array with particle properties
                        # 0 is for frequency which we are ignoring here.
                        particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                        n_particles[0] += 1
                    else:
                        print("Warning: Maximum number of particles reached!")
    
    return n_particles, particle_prop


def create_graziani_census_particles(n_particles, particle_prop):
    """Creates census particles for the graziani slab problem"""
    # Group Opacities (given)
    sigma_g = np.array([9.16000e04, 9.06781e04, 6.08939e04, 4.08607e04, 2.72149e04, 
                    1.86425e04, 1.24389e04, 8.19288e03, 5.79710e03, 5.14390e03, 
                    5.20350e03, 8.69569e03, 6.67314e03, 4.15912e03, 2.62038e03, 
                    1.64328e03, 1.01613e03, 6.19069e02, 3.75748e02, 2.97349e02, 
                    8.21172e02, 4.01655e03, 4.54828e03, 3.50487e03, 3.02359e03, 
                    4.34203e03, 2.98594e03, 1.55364e03, 9.42213e02, 5.76390e02, 
                    3.52953e02, 2.09882e02, 1.26546e02, 7.80087e01, 9.97421e01, 
                    1.48848e02, 8.22907e01, 4.86915e01, 2.91258e01, 1.68133e01, 
                    9.92194e00, 5.18722e00, 2.24699e00, 1.29604e00, 7.46975e-01, 
                    8.43058e-01, 2.43746e00, 1.50509e00, 9.01762e-01, 5.38182e-01])
    normalized_planck_integral = np.zeros(part.Ng, dtype=np.float64)
    # Set up frequency group structure
    # 50 groups, logarithmically spaced between 3.0 × 10−3 keV and 30.0 keV
    E_min = 3.0e-3  # keV
    E_max = 30.0  # keV

    # Generate logarithmically spaced edges
    edges = np.logspace(np.log10(E_min), np.log10(E_max), part.Ng + 1)

    # Compute the group center points (geometric mean of adjacent edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # Compute normalized Planck integral for each frequency group
    P_g = np.array([imc_util.normalizedPlanckIntegral(edges[i], edges[i+1], 0.03) 
                    for i in range(len(edges)-1)])

    # Normalize P_g so they sum to 1
    P_g /= np.sum(P_g)

    # P_g now contains the fraction of blackbody radiation in each frequency group

    energy_per_group = phys.a * (0.03) ** 4 * P_g
    # print(f'body energy in each group = {energy_per_group}')

    for icell in range(mesh.ncells):
        for ifreq in range(part.Ng):
            # Define positions, angles, and emission times, and frequency group.
            x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
            angles = -1.0 + (np.arange(part.Nmu[icell]) + 0.5) * 2 / part.Nmu[icell]
            ttt = 0
            # print(f'emission_times = {emission_times}')
            frq = ifreq
            # Calculate energy weight
            n_source_ptcls = part.Nx * part.Nmu[icell] * part.Nt # number of census particles per group
            nrg = energy_per_group[frq] / n_source_ptcls
            startnrg = nrg
            origin = icell
            # Loop to create particles and add them to the preallocated array
            for xpos in x_positions:
                for mu in angles:
                    if n_particles[0] < part.max_array_size:
                        # Fill the preallocated array with particle properties
                        particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, frq, nrg, startnrg]
                        n_particles[0] += 1
                    else:
                        print("Warning: Maximum number of particles reached!")
        
    return n_particles, particle_prop


def create_graziani_body_source_particles(n_particles, particle_prop, temp, current_time, dt):
    """Creates body source particles for the graziani slab problem"""
    # Group Opacities (given)
    sigma_g = np.array([9.16000e04, 9.06781e04, 6.08939e04, 4.08607e04, 2.72149e04, 
                    1.86425e04, 1.24389e04, 8.19288e03, 5.79710e03, 5.14390e03, 
                    5.20350e03, 8.69569e03, 6.67314e03, 4.15912e03, 2.62038e03, 
                    1.64328e03, 1.01613e03, 6.19069e02, 3.75748e02, 2.97349e02, 
                    8.21172e02, 4.01655e03, 4.54828e03, 3.50487e03, 3.02359e03, 
                    4.34203e03, 2.98594e03, 1.55364e03, 9.42213e02, 5.76390e02, 
                    3.52954e02, 2.09882e02, 1.26546e02, 7.80087e01, 9.97421e01, 
                    1.48848e02, 8.22907e01, 4.86915e01, 2.91258e01, 1.68133e01, 
                    9.92194e00, 5.18722e00, 2.24699e00, 1.29604e00, 7.46975e-01, 
                    8.43058e-01, 2.43746e00, 1.50509e00, 9.01762e-01, 5.38182e-01])
    
    # Set up frequency group structure
    # 50 groups, logarithmically spaced between 3.0 × 10−3 keV and 30.0 keV
    # Define the energy range
    E_min = 3.0e-3  # keV
    E_max = 30.0  # keV
    # Generate logarithmically spaced edges
    edges = np.logspace(np.log10(E_min), np.log10(E_max), part.Ng + 1)

    # Compute the group center points (geometric mean of adjacent edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # Compute normalized Planck integral for each frequency group
    P_g = np.array([imc_util.normalizedPlanckIntegral(edges[i], edges[i+1], 0.03) / (edges[i+1]-edges[i])
                    for i in range(len(edges)-1)])


    # P_g now contains the fraction of blackbody radiation in each frequency group
    energy_per_group = phys.a * phys.c * (0.03) ** 4 * dt * sigma_g * P_g
    # print(f'body energy in each group = {energy_per_group}')

    for icell in range(mesh.ncells):
        for ifreq in range(part.Ng):
            # Define positions, angles, and emission times, and frequency group.
            x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
            angles = -1.0 + (np.arange(part.Nmu[icell]) + 0.5) * 2 / part.Nmu[icell]
            emission_times = current_time + (np.arange(part.Nt) + 0.5) * dt / part.Nt
            # print(f'emission_times = {emission_times}')
            frq = ifreq
            # Calculate energy weight
            n_source_ptcls = part.Nx * part.Nmu[icell] * part.Nt # number of source particles per freq group
            nrg = energy_per_group[ifreq] / n_source_ptcls
            startnrg = nrg
            origin = icell
            # Loop to create particles and add them to the preallocated array
            for xpos in x_positions:
                for mu in angles:
                    for ttt in emission_times:
                        if n_particles[0] < part.max_array_size:
                            # Fill the preallocated array with particle properties
                            particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, frq, nrg, startnrg]
                            n_particles[0] += 1
                        else:
                            print("Warning: Maximum number of particles reached!")
        
    return n_particles, particle_prop
        
@njit
def create_graziani_left_surface_source_particles(n_particles, particle_prop, current_time, dt, T_surf=0.3):
    """Creates surface-source particles for the left face source (emitting rightward)"""
    
    E_min = 3.0e-3  # keV
    E_max = 30.0  # keV

    # Generate logarithmically spaced edges
    edges = np.logspace(np.log10(E_min), np.log10(E_max), part.Ng + 1)
    # Compute the group center points (geometric mean of adjacent edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    d_nu = np.empty(len(edges)-1)
    for i in range(len(edges)-1):
        d_nu[i] = (edges[i+1] - edges[i])

    planck_ED_average = np.empty(len(edges)-1, dtype=np.float64)
    with objmode:
        for i in range(len(edges)-1):
            planck_ED_average[i] = imc_util.PlanckianEnergyDensityAverage(T_surf, edges[i], edges[i+1])
            # with objmode:
            #     print(f' Group {i} radiation energy density = {planck_ED_average[i]}')

    
    # P_g = np.empty(len(edges) - 1, dtype=np.float64)
    # with objmode:
    #     for i in range(len(edges) - 1):
    #         P_g[i] = imc_util.normalizedPlanckIntegral(edges[i], edges[i+1], T_surf) #* (edges[i+1] - edges[i])

    # plt.figure()
    # plt.plot(centers, planck_ED_average)
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.xlim(1e-3, 10)
    # # plt.ylim(1e5,1e11)
    # plt.xlabel("Frequency (keV)")
    # plt.show()
    # print(f'sum = {np.sum(P_g)}')

    energy_per_group = planck_ED_average / 4
    # Create source particles for the surface
    xpos = 0.0
    Nmu = 8
    angles = (np.arange(Nmu) + 0.5) / (Nmu)
    angles = np.sqrt(angles)
    # print(f'angles = {angles}')
    Nt = 1000
    emission_times = current_time + (np.arange(Nt) + 0.5) * dt / Nt
    print(f'Number of surface source particles per group = {len(angles) * len(emission_times)}')

    # # Create energy-weights
    # for ifreq in range(part.Ng):
    #     if ifreq == 30:
    #         angles = (np.arange(1000) + 0.5) / (1000)
    #     n_source_ptcls = len(angles) * len(emission_times)
    #     nrg = energy_per_group[ifreq] / n_source_ptcls
    #     icell = 0  # starts in leftmost cell
    #     origin = 0
    # xpos = 0.0
    icell = 0
    origin = 0
    # ifreq = 30
    # n_source_ptcls = 5_000_000
    # nrg = energy_per_group / n_source_ptcls

    # for _ in range(n_source_ptcls):
    #     if n_particles[0] < part.max_array_size:
    #             # Fill the preallocated array with particle properties
    #             mu = np.sqrt(np.random.uniform())  # Random angle
    #             ttt = current_time + dt * np.random.uniform()  # Random time
    #             frq = ifreq
    #             ptcl_nrg = nrg * 0.5
    #             startnrg = ptcl_nrg
    #             # print(f'energy of a surface source particle = {ptcl_nrg}')
    #             particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, frq, ptcl_nrg, startnrg]
    #             n_particles[0] += 1
    #     else:
    #         print("Warning: Maximum number of particles reached!")
    # Create particles and add them to global list
    for ifreq in range(part.Ng):
        for mu in angles:
            for ttt in emission_times:
                if n_particles[0] < part.max_array_size:
                    # Fill the preallocated array with particle properties
                    frq = ifreq
                    ptcl_nrg = energy_per_group[ifreq] / (len(angles) * len(emission_times))
                    startnrg = ptcl_nrg
                    # print(f'energy of a surface source particle = {ptcl_nrg}')
                    particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, frq, ptcl_nrg, startnrg]
                    n_particles[0] += 1
                else:
                    print("Warning: Maximum number of particles reached!")

    return n_particles, particle_prop


def create_surface_source_particles(n_particles, particle_prop, current_time, dt):
    """Creates source particles for the boundary condition."""
    e_surf = phys.sb * (bcon.T0 ** 4) * dt
    print(f'Energy emitted by the surface = {e_surf}')

    # Create source particles for the surface
    xpos = 0.0
    angles = (np.arange(part.Nmu[0]) + 0.5) / (part.Nmu[0])
    angles = np.sqrt(angles)
    emission_times = current_time + (np.arange(part.Nt) + 0.5) * dt / part.Nt

    # Create energy-weights
    n_source_ptcls = len(angles) * len(emission_times)
    print(f'Number of surface source particles = {n_source_ptcls}')
    
    nrg = e_surf / n_source_ptcls
    icell = 0  # starts in leftmost cell
    origin = -1

    # Create particles and add them to global list
    for mu in angles:
        for ttt in emission_times:
            if n_particles[0] < part.max_array_size:
                # Fill the preallocated array with particle properties
                # 0 is for frequency which we are ignoring here.
                ptcl_nrg =  nrg
                startnrg = ptcl_nrg
                # print(f'energy of a surface source particle = {ptcl_nrg}')
                particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles[0] += 1
            else:
                print("Warning: Maximum number of particles reached!")

    return n_particles, particle_prop
    

def create_volume_source_particles(n_particles, particle_prop, dt):
    """ Creates source particles for the volume source."""
    # Calculate the numbers of cells the source will span
    source_cells = int((np.ceil(vol.x_0/mesh.dx)))
    
    # Create zeros vector spanning mesh.ncells
    source = np.zeros(source_cells)

    # The source spans from 0 to x_0
    source[0:source_cells] = 1.0 * phys.a * phys.c

    # Formula for radiation source
    e_source = source[:] * dt * mesh.dx
    
    e_total_vol = np.sum(e_source)
    print(f'e_total_vol = {e_total_vol}')

    # Make particles from volume source
    for icell in range(source_cells):
        # Create position, angle, and time arrays
        x_positions = mesh.nodepos[icell] + ((np.arange(part.Nx) + 0.5) * mesh.dx) / part.Nx
        angles = -1.0 + ((np.arange(part.Nmu[icell]) + 0.5) * 2) / part.Nmu[icell]
        emission_times = time.time + (np.arange(part.Nt) + 0.5) * dt / part.Nt
        
        # Calculate energy weight per particle
        n_source_ptcls = part.Nx * part.Nmu[icell] * part.Nt
        nrg = e_source[icell] / n_source_ptcls
        startnrg = nrg
        origin = icell

        # Create particles and add them to the preallocated array
        for xpos in x_positions:
            for mu in angles:
                for ttt in emission_times:
                    if n_particles[0] < part.max_array_size:
                        # Fill the preallocated array with particle properties
                        # 0 is for frequency which we are ignoring here
                        particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                        n_particles[0] += 1
                    else:
                        print("Warning: Maximum number of particles reached!")
                        break  # Exit if we exceed the max_particles
    return n_particles, particle_prop


"""These functions below use random numbers."""


def create_census_particles_random(n_particles, particle_prop, radtemp):
    """Creates census particles for the first time-step"""
    n_census_ptcls = 10
    for icell in range(mesh.ncells):
        # Assign energy-weights
        nrg = phys.a * (radtemp[icell] ** 4) * mesh.dx / n_census_ptcls
        startnrg = nrg

        # Assign origin and time of emission
        ttt = 0.0
        origin = icell

        # Create particles and add them to the global list
        for _ in range(n_census_ptcls):
            xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
            mu = -1.0 + 2 * np.random.uniform()
            if n_particles[0] < part.max_array_size:
                # Fill the preallocated array with particle properties
                # 0 is for frequency which we are ignoring here.
                particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles[0] += 1
            else:
                print("Warning: Maximum number of particles reached!")

    return n_particles, particle_prop


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

"""IMC sourcing routine"""


def imc_get_energy_sources(radiation_source, body_source, surface_source, fleck, temp, dt, sigma_a):
    """Get energy source terms"""

    e_rad = 0.0
    if radiation_source:
        e_rad = np.ones(mesh.source_cells) * phys.a * phys.c * dt * mesh.dx

    e_body = 0.0 
    if body_source:
        # Emission source term
        e_body = np.zeros(mesh.ncells)  # Energy emitted per cell per time-step
        e_body[:] = (
            fleck[:]
            * sigma_a[:]
            * phys.a
            * phys.c
            * temp[:] ** 4
            * mesh.dx
            * dt
        )

    e_surf = 0.0
    if surface_source:
        e_surf = phys.sb * bcon.T0 ** 4 * time.dt

    # Total energy emitted
    e_total = np.sum(e_rad) + e_surf + np.sum(e_body)

    print("\nEnergy radiated in timestep:")
    print(f'Energy emitted by body-source: {np.sum(e_body)}')
    print(f'Energy emitted by rad source: {np.sum(e_rad)}')
    print(f'Energy emitted by surface source: {e_surf}')
    print("Total energy emitted: {:24.16E}".format(e_total))

    return e_rad, e_surf, e_body, e_total


def imc_get_emission_probabilities(e_rad, e_surf, e_body, e_total):
    """Convert energy source terms to particle emission probabilities."""
    print(f'e_body = {e_body}')
    print(f'e_surf = {e_surf}')
    print(f'e_rad = {e_rad}')
    # Initialize probabilities
    p_rad = np.zeros(10)  # Assuming 10 radiation source cells, update if different
    p_surf = 0.0
    p_body = np.zeros(mesh.ncells)

    if e_total > 0:
        # Probability of emission from the radiation source
        if np.sum(e_rad) > 0:
            p_rad[:] = np.cumsum(e_rad[:]) / np.sum(e_rad[:])

        # Probability of emission from the surface source
        p_surf = e_surf / e_total

        # Probability of emission from the body source
        if np.sum(e_body) > 0:
            p_body[:] = np.cumsum(e_body[:]) / np.sum(e_body[:])
    print(f'p_rad = {p_rad}, p_surf = {p_surf}, p_body = {p_body}')
    return p_rad, p_surf, p_body


def imc_get_source_particle_numbers(p_rad, p_surf, p_body):
    """Calculate number of source particles to create from radiation, surface, and body sources."""
    
    n_input = part.n_input
    print("(User requested {:8d} per timestep)".format(n_input))

    # Initialize counts for each source type
    
    n_rad = np.zeros(mesh.source_cells, dtype=np.uint64)
    n_surf = 0
    n_body = np.zeros(mesh.ncells, dtype=np.uint64)

    # Ensure at least one particle from each active source
    if np.sum(p_body) > 0:
        n_body[:] = 1
    if np.sum(p_rad) > 0:
        n_rad[:] = 1
    if p_surf > 0:
        n_surf = 1
    
    # Subtract allocated particles from total input count
    n_input -= (np.sum(n_body) + np.sum(n_rad) + n_surf)
    
    # Sample the remaining particles based on probabilities
    for _ in range(n_input):
        eta = np.random.uniform()

        if eta <= p_surf:
            n_surf += 1
        elif p_rad.size > 0 and np.sum(p_rad) > 0:
            for irad in range(len(p_rad)):
                if eta <= p_rad[irad]:
                    n_rad[irad] += 1
                    break
        else:
            for icell in range(mesh.ncells):
                if eta <= p_body[icell]:
                    n_body[icell] += 1
                    break

    print("\nRadiation source:", n_rad)
    print("Surface source:", n_surf)
    print("Body source:")
    print(n_body)

    return n_rad, n_surf, n_body


def imc_source_particles(e_rad, n_rad, e_surf, n_surf, e_body, n_body, particle_prop, n_particles, current_time, dt):
    """For known energy distribution, create source particles in a preallocated array."""
    
    max_particles = part.max_array_size  # Maximum allowed particles

    # Create the surface-source particles
    if n_surf > 0:
        nrg = e_surf / float(n_surf)
        startnrg = nrg
        for _ in range(n_surf):
            if n_particles[0] < max_particles:
                origin = -1
                xpos = 0.0
                mu = np.sqrt(np.random.uniform())  # Corresponds to f(mu) = 2mu
                ttt = current_time + np.random.uniform() * dt
                # Fill the preallocated array with particle properties
                particle_prop[n_particles[0]] = [origin, ttt, 0, xpos, mu, 0, nrg, startnrg]
                n_particles[0] += 1
            else:
                print("Warning: Maximum number of particles reached!")
                break

    # Create the body-source particles
    for icell in range(mesh.ncells):
        if n_body[icell] <= 0:
            continue
        nrg = e_body[icell] / float(n_body[icell])
        startnrg = nrg
        for _ in range(n_body[icell]):
            if n_particles[0] < max_particles:
                origin = icell
                xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
                mu = 1.0 - 2.0 * np.random.uniform()
                ttt = current_time + np.random.uniform() * dt
                # Fill the preallocated array
                particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles[0] += 1
            else:
                print("Warning: Maximum number of particles reached!")
                break

    # Create the rad-source particles
    for icell in range(mesh.source_cells):
        if n_rad[icell] <= 0:
            continue
        nrg = e_rad[icell] / float(n_rad[icell])
        startnrg = nrg
        for _ in range(n_rad[icell]):
            if n_particles[0] < max_particles:
                origin = icell
                xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
                mu = 1.0 - 2.0 * np.random.uniform()
                ttt = current_time + np.random.uniform() * dt
                # Fill the preallocated array
                particle_prop[n_particles[0]] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles[0] += 1
            else:
                print("Warning: Maximum number of particles reached!")
                break

    return n_particles, particle_prop


def run(fleck, temp, sigma_a, particle_prop, n_particles, current_time, dt):
    """
    Source new IMC particles.

    This routine calculates the energy sources for:
    - The left-hand boundary (which is currently held at a constant temperature, T0).
    - The body cells.
    - The radiation source, if specified.

    These are then converted into particle emission probabilities.
    The number of particles to source in this time-step is determined
    (ensuring that the total number in the system does not exceed
    some pre-defined maximum), and then these are attributed either to
    the boundary, the mesh cells, or the radiation cells, according to the probabilities
    calculated earlier. The particles are then created.
    """
    print("\n" + "-" * 79)
    print("Source step ({:4d})".format(time.step))
    print("-" * 79)

    # Get the energy source terms
    e_rad, e_surf, e_body, e_total = imc_get_energy_sources(vol.radiation_source, mesh.body_source, bcon.surface_source, fleck, temp, dt, sigma_a)

    # Get emission probabilities
    p_rad, p_surf, p_body = imc_get_emission_probabilities(e_rad, e_surf, e_body, e_total)

    # Determine number of source particles
    n_rad, n_surf, n_body = imc_get_source_particle_numbers(p_rad, p_surf, p_body)

    # Create particles
    n_particles, particle_prop = imc_source_particles(e_rad, n_rad, e_surf, n_surf, e_body, n_body, particle_prop, n_particles, current_time, dt)

    # Final particle count in system
    print("Number of particles in the system = {:12d}".format(n_particles[0]))

    return n_particles, particle_prop
