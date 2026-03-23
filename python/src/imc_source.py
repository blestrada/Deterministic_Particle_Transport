"""Source IMC particles"""

import numpy as np
from numba import njit, objmode
import matplotlib.pyplot as plt
from numpy.typing import NDArray

import imc_global_part_data as part
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_bcon_data as bcon
import imc_global_time_data as time
import imc_global_volsource_data as vol
import imc_utilities as imc_util

# Random Sampling Functions
@njit
def sample_radius(r_min: float,
                  r_max: float
) -> float:
    """Given r_min and r_max, sample an r position in the cell.
    Since emission is uniform in volume, the pdf for sampling r is f(r)=2r/r_max^2."""
    r = np.sqrt(r_min**2 + np.random.uniform()*(r_max**2 - r_min**2))
    return r

@njit
def sample_z(z_min: float,
             z_max: float
) -> float:
    """Given z_min and z_max, sample a z position in the cell"""
    z = z_min + np.random.uniform()*(z_max - z_min)
    return z

@njit
def sample_mu_isotropic() -> float:
    return 2 * np.random.uniform() - 1.0

@njit
def sample_phi_isotropic() -> float:
    return 2 * np.pi * np.random.uniform()

@njit
def sample_mu_lambertian() -> float:
    """
    Produce mu between 0 and 1 with a lambertian distribution
    """
    return np.sqrt(np.random.uniform())
# Deterministic Sampling Functions

@njit
def deterministic_sample_mu_boundary(n_samples: int) -> NDArray:
    """
    Samples mu between 0 and 1, explicitly including the 
    boundaries 0.0 and 1.0.
    """
    if n_samples == 1:
        return np.array([0.5]) # Fallback for single particle
    
    # Linearly spaced from 0 to 1
    return np.linspace(-1.0, 1.0, n_samples)

@njit
def deterministic_sample_phi_boundary(n_samples: int) -> tuple[NDArray, NDArray]:
    """
    Sample phi deterministically, ensuring phi=0 is included to 
    hit the r=0 axis directly.
    """
    if n_samples <= 1:
        # Fallback: a single particle is always aimed at the axis
        return np.array([0.0]), np.array([1.0])
    
    # 1. Create points starting exactly at 0.0
    # We stop just before 2*pi because 2*pi == 0.0
    phi_values = np.linspace(0.0, 2 * np.pi, n_samples + 1)[:-1]
    
    # 2. Weights for periodic sampling are uniform
    # (Each point represents a 2*pi / n_samples slice)
    weights = np.ones(n_samples)
    
    # Normalize weights so the mean is 1.0 (standard for your loop)
    # Since they are all 1.0, mean is already 1.0, but this is safe:
    weights = weights / np.mean(weights)
    
    return phi_values, weights

@njit
def deterministic_sample_radius(r_min: float,
                                r_max: float,
                                n_samples: int
) -> NDArray:
    """
    Deterministically sample R for a cylindrical shell.
    """
    # centered uniform points
    p_values = (np.arange(n_samples) + 0.5) / n_samples
    # apply the PDF
    r_values = np.sqrt(r_min**2 + p_values * (r_max**2 - r_min**2))
    return r_values

@njit
def weighted_sample_radius(r_min: float,
                           r_max: float,
                           n_samples: int
) -> tuple[NDArray, NDArray]:
    """
    Uniformly sample R and return weights based on the 
    area-correction required for a cylindrical shell.
    """
    dr = (r_max - r_min) / n_samples
    r_values = np.linspace(r_min + 0.5*dr, r_max - 0.5*dr, n_samples)
    
    area_norm = r_max**2 - r_min**2
    
    weights = (2 * r_values * (r_max - r_min)) / area_norm

    weights = weights / np.mean(weights)
    
    return r_values, weights

@njit
def deterministic_sample_z(z_min: float,
                           z_max:float,
                           n_samples: int
) -> NDArray:
    """
    Determinstically sample Z for a cylindrical shell.
    """
    # Assume a uniform distribution
    z_values = z_min + (np.arange(n_samples)+0.5) * (z_max - z_min) / n_samples
    return z_values

@njit
def deterministic_sample_mu_isotropic(n_samples:int) -> NDArray:
    return -1.0 + ((np.arange(n_samples)) + 0.5) * 2 / n_samples


def deterministic_sample_mu_isotropic_offset(n_samples: int, offset: float = 0.75) -> np.ndarray:
    """
    n_samples: Number of mu bins
    offset: A value between 0 and 1. 
            0.5 is the standard midpoint.
            Values like 0.25 or 0.75 shift the points within the bins.
    """
    # Spacing remains 2 / n_samples
    delta_mu = 2.0 / n_samples
    # Start at -1.0, and move by (index + offset) steps of delta_mu
    return -1.0 + (np.arange(n_samples) + offset) * delta_mu


@njit
def deterministic_sample_mu_lambertian(n_samples) -> NDArray:
    """
    Produce n_samples of mu between 0 and 1 with a lambertian distribution
    """
    return np.sqrt(((np.arange(n_samples)) + 0.5) / n_samples)

@njit
def deterministic_sample_mu_lambertian_weighted(n_samples: int, offset: float = 0.5):
    """
    Produce n_samples of mu uniformly between 0 and 1, 
    with corresponding weights for a Lambertian distribution.
    """
    # 1. Sample mu uniformly between 0 and 1
    # Spacing is 1/n_samples
    mu_vals = (np.arange(n_samples) + offset) / n_samples
    
    # 2. Lambertian weights are proportional to mu
    # Weight(mu) = 2 * mu (normalized so the integral from 0 to 1 is 1)
    weights = 2.0 * mu_vals
    
    return mu_vals, weights


@njit
def deterministic_sample_mu_lambertian_offset(n_samples: int, offset: float = 0.5) -> np.ndarray:
    """
    Produce n_samples of mu between 0 and 1 with a Lambertian distribution.
    
    n_samples: Number of samples.
    offset: A value between 0 and 1 (0.5 is the standard center-of-bin).
    """
    # Create the uniform spacing in the CDF space [0, 1]
    # Then map back to mu space using the inverse CDF (sqrt)
    return np.sqrt((np.arange(n_samples) + offset) / n_samples)

@njit
def weighted_sample_mu_lambertian(n_samples: int) -> tuple[NDArray, NDArray]:
    """
    Uniformly sample mu and return weights based on the 
    Lambertian (cosine) distribution PDF f(mu) = 2*mu.
    """
    # 1. Create uniformly spaced mu values
    dmu = 1.0 / n_samples
    # Centered points to avoid 0.0 or 1.0 boundary issues
    mu_values = np.linspace(0.5 * dmu, 1.0 - 0.5 * dmu, n_samples)
    
    # 2. Calculate the weights
    # The physical PDF is f(mu) = 2*mu
    # The sampling PDF is g(mu) = 1 (uniform)
    # Weight = f(mu) / g(mu)
    weights = 2.0 * mu_values
    
    # Normalize weights so the mean is 1.0
    # This ensures the total number of particles/intensity is preserved
    weights = weights / np.mean(weights)
    
    return mu_values, weights


@njit          
def deterministic_sample_phi_isotropic(n_samples) -> NDArray:
    return (0.0 + (np.arange(n_samples) + 0.5)) * 2 * np.pi / n_samples

@njit
def deterministic_sample_phi_isotropic_offset(n_samples: int, offset: float = 0.0) -> NDArray:
    """
    Generates n_samples of phi evenly spaced between 0 and 2pi, 
    starting from a specific angular offset.
    """
    # Calculate the base mid-point samples
    base_phi = (np.arange(n_samples) + 0.5) * (2 * np.pi / n_samples)
    
    # Apply offset and wrap around 2*pi
    return (base_phi + offset) % (2 * np.pi)

@njit
def deterministic_sample_phi_tanh(n_samples, alpha) -> tuple[NDArray, NDArray]:
    """
    Sample angles non-uniformly using a tanh distribution to cluster
    points near 0 and pi.
    """
    # 1. Create normalized index j in range [-1, 1]
    i = np.arange(n_samples)
    j_scaled = ((i + 0.5) / n_samples - 0.5) * 2
    
    # 2. Map to [0, 2*pi]
    # The tanh term spans [-1, 1], so we shift and scale it
    phi_values = np.pi * (1 + np.tanh(alpha * j_scaled) / np.tanh(alpha))
    
    # 3. Calculate weights W(i)
    # Derivative of tanh is sech^2. We normalize so mean weight is 1.0
    weights = 1.0 / (np.cosh(alpha * j_scaled)**2)
    weights = weights / np.mean(weights)
    
    return phi_values, weights

@njit
def deterministic_sample_mu_tanh(n_samples, alpha) -> tuple[NDArray, NDArray]:
    """
    Sample mu non-uniformly using a tanh distribution to cluster
    points near -1 and 1.
    """
    # 1. Create normalized index j in range [-1, 1]
    i = np.arange(n_samples)
    j_scaled = ((i + 0.5) / n_samples - 0.5) * 2
    
    # 2. Map to [-1, 1]
    # np.tanh(alpha * j_scaled) / np.tanh(alpha) spans [-1, 1]
    # This clusters points near the poles -1 and 1.
    mu_values = np.tanh(alpha * j_scaled) / np.tanh(alpha)
    
    # 3. Calculate weights W(i)
    # The weight is proportional to the inverse of the density (the derivative).
    # d/dx [tanh(ax)/tanh(a)] = a * sech^2(ax) / tanh(a)
    # Since we normalize by the mean anyway, we only need the sech^2 part.
    weights = 1.0 / (np.cosh(alpha * j_scaled)**2)
    weights = weights / np.mean(weights)
    
    return mu_values, weights

@njit
def deterministic_sample_mu_lobatto(n_samples: int) -> tuple[NDArray, NDArray]:
    if n_samples == 1: return np.array([0.0]), np.array([1.0])
    
    # Include -1 and 1 explicitly
    mu_values = np.linspace(-1.0, 1.0, n_samples)
    
    # Trapezoidal weights: Endpoints represent half the "width" of internal points
    weights = np.ones(n_samples)
    weights = weights / np.mean(weights) # Normalize mean to 1.0
    
    return mu_values, weights


@njit
def deterministic_sample_t_tanh_dist(n_samples, 
                                     alpha, 
                                     t_lower, 
                                     t_upper
) -> tuple[NDArray, NDArray]:
    """
    Sample T non-uniformly to get better clustering near the edges.
    
    Based on the tanh mapping: x = 0.5 * [tanh(alpha * j) / tanh(alpha)]
    """
    # 1. Calculate normalized index j in range [-(N-1)/N, (N-1)/N] 
    # as per Equation (27) in the notes
    i = np.arange(n_samples)
    j = (i + 0.5) / n_samples - 0.5
    j_scaled = j * 2  # Scale to approximately [-1, 1]
    
    dt = t_upper - t_lower
    t_mid = (t_upper + t_lower) / 2
    
    # 2. Map j to time values using Equation (30)
    # We normalize by tanh(alpha) to ensure the range stays within [t_lower, t_upper]
    t_values = t_mid + (dt / 2) * (np.tanh(alpha * j_scaled) / np.tanh(alpha))
    
    # 3. Calculate weights W(i) = dx/di per Equation (31)
    # The derivative of tanh(u) is sech^2(u) = 1 / cosh^2(u)
    # This compensates for the high density of points at the edges.
    weights = 1.0 / (np.cosh(alpha * j_scaled)**2)
    
    # 4. Normalize weights so their sum equals n_samples
    # This ensures the 'total energy' is conserved
    weights = weights / np.mean(weights)
    
    return t_values, weights

@njit
def deterministic_sample_r_tanh_dist(
    n_samples: int,
    alpha: float,
    r_min: float,
    r_max: float
) -> tuple[NDArray, NDArray]:
    """
    Sample radius R using a tanh distribution to cluster points near r_min and r_max,
    while maintaining the physical PDF weights of a cylindrical shell.
    """
    # 1. Generate the tanh-mapped values (p_values) in the range [0, 1]
    # We use the same logic as the time-sampling but shifted to [0, 1]
    i = np.arange(n_samples)
    j = (i + 0.5) / n_samples - 0.5  # Range [-0.5, 0.5]
    j_scaled = j * 2                 # Range [-1, 1]
    
    # This maps j_scaled to a non-linear range [0, 1]
    # p_values will be clustered near 0 and 1
    p_values = 0.5 + 0.5 * (np.tanh(alpha * j_scaled) / np.tanh(alpha))
    
    # 2. Map these p_values to Radius using the physical PDF (Cylindrical shell)
    # r = sqrt(r_min^2 + p * (r_max^2 - r_min^2))
    r_sq_diff = r_max**2 - r_min**2
    r_values = np.sqrt(r_min**2 + p_values * r_sq_diff)
    
    # 3. Calculate weights
    # The weight is composed of two parts:
    # A) The tanh Jacobian: 1 / cosh^2(alpha * j_scaled)
    # B) The Radial Jacobian: The uniform-to-radius transformation is self-correcting 
    #    if we use the tanh weights directly.
    weights = 1.0 / (np.cosh(alpha * j_scaled)**2)
    
    # 4. Normalize weights so their mean is 1.0
    weights = weights / np.mean(weights)
    
    return r_values, weights


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
    total_energy_emitted = np.sum(phys.c * fleck * sigma_a * phys.a * (temp ** 4) * dt * mesh.dx)
    print(f'total energy emitted by the body-source = {total_energy_emitted}')
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
                    if n_particles < part.max_array_size:
                        # Fill the preallocated array with particle properties
                        # 0 is for frequency which we are ignoring here.
                        particle_prop[n_particles] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                        n_particles += 1
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
            if n_particles < part.max_array_size:
                # Fill the preallocated array with particle properties
                # 0 is for frequency which we are ignoring here.
                ptcl_nrg =  nrg
                startnrg = ptcl_nrg
                # print(f'energy of a surface source particle = {ptcl_nrg}')
                particle_prop[n_particles] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles += 1
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
                    if n_particles < part.max_array_size:
                        # Fill the preallocated array with particle properties
                        # 0 is for frequency which we are ignoring here
                        particle_prop[n_particles] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                        n_particles += 1
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
            if n_particles < max_particles:
                origin = -1
                xpos = 0.0
                mu = np.sqrt(np.random.uniform())  # Corresponds to f(mu) = 2mu
                ttt = current_time + np.random.uniform() * dt
                # Fill the preallocated array with particle properties
                particle_prop[n_particles] = [origin, ttt, 0, xpos, mu, 0, nrg, startnrg]
                n_particles += 1
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
            if n_particles < max_particles:
                origin = icell
                xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
                mu = 1.0 - 2.0 * np.random.uniform()
                ttt = current_time + np.random.uniform() * dt
                # Fill the preallocated array
                particle_prop[n_particles] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles += 1
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
            if n_particles < max_particles:
                origin = icell
                xpos = mesh.nodepos[icell] + np.random.uniform() * mesh.dx
                mu = 1.0 - 2.0 * np.random.uniform()
                ttt = current_time + np.random.uniform() * dt
                # Fill the preallocated array
                particle_prop[n_particles] = [origin, ttt, icell, xpos, mu, 0, nrg, startnrg]
                n_particles += 1
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
    print("Number of particles in the system = {:12d}".format(n_particles))

    return n_particles, particle_prop


def imc_get_energy_sources_2D(body_source, surface_source, 
                              fleck, temp, dt, sigma_a, mesh_z_edges, mesh_r_edges):
    """
    Calculates RZ energy source terms using cylindrical volumes.
    """
    num_z = len(mesh_z_edges) - 1
    num_r = len(mesh_r_edges) - 1
    
    # Initialize body energy array
    e_body = np.zeros((num_z, num_r))

    if body_source:
        # Pre-calculate RZ volumes for the entire mesh
        dz = np.diff(mesh_z_edges)
        dr2 = np.diff(mesh_r_edges**2)
        # volumes[i, j] = pi * (r_out^2 - r_in^2) * dz
        volumes = np.pi * np.outer(dz, dr2)

        # Volumetric emission rate: j = f * sigma_a * a * c * T^4
        # Total energy: E = j * Volume * dt
        e_body = (
            fleck * 
            sigma_a * 
            phys.a * 
            phys.c * 
            temp**4 * 
            volumes * 
            dt
        )

    e_surf = 0.0
    if surface_source:
        print(f'Surface at temp = {bcon.T0}')
        surface_length = 0.04
        e_surf = phys.sb * bcon.T0 ** 4 * dt * (np.pi * surface_length**2)

    # Total energy emitted
    e_total = e_surf + np.sum(e_body)
    print("\nEnergy radiated in timestep:")
    print(f'Energy emitted by body-source: {np.sum(e_body)}')
    print(f'Energy emitted by surface source: {e_surf}')
    print("Total energy emitted: {:24.16E}".format(e_total))
    return e_surf, e_body, e_total

def imc_get_emission_probabilities2D(e_surf, e_body, e_total):
    """Convert energy source terms into particle emission probabilities"""
    # Initialize probabilities
    p_surf = 0.0
    p_body = np.zeros_like(e_body)

    if e_total > 0.0:
        # Probability of emission from the surface source
        p_surf = e_surf / e_total

        # Probability of emission from each cell in the body source
        if np.sum(e_body) > 0.0:
            p_body = e_body / e_total
    print(f'p_surf = {p_surf}')
    print(f'p_body (sum) = {np.sum(p_body)}')
    return p_surf, p_body


def imc_get_source_particle_numbers2D(p_surf, p_body):
    """Calculate the number of source particles to create from the surface and body"""
    n_input = part.n_input
    print(f'User requested {part.n_input} particles per time-step')

    # initialize counts for each source type
    n_surf = 0
    n_body = np.zeros_like(p_body, dtype=int)

    # --- Step 1: Ensure at least 1 particle for each source ---
    if p_surf > 0.0:
        n_surf = 1
    n_body[p_body > 0.0] = 1

    # Subtract allocated particles from budget
    allocated = n_surf + np.sum(n_body)
    remaining = n_input - allocated
    if remaining < 0:
        raise ValueError("Requested too few particles to satisfy minimum allocation.")

    # --- Step 2: Sample the remaining particles ---
    body_flat = p_body.flatten()
    body_cum = np.cumsum(body_flat)
    if body_cum[-1] > 0.0:
        body_cum /= body_cum[-1]

    for _ in range(remaining):
        eta = np.random.rand()
        if eta <= p_surf:
            n_surf += 1
        else:
            eta_body = (eta - p_surf) / (1 - p_surf) if p_surf < 1.0 else 0.0
            idx = np.searchsorted(body_cum, eta_body)
            i, j = divmod(idx, p_body.shape[1])
            n_body[i, j] += 1

    # --- Step 3: Consistency check ---
    total_particles = n_surf + np.sum(n_body)
    if total_particles != n_input:
        raise RuntimeError(
            f"Particle allocation mismatch: expected {n_input}, got {total_particles}"
        )

    print("Surface source:", n_surf)
    print("Body source (per cell):")
    print(n_body)

    return n_surf, n_body


def imc_source_particles2D(
    e_surf, n_surf, e_body, n_body, particle_prop, n_particles,
    current_time, dt, mesh_z_edges, mesh_r_edges
):
    """For known energy distribution, create source particles and add to pre-allocated array.
    
    particle_prop format:
    [emission_time, x_idx, y_idx, xpos, ypos, theta, frq, nrg, startnrg]
    """
    max_particles = part.max_array_size

    # --- Surface source particles ---
    if n_surf > 0 and e_surf > 0.0:
        nrg = e_surf / float(n_surf)
        startnrg = nrg
        for _ in range(n_surf):
            if n_particles >= max_particles:
                raise RuntimeError("Maximum number of particles reached in surface source.")
            z = 1e-12
            r = sample_radius(0.0, 0.04)
            # Cell indices
            z_idx = 0
            r_idx = np.searchsorted(mesh_r_edges, r) - 1
            # print(f'r_idx = {r_idx}')
            # Time, frequency, angle
            ttt = sample_z(current_time, current_time + dt)
            frq = 0.0
            mu = sample_mu_lambertian()  # Lambertian sampling for mu
            phi = sample_phi_isotropic()
            # Store
            particle_prop[n_particles] = [
                ttt, z_idx, r_idx, z, r, mu, phi, frq, nrg, startnrg
            ]
            n_particles += 1

    # --- Body source particles ---
    if np.sum(n_body) > 0 and np.sum(e_body) > 0.0:
        for i in range(n_body.shape[0]): # Z index
            for j in range(n_body.shape[1]): # R index
                if n_body[i, j] > 0:
                    nrg = e_body[i, j] / float(n_body[i, j])
                    startnrg = nrg
                    for _ in range(n_body[i, j]):
                        if n_particles >= max_particles:
                            raise RuntimeError("Maximum number of particles reached in body source.")
                        # Sample uniformly inside the cell
                        z = sample_z(mesh_z_edges[i], mesh_z_edges[i+1])
                        r = sample_radius(mesh_r_edges[j], mesh_r_edges[j+1])
                        # Indices
                        z_idx, r_idx = i, j
                        # Time, frequency, angle
                        ttt = sample_z(current_time, current_time + dt)
                        frq = 0.0
                        mu = sample_mu_isotropic()
                        phi = sample_phi_isotropic()
                        # Store
                        particle_prop[n_particles] = [
                            ttt, z_idx, r_idx, z, r, mu, phi, frq, nrg, startnrg
                        ]
                        n_particles += 1

    return n_particles, particle_prop


def run2D(fleck, temp, sigma_a, particle_prop, n_particles, current_time, dt, mesh_z_edges, mesh_r_edges):
    print("\n" + "-" * 79)
    print("Source step ({:4d})".format(time.step))
    print("-" * 79)

    # Get the energy source terms
    e_surf, e_body, e_total = imc_get_energy_sources_2D(True, True, fleck, temp, dt, sigma_a, mesh_z_edges, mesh_r_edges)

    # Get emission probabilities
    p_surf, p_body = imc_get_emission_probabilities2D(e_surf, e_body, e_total)

    # Determine number of source particles
    n_surf, n_body = imc_get_source_particle_numbers2D(p_surf, p_body)

    # Create particles
    n_particles, particle_prop = imc_source_particles2D(e_surf, n_surf, e_body, n_body, particle_prop, n_particles, current_time, dt, mesh_z_edges, mesh_r_edges)

    # Final particle count in system
    print("Number of particles in the system after sourcing = {:12d}".format(n_particles))

    return n_particles, particle_prop

@njit
def crooked_pipe_surface_particles(T_surf, n_particles, particle_prop, 
                                   surface_Nr, surface_Nmu, 
                                   surface_N_phi, surface_Nt, 
                                   current_time, dt, mesh_r_edges):
    """Creates surface source particles for the boundary condition"""
    surface_length = 0.04 # cm
    e_surf = phys.sb * T_surf ** 4 * dt * (np.pi * surface_length**2)
    print('Total energy emitted by the surface =', e_surf)

    # Generate radii
    r_values, r_weights  = weighted_sample_radius(0.0+0.005, 0.005+surface_length, surface_Nr)
    # print(f'r_values = {r_values}')

    # Generate mu
    mu_values, mu_weights = deterministic_sample_mu_lambertian_weighted(surface_Nmu, offset=0.75)
    # print(f'mu_values = {mu_values}')

    # Generate Phi
    phi_values = deterministic_sample_phi_isotropic(surface_N_phi)
    # phi_values = np.array([0.0])
    # print(f'phi_values = {phi_values}')

    # Emission times evenly spaced over dt
    t_values = deterministic_sample_z(current_time, current_time + dt, surface_Nt)

    # Total number of surface source particles
    n_source_ptcls = len(r_values) * len(mu_values) * len(phi_values) * len(t_values)
    print('Number of surface source particles =', n_source_ptcls)

    # Energy per particle
    base_nrg = e_surf / n_source_ptcls

    z = 1e-12
    z_idx = 0

    for i, r in enumerate(r_values):
        r_idx = np.searchsorted(mesh_r_edges, r) - 1 
        for i_m, mu in enumerate(mu_values):
            for i_p,phi in enumerate(phi_values):
                for ttt in t_values:
                    if n_particles < part.max_array_size:
                        weighted_nrg = base_nrg * r_weights[i] * mu_weights[i_m]
                        startnrg = weighted_nrg
                        # Assign: [emission_time, z_idx, r_idx, z, r, mu, phi, frq, nrg, startnrg]
                        particle_prop[n_particles] = [ttt, z_idx, r_idx, z, r, mu, phi, 0, weighted_nrg, startnrg]
                        n_particles += 1
                    else:
                        print("Warning: Maximum number of particles reached!")
                        return n_particles, particle_prop

    return n_particles, particle_prop

@njit
def crooked_pipe_surface_particles_rn(T_surf, n_particles, particle_prop, 
                                   surface_Nr, surface_Nmu, 
                                   surface_N_phi, surface_Nt, 
                                   current_time, dt, mesh_r_edges):
    """Creates surface source particles for the boundary condition"""
    surface_length = 0.04 + 0.005 # cm
    e_surf = phys.sb * T_surf ** 4 * dt * (np.pi * surface_length**2)
    print('Total energy emitted by the surface =', e_surf)

    n_surface_particles = 2_000_000
    nrg = e_surf / n_surface_particles
    for _ in range(n_surface_particles):
        z = 1e-12
        r = sample_radius(0.0, surface_length)
        # Cell indices
        z_idx = 0
        r_idx = np.searchsorted(mesh_r_edges, r) - 1
        # print(f'r_idx = {r_idx}')
        # Time, frequency, angle
        ttt = sample_z(current_time, current_time + dt)
        frq = 0.0
        mu = sample_mu_lambertian()  # Lambertian sampling for mu
        phi = sample_phi_isotropic()
        # Store
        particle_prop[n_particles] = [
            ttt, z_idx, r_idx, z, r, mu, phi, frq, nrg, nrg
        ]
        n_particles += 1
    

    return n_particles, particle_prop





def crooked_pipe_body_particles(n_particles, particle_prop, 
                                part_Nx, part_Ny, part_Nmu, part_Nphi, part_Nt,
                                current_time, dt, 
                                mesh_r_edges, mesh_z_edges, 
                                mesh_temp, mesh_fleck, 
                                mesh_sigma_a):
    nz_cells = len(mesh_z_edges) - 1
    nr_cells = len(mesh_r_edges) - 1
    # Pre-calculate RZ volumes for the entire mesh
    dz = np.diff(mesh_z_edges)
    dr2 = np.diff(mesh_r_edges**2)
    # volumes[i, j] = pi * (r_out^2 - r_in^2) * dz
    volumes = np.pi * np.outer(dz, dr2)
    
    start_count = n_particles
    for iz in range(nz_cells):
        for ir in range(nr_cells):
            if mesh_temp[iz, ir] <= 2.59E-5:
                continue
            # Cell volume
            zone_volume = volumes[iz, ir]

            # Generate z
            z_min = mesh_z_edges[iz]
            z_max = mesh_z_edges[iz+1]
            z_samples = part_Nx[iz, ir]
            z_values = deterministic_sample_z(z_min, z_max, z_samples)

            # Generate r
            r_min = mesh_r_edges[ir]
            r_max = mesh_r_edges[ir+1]
            r_samples = part_Ny[iz, ir]
            r_values = deterministic_sample_radius(r_min, r_max, r_samples)

            # Generate mu
            mu_samples = part_Nmu[iz, ir]
            mu_values, mu_weights = deterministic_sample_mu_tanh(mu_samples, alpha=4.0)
            
            # Generate phi
            phi_samples = part_Nphi[iz, ir]
            phi_values, phi_weights = deterministic_sample_phi_tanh(phi_samples, alpha=4.0)
            
            # Generate t
            t_samples = part_Nt[iz, ir]
            t_min = current_time
            t_max = current_time + dt
            t_values = deterministic_sample_z(t_min, t_max, t_samples)

            # The total number of source particle in the cell
            n_cell_ptcls = len(z_values) * len(r_values) * len(mu_values) * len(phi_values) * len(t_values)

            # Total energy emitted by the cell
            e_cell = (
            mesh_fleck[iz,ir] * 
            mesh_sigma_a[iz,ir] * 
            phys.a * 
            phys.c * 
            mesh_temp[iz,ir] **4 * 
            zone_volume * 
            dt)
            
            base_nrg = e_cell / n_cell_ptcls

            # Loop to create particles
            for z in z_values:
                for i_r, r in enumerate(r_values):
                    for i_m, mu in enumerate(mu_values):
                        for i_p,phi in enumerate(phi_values):
                            for i_t, ttt in enumerate(t_values):
                                if n_particles < part.max_array_size:
                                    weighted_nrg = base_nrg * phi_weights[i_p] * mu_weights[i_m]
                                    # Assign: [emission_time, z_idx, r_idx, z, r, mu, phi, frq, nrg, startnrg]
                                    particle_prop[n_particles] = [ttt, iz, ir, z, r, mu, phi, 0, weighted_nrg, weighted_nrg]
                                    n_particles += 1
                                else:
                                    print("Warning: Maximum number of particles reached!")
    actual_sum = np.sum(particle_prop[start_count:n_particles, 8])
    theoretical_sum = np.sum(mesh_fleck * mesh_sigma_a * phys.a * phys.c * mesh_temp**4 * volumes * dt)
    
    print("Added", n_particles - start_count, "body-source particles.")    
    print("Sum of generated particles energies =", actual_sum)
    print("Total energy emitted by body-source =", theoretical_sum)
    
    return n_particles, particle_prop


def RZ_body_particles_rn(n_particles, particle_prop, 
                         part_Nx, part_Ny, part_Nmu, part_Nphi, part_Nt,
                         current_time, dt, 
                         mesh_r_edges, mesh_z_edges, 
                         mesh_temp, mesh_fleck, 
                         mesh_sigma_a):
    
    nz_cells = len(mesh_z_edges) - 1
    nr_cells = len(mesh_r_edges) - 1
    
    # Pre-calculate RZ volumes: V = pi * (r_out^2 - r_in^2) * dz
    dz = np.diff(mesh_z_edges)
    dr2 = np.diff(mesh_r_edges**2)
    volumes = np.pi * np.outer(dz, dr2)
    
    start_count = n_particles
    
    for iz in range(nz_cells):
        for ir in range(nr_cells):
            # Skip cold cells to save performance
            if mesh_temp[iz, ir] <= 2.59E-5:
                continue

            # Calculate total particles for this specific cell
            n_cell_ptcls = (part_Nx[iz, ir] * part_Ny[iz, ir] * part_Nmu[iz, ir] * part_Nphi[iz, ir] * part_Nt[iz, ir])
            
            if n_cell_ptcls <= 0:
                continue

            # Total energy emitted by the cell
            e_cell = (mesh_fleck[iz, ir] * mesh_sigma_a[iz, ir] * phys.a * phys.c * mesh_temp[iz, ir]**4 * volumes[iz, ir] * dt)
            
            # Every particle gets an equal share of the cell energy
            # because the random sampling handles the spatial/angular distribution
            base_nrg = e_cell / n_cell_ptcls

            for _ in range(int(n_cell_ptcls)):
                if n_particles >= part.max_array_size:
                    print("Warning: Maximum number of particles reached!")
                    # Break the inner loops and return what we have
                    return n_particles, particle_prop

                # 1. Random Spatial Sampling
                z = sample_z(mesh_z_edges[iz], mesh_z_edges[iz+1])
                r = sample_radius(mesh_r_edges[ir], mesh_r_edges[ir+1])
                
                # 2. Random Time Sampling
                ttt = sample_z(current_time, current_time + dt)
                
                # 3. Random Angular Sampling
                mu = sample_mu_isotropic()
                phi = sample_phi_isotropic()
                
                # 4. Energy and Frequency
                frq = 0.0 # Assuming gray/frequency-integrated for now
                
                # Assign to properties array
                # Format: [time, z_idx, r_idx, z, r, mu, phi, freq, nrg, start_nrg]
                particle_prop[n_particles] = [
                    ttt, float(iz), float(ir), z, r, mu, phi, frq, base_nrg, base_nrg
                ]
                n_particles += 1

    # Consistency Check
    actual_sum = np.sum(particle_prop[start_count:n_particles, 8])
    theoretical_sum = np.sum(mesh_fleck * mesh_sigma_a * phys.a * phys.c * mesh_temp**4 * volumes * dt)
    
    print(f"Added {n_particles - start_count} body-source particles.")    
    print(f"Sum of generated particles energies = {actual_sum:.6e}")
    print(f"Total energy emitted by body-source   = {theoretical_sum:.6e}")
    
    return n_particles, particle_prop


def RZ_body_particles_nrn(n_particles, particle_prop, 
                          part_Nx, part_Ny, part_Nmu, part_Nphi, part_Nt,
                          current_time, dt, 
                          mesh_r_edges, mesh_z_edges, 
                          mesh_temp, mesh_fleck, 
                          mesh_sigma_a):
    
    nz_cells = len(mesh_z_edges) - 1
    nr_cells = len(mesh_r_edges) - 1
    
    # Pre-calculate RZ volumes: V = pi * (r_out^2 - r_in^2) * dz
    dz = np.diff(mesh_z_edges)
    dr2 = np.diff(mesh_r_edges**2)
    volumes = np.pi * np.outer(dz, dr2)
    
    start_count = n_particles
    
    for iz in range(nz_cells):
        for ir in range(nr_cells):
            if mesh_temp[iz, ir] <= 2.59E-5:
                continue

            # 1. Total energy emitted by this cell
            zone_volume = volumes[iz, ir]
            e_cell = (mesh_fleck[iz, ir] * mesh_sigma_a[iz, ir] * phys.a * phys.c * mesh_temp[iz, ir]**4 * zone_volume * dt)

            # 2. Get deterministic samples and weights for each dimension
            r_vals, r_weights = weighted_sample_radius(mesh_r_edges[ir], mesh_r_edges[ir+1], part_Ny[iz, ir])
            
            # Spatial Z (Uniform spacing)
            z_vals = deterministic_sample_z(mesh_z_edges[iz], mesh_z_edges[iz+1], part_Nx[iz, ir])
            
            mu_vals = deterministic_sample_mu_isotropic(part_Nmu[iz,ir])
            phi_vals = deterministic_sample_phi_isotropic(part_Nphi[iz, ir])
            # print(f'mu_vals = {mu_vals}')
            # print(f'phi_vals = {phi_vals}')
            # Time (Uniform or Tanh)
            t_vals = deterministic_sample_z(current_time, current_time + dt, part_Nt[iz, ir])

            # Total particles in this cell
            n_cell_ptcls = len(r_vals) * len(z_vals) * len(mu_vals) * len(phi_vals) * len(t_vals)
            if n_cell_ptcls == 0: continue
            
            # Base energy per particle
            base_nrg = e_cell / n_cell_ptcls

            # 3. Nested loops to "tile" the phase space
            for z in z_vals:
                for i_r, r in enumerate(r_vals):
                    for i_m, mu in enumerate(mu_vals):
                        for i_p, phi in enumerate(phi_vals):
                            for ttt in t_vals:
                                if n_particles >= part.max_array_size:
                                    print("Warning: Max particle size reached.")
                                    return n_particles, particle_prop
                                weighted_nrg = base_nrg * r_weights[i_r]
                                particle_prop[n_particles] = [
                                    ttt, float(iz), float(ir), z, r, mu, phi, 0.0, weighted_nrg, weighted_nrg
                                ]
                                n_particles += 1

    # Print diagnostics
    actual_sum = np.sum(particle_prop[start_count:n_particles, 8])
    theoretical_sum = np.sum(mesh_fleck * mesh_sigma_a * phys.a * phys.c * mesh_temp**4 * volumes * dt)
    print(f"Added {n_particles - start_count} deterministic particles.")
    print(f"Energy Actual: {actual_sum:.6e} | Theoretical: {theoretical_sum:.6e}")
    
    return n_particles, particle_prop


def RZ_body_particles_nrn_angles_rn(n_particles, particle_prop, 
                          part_Nx, part_Ny, part_Nmu, part_Nphi, part_Nt,
                          current_time, dt, 
                          mesh_r_edges, mesh_z_edges, 
                          mesh_temp, mesh_fleck, 
                          mesh_sigma_a):
    
    nz_cells = len(mesh_z_edges) - 1
    nr_cells = len(mesh_r_edges) - 1
    
    # Pre-calculate RZ volumes: V = pi * (r_out^2 - r_in^2) * dz
    dz = np.diff(mesh_z_edges)
    dr2 = np.diff(mesh_r_edges**2)
    volumes = np.pi * np.outer(dz, dr2)
    
    start_count = n_particles
    
    for iz in range(nz_cells):
        for ir in range(nr_cells):
            if mesh_temp[iz, ir] <= 2.59E-5:
                continue

            # 1. Total energy emitted by this cell
            zone_volume = volumes[iz, ir]
            e_cell = (mesh_fleck[iz, ir] * mesh_sigma_a[iz, ir] * phys.a * phys.c * mesh_temp[iz, ir]**4 * zone_volume * dt)

            # 2. Spatial/Temporal Deterministic Samples
            r_vals, r_weights = weighted_sample_radius(mesh_r_edges[ir], mesh_r_edges[ir+1], part_Ny[iz, ir])
            z_vals = deterministic_sample_z(mesh_z_edges[iz], mesh_z_edges[iz+1], part_Nx[iz, ir])
            t_vals = deterministic_sample_z(current_time, current_time + dt, part_Nt[iz, ir])

            # Total particles in this cell (Using Nmu/Nphi from your grid as multipliers)
            # This ensures the energy is divided correctly among the random samples
            n_mu = int(part_Nmu[iz, ir])
            n_phi = int(part_Nphi[iz, ir])
            n_cell_ptcls = len(r_vals) * len(z_vals) * len(t_vals) * n_mu * n_phi
            
            if n_cell_ptcls == 0: continue
            
            # Base energy per particle
            base_nrg = e_cell / n_cell_ptcls

            # 3. Nested loops for Space/Time
            for z in z_vals:
                for i_r, r in enumerate(r_vals):
                    for ttt in t_vals:
                        # For every space-time point, we generate Nmu * Nphi random angle pairs
                        for _ in range(n_mu * n_phi):
                            if n_particles >= particle_prop.shape[0]: # Use .shape[0] for safety
                                print("Warning: Max particle size reached.")
                                return n_particles, particle_prop
                            
                            # Randomly sample the angles
                            mu = sample_mu_isotropic()   # Typically np.random.uniform(-1, 1)
                            phi = sample_phi_isotropic() # Typically np.random.uniform(0, 2*np.pi)
                            
                            weighted_nrg = base_nrg * r_weights[i_r]
                            
                            particle_prop[n_particles] = [
                                ttt, float(iz), float(ir), z, r, mu, phi, 0.0, weighted_nrg, weighted_nrg
                            ]
                            n_particles += 1

    # Print diagnostics
    actual_sum = np.sum(particle_prop[start_count:n_particles, 8])
    theoretical_sum = np.sum(mesh_fleck * mesh_sigma_a * phys.a * phys.c * mesh_temp**4 * volumes * dt)
    print(f"Added {n_particles - start_count} particles (Deterministic Space/Time, Random Angles).")
    print(f"Energy Actual: {actual_sum:.6e} | Theoretical: {theoretical_sum:.6e}")
    
    return n_particles, particle_prop

def RZ_volume_source_DPT(n_particles, particle_prop,
                         part_Nx, part_Ny, part_Nmu, part_Nphi, part_Nt,
                         current_time, dt,
                         mesh_r_edges, mesh_z_edges,
                         vol_x_0, dz, phys_a, phys_c, max_array_size
):
    """Create Source Particles for an RZ volume Source that spans Z. Assume 1 R cell"""
    # Define the source intensity and total energy per cell
    z_source_cells = int(np.ceil(0.5/0.05))
    
    dz = np.diff(mesh_z_edges)
    dr2 = mesh_r_edges[1]**2 - mesh_r_edges[0]**2
    volumes = np.pi * dr2 * dz  # Actual volume in cm^3

    # 2. Physics Constants
    # unit radiation source
    S = 1.0 * phys.a * phys.c 

    # 3. Energy per cell
    # In 1D: E = S * dt * dx
    # In 2D: E = S * dt * V
    e_source = S * dt * volumes[:z_source_cells]

    e_total_vol = np.sum(e_source)
    print(f'2D Total Energy: {e_total_vol}')

    start_count = n_particles

    for iz in range(z_source_cells):
        # Generate z
        z_min = mesh_z_edges[iz]
        z_max = mesh_z_edges[iz+1]
        z_samples = part_Nx[iz,0]
        z_values = deterministic_sample_z(z_min, z_max, z_samples)

        # Generate r
        r_min = mesh_r_edges[0]
        r_max = mesh_r_edges[1]
        r_samples = part_Ny[iz,0]
        r_values = deterministic_sample_radius(r_min, r_max, r_samples)
        
        # Generate mu
        mu_samples = part_Nmu[iz,0]
        mu_values = deterministic_sample_mu_isotropic(mu_samples)

        # Generate phi
        phi_samples = part_Nphi[iz,0]
        phi_values = deterministic_sample_phi_isotropic(phi_samples)
        
        # Generate t
        t_samples = part_Nt[iz,0]
        t_min = current_time
        t_max = current_time + dt
        t_values = deterministic_sample_z(t_min, t_max, t_samples)


        # Calculate energy weight per particle
        n_source_ptcls = len(z_values) * len(r_values) * len(mu_values) * len(phi_values) * len(t_values)
        base_nrg = e_source[iz] / n_source_ptcls

        # Create particles and add them to the preallocated array
        for z in z_values:
                for i_r, r in enumerate(r_values):
                    for mu in mu_values:
                        for phi in phi_values:
                            for i_t, ttt in enumerate(t_values):
                                if n_particles < max_array_size:
                                    weighted_nrg = base_nrg
                                    # Assign: [emission_time, z_idx, r_idx, z, r, mu, phi, frq, nrg, startnrg]
                                    particle_prop[n_particles] = [ttt, iz, 0, z, r, mu, phi, 0, weighted_nrg, weighted_nrg]
                                    n_particles += 1
                                else:
                                    print("Warning: Maximum number of particles reached!")
    print("Added", n_particles - start_count, "volume-source particles.")
    return n_particles, particle_prop