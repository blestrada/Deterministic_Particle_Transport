"""Source IMC particles"""

import numpy as np

import imc_global_part_data as part
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_bcon_data as bcon
import imc_global_time_data as time


def create_census_particles():
    """Creates census particles for the first time-step"""
    for icell in range(mesh.ncells):
        # Create position, angle, and scattering arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
        xi_values = (np.arange(part.Nxi) + 0.5) / part.Nxi

        # Assign energy-weights
        n_census_ptcls = part.Nx * part.Nmu * part.Nxi
        nrg = phys.a * (mesh.radtemp[icell] ** 4) * mesh.dx / n_census_ptcls
        startnrg = nrg

        # Assign origin and time of emission
        ttt = time.time
        origin = icell

        # Create particles and add them to the global list
        for xpos in x_positions:
            for mu in angles:
                for xi in xi_values:
                    part.particle_prop.append([origin, ttt, icell, xpos, mu, xi, nrg, startnrg])
                    part.census_grid.append([icell, xpos, mu, xi, 0])

def create_body_source_particles():
    """Creates source particles for the mesh"""
    for icell in range(mesh.ncells):
        # Create position, angle, time, and scattering arrays
        x_positions = mesh.nodepos[icell] + (np.arange(part.Nx) + 0.5) * mesh.dx / part.Nx
        angles = -1 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
        emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt
        xi_values = (np.arange(part.Nxi) + 0.5) / part.Nxi

        # Assign energy-weights
        n_source_ptcls = part.Nx * part.Nmu * part.Nt * part.Nxi
        nrg = phys.c * mesh.sigma_a[icell] * phys.a * (mesh.temp[icell] ** 4) * time.dt * mesh.dx / n_source_ptcls
        startnrg = nrg
        # Create particles and add them to global list
        origin = icell
        for xpos in x_positions:
            for mu in angles:
                for ttt in emission_times:
                    for xi in xi_values:
                        part.particle_prop.append([origin, ttt, icell, xpos, mu, xi, nrg, startnrg])


def create_surface_source_particles():
    """Creates source particles for the boundary condition"""
    e_surf = phys.sb * bcon.T0 ** 4 * time.dt

    # Create source particles for the surface
    xpos = 1e-5
    angles = -1 + (np.arange(part.Nmu) + 0.5) * 2 / part.Nmu
    emission_times = time.time + (np.arange(part.Nt) + 0.5) * time.dt / part.Nt
    xi_values = (np.arange(part.Nxi) + 0.5) / part.Nxi

    # Create energy-weights
    n_source_ptcls = len(angles) * len(emission_times) * len(xi_values)
    nrg = e_surf / n_source_ptcls
    startnrg = nrg
    icell = 0  # starts in leftmost cell
    origin = icell
    # Create particles and add them to global list
    for mu in angles:
        for ttt in emission_times:
            for xi in xi_values:
                part.particle_prop.append([origin, ttt, icell, xpos, mu, xi, nrg, startnrg])
