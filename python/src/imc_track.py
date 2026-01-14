"""Advance particles over a time-step"""

import numpy as np
from numba import njit, jit, objmode, prange, get_num_threads, config
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numba

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_part_data as ptcl
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_source

@njit
def chi_equation(chi, x_0, x_1, dx, X_s):
    chi = np.asarray(chi).item()
    X_s = np.asarray(X_s).item()

    chi_dx = chi * dx

    if np.abs(chi_dx) > 100:
        exp_chi_dx = np.exp(50)  # Limit exponential growth
    elif np.abs(chi_dx) > 1e-5:
        exp_chi_dx = np.exp(chi_dx).item()  
    else:
        exp_chi_dx = 1.0 + chi_dx + 0.5 * chi_dx ** 2  # Taylor expansion

    # Compute numerator and denominator 
    numerator = 1.0 - chi * x_0 + exp_chi_dx * (chi * x_1 - 1.0)
    denominator = chi * (exp_chi_dx - 1.0)

    # Compute result
    scalar_result = numerator / denominator
    return scalar_result - X_s

@njit
def gamma_equation(gamma, y_0, y_1, dy, Y_s):
    gamma = np.asarray(gamma).item()  
    Y_s = np.asarray(Y_s).item()  

    gamma_dy = gamma * dy

    if np.abs(gamma_dy) > 100:
        exp_gamma_dy = np.exp(50)  # Limit exponential growth
    elif np.abs(gamma_dy) > 1e-5:
        exp_gamma_dy = np.exp(gamma_dy).item()  
    else:
        exp_gamma_dy = 1.0 + gamma_dy + 0.5 * gamma_dy ** 2  # Taylor expansion

    # Compute numerator and denominator 
    numerator = 1.0 - gamma * y_0 + exp_gamma_dy * (gamma * y_1 - 1.0)
    denominator = gamma * (exp_gamma_dy - 1.0)

    # Compute result
    scalar_result = numerator / denominator
    return scalar_result - Y_s


@njit
def tau_equation(tau, t_0, t_1, dt, T_s):
    tau = np.asarray(tau).item()  
    T_s = np.asarray(T_s).item()  
    if np.abs(tau * dt) > 100:
        exp_tau_dt = np.exp(50)
    elif np.abs(tau * dt) > 1e-5:
        exp_tau_dt = np.exp(tau * dt)
    else:
        exp_tau_dt = 1.0 + (tau * dt) + 0.5 * (tau * dt) ** 2

    return (1 - tau * t_0 + exp_tau_dt * (tau * t_1 - 1.0)) / (tau * (exp_tau_dt - 1.0)) - T_s


@njit
def p_x_t_solve(chi, tau, dx, x_0, x, t, t_0, dt):
    
    if np.abs(chi * dx) > 100:
        exp_chi_dx = np.exp(50)
    elif np.abs(chi * dx) > 1e-5:
        exp_chi_dx = np.exp(chi * dx)
    else:
        exp_chi_dx = 1.0 + (chi * dx) + 0.5 * (chi * dx) ** 2
    if np.abs(tau * dt) > 100:
        exp_tau_dt = np.exp(50)
    elif np.abs(tau * dt) > 1e-5:
        exp_tau_dt = np.exp(tau * dt)
    else:
        exp_tau_dt = 1.0 + (tau * dt) + 0.5 * (tau * dt) ** 2


    if np.abs(chi * (x - x_0)) > 100:
        exp_chi = np.exp(50)
    elif np.abs(chi * (x - x_0)) > 1e-5:
        exp_chi = np.exp(chi * (x - x_0))
    else:
        exp_chi = 1.0 + (chi * (x - x_0)) + 0.5 * (chi * (x - x_0)) ** 2

    if np.abs(tau * (t - t_0)) > 100:
        exp_tau = np.exp(50)
    elif np.abs(tau * (t - t_0)) > 1e-5:
        exp_tau = np.exp(tau * (t - t_0))
    else:
        exp_tau = 1.0 + (tau * (t - t_0)) + 0.5 * (tau * (t - t_0)) ** 2

    P = chi * tau * (exp_chi * exp_tau) / \
        ((exp_chi_dx - 1.0) * (exp_tau_dt - 1.0))
    return P


@njit
def p_x_y_t_solve(chi, gamma, tau, dx, dy, dt, x_0, x, y_0, y, t_0, t):
    
    if np.abs(chi * dx) > 100:
        exp_chi_dx = np.exp(50)
    elif np.abs(chi * dx) > 1e-5:
        exp_chi_dx = np.exp(chi * dx)
    else:
        exp_chi_dx = 1.0 + (chi * dx) + 0.5 * (chi * dx) ** 2

    if np.abs(gamma * dy) > 100:
        exp_gamma_dy = np.exp(50)
    elif np.abs(gamma * dy) > 1e-5:
        exp_gamma_dy = np.exp(gamma * dy)
    else:
        exp_gamma_dy = 1.0 + (gamma * dy) + 0.5 * (gamma * dy) ** 2

    if np.abs(tau * dt) > 100:
        exp_tau_dt = np.exp(50)
    elif np.abs(tau * dt) > 1e-5:
        exp_tau_dt = np.exp(tau * dt)
    else:
        exp_tau_dt = 1.0 + (tau * dt) + 0.5 * (tau * dt) ** 2


    if np.abs(chi * (x - x_0)) > 100:
        exp_chi = np.exp(50)
    elif np.abs(chi * (x - x_0)) > 1e-5:
        exp_chi = np.exp(chi * (x - x_0))
    else:
        exp_chi = 1.0 + (chi * (x - x_0)) + 0.5 * (chi * (x - x_0)) ** 2

    if np.abs(gamma * (y - y_0)) > 100:
        exp_gamma = np.exp(50)
    elif np.abs(gamma * (y - y_0)) > 1e-5:
        exp_gamma = np.exp(gamma * (y - y_0))
    else:
        exp_gamma = 1.0 + (gamma * (y - y_0)) + 0.5 * (gamma * (y - y_0)) ** 2

    if np.abs(tau * (t - t_0)) > 100:
        exp_tau = np.exp(50)
    elif np.abs(tau * (t - t_0)) > 1e-5:
        exp_tau = np.exp(tau * (t - t_0))
    else:
        exp_tau = 1.0 + (tau * (t - t_0)) + 0.5 * (tau * (t - t_0)) ** 2

    P = chi * gamma * tau * (exp_chi * exp_gamma * exp_tau) / \
        ((exp_chi_dx - 1.0) * (exp_tau_dt - 1.0)) * ((exp_gamma_dy - 1.0))
    return P

@njit
def run_random(n_particles, particle_prop, current_time, dt, mesh_sigma_a, mesh_fleck, mesh_sigma_s):
    """Advance particles over a time-step"""
    # Create local storage for the energy deposited this time-step
    nrgdep = np.zeros(mesh.ncells, dtype=np.float64)

    # optimizations
    endsteptime = current_time + dt
    phys_c = phys.c
    mesh_nodepos = mesh.nodepos
    top_cell = mesh.ncells - 1
    phys_invc = phys.invc
    mesh_rightbc = mesh.right_bc
    mesh_leftbc = mesh.left_bc

    print(f'Particle Loop')

    # Loop over all particles
    for iptcl in range(n_particles):
        # Get particle's initial properties at start of time-step
        ttt = particle_prop[iptcl, 1]
        icell = int(particle_prop[iptcl, 2])  # Convert to int
        xpos = particle_prop[iptcl, 3]
        mu = particle_prop[iptcl, 4]
        frq = particle_prop[iptcl, 5]
        nrg = particle_prop[iptcl, 6]
        startnrg = particle_prop[iptcl, 7]
        
        # Loop over segments in the history (between boundary-crossings and collisions)
        while True:
            # with objmode:
            #     print(f'iptcl = {iptcl}')
            #     print(f'icell = {icell}')
            #     print(f'xpos = {xpos}')
            #     print(f'nrg = {nrg}')
            #     print(f'ttt = {ttt}')
            #     print(f'endsteptime = {endsteptime}')

            # Calculate distance to boundary
            dist_b = distance_to_boundary_1D(xpos, mu, mesh_nodepos[icell], mesh_nodepos[icell+1])
        
            # Calculate distance to census
            dist_cen = distance_to_census(phys_c, ttt, dt)

            # Calculate distance to collision
            dist_coll = distance_to_collision(mesh_sigma_a[icell], mesh_sigma_s[icell], mesh_fleck[icell])

            # Actual distance - whichever happens first
            dist = min(dist_b, dist_cen, dist_coll)

            # Calculate new particle energy
            newnrg = nrg * np.exp(-mesh_sigma_a[icell] * mesh_fleck[icell] * dist)

            # print(f'newnrg = {newnrg}')

            nrgdep[icell] += nrg - newnrg
    
            # Advance position, time, and energy
            xpos += mu * dist
            ttt += dist * phys_invc
            nrg = newnrg

            # Boundary treatment
            if dist == dist_b:
                # Left boundary treatment
                if mu < 0: # If going left
                    if icell == 0: # If at the leftmost cell
                        if mesh_leftbc == 'vacuum':
                            # Flag particle for later destruction
                            particle_prop[iptcl][6] = -1.0
                            break
                        elif mesh_leftbc == 'reflecting':
                            mu *= -1.0  # Reverse direction
                    else:  # If not at the leftmost cell
                        icell -= 1  # Move to the left cell

                # Right boundary treatment
                elif mu > 0: # If going right
                    if icell == top_cell:
                        if mesh_rightbc == 'vacuum':
                            # Flag particle for later destruction
                            particle_prop[iptcl][6] = -1.0
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu *= -1.0  # Reverse direction
                    else:  # If not at the top cell
                        icell += 1  # Move to the right cell
            
            # If the event was census, finish this history
            if dist == dist_cen:
                # Update the particle's properties in the array
                particle_prop[iptcl, 1] = ttt
                particle_prop[iptcl, 2] = icell
                particle_prop[iptcl, 3] = xpos
                particle_prop[iptcl, 4] = mu
                particle_prop[iptcl, 5] = frq
                particle_prop[iptcl, 6] = nrg
                break  # Finish history for this particle 
                
            # If event was collision, also update and direction
            if dist == dist_coll:
                # Collision (i.e. absorption, but treated as pseudo-scattering)
                mu = 1.0 - 2.0 * np.random.uniform()
                

        # End loop over history segments

    # End loop over particles
    return nrgdep, n_particles, particle_prop

@njit(parallel=True)
def run_random_parallel(n_particles,
                        particle_prop,
                        current_time,
                        dt,
                        mesh_sigma_a,
                        mesh_fleck,
                        mesh_sigma_s):
    """
    Parallel version of run_random using numba.prange and per-thread accumulators.

    Returns
    -------
    nrgdep : np.ndarray (ncells,)
        Energy deposited per cell (summed over all particles).
    n_particles : int
        Same as input (for API compatibility).
    particle_prop : np.ndarray
        Updated particle properties (some particles may be flagged with negative energy).
    """
    #-------------------------------------------------------
    # Local references (assumes these globals exist)
    #-------------------------------------------------------
    ncells = mesh.ncells
    mesh_nodepos = mesh.nodepos
    top_cell = ncells - 1
    mesh_rightbc = mesh.right_bc
    mesh_leftbc = mesh.left_bc

    phys_c = phys.c
    phys_invc = phys.invc

    # number of threads and per-thread accumulation buffer
    nthreads = get_num_threads()
    # shape: (nthreads, ncells)
    nrgdep_local = np.zeros((nthreads, ncells), dtype=np.float64)

    # MAIN parallel loop over particles
    for iptcl in prange(n_particles):
        # thread id for indexing per-thread local buffer
        tid = numba.np.ufunc.parallel._get_thread_id()  # thread-local index

        # load particle properties (keep local copies)
        ttt = particle_prop[iptcl, 1]
        icell = int(particle_prop[iptcl, 2])
        xpos = particle_prop[iptcl, 3]
        mu = particle_prop[iptcl, 4]
        frq = particle_prop[iptcl, 5]
        nrg = particle_prop[iptcl, 6]
        startnrg = particle_prop[iptcl, 7]

        # advance this particle's history for the timestep
        while True:
            # distance to boundary (1D cell boundaries)
            dist_b = distance_to_boundary_1D(xpos, mu, mesh_nodepos[icell], mesh_nodepos[icell+1])

            # distance to census
            dist_cen = distance_to_census(phys_c, ttt, dt)

            # distance to collision
            dist_coll = distance_to_collision(mesh_sigma_a[icell], mesh_sigma_s[icell], mesh_fleck[icell])

            # actual event distance
            # use min with explicit comparisons to avoid ambiguous equalities causing numerical branching surprises
            dist = dist_b
            if dist_cen < dist:
                dist = dist_cen
            if dist_coll < dist:
                dist = dist_coll

            # compute attenuated energy after travelling dist
            newnrg = nrg * np.exp(-mesh_sigma_a[icell] * mesh_fleck[icell] * dist)

            # accumulate deposited energy into thread-local buffer
            nrgdep_local[tid, icell] += (nrg - newnrg)

            # advance particle
            xpos += mu * dist
            ttt += dist * phys_invc
            nrg = newnrg

            # boundary handling
            if dist == dist_b:
                # left-going
                if mu < 0.0:
                    if icell == 0:
                        if mesh_leftbc == 'vacuum':
                            particle_prop[iptcl, 6] = -1.0  # flag for deletion
                            break
                        elif mesh_leftbc == 'reflecting':
                            mu = -mu
                    else:
                        icell -= 1
                # right-going
                elif mu > 0.0:
                    if icell == top_cell:
                        if mesh_rightbc == 'vacuum':
                            particle_prop[iptcl, 6] = -1.0
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu = -mu
                    else:
                        icell += 1

            # census event: stash updated properties and finish particle
            if dist == dist_cen:
                particle_prop[iptcl, 1] = ttt
                particle_prop[iptcl, 2] = icell
                particle_prop[iptcl, 3] = xpos
                particle_prop[iptcl, 4] = mu
                particle_prop[iptcl, 5] = frq
                particle_prop[iptcl, 6] = nrg
                break

            # collision event: update direction and continue history
            if dist == dist_coll:
                mu = 1.0 - 2.0 * np.random.uniform()

            # loop continues until break
        # end while per-particle history
    # end parallel loop over particles

    # Reduce per-thread accumulators into single nrgdep array
    nrgdep = np.zeros(ncells, dtype=np.float64)
    for ic in range(ncells):
        s = 0.0
        for th in range(nthreads):
            s += nrgdep_local[th, ic]
        nrgdep[ic] = s

    return nrgdep, n_particles, particle_prop

@njit
def run(n_particles, particle_prop, current_time, dt, Nmu, mesh_sigma_a, mesh_sigma_s, mesh_sigma_t, mesh_fleck):
    """Advance particles over a time-step, including implicit scattering."""

    nrgdep = np.zeros(mesh.ncells, dtype=np.float64)
    nrgscattered = np.zeros(mesh.ncells, dtype=np.float64)
    xEs = np.zeros(mesh.ncells, dtype=np.float64)
    tEs = np.zeros(mesh.ncells, dtype=np.float64)
    w_average = np.zeros(mesh.ncells, dtype=np.float64)
    w_average_times_mu_squared = np.zeros(mesh.ncells, dtype=np.float64)
    # Optimizations
    endsteptime = current_time + dt
    mesh_nodepos = mesh.nodepos
    phys_c = phys.c
    top_cell = mesh.ncells - 1
    phys_invc = phys.invc
    mesh_rightbc = mesh.right_bc
    mesh_leftbc = mesh.left_bc

    print(f'Particle Loop')

    # Loop over all active particles
    for iptcl in range(n_particles):
        # Get particle's initial properties at start of time-step
        ttt = particle_prop[iptcl, 1]
        icell = int(particle_prop[iptcl, 2])  # Convert to int
        xpos = particle_prop[iptcl, 3]
        mu = particle_prop[iptcl, 4]
        frq = particle_prop[iptcl, 5]
        nrg = particle_prop[iptcl, 6]
        startnrg = particle_prop[iptcl, 7]
            
        # Loop over segments in the history (between boundary-crossings and collisions)
        while True:
            # with objmode:
            #     print(f'iptcl = {iptcl}')
            #     print(f'icell = {icell}')
            #     print(f'xpos = {xpos}')
            #     print(f'nrg = {nrg}')
            #     print(f'ttt = {ttt}')
            #     print(f'endsteptime = {endsteptime}')
            #     print(f'time.time = {time.time}')
            #     print(f'time.dt = {time.dt}')
            #     print(f'calc = {time.time + time.dt}')
            # Calculate distance to boundary
            if mu > 0.0:
                dist_b = (mesh_nodepos[icell + 1] - xpos) / mu
            else:
                dist_b = (mesh_nodepos[icell] - xpos) / mu
            # with objmode:
            #     print(f'dist_b = {dist_b}')
            # Calculate distance to census
            dist_cen = phys_c * (endsteptime - ttt)
            # with objmode:
            #     print(f'dist_cen = {dist_cen}')
            # Actual distance - whichever happens first
            dist = min(dist_b, dist_cen)
            # with objmode:
            #     print(f'dist = {dist}')
            # Calculate new particle energy
            newnrg = nrg * np.exp(-mesh_sigma_t[icell] * dist)
            # # Check if the particle's energy falls below 0.01 * startnrg
            # if newnrg <= 0.01 * startnrg:
            #     newnrg = 0.0

            # Calculate energy change
            nrg_change = nrg - newnrg
            # with objmode:
            #     print(nrg_change)
            # Calculate fractions for absorption and scattering
            frac_absorbed = mesh_sigma_a[icell] * mesh_fleck[icell] / mesh_sigma_t[icell]
            frac_scattered = ((1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell] + mesh_sigma_s[icell]) / mesh_sigma_t[icell]

            # Update energy deposition tallies
            nrgdep[icell] += nrg_change * frac_absorbed
            nrgscattered[icell] += nrg_change * frac_scattered

            # Calculate the average length of scatter
            average_scatter_length = 1 / mesh_sigma_t[icell] * (1 - (1 + mesh_sigma_t[icell] * dist) * np.exp(-mesh_sigma_t[icell] * dist))/(1 - np.exp(-mesh_sigma_t[icell] * dist))
            average_position_of_scatter = xpos + mu * average_scatter_length
            # if iptcl== 0:
            #     # print(f'average scattering length = {average_scatter_length}')
            #     # print(f'average position of scatter = {average_position_of_scatter}')
            average_time_of_scatter = ttt + average_scatter_length / phys_c
            xEs[icell] += nrg_change * frac_scattered * average_position_of_scatter
            tEs[icell] += nrg_change * frac_scattered * average_time_of_scatter
            
            # Calculate average weight over the path
            w_average[icell] += nrg / mesh_sigma_t[icell] / dist * (1 - np.exp(-mesh_sigma_t[icell] * dist))
            w_average_times_mu_squared[icell] += nrg / mesh_sigma_t[icell] / dist * (1 - np.exp(-mesh_sigma_t[icell] * dist)) * mu ** 2

            # if newnrg == 0.0:
            #     # Flag particle for later destruction
            #     particle_prop[iptcl, 6] = -1.0
            #     break
            #print(f'nrgdep = {nrgdep[:10]}')
            

            # Advance position, time, and energy
            xpos += mu * dist
            ttt += dist * phys_invc
            nrg = newnrg

            # Boundary treatment
            if dist == dist_b:
                # Left boundary treatment
                if mu < 0:  # If going left
                    if icell == 0:  # At the leftmost cell
                        if mesh_leftbc == 'vacuum':
                            particle_prop[iptcl, 6] = -1.0  # Mark as destroyed
                            break
                        elif mesh_leftbc == 'reflecting':
                            mu *= -1.0  # Reflect particle
                            if mu == 0: raise ValueError
                    else:  # Move to the left cell
                        icell -= 1

                # Right boundary treatment
                elif mu > 0:  # If going right
                    if icell == top_cell:  # At the rightmost cell
                        if mesh_rightbc == 'vacuum':
                            particle_prop[iptcl, 6] = -1.0  # Mark as destroyed
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu *= -1.0  # Reflect particle
                            if mu == 0: raise ValueError
                    else:  # Move to the right cell
                        icell += 1

            # Check if event was census
            if dist == dist_cen:
                # Update the particle's properties in the array
                particle_prop[iptcl, 1] = ttt
                particle_prop[iptcl, 2] = icell
                particle_prop[iptcl, 3] = xpos
                particle_prop[iptcl, 4] = mu
                particle_prop[iptcl, 5] = frq
                particle_prop[iptcl, 6] = nrg
                break  # Finish history for this particle                    
    
    # Start implicit scattering process
    epsilon = 1e-3
    iterations = 0
    converged = False

    # Calculate zone-wise average position of scatter and average time of scatter
    X_s = np.zeros(mesh.ncells, dtype=np.float64)
    T_s = np.zeros(mesh.ncells, dtype=np.float64)
    # with objmode:
    #     print(f'xEs = {xEs[0]}')
    #     print(f'nrgscattered = {nrgscattered[:10]}')
    #     print(f'nrgscattered last 10 = {nrgscattered[-10:]}')
    X_s = xEs / nrgscattered # average position of scatter in a zone
    T_s = tEs / nrgscattered # average time of scatter in a zone
    # with objmode:
    #     print(f'average position of scatter = {X_s[0]}')
    # Create arrays for cell-valued chi and tau
    chi = np.zeros(mesh.ncells)
    tau = np.zeros(mesh.ncells)

    old_nrgscattered = np.copy(nrgscattered)

    while not converged:
        scattered_particles = np.zeros((ptcl.max_array_size, 8), dtype=np.float64)
        n_scattered_particles = 0
        # Store the old nrgscattered

        # Create source particles based on energy scattered in each cell
        P_tally = np.zeros(mesh.ncells, dtype=np.float64)
        
        for icell in range(mesh.ncells):
            # print(f'icell = {icell}')
            # print(f'current_time = {current_time}')
            X_s[icell] = np.round(X_s[icell] - mesh_nodepos[icell], 8)
            # print(f'absolute T_s = {T_s[icell]}')
            T_s[icell] = np.round(T_s[icell] - current_time, 8)
            # print(f'relative T_s = {T_s[icell]}')
            # print(f'icell = {icell}')
            # print(f'x_0 = {mesh_nodepos[icell]}, dx = {mesh.dx}, x_1 = {mesh_nodepos[icell + 1]}, X_s = {X_s[icell]}')
            
            # Create position, angle, time arrays
            x_positions = mesh.nodepos[icell] + (np.arange(ptcl.Nx) + 0.5) * mesh.dx / ptcl.Nx
            angles = -1.0 + (np.arange(Nmu[icell]) + 0.5) * 2 / Nmu[icell]
            emission_times = current_time + (np.arange(ptcl.Nt) + 0.5) * dt / ptcl.Nt
            # Assign energy-weights
            n_source_ptcls = ptcl.Nx * Nmu[icell] * ptcl.Nt
            nrg = nrgscattered[icell] / n_source_ptcls
            startnrg = nrg
            # Solve for chi and tau in each cell
            with objmode:
                method = 'lm'
                solution = root(chi_equation, 1, args=(float(0), mesh.dx, mesh.dx,  X_s[icell]), method=method, tol=1e-10)       
                chi[icell] = solution.x
                # print(f'chi solved = {chi[icell]}')
                solution = root(tau_equation, 1, args=(float(0), dt, dt,  T_s[icell]), method=method, tol=1e-10)
                tau[icell] = solution.x
                # print(f'tau solved = {tau[icell]}'

            # Create scattered particles
            for xpos in x_positions:
                for mu in angles:
                    for ttt in emission_times:
                        if n_scattered_particles < ptcl.max_array_size:
                            rel_x = xpos - mesh_nodepos[icell]
                            rel_t = ttt - current_time 
                            P = p_x_t_solve(chi[icell], tau[icell], mesh.dx, 0, rel_x, rel_t, 0, dt)
                            if P < 0:
                                print(f' P = {P}')
                                raise ValueError
                            P_tally[icell] += P
                            
                            idx = n_scattered_particles
                            scattered_particles[idx, 0] = icell  # origin
                            scattered_particles[idx, 1] = ttt  # time
                            scattered_particles[idx, 2] = icell  # cell index
                            scattered_particles[idx, 3] = xpos  # position
                            scattered_particles[idx, 4] = mu  # direction
                            scattered_particles[idx, 5] = frq
                            scattered_particles[idx, 6] = nrg  # energy
                            scattered_particles[idx, 7] = P  # start energy
                            n_scattered_particles += 1
                        else:
                            print("Warning: Maximum number of scattered particles reached!")

        # Put correct energy
        for i in range(n_scattered_particles):
            # Get P and icell
            icell = int(scattered_particles[i, 2])
            P = scattered_particles[i, 7]
            # Set particle energy
            nrg = nrgscattered[icell] * P / P_tally[icell]
            # Set particle startnrg
            scattered_particles[idx, 7] = nrg  # start energy

        # Reset nrgscattered
        nrgscattered[:] = 0.0
        xEs[:] = 0.0
        tEs[:] = 0.0
        # Loop over scattered particles
        for iptcl in range(n_scattered_particles):
            
            # Get particle's initial properties at start of time-step
            ttt = scattered_particles[iptcl, 1]
            icell = int(scattered_particles[iptcl, 2])  # Convert to int
            xpos = scattered_particles[iptcl, 3]
            mu = scattered_particles[iptcl, 4]
            frq = scattered_particles[iptcl, 5]
            nrg = scattered_particles[iptcl, 6]
            startnrg = scattered_particles[iptcl, 7]

            while True:
                # Calculate distance to boundary
                if mu > 0.0:
                    dist_b = (mesh_nodepos[icell + 1] - xpos) / mu
                else:
                    dist_b = (mesh_nodepos[icell] - xpos) / mu

                # Distance to census
                dist_cen = phys_c * (endsteptime - ttt)
                
                dist = min(dist_b, dist_cen)
                
                # Update energy
                newnrg = nrg * np.exp(-mesh_sigma_t[icell] * dist)

                # # Check if the particle's energy falls below 0.01 * startnrg
                # if newnrg <= 0.01 * startnrg:
                #     newnrg = 0.0

                nrg_change = nrg - newnrg
                frac_absorbed = mesh_sigma_a[icell] * mesh_fleck[icell] / mesh_sigma_t[icell]
                frac_scattered = (1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell] / mesh_sigma_t[icell] + mesh_sigma_s[icell] / mesh_sigma_t[icell]
                nrgdep[icell] += nrg_change * frac_absorbed
                nrgscattered[icell] += nrg_change * frac_scattered

                # Calculate the average length of scatter
                average_scatter_length = 1 / mesh_sigma_t[icell] * (1 - (1 + mesh_sigma_t[icell] * dist) * np.exp(-mesh_sigma_t[icell] * dist))/(1 - np.exp(-mesh_sigma_t[icell] * dist))
                average_position_of_scatter = xpos + mu * average_scatter_length
                average_time_of_scatter = ttt + average_scatter_length / phys_c
                xEs += nrg_change * frac_scattered * average_position_of_scatter
                tEs += nrg_change * frac_scattered * average_time_of_scatter

                # Calculate average weight over the path
                w_average[icell] += nrg / mesh_sigma_t[icell] / dist * (1 - np.exp(-mesh_sigma_t[icell] * dist))
                w_average_times_mu_squared[icell] += nrg / mesh_sigma_t[icell] / dist * (1 - np.exp(-mesh_sigma_t[icell] * dist)) * mu ** 2

                # if newnrg == 0.0:
                #     # Flag particle for later destruction
                #     scattered_particles[iptcl, 6] = -1.0
                #     break

                
                # Advance position and time
                xpos += mu * dist
                ttt += dist * phys_invc
                nrg = newnrg

                # Boundary treatment
                if dist == dist_b:
                    # Left boundary treatment
                    if mu < 0:  # If going left
                        if icell == 0:  # At the leftmost cell
                            if mesh_leftbc == 'vacuum':
                                scattered_particles[iptcl, 6] = -1.0  # Mark as destroyed
                                break
                            elif mesh_leftbc == 'reflecting':
                                mu *= -1.0  # Reflect particle
                                if mu == 0: raise ValueError
                        else:  # Move to the left cell
                            icell -= 1

                    # Right boundary treatment
                    elif mu > 0:  # If going right
                        if icell == top_cell:  # At the rightmost cell
                            if mesh_rightbc == 'vacuum':
                                scattered_particles[iptcl, 6] = -1.0  # Mark as destroyed
                                break
                            elif mesh_rightbc == 'reflecting':
                                mu *= -1.0  # Reflect particle
                                if mu == 0: raise ValueError
                        else:  # Move to the right cell
                            icell += 1

                # Census check
                if dist == dist_cen:
                    scattered_particles[iptcl, 1] = ttt
                    scattered_particles[iptcl, 2] = icell
                    scattered_particles[iptcl, 3] = xpos
                    scattered_particles[iptcl, 4] = mu
                    scattered_particles[iptcl, 5] = frq
                    scattered_particles[iptcl, 6] = nrg
                    break
        
        X_s = xEs / nrgscattered # average position of scatter in a zone
        T_s = tEs / nrgscattered # average time of scatter in a zone
        iterations += 1
        total_scattered_energy = np.sum(nrgscattered)
        total_old_scattered_energy = np.sum(old_nrgscattered)
        # with objmode:
        #     print(f'scattering iteration done.')
        #     print(f'total scattered energy = {total_scattered_energy}')
        #     print(f'total original scattered energy = {total_old_scattered_energy}')
        # with objmode:
        #     print(f' average position of scatter in iterative loop: {X_s[0]}')
        #     print(f' average time of scatter in iterative loop: {T_s[0]}')
        # Now we need to move all the processed scattered particles to the global particle array
        # Calculate how many particles we are going to add
        n_existing_particles = n_particles
        n_total_particles = n_existing_particles + n_scattered_particles

        # Check if the combined number of particles exceeds the maximum allowed size
        if n_total_particles > ptcl.max_array_size:
            print("Warning: Not enough space in the global array for all scattered particles.")
            raise ValueError
        else:
            n_to_add = n_scattered_particles

        # Copy particles to the global particle array
        if n_to_add > 0:
            # Copy relevant fields from scattered_particles to the global particle_prop array
            particle_prop[n_existing_particles:n_existing_particles + n_to_add, :] = scattered_particles[:n_to_add, :]

            # Update the global number of particles
            n_particles = n_existing_particles + n_to_add

        if abs(total_scattered_energy / total_old_scattered_energy) < epsilon:
            converged = True
    print(f'Number of scattering iterations = {iterations}')

    # Deposit left over scattered energy to conserve energy
    nrgdep[:] += nrgscattered[:]

    # Calculate Eddington Factor
    # with objmode:
    #     print(f'w_average_times_mu_squared = {w_average_times_mu_squared[:10]}')
    #     print(f'w_average = {w_average[:10]}')
    eddington = w_average_times_mu_squared / w_average
    eddington = 0
    # with objmode:
    #     print(f'eddington = {eddington[:10]}')
    #     if time.step % 100 == 0:  # Check if the current time step is a multiple of 25
    #         plt.figure()
    #         plt.plot(mesh.cellpos, eddington)
    #         plt.title(f'Eddington Factor @ time={endsteptime}')
    #         plt.xlabel('x')
    #         y_ticks = np.linspace(0, 1, 10)
    #         x_ticks = np.linspace(0, 10, 10)
    #         plt.yticks(y_ticks)
    #         plt.xticks(x_ticks)
    #         plt.ylim(0, 1.0)
            
    #         # Save the figure with the time in the filename
    #         filename = 'eddington_time_{:.4f}.png'.format(endsteptime)
    #         plt.savefig(filename, format='png')
    #         plt.close()

    # Update global energy deposited mesh
    return nrgdep, n_particles, particle_prop, eddington


@njit(parallel=True)
def run_parallel_firstloop(
    n_particles,
    particle_prop,
    current_time,
    dt,
    Nmu,
    mesh_sigma_a,
    mesh_sigma_s,
    mesh_sigma_t,
    mesh_fleck
):
    
    # local aliases / constants
    ncells = mesh.ncells
    mesh_nodepos = mesh.nodepos
    top_cell = ncells - 1
    phys_c = phys.c
    phys_invc = phys.invc
    mesh_rightbc = mesh.right_bc
    mesh_leftbc = mesh.left_bc
    endsteptime = current_time + dt

    

    # Global tallies
    nrgdep = np.zeros(ncells, dtype=np.float64)
    nrgscattered = np.zeros(ncells, dtype=np.float64)
    x_Es = np.zeros(ncells, dtype=np.float64)
    tEs = np.zeros(ncells, dtype=np.float64)

    # Private tallies for each thread
    n_threads = get_num_threads()
    priv_nrgdep = np.zeros((n_threads, ncells), dtype=np.float64)
    priv_nrgscat = np.zeros((n_threads, ncells), dtype=np.float64)
    priv_xEs = np.zeros((n_threads, ncells), dtype=np.float64)
    priv_tEs = np.zeros((n_threads, ncells), dtype=np.float64)


    # Parallel loop over particles
    for iptcl in prange(n_particles):
        tid = numba.np.ufunc.parallel._get_thread_id()  # thread-local index

        # unpack particle
        ttt = particle_prop[iptcl, 1]
        icell = int(particle_prop[iptcl, 2])
        xpos = particle_prop[iptcl, 3]
        mu = particle_prop[iptcl, 4]
        frq = particle_prop[iptcl, 5]
        nrg = particle_prop[iptcl, 6]
        startnrg = particle_prop[iptcl, 7]

        # advance this particle for the timestep
        while True:
            # distance to boundary
            if mu > 0.0:
                dist_b = (mesh_nodepos[icell + 1] - xpos) / mu
            else:
                dist_b = (mesh_nodepos[icell] - xpos) / mu

            # distance to census
            dist_cen = phys_c * (endsteptime - ttt)

            # min event distance
            dist = dist_b
            if dist_cen < dist:
                dist = dist_cen

            # attenuation using total opacity
            newnrg = nrg * np.exp(-mesh_sigma_t[icell] * dist)

            nrg_change = nrg - newnrg

            # fractions
            frac_absorbed = mesh_sigma_a[icell] * mesh_fleck[icell] / mesh_sigma_t[icell]
            frac_scattered = ((1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell] + mesh_sigma_s[icell]) / mesh_sigma_t[icell]

            # update PRIVATE tallies
            priv_nrgdep[tid, icell] += nrg_change * frac_absorbed
            priv_nrgscat[tid, icell] += nrg_change * frac_scattered
            
            # implicit scattering tallies
            average_scatter_length = (1 / mesh_sigma_t[icell] *
                                      (1 - (1 + mesh_sigma_t[icell] * dist) *
                                       np.exp(-mesh_sigma_t[icell] * dist)) /
                                      (1 - np.exp(-mesh_sigma_t[icell] * dist)))
            avg_x = xpos + mu * average_scatter_length
            avg_t = ttt + average_scatter_length / phys_c

            priv_xEs[tid, icell] += nrg_change * frac_scattered * avg_x
            priv_tEs[tid, icell] += nrg_change * frac_scattered * avg_t


            # advance particle
            xpos += mu * dist
            ttt += dist * phys_invc
            nrg = newnrg

            # boundary handling
            if dist == dist_b:
                if mu < 0.0:
                    if icell == 0:
                        if mesh_leftbc == 'vacuum':
                            particle_prop[iptcl, 6] = -1.0
                            break
                        elif mesh_leftbc == 'reflecting':
                            mu = -mu
                    else:
                        icell -= 1
                elif mu > 0.0:
                    if icell == top_cell:
                        if mesh_rightbc == 'vacuum':
                            particle_prop[iptcl, 6] = -1.0
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu = -mu
                    else:
                        icell += 1

            # census: write back and finish this history
            if dist == dist_cen:
                particle_prop[iptcl, 1] = ttt
                particle_prop[iptcl, 2] = icell
                particle_prop[iptcl, 3] = xpos
                particle_prop[iptcl, 4] = mu
                particle_prop[iptcl, 5] = frq
                particle_prop[iptcl, 6] = nrg
                break

    # --- Reduction step ---
    for t in range(n_threads):
        nrgdep += priv_nrgdep[t]
        nrgscattered += priv_nrgscat[t]
        x_Es += priv_xEs[t]
        tEs += priv_tEs[t]

    return nrgdep, nrgscattered, x_Es, tEs

def generate_scattered_particles1D(
    nrgscattered,
    x_Es, t_Es,
    mesh_nodepos, mesh_dx,
    ptcl_max_array_size, Nx, Nt, Nmu,
    current_time, dt
):
    ncells= len(mesh_nodepos) - 1
    mesh_dx = np.diff(mesh_nodepos)
    # allocate scattered particle array
    scattered_particles = np.zeros((ptcl_max_array_size, 8), dtype=np.float64)
    n_scattered_particles = 0

    # Average scatter positions and times
    X_s = np.divide(x_Es, nrgscattered, out=np.zeros_like(x_Es), where=nrgscattered > 0.0)
    T_s = np.divide(t_Es, nrgscattered, out=np.zeros_like(t_Es), where=nrgscattered > 0.0)

    # Allocate chi, gamma, tau per cell
    chi   = np.zeros(ncells, dtype=np.float64)
    tau   = np.zeros(ncells, dtype=np.float64)

    P_tally = np.zeros(ncells, dtype=np.float64)

    # Loop over cells
    for ix in range(ncells):
        if nrgscattered[ix] <= 0.0:
            continue  # no energy to scatter

        # Shifted relative averages
        rel_X = np.round(X_s[ix] - mesh_nodepos[ix], 8)
        rel_T = np.round(T_s[ix] - current_time, 8)

        dx_cell = mesh_dx[ix]

        # Space, angle, and time grids
        x_positions = mesh_nodepos[ix] + (np.arange(Nx) + 0.5) * dx_cell / Nx
        angles = -1 + (np.arange(Nmu[ix]) + 0.5) * 2 / Nmu[ix]
        emission_times = current_time + (np.arange(Nt) + 0.5) * dt / Nt

        n_source_ptcls = Nx * Nmu[ix] * Nt
        nrg = nrgscattered[ix] / n_source_ptcls

        # Solve for chi, gamma, tau
        sol = root(chi_equation, 1, args=(0.0, dx_cell, dx_cell, rel_X), method='hybr')
        chi[ix] = sol.x
        sol = root(tau_equation, 1, args=(0.0, dt, dt, rel_T), method='hybr')
        tau[ix] = sol.x

        # Create scattered particles
        for xpos in x_positions:
            for mu in angles:
                for ttt in emission_times:
                    if n_scattered_particles >= ptcl_max_array_size:
                        raise RuntimeError("Maximum number of scattered particles reached")

                    rel_x = xpos - mesh_nodepos[ix]

                    P = p_x_t_solve(
                        chi[ix], tau[ix],
                        dx_cell, 0,
                        rel_x, rel_T, 0, dt
                    )

                    if P < 0:
                        raise ValueError(f"Negative probability P={P}")
                    P_tally[ix] += P

                    idx = n_scattered_particles
                    scattered_particles[idx, 1] = ttt   # emission time
                    scattered_particles[idx, 2] = ix    # x cell index
                    scattered_particles[idx, 3] = xpos  # x position
                    scattered_particles[idx, 4] = mu # angle
                    scattered_particles[idx, 5] = 0     # frequency (placeholder)
                    scattered_particles[idx, 6] = nrg   # particle energy
                    scattered_particles[idx, 7] = P     # probability weight

                    n_scattered_particles += 1
    # Put correct energy
    for i in range(n_scattered_particles):
        # Get P and icell
        icell = int(scattered_particles[i, 2])
        P = scattered_particles[i, 7]
        # Set particle energy
        nrg = nrgscattered[icell] * P / P_tally[icell]
        # Set particle startnrg
        scattered_particles[idx, 6] = nrg  # start energy

    return scattered_particles, n_scattered_particles


@njit
def run2D(n_particles, particle_prop, current_time, dt, sigma_a, sigma_s, sigma_t, fleck, mesh_x_edges, mesh_y_edges):
    """Advance particles over a time-step"""

    num_x_cells = len(mesh_x_edges) - 1
    num_y_cells = len(mesh_y_edges) - 1
    nrgdep = np.zeros((num_x_cells, num_y_cells), dtype=np.float64)

    endsteptime = current_time + dt
    phys_c = phys.c
    phys_invc = phys.invc

    with objmode:
        print(f'Particle Loop')

    # [emission_time, x_idx, y_idx, xpos, ypos, mu, theta, frq, nrg, startnrg]

    for iptcl in range(n_particles):

        # Get particle's initial properties at start of time-step
        ttt = particle_prop[iptcl, 0]
        x_cell_idx = int(particle_prop[iptcl, 1])
        y_cell_idx = int(particle_prop[iptcl, 2])
        xpos = particle_prop[iptcl, 3]
        ypos = particle_prop[iptcl, 4]
        mu   = particle_prop[iptcl, 5]
        omega = particle_prop[iptcl, 6]
        frq = particle_prop[iptcl, 7]
        nrg = particle_prop[iptcl, 8]
        startnrg = particle_prop[iptcl, 9]

        # Loop over segments in the particle history
        while True:

            # Calculate distance to boundary
            dist_b = distance_to_boundary_2D(xpos, ypos, mu, omega, mesh.x_edges[x_cell_idx], mesh.x_edges[x_cell_idx+1], mesh.y_edges[y_cell_idx], mesh.y_edges[y_cell_idx+1] )
            # Calculate distance to census
            dist_cen = phys_c * (endsteptime - ttt)
            if dist_cen < 0:
                raise ValueError
            

            dist_coll = -np.log(np.random.uniform()) / ( (1 - fleck[x_cell_idx, y_cell_idx]) * sigma_a[x_cell_idx, y_cell_idx])
            if dist_coll < 0:
                print(f'fleck = {fleck[x_cell_idx, y_cell_idx]}')
                print(f'sigma_a = {sigma_a[x_cell_idx, y_cell_idx]}')
                raise ValueError
            
            distances = np.array([dist_b, dist_cen, dist_coll])
            dist = np.min(distances)
            event = np.argmin(distances) # 0=dist_b, 1=census, 2=collision
            if dist < 0:
                raise ValueError

            # Calculate new particle energy
            newnrg = nrg * np.exp(-fleck[x_cell_idx, y_cell_idx] * sigma_a[x_cell_idx, y_cell_idx] * dist)
            newnrg = max(newnrg, 0.0)  # prevent negative energy
            if newnrg < 0:
                print(f'fleck = {fleck[x_cell_idx, y_cell_idx]}')
                raise RuntimeError(f"New particle energy is negative! newnrg = {newnrg}")

            nrg_change = nrg - newnrg
            nrgdep[x_cell_idx, y_cell_idx] += nrg_change

            # Compute CORRECTED 3D projected direction unit vector for movement
            # You must use mu and omega to calculate the projected components (u, v).
            rho = np.sqrt(1.0 - mu**2) # This is the X-Y plane projection factor
            dx = rho * np.cos(omega)   # u component
            dy = rho * np.sin(omega)   # v component

            # Advance position
            xpos += dx * dist
            ypos += dy * dist

            # Advance time
            ttt += dist * phys_invc

            # Update energy
            nrg = newnrg

            if event == 0:
                # boundary
                xpos, ypos, mu, omega, x_cell_idx, y_cell_idx, alive = treat_boundary_2D(
                    xpos, ypos, mu, omega,
                    x_cell_idx, y_cell_idx,
                    mesh_x_edges, mesh_y_edges, # Use mesh_x_edges, mesh_y_edges as passed to run2D
                    left_bc='vacuum', right_bc='vacuum', bottom_bc='reflecting', top_bc='vacuum'
                )
                
                # IMPORTANT: If particle is still alive, it means it was a reflection 
                # or an internal crossing. In either case, we must stop the current
                # segment and start a new one from the new position/cell.
                if not alive:
                    # mark particle dead
                    particle_prop[iptcl, 8] = -1.0
                    break # Finished history (e.g., vacuum BC)
                else:
                    # Internal crossing or reflection: continue to the next iteration 
                    # to calculate new distances from the new cell.
                    # No need to update particle_prop here, as ttt is still correct, 
                    # and x/y_cell_idx are updated from treat_boundary_2D.
                    continue # Start next segment
# ...

            elif event == 1:
                # census
                # Update the particle's properties [emission_time, x_idx, y_idx, xpos, ypos, mu, omega, frq, nrg, startnrg]
                particle_prop[iptcl, 0] = ttt
                particle_prop[iptcl, 1] = x_cell_idx
                particle_prop[iptcl, 2] = y_cell_idx
                particle_prop[iptcl, 3] = xpos
                particle_prop[iptcl, 4] = ypos
                particle_prop[iptcl, 5] = mu
                particle_prop[iptcl, 6] = omega
                particle_prop[iptcl, 7] = frq
                particle_prop[iptcl, 8] = nrg
                particle_prop[iptcl, 9] = startnrg
                break  # Finish history for this particle
        
            else:
                # Collision (i.e. absorption, but treated as pseudo-scattering)
                mu = 2 * np.random.uniform() - 1.0
                omega = 2 * np.pi * np.random.uniform()

            # End loop over history segments
    # End loop over particles
    return nrgdep, n_particles, particle_prop


@njit
def track_single_particle(
    iptcl, particle_prop, mesh_z_edges, mesh_r_edges, 
    mesh_sigma_a, mesh_sigma_s, mesh_sigma_t, mesh_fleck, 
    phys_c, phys_invc, endsteptime,
    tid, priv_dep, priv_scat, priv_xEs, priv_yEs, priv_tEs
):
    # Get Particle Parameters
    ttt = particle_prop[iptcl, 0]
    z_cell_idx = int(particle_prop[iptcl, 1])
    r_cell_idx = int(particle_prop[iptcl, 2])
    z = particle_prop[iptcl, 3]
    r = particle_prop[iptcl, 4]
    mu   = particle_prop[iptcl, 5]
    phi = particle_prop[iptcl, 6]
    frq = particle_prop[iptcl, 7]
    nrg = particle_prop[iptcl, 8]
    startnrg = particle_prop[iptcl, 9]

    # Particle History Loop
    history_continues = True
    while history_continues:
        
        d_z = distance_to_z_boundary(z, mu, 
                                     mesh_z_edges[z_cell_idx], 
                                     mesh_z_edges[z_cell_idx+1])
        
        d_rmin, d_rmax = get_individual_r_distances(r, 
                                                    mu, phi, 
                                                    mesh_r_edges[r_cell_idx], 
                                                    mesh_r_edges[r_cell_idx+1])
        
        dist_b = min(d_z, d_rmin, d_rmax)
        dist_cen = phys_c * (endsteptime - ttt)
        
        if dist_cen < 0:
            dist_cen = 0.0 # Should not happen if endsteptime > ttt, but provides safety
        
        distances = np.array([dist_b, dist_cen])

        dist = np.min(distances)
        event = np.argmin(distances) # 0=dist_b, 1=census

        # Energy change, absorption, scattering fractions
        newnrg = nrg * np.exp(-mesh_sigma_t[z_cell_idx, r_cell_idx] * dist)
        newnrg = max(newnrg, 0.0)
        nrg_change = nrg - newnrg

        frac_absorbed = (mesh_sigma_a[z_cell_idx, r_cell_idx] * mesh_fleck[z_cell_idx, r_cell_idx] / mesh_sigma_t[z_cell_idx, r_cell_idx])
        frac_scattered = (((1.0 - mesh_fleck[z_cell_idx, r_cell_idx]) * mesh_sigma_a[z_cell_idx, r_cell_idx] + mesh_sigma_s[z_cell_idx, r_cell_idx]) / mesh_sigma_t[z_cell_idx, r_cell_idx])

        # Update private tallies
        priv_dep[tid, z_cell_idx, r_cell_idx] += nrg_change * frac_absorbed
        priv_scat[tid, z_cell_idx, r_cell_idx] += nrg_change * frac_scattered

        # Average scatter length
        average_scatter_length = (1 / mesh_sigma_t[z_cell_idx, r_cell_idx] *
                                      (1 - (1 + mesh_sigma_t[z_cell_idx, r_cell_idx] * dist) *
                                       np.exp(-mesh_sigma_t[z_cell_idx, r_cell_idx] * dist)) /
                                      (1 - np.exp(-mesh_sigma_t[z_cell_idx, r_cell_idx] * dist)))
        
        # Move particle
        r_next, z_next, phi_next = move_particle_RZ(r, z, phi, mu, dist)

        avg_z = z + (z_next - z) * average_scatter_length
        avg_t = ttt + (dist * phys_invc) * average_scatter_length
        avg_r = r + (r_next - r) * average_scatter_length

        priv_xEs[tid, z_cell_idx, r_cell_idx] += nrg_change * frac_scattered * avg_z
        priv_yEs[tid, z_cell_idx, r_cell_idx] += nrg_change * frac_scattered * avg_r
        priv_tEs[tid, z_cell_idx, r_cell_idx] += nrg_change * frac_scattered * avg_t

        # Advance particle time
        ttt += dist * phys_invc

        # update particle
        r,z,phi = r_next, z_next, phi_next

        # Update particle energy
        nrg = newnrg

        # Energy cutoff echeck
        # Kill particle if nrg < 0.01 * startnrg
        if nrg < 0.01 * startnrg:
            priv_dep[tid, z_cell_idx, r_cell_idx] += nrg  # Deposit remaining energy
            particle_prop[iptcl, 8] = -1.0               # Mark as dead
            history_continues = False
            continue # Exit loop for this particle

        # Event Handling
        if event == 0:
            # Boundary
            r_cell_idx, z_cell_idx, mu, phi, alive = treat_boundary_RZ(
                mu, phi, r_cell_idx, z_cell_idx, 
                len(mesh_r_edges)-1, len(mesh_z_edges)-1,
                dist_b, d_z, d_rmin, d_rmax
            )
            
            if not alive:
                particle_prop[iptcl, 8] = -1.0
                history_continues = False 
            
        elif event == 1:
            # Census: save properties and exit
            
            particle_prop[iptcl, 0] = ttt
            particle_prop[iptcl, 1] = z_cell_idx
            particle_prop[iptcl, 2] = r_cell_idx
            particle_prop[iptcl, 3] = z
            particle_prop[iptcl, 4] = r
            particle_prop[iptcl, 5] = mu
            particle_prop[iptcl, 6] = phi
            particle_prop[iptcl, 7] = frq
            particle_prop[iptcl, 8] = nrg
            particle_prop[iptcl, 9] = startnrg
            
            history_continues = False

    return


@njit(parallel=True)
def run_crooked_pipe_firstloop(n_particles, particle_prop, current_time, dt,
                               mesh_sigma_a, mesh_sigma_s, mesh_sigma_t,
                               mesh_fleck, mesh_z_edges, mesh_r_edges,
                               ):
    num_z_cells = len(mesh_z_edges) - 1
    num_r_cells = len(mesh_r_edges) - 1

    n_threads = config.NUMBA_NUM_THREADS

    # Global tallies
    nrgdep = np.zeros((num_z_cells, num_r_cells), dtype=np.float64)
    nrgscattered = np.zeros((num_z_cells, num_r_cells), dtype=np.float64)
    x_Es = np.zeros((num_z_cells, num_r_cells), dtype=np.float64)
    y_Es = np.zeros((num_z_cells, num_r_cells), dtype=np.float64)
    tEs = np.zeros((num_z_cells, num_r_cells), dtype=np.float64)

    # Private tallies for each thread
    priv_dep = np.zeros((n_threads, num_z_cells, num_r_cells), dtype=np.float64)
    priv_scat = np.zeros((n_threads, num_z_cells, num_r_cells), dtype=np.float64)
    priv_xEs = np.zeros((n_threads, num_z_cells, num_r_cells), dtype=np.float64)
    priv_yEs = np.zeros((n_threads, num_z_cells, num_r_cells), dtype=np.float64)
    priv_tEs = np.zeros((n_threads, num_z_cells, num_r_cells), dtype=np.float64)

    # Constants needed inside the tracking function
    endsteptime = current_time + dt
    # NOTE: Assuming 'phys' object is globally or externally available to Numba
    phys_c = phys.c
    phys_invc = phys.invc

    # Parallel loop over particles (Simple dispatch)
    for iptcl in prange(n_particles):
        tid = numba.np.ufunc.parallel._get_thread_id()
        
        # Call the tracking function, isolating the complex control flow
        track_single_particle(
            iptcl, particle_prop, mesh_z_edges, mesh_r_edges, 
            mesh_sigma_a, mesh_sigma_s, mesh_sigma_t, mesh_fleck, 
            phys_c, phys_invc, endsteptime,
            tid, priv_dep, priv_scat, priv_xEs, priv_yEs, priv_tEs
        )

    # --- Reduction step ---
    for t in range(n_threads):
        nrgdep += priv_dep[t]
        nrgscattered += priv_scat[t]
        x_Es += priv_xEs[t]
        y_Es += priv_yEs[t]
        tEs += priv_tEs[t]
    # with objmode:
    #     print(f'In inner loop:')
    #     print(f'nrgdep = {nrgdep}')  
    #     print(f'nrgscattered = {nrgscattered}')
    #     print(f'x_Es = {x_Es}')
    #     print(f'y_Es = {y_Es}')
    #     print(f'tEs = {tEs}')  

    return nrgdep, nrgscattered, x_Es, y_Es, tEs


def generate_scattered_particles(
    nrgscattered,
    x_Es, y_Es, t_Es,
    mesh_z_edges, mesh_r_edges, mesh_dz, mesh_dr,
    ptcl_max_array_size, ptcl_Nx, ptcl_Ny, ptcl_Nt, ptcl_Nmu, ptcl_N_phi,
    current_time, dt,
):
    """
    Generate new scattered particles based on scattered energy tallies.

    Returns
    -------
    scattered_particles : ndarray
        Array of scattered particle properties [time, iz, ir, z, r, mu, phi, frq, nrg, startnrg].
    n_scattered_particles : int
        Number of scattered particles generated.
    """
    # print(f'sum of nrgscattered = {np.sum(nrgscattered)}')
    nz_cells = len(mesh_z_edges) - 1
    nr_cells = len(mesh_r_edges) - 1

    # Pre-calculate RZ volumes for the entire mesh
    dz = np.diff(mesh_z_edges)
    dr2 = np.diff(mesh_r_edges**2)

    # Allocate scattered particle array
    scattered_particles = np.zeros((ptcl_max_array_size, 10), dtype=np.float64)
    n_scattered_particles = 0

    # Average scatter positions and times
    X_s = np.divide(x_Es, nrgscattered, out=np.zeros_like(x_Es), where=nrgscattered > 0.0)
    Y_s = np.divide(y_Es, nrgscattered, out=np.zeros_like(y_Es), where=nrgscattered > 0.0)
    T_s = np.divide(t_Es, nrgscattered, out=np.zeros_like(t_Es), where=nrgscattered > 0.0)

    # Allocate chi, gamma, tau per cell
    chi   = np.zeros((nz_cells, nr_cells), dtype=np.float64)
    gamma = np.zeros((nz_cells, nr_cells), dtype=np.float64)
    tau   = np.zeros((nz_cells, nr_cells), dtype=np.float64)

    P_tally = np.zeros((nz_cells, nr_cells), dtype=np.float64)

    # Loop over cells
    for iz in range(nz_cells):
        for ir in range(nr_cells):

            if nrgscattered[iz, ir] <= 0.0:
                continue  # no energy to scatter

            # Shifted relative averages
            rel_X = np.round(X_s[iz, ir] - mesh_z_edges[iz], 8)
            rel_Y = np.round(Y_s[iz, ir] - mesh_r_edges[ir], 8)
            rel_T = np.round(T_s[iz, ir] - current_time, 8)

            dz_cell = mesh_dz[iz]
            dr_cell = mesh_dr[ir]

            # Space, angle, and time grids
            z_values = imc_source.deterministic_sample_z(mesh_z_edges[iz], mesh_z_edges[iz+1], ptcl_Nx[iz,ir])
            r_values, r_weights = imc_source.weighted_sample_radius(mesh_r_edges[ir], mesh_r_edges[ir+1], ptcl_Ny[iz,ir])
            mu_values = imc_source.deterministic_sample_mu_isotropic(ptcl_Nmu[iz,ir])
            phi_values = imc_source.deterministic_sample_phi_isotropic(ptcl_N_phi[iz,ir])
            t_values = current_time + (np.arange(ptcl_Nt[iz, ir]) + 0.5) * dt / ptcl_Nt[iz, ir]

            n_source_ptcls = len(z_values) * len(r_values) * len(mu_values) * len(phi_values) * len(t_values)

            nrg = nrgscattered[iz, ir] / n_source_ptcls

            # Solve for chi, gamma, tau
            with objmode:
                sol = root(chi_equation, 1, args=(0.0, dz_cell, dz_cell, rel_X), method='hybr')
                chi[iz, ir] = sol.x
                sol = root(gamma_equation, 1, args=(0.0, dr_cell, dr_cell, rel_Y), method='hybr')
                gamma[iz, ir] = sol.x
                sol = root(tau_equation, 1, args=(0.0, dt, dt, rel_T), method='hybr')
                tau[iz, ir] = sol.x

            # Create scattered particles
            for z in z_values:
                for i_r, r in enumerate(r_values):
                    w_r = r_weights[i_r]
                    for mu in mu_values:
                        for phi in phi_values:
                            for ttt in t_values:
                                if n_scattered_particles >= ptcl_max_array_size:
                                    raise RuntimeError("Maximum number of scattered particles reached")

                                rel_z = z - mesh_z_edges[iz]
                                rel_r = r - mesh_r_edges[ir]

                                P = p_x_y_t_solve(
                                    chi[iz, ir], gamma[iz, ir], tau[iz, ir],
                                    dz_cell, dr_cell, dt,
                                    0, rel_z, 0, rel_r, 0, dt
                                )
                                if P < 0:
                                    raise ValueError(f"Negative probability P={P}")
                                combined_w = P * w_r
                                P_tally[iz, ir] += combined_w

                                idx = n_scattered_particles
                                scattered_particles[idx, 0] = ttt   # emission time
                                scattered_particles[idx, 1] = iz    # z cell index
                                scattered_particles[idx, 2] = ir    # r cell index
                                scattered_particles[idx, 3] = z     # z position
                                scattered_particles[idx, 4] = r     # r position
                                scattered_particles[idx, 5] = mu    # mu
                                scattered_particles[idx, 6] = phi   # phi
                                scattered_particles[idx, 7] = 0     # frequency (placeholder)
                                scattered_particles[idx, 8] = nrg   # particle energy
                                scattered_particles[idx, 9] = combined_w     # probability weight

                                n_scattered_particles += 1
    # Final energy correction loop
    for i in range(n_scattered_particles):
        iz = int(scattered_particles[i, 1])
        ir = int(scattered_particles[i, 2])
        
        # This is the (P * w_r) we stored in index 9
        combined_w = scattered_particles[i, 9]
        
        # Corrected Energy Calculation
        # Energy = Total Cell Energy * (Individual Weight / Sum of Weights in Cell)
        final_nrg = nrgscattered[iz, ir] * (combined_w / P_tally[iz, ir])
        
        scattered_particles[i, 8] = final_nrg
        # Typically, you'd store the weight here or the energy, depending on your solver
        scattered_particles[i, 9] = final_nrg
    
    # print(f'Sum of generated particles energies = {np.sum(scattered_particles[:, 8])}')
    return scattered_particles, n_scattered_particles

@njit
def generate_scattered_particles_no_distribution(
    nrgscattered,
    x_Es, y_Es, t_Es,
    mesh_z_edges, mesh_r_edges, mesh_dz, mesh_dr,
    ptcl_max_array_size, ptcl_Nx, ptcl_Ny, ptcl_Nt, ptcl_Nmu, ptcl_N_phi,
    current_time, dt
):
    """
    Generate new scattered particles based on scattered energy tallies.

    Returns
    -------
    scattered_particles : ndarray
        Array of scattered particle properties [time, iz, ir, z, r, mu, phi, frq, nrg, startnrg].
    n_scattered_particles : int
        Number of scattered particles generated.
    """
    # with objmode:
    #     print(f'sum of nrgscattered = {np.sum(nrgscattered)}')
    nz_cells = len(mesh_z_edges) - 1
    nr_cells = len(mesh_r_edges) - 1

    # Allocate scattered particle array
    max_size = np.sum(ptcl_Nx * ptcl_Ny * ptcl_Nmu * ptcl_Nt * ptcl_N_phi)
    scattered_particles = np.zeros((int(max_size), 10), dtype=np.float64)
    n_scattered_particles = 0

    # Loop over cells
    for iz in range(nz_cells):
        for ir in range(nr_cells):

            if nrgscattered[iz, ir] <= 0.0:
                continue  # no energy to scatter

            # Space, angle, and time grids
            z_values = imc_source.deterministic_sample_z(mesh_z_edges[iz], mesh_z_edges[iz+1], ptcl_Nx[iz,ir])
            r_values, r_weights = imc_source.weighted_sample_radius(mesh_r_edges[ir], mesh_r_edges[ir+1], ptcl_Ny[iz,ir])
            mu_values = imc_source.deterministic_sample_mu_isotropic(ptcl_Nmu[iz,ir])
            phi_values = imc_source.deterministic_sample_phi_isotropic(ptcl_N_phi[iz,ir])
            t_values, t_weights = imc_source.deterministic_sample_t_tanh_dist(ptcl_Nt[iz,ir], 2.0, current_time, current_time + dt)

            n_cell_ptcls = len(z_values) * len(r_values) * len(mu_values) * len(phi_values) * len(t_values)


            base_nrg = nrgscattered[iz, ir] / n_cell_ptcls

            # Create scattered particles
            for z in z_values:
                for i_r, r in enumerate(r_values):
                    for mu in mu_values:
                        for phi in phi_values:
                            for i_t, ttt in enumerate(t_values):
                                if n_scattered_particles >= ptcl_max_array_size:
                                    raise RuntimeError("Maximum number of scattered particles reached")
                                weighted_nrg = base_nrg * r_weights[i_r] * t_weights[i_t]
                                idx = n_scattered_particles
                                scattered_particles[idx, 0] = ttt   # emission time
                                scattered_particles[idx, 1] = iz    # z cell index
                                scattered_particles[idx, 2] = ir    # r cell index
                                scattered_particles[idx, 3] = z     # z position
                                scattered_particles[idx, 4] = r     # r position
                                scattered_particles[idx, 5] = mu    # mu
                                scattered_particles[idx, 6] = phi   # phi
                                scattered_particles[idx, 7] = 0     # frequency (placeholder)
                                scattered_particles[idx, 8] = weighted_nrg   # particle energy
                                scattered_particles[idx, 9] = weighted_nrg   # starnrg

                                n_scattered_particles += 1
    # with objmode:
    #     print(f'Sum of generated particles energies = {np.sum(scattered_particles[:, 8])}')
    return scattered_particles, n_scattered_particles


@njit
def track_single_particle_RN(iptcl, particle_prop, mesh_z_edges, mesh_r_edges,
                             mesh_sigma_a, mesh_sigma_s, mesh_fleck,
                             phys_c, phys_invc, endsteptime,
                             tid, priv_dep
):
    # Get Particle Parameters
    ttt = particle_prop[iptcl, 0]
    z_cell_idx = int(particle_prop[iptcl, 1])
    r_cell_idx = int(particle_prop[iptcl, 2])
    z = particle_prop[iptcl, 3]
    r = particle_prop[iptcl, 4]
    mu   = particle_prop[iptcl, 5]
    phi = particle_prop[iptcl, 6]
    frq = particle_prop[iptcl, 7]
    nrg = particle_prop[iptcl, 8]
    startnrg = particle_prop[iptcl, 9]

    history_continues = True
    while history_continues:
        
        d_z = distance_to_z_boundary(z, mu, 
                                     mesh_z_edges[z_cell_idx], 
                                     mesh_z_edges[z_cell_idx+1])
        
        d_rmin, d_rmax = get_individual_r_distances(r, 
                                                    mu, phi, 
                                                    mesh_r_edges[r_cell_idx], 
                                                    mesh_r_edges[r_cell_idx+1])
        
        dist_b = min(d_z, d_rmin, d_rmax)
        dist_cen = phys_c * (endsteptime - ttt)
        
        if dist_cen < 0:
            dist_cen = 0.0 # Should not happen if endsteptime > ttt, but provides safety
        
        dist_col = distance_to_collision(mesh_sigma_a[z_cell_idx,r_cell_idx],
                                         mesh_sigma_s[z_cell_idx,r_cell_idx], 
                                         mesh_fleck[z_cell_idx,r_cell_idx])

        distances = np.array([dist_b, dist_cen, dist_col])

        dist = np.min(distances)
        event = np.argmin(distances) # 0=dist_b, 1=census, 2=collision

        # Energy change
        newnrg = nrg * np.exp(-mesh_sigma_a[z_cell_idx, r_cell_idx] * mesh_fleck[z_cell_idx, r_cell_idx] * dist)
        newnrg = max(newnrg, 0.0)
        nrg_change = nrg - newnrg

        # Update private tallies
        priv_dep[tid, z_cell_idx, r_cell_idx] += nrg_change

        # Move particle
        r, z, phi = move_particle_RZ(r, z, phi, mu, dist)
        
        # Advance particle time
        ttt += dist * phys_invc

        # Update particle energy
        nrg = newnrg

        # --- NEW ROULETTE / THRESHOLD CHECK ---
        if nrg < 0.01 * startnrg:
            priv_dep[tid, z_cell_idx, r_cell_idx] += nrg  # Deposit remaining energy
            particle_prop[iptcl, 8] = -1.0               # Mark as dead
            history_continues = False
            continue # Exit loop for this particle
        # --------------------------------------

        # Event Handling
        if event == 0:
            # Boundary
            r_cell_idx, z_cell_idx, mu, phi, alive = treat_boundary_RZ(
                mu, phi, r_cell_idx, z_cell_idx, 
                len(mesh_r_edges)-1, len(mesh_z_edges)-1,
                dist_b, d_z, d_rmin, d_rmax
            )
            
            if not alive:
                particle_prop[iptcl, 8] = -1.0
                history_continues = False 
            
        elif event == 1:
            # Census: save properties and exit
            
            particle_prop[iptcl, 0] = ttt
            particle_prop[iptcl, 1] = z_cell_idx
            particle_prop[iptcl, 2] = r_cell_idx
            particle_prop[iptcl, 3] = z
            particle_prop[iptcl, 4] = r
            particle_prop[iptcl, 5] = mu
            particle_prop[iptcl, 6] = phi
            particle_prop[iptcl, 7] = frq
            particle_prop[iptcl, 8] = nrg
            particle_prop[iptcl, 9] = startnrg
            
            history_continues = False
        
        elif event == 2: # Collision
            # Sample mu
            mu = imc_source.sample_mu_isotropic()

            # Sample phi
            phi = imc_source.sample_phi_isotropic()

    return

@njit(parallel=True)
def run_crooked_pipe_loop_RN(n_particles, particle_prop, current_time, dt,
                             mesh_sigma_a, mesh_sigma_s, mesh_fleck,
                             mesh_z_edges, mesh_r_edges):
    
    num_z_cells = len(mesh_z_edges) - 1
    num_r_cells = len(mesh_r_edges) - 1

    # Global tallies
    nrgdep = np.zeros((num_z_cells, num_r_cells), dtype=np.float64)

    # Private tallies for each thread
    n_threads = numba.get_num_threads()
    priv_dep = np.zeros((n_threads, num_z_cells, num_r_cells), dtype=np.float64)

    endsteptime = current_time + dt

    phys_c = phys.c
    phys_invc = phys.invc

    # Parallel loop over particles
    for iptcl in prange(n_particles):
        tid = numba.np.ufunc.parallel._get_thread_id()
        # Call the tracking function
        track_single_particle_RN(
            iptcl, particle_prop, mesh_z_edges, mesh_r_edges,
            mesh_sigma_a, mesh_sigma_s, mesh_fleck,
            phys_c, phys_invc, endsteptime,
            tid, priv_dep
        )
    
    # --- Reduction step ---
    for t in range(n_threads):
        nrgdep += priv_dep[t]

    return nrgdep


@njit
def clean(n_particles, particle_prop):
    """Tidy up the particle list by removing leaked and absorbed particles with energy < 0.0"""
    
    # Count the number of particles flagged for deletion
    n_to_remove = 0
    for i in range(n_particles):
        if particle_prop[i][6] < 0.0:
            n_to_remove += 1

    # Create a new index to track the valid particles
    valid_index = 0
    for i in range(n_particles):
        if particle_prop[i][6] >= 0.0:
            # If particle is valid, move it to the position `valid_index`
            if valid_index != i:
                particle_prop[valid_index] = particle_prop[i]
            valid_index += 1

    # Update the total number of active particles
    n_particles = valid_index

    with objmode:
        print(f'Number of particles removed = {n_to_remove}')
        print(f'Number of particles in the system = {n_particles}')
    return n_particles, particle_prop

@njit
def clean2D(n_particles, particle_prop, energy_col=8):
    """
    Remove particles with negative energy and compact the particle array.
    
    Parameters
    ----------
    n_particles : int
        Number of active particles in the system.
    particle_prop : np.ndarray
        2D array of particle properties (shape: n_particles x n_attributes)
    energy_col : int
        Column index where particle energy is stored.
    
    Returns
    -------
    n_particles : int
        Updated number of active particles
    particle_prop : np.ndarray
        Updated particle array
    """
    n_to_remove = 0
    valid_index = 0

    for i in range(n_particles):
        if particle_prop[i, energy_col] < 0.0:
            n_to_remove += 1
        else:
            # Move valid particle to `valid_index`
            if valid_index != i:
                particle_prop[valid_index] = particle_prop[i]
            valid_index += 1

    n_particles = valid_index

    # Debug print (Numba-safe)
    # print("Number of particles removed =", n_to_remove)
    # print("Number of particles remaining =", n_particles)

    return n_particles, particle_prop

@njit
def distance_to_boundary_1D(xpos: float,
                            mu: float,
                            left_boundary: float,
                            right_boundary: float
) -> float:
    """
    Calculate the distance to the spatial boundary in 1D geometry.

    Parameters
    ----------
    xpos: float
        Current particle position.

    mu: float
        Direction cosine of the particle's motion.
        Positive mu means motion toward the right boundary.
        Negative mu means motion toward the left boundary.

    left_boundary: float
        Position of the left spatial boundary.

    right_boundary: float
        Position of the rightmost spatial boundary.

    Returns
    -------

    dist_b: float
        Distance from the particle to the next spatial boundary along its
        direction of motion.
    
    Raises
    ------
    ValueError
        If the calculated distance is zero or negative.

    """
    if mu > 0.0:
        dist_b = (right_boundary - xpos) / mu
    else:
        dist_b = (left_boundary - xpos) / mu

    if dist_b <= 0.0:
        print("Bad distance:", dist_b, xpos, mu, left_boundary, right_boundary)
        raise ValueError
    return dist_b

@njit
def distance_to_boundary_2D(
    xpos: float,
    ypos: float,
    mu: float,    # Polar cosine (w or cos(theta))
    omega: float, # Azimuthal angle (phi)
    left_boundary: float,
    right_boundary: float,
    bottom_boundary: float,
    top_boundary: float
) -> float:
    """
    Calculate distance to nearest boundary in 2D geometry, using 3D projected components.
    """

    # 1. Calculate the magnitude of the velocity vector projected onto the x-y plane (rho)
    # This is sqrt(1 - w^2) = sqrt(1 - mu^2)
    rho = np.sqrt(1.0 - mu**2) 
    
    # 2. Calculate the CORRECT projected direction cosines (u and v)
    # u = rho * cos(omega) is the x-component
    # v = rho * sin(omega) is the y-component
    u = rho * np.cos(omega)
    v = rho * np.sin(omega)

    # 3. Calculate distances to faces (using u and v)
    
    # X-distances
    d_left  = (left_boundary  - xpos) / u if u < 0.0 else np.inf
    d_right = (right_boundary - xpos) / u if u > 0.0 else np.inf
    
    # Y-distances
    d_bottom = (bottom_boundary - ypos) / v if v < 0.0 else np.inf
    d_top    = (top_boundary   - ypos) / v if v > 0.0 else np.inf

    # 4. Filter out small/negative values caused by floating-point errors
    eps = 1e-12
    if d_left  <= eps: d_left  = np.inf
    if d_right <= eps: d_right = np.inf
    if d_bottom <= eps: d_bottom = np.inf
    if d_top    <= eps: d_top    = np.inf

    dist_b = min(d_left, d_right, d_bottom, d_top)

    if not np.isfinite(dist_b):
        # The particle is far outside the cell, or rho is near zero (moving vertically).
        # This is where your previous errors were caught.
        with objmode:
            print("\n[distance_to_boundary_2D] ⚠️ Invalid distance detected")
            print("  xpos, ypos =", xpos, ypos)
            print("  mu, omega =", mu, omega)
            print("  u, v =", u, v) # Added u and v for better debugging
            print("  left, right =", left_boundary, right_boundary)
            print("  bottom, top =", bottom_boundary, top_boundary)
            print("  d_left, d_right, d_bottom, d_top =", d_left, d_right, d_bottom, d_top)
        dist_b = 0.0 # Clamp to zero distance

    return dist_b


@njit
def distance_to_census(c: float,
                       current_time: float,
                       end_time: float
) -> float:
    """
    Calculate the distance for a particle to reach census.

    Parameters
    ----------
    c: float
        The speed of light.

    current_time: float
        The particle's current time.

    end_time: float
        The time at the end of the time step.

    Returns
    -------

    dist_census: float
        Distance travelled for the particle to reach census.
    
    Raises
    ------
    ValueError
        If the calculated distance is zero or negative.

    """
    dist_census = c * (end_time - current_time)

    if dist_census <= 0.0:
        print("Bad distance:", dist_census, c, end_time, current_time)
        raise ValueError
    
    return dist_census

@njit
def distance_to_collision(sigma_a: float,
                          sigma_s: float,
                          fleck: float
) -> float:
    """
    Calculate the distance for a particle to collide.

    Parameters
    ----------
    sigma_a: float
        The absorption opacity in the cell.

    sigma_s: float
        The real scattering opacity in the cell (as opposed to effective scatter).

    fleck: float
        The fleck factor in the cell, which should be between 0 and 1.

    Returns
    -------

    dist_coll: float
        Distance travelled for the particle to collide.
    
    Raises
    ------
    ValueError
        If the calculated distance is zero or negative.

    """
    # Compute the total scattering opacity (both real and effective)
    sigma_s_total = sigma_s + (1.0 - fleck) * sigma_a

    # Check for invalid opacity
    if sigma_s_total <= 0.0:
        raise ValueError

    # Sample distance to collision
    xi = np.random.uniform()
    dist_coll = -np.log(xi) / sigma_s_total

    if dist_coll <= 0.0:
        raise ValueError
    
    return dist_coll

@njit
def treat_boundary_2D(
    xpos, ypos, mu, omega,
    x_cell_idx, y_cell_idx,
    x_edges, y_edges,
    left_bc="vacuum", right_bc="vacuum",
    bottom_bc="reflecting", top_bc="vacuum"
):
    """
    Handles both internal and external boundary crossings for 2D geometry,
    preserving mu (the 3D Z-direction cosine) during X-Y reflections.
    """

    num_x_cells = len(x_edges) - 1
    num_y_cells = len(y_edges) - 1
    alive = True

    # 1. Calculate the CORRECT projected direction components (u, v)
    rho = np.sqrt(1.0 - mu**2) 
    dx = rho * np.cos(omega) # The projected X-component (u)
    dy = rho * np.sin(omega) # The projected Y-component (v)

    # --- X-direction boundary handling ---
    if xpos <= x_edges[x_cell_idx]:  # crossed left face of current cell
        if x_cell_idx == 0:  # global left boundary
            if left_bc == "vacuum":
                alive = False
            elif left_bc == "reflecting":
                # Specular reflection: reverse X-component (u -> -u)
                dx = abs(dx)
                # Recalculate omega (azimuthal angle)
                omega = np.arctan2(dy, dx)
                xpos = x_edges[0] + 1e-12
        else:
            # internal boundary: move to left cell
            x_cell_idx -= 1
            xpos = x_edges[x_cell_idx + 1] - 1e-12

    elif xpos >= x_edges[x_cell_idx + 1]:  # crossed right face of current cell
        if x_cell_idx == num_x_cells - 1:  # global right boundary
            if right_bc == "vacuum":
                alive = False
            elif right_bc == "reflecting":
                # Specular reflection: reverse X-component (u -> -u)
                dx = -abs(dx)
                # Recalculate omega
                omega = np.arctan2(dy, dx)
                xpos = x_edges[-1] - 1e-12
        else:
            # internal boundary: move to right cell
            x_cell_idx += 1
            xpos = x_edges[x_cell_idx] + 1e-12

    # --- Y-direction boundary handling ---
    # Note: These checks are independent of the X-checks and handle diagonal crossings
    if ypos <= y_edges[y_cell_idx]:  # crossed bottom face
        if y_cell_idx == 0:
            if bottom_bc == "vacuum":
                alive = False
            elif bottom_bc == "reflecting":
                # Specular reflection: reverse Y-component (v -> -v)
                dy = abs(dy)
                # Recalculate omega
                omega = np.arctan2(dy, dx)
                ypos = y_edges[0] + 1e-12
        else:
            # internal boundary
            y_cell_idx -= 1
            ypos = y_edges[y_cell_idx + 1] - 1e-12

    elif ypos >= y_edges[y_cell_idx + 1]:  # crossed top face
        if y_cell_idx == num_y_cells - 1:
            if top_bc == "vacuum":
                alive = False
            elif top_bc == "reflecting":
                # Specular reflection: reverse Y-component (v -> -v)
                dy = -abs(dy)
                # Recalculate omega
                omega = np.arctan2(dy, dx)
                ypos = y_edges[-1] - 1e-12
        else:
            # internal boundary
            y_cell_idx += 1
            ypos = y_edges[y_cell_idx] + 1e-12

    # Clamp indices safely
    x_cell_idx = max(0, min(num_x_cells - 1, x_cell_idx))
    y_cell_idx = max(0, min(num_y_cells - 1, y_cell_idx))

    # IMPORTANT: mu (the Z-direction cosine) remains unchanged for reflections
    # in the X-Y plane. The previous line 'mu = np.cos(omega)' is REMOVED.

    return xpos, ypos, mu, omega, x_cell_idx, y_cell_idx, alive

@njit
def distance_to_z_boundary(z: float,
                           mu: float,
                           z_min: float,
                           z_max: float
):
    """
    Calculates the distance to the nearest axial (Z) boundary,

    
    :param z (float): Current Z coordinate
    :param mu (float): Cosine of the polar angle (wrt Z axis)
    :param z_min (float): Lower Z boundary of the cell
    :param z_max (float): Upper Z boundary of the cell
    """
    if mu == 0:
        return float('inf')
    if mu > 0:
        # Particle is moving "up" toward z_max
        return (z_max - z) / mu
    else:
        # Particle is moving "down" toward z_min
        return (z_min - z) / mu
    
@njit
def distance_to_r_boundary(r, mu, phi, r_min, r_max):
    """
    Calculates the distance to the nearest radial (R) boundary.
    """
    # 1. Pre-calculate directional components
    # sin(theta)^2 = 1 - cos(theta)^2
    sin_theta_sq = 1.0 - mu**2
    # Omega_r is the component of the direction vector along the current radius
    omega_r = np.sqrt(sin_theta_sq) * np.cos(phi)
    
    # Quadratic coefficients: Ad^2 + Bd + C = 0
    # A is the projection of the direction onto the r-theta plane squared
    a = sin_theta_sq
    b = 2.0 * r * omega_r
    
    # If a is 0, the particle is moving parallel to the Z-axis
    if a <= 1e-15:
        return float('inf')

    # 2. Check outer boundary (r_max)
    c_out = r**2 - r_max**2
    disc_out = b**2 - 4.0 * a * c_out
    
    # For r_max, we always take the positive root (heading outward)
    # Using the more stable form of the quadratic solution
    d_rmax = (-b + np.sqrt(max(0, disc_out))) / (2.0 * a)
    
    # 3. Check inner boundary (r_min)
    d_rmin = float('inf')
    if r_min > 0:
        c_in = r**2 - r_min**2
        disc_in = b**2 - 4.0 * a * c_in
        
        # If discriminant is negative, we miss the inner cylinder entirely
        if disc_in >= 0:
            # The smaller root is the entry point to the inner cylinder
            res = (-b - np.sqrt(disc_in)) / (2.0 * a)
            if res > 0:
                d_rmin = res
                
    return min(d_rmax, d_rmin)

@njit
def get_distance_to_nearest_boundary(r, z, mu, phi, r_min, r_max, z_min, z_max):
    """
    Calculates the minimum distance to any boundary in RZ geometry.
    """
    # Calculate distance to axial walls
    d_z = distance_to_z_boundary(z, mu, z_min, z_max)
    
    # Calculate distance to radial walls
    d_r = distance_to_r_boundary(r, mu, phi, r_min, r_max)
    
    # The nearest boundary is the minimum of the two
    return min(d_z, d_r)

@njit
def move_particle_RZ(r, z, phi, mu, d):
    """
    Moves a particle a distance d and updates R, Z, and Phi.
    """
    sin_theta = np.sqrt(1.0 - mu**2)
    projection_xy = d * sin_theta
    
    # 1. Update Z (Linear)
    z_new = z + d * mu
    
    # 2. Update R (Law of Cosines)
    # r_new^2 = r^2 + dist_xy^2 + 2 * r * dist_xy * cos(phi)
    r_new_sq = r**2 + projection_xy**2 + 2 * r * projection_xy * np.cos(phi)
    r_new = np.sqrt(max(0, r_new_sq)) # max(0) handles tiny precision errors
    
    # 3. Update Phi (Change of local basis)
    if r_new > 0:
        cos_phi_new = (r * np.cos(phi) + projection_xy) / r_new
        # Clamp value to [-1, 1] for safety before acos
        cos_phi_new = max(-1.0, min(1.0, cos_phi_new))
        phi_new = np.acos(cos_phi_new)
    else:
        # If we hit the exact center, phi is technically undefined; 
        # usually preserved or reset to 0.
        phi_new = phi
        
    return r_new, z_new, phi_new

@njit
def treat_boundary_RZ(mu, phi, r_idx, z_idx, 
                      Nr, Nz, dist_b, d_z, d_rmin, d_rmax):
    """
    Logic for crossing cell faces or reflecting.
    """
    alive = True
    
    # Check Axial Faces
    if dist_b == d_z:
        if mu > 0:
            z_idx += 1
            if z_idx >= Nz:
                alive = False # Vacuum Top
        else:
            z_idx -= 1
            if z_idx < 0:
                # Bottom Reflection
                alive = False
                
    # Check Radial Faces
    elif dist_b == d_rmax:
        r_idx += 1
        if r_idx >= Nr: alive = False # Vacuum Outer
        
    elif dist_b == d_rmin:
        r_idx -= 1
        if r_idx < 0:
            # Hitting Centerline (r=0)
            # Naturally reflective: change phi to point outward
            r_idx = 0
            phi = phi + np.pi 
            
    return r_idx, z_idx, mu, phi, alive

@njit
def get_individual_r_distances(r, mu, phi, r_min, r_max):
    sin_theta_sq = 1.0 - mu**2
    # Ensure sin_theta_sq is not negative due to precision
    a = max(0.0, sin_theta_sq)
    
    if a <= 1e-15:
        return float('inf'), float('inf')

    omega_r = np.sqrt(a) * np.cos(phi)
    b = 2.0 * r * omega_r
    
    # Distance to Outer (r_max)
    c_out = r**2 - r_max**2
    disc_out = max(0.0, b**2 - 4.0 * a * c_out)
    d_rmax = (-b + np.sqrt(disc_out)) / (2.0 * a)
    
    # Distance to Inner (r_min)
    d_rmin = float('inf')
    if r_min > 0:
        c_in = r**2 - r_min**2
        disc_in = b**2 - 4.0 * a * c_in
        if disc_in >= 0:
            res = (-b - np.sqrt(disc_in)) / (2.0 * a)
            if res > 1e-13: # Nudge for precision
                d_rmin = res
                
    return d_rmin, d_rmax