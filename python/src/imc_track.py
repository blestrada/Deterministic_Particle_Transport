"""Advance particles over a time-step"""

import numpy as np
from numba import njit, jit, objmode
from scipy.optimize import root
import matplotlib.pyplot as plt

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_part_data as ptcl
import imc_global_phys_data as phys
import imc_global_time_data as time

@njit
def chi_equation(chi, x_0, x_1, dx, X_s):
    chi = np.asarray(chi).item()  # Ensure chi is a scalar
    X_s = np.asarray(X_s).item()  # Ensure X_s is a scalar

    # Ensure chi * dx is treated as a scalar
    chi_dx = chi * dx

    if np.abs(chi_dx) > 100:
        exp_chi_dx = np.exp(50)  # Limit exponential growth
    elif np.abs(chi_dx) > 1e-5:
        exp_chi_dx = np.exp(chi_dx).item()  # Ensure it's a scalar
    else:
        exp_chi_dx = 1.0 + chi_dx + 0.5 * chi_dx ** 2  # Taylor expansion

    # Compute numerator and denominator as scalars
    numerator = 1.0 - chi * x_0 + exp_chi_dx * (chi * x_1 - 1.0)
    denominator = chi * (exp_chi_dx - 1.0)

    # Compute result and ensure it's a scalar
    scalar_result = numerator / denominator
    return scalar_result - X_s


@njit
def tau_equation(tau, t_0, t_1, dt, T_s):
    tau = np.asarray(tau).item()  # Ensure chi is a scalar
    T_s = np.asarray(T_s).item()  # Ensure X_s is a scalar
    if np.abs(tau * dt) > 100:
        exp_tau_dt = np.exp(50)
    elif np.abs(tau * dt) > 1e-5:
        exp_tau_dt = np.exp(tau * dt)
    else:
        exp_tau_dt = 1.0 + (tau * dt) + 0.5 * (tau * dt) ** 2

    return (1 - tau * t_0 + exp_tau_dt * (tau * t_1 - 1.0)) / (tau * (exp_tau_dt - 1.0)) - T_s


@njit
def p_x_t_solve(chi, tau, dx, x_0, x, t, t_0, dt):
    # print(f'chi = {chi}')
    # print(f'tau = {tau}')
    # print(f'dx = {dx}')
    # print(f'x_0 = {x_0}')
    # print(f'x = {x}')
    # print(f't = {t}')
    # print(f't_0 = {t_0}')
    # print(f'dt = {dt}')

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
    for iptcl in range(n_particles[0]):
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
            if mu > 0.0:
                dist_b = (mesh_nodepos[icell + 1] - xpos) / mu
            else:
                dist_b = (mesh_nodepos[icell] - xpos) / mu
        
            # Calculate distance to census
            dist_cen = phys_c * (endsteptime - ttt)

            # Calculate distance to collision
            d_coll = -np.log(np.random.uniform()) / (mesh_sigma_s[icell] + (1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell])
            if d_coll <= 0.0:
                raise ValueError(f"d_coll {d_coll} less than zero.")

            # Actual distance - whichever happens first
            dist = min(dist_b, dist_cen, d_coll)

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
            if dist == d_coll:
                # Collision (i.e. absorption, but treated as pseudo-scattering)
                mu = 1.0 - 2.0 * np.random.uniform()
                

        # End loop over history segments

    # End loop over particles
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
    for iptcl in range(n_particles[0]):
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
    epsilon = 1e-9
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
    # initialize nrgdep and nrgscattered variables


    while not converged:
        scattered_particles = np.zeros((ptcl.max_array_size, 8), dtype=np.float64)
        n_scattered_particles = 0
        # Store the old nrgscattered
        old_nrgscattered = np.zeros(mesh.ncells, dtype=np.float64)
        old_nrgscattered[:] = nrgscattered[:]
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
        # with objmode:
        #     print(f' average position of scatter in iterative loop: {X_s[0]}')
        #     print(f' average time of scatter in iterative loop: {T_s[0]}')
        # Now we need to move all the processed scattered particles to the global particle array
        # Calculate how many particles we are going to add
        n_existing_particles = n_particles[0]
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
            n_particles[0] = n_existing_particles + n_to_add

        if np.all(np.abs(nrgscattered - old_nrgscattered) < epsilon):
            converged = True
    print(f'Number of scattering iterations = {iterations}')

    # Deposit left over scattered energy to conserve energy
    nrgdep[:] += nrgscattered[:]

    # Calculate Eddington Factor
    # with objmode:
    #     print(f'w_average_times_mu_squared = {w_average_times_mu_squared[:10]}')
    #     print(f'w_average = {w_average[:10]}')
    eddington = w_average_times_mu_squared / w_average
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


@njit
def run_multigroup(n_particles, particle_prop, current_time, dt, mesh_sigma_a):
    """Advance particles over a time-step."""

    nrgdep = np.zeros((mesh.ncells, 50), dtype=np.float64)
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
    for iptcl in range(n_particles[0]):
        # Get particle's initial properties at start of time-step
        ttt = particle_prop[iptcl, 1]
        icell = int(particle_prop[iptcl, 2])  # Convert to int
        xpos = particle_prop[iptcl, 3]
        mu = particle_prop[iptcl, 4]
        frq = int(particle_prop[iptcl, 5]) # Convert to int
        nrg = particle_prop[iptcl, 6]
        startnrg = particle_prop[iptcl, 7]
            
        # Loop over segments in the history (between boundary-crossings and collisions)
        while True:
            # with objmode:
            #     print(f'iptcl = {iptcl}')
            #     print(f'icell = {icell}')
            #     print(f'xpos = {xpos}')
            #     print(f'mu = {mu}')
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
            newnrg = nrg * np.exp(-mesh_sigma_a[frq] * dist)

            # Calculate energy change
            nrg_change = nrg - newnrg
            if nrg_change < 0:
                raise ValueError
            # Update energy deposition tallies
            nrgdep[icell][frq] += nrg_change

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

    # Update global energy deposited mesh
    return nrgdep, n_particles, particle_prop


@njit
def clean(n_particles, particle_prop):
    """Tidy up the particle list by removing leaked and absorbed particles with energy < 0.0"""
    
    # Count the number of particles flagged for deletion
    n_to_remove = 0
    for i in range(n_particles[0]):
        if particle_prop[i][6] < 0.0:
            n_to_remove += 1

    # Create a new index to track the valid particles
    valid_index = 0
    for i in range(n_particles[0]):
        if particle_prop[i][6] >= 0.0:
            # If particle is valid, move it to the position `valid_index`
            if valid_index != i:
                particle_prop[valid_index] = particle_prop[i]
            valid_index += 1

    # Update the total number of active particles
    n_particles[0] = valid_index

    with objmode:
        print(f'Number of particles removed = {n_to_remove}')
        print(f'Number of particles in the system = {n_particles}')
    return n_particles, particle_prop