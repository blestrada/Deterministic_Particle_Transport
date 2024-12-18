"""Advance particles over a time-step"""

import numpy as np
from numba import njit, jit, objmode, gdb
from scipy.optimize import newton
import matplotlib.pyplot as plt

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_part_data as ptcl
import imc_global_phys_data as phys
import imc_global_time_data as time

@njit
def brent_solver(f, a, b, tol=1e-5, max_iter=100):
    """
    Brent's Method for finding roots of a scalar function.

    Parameters:
        f (function): The function for which the root is sought.
        a (float): Lower bound of the initial bracket.
        b (float): Upper bound of the initial bracket.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: The root of the function.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("The function must have different signs at a and b.")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = e = b - a  # Distance between a and b
    for _ in range(max_iter):
        if fb != 0 and fc != 0 and fa != 0:  # Use inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) + \
                b * fa * fc / ((fb - fa) * (fb - fc)) + \
                c * fa * fb / ((fc - fa) * (fc - fb))
        else:  # Fall back to secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions to accept the interpolation point
        cond1 = not (a < s < b if a < b else b < s < a)
        cond2 = abs(s - b) >= abs(b - c) / 2
        cond3 = abs(s - b) >= abs(c - d) / 2 if e != d else False
        cond4 = abs(b - c) < tol

        if cond1 or cond2 or cond3 or cond4:
            # Bisection step
            s = (a + b) / 2
            d = e = b - a

        fs = f(s)
        d, e = e, d  # Update step sizes

        # Update points for next iteration
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fb) < abs(fa):
            a, b = b, a
            fa, fb = fb, fa

        # Check convergence
        if abs(b - a) < tol or abs(fb) < tol:
            return b

    raise RuntimeError("Brent's method did not converge.")

@njit
def p_x_t_solve(dx, x_0, x_1, x, t, t_0, t_1, dt, X_s, T_s):
    @njit
    def f(chi):
        numerator = 1.0 - chi * x_0 + np.exp(chi * dx) * (chi * x_1 - 1.0)
        denominator = chi * (np.exp(chi * dx) - 1.0)
        return numerator / denominator - X_s

    @njit
    def g(tau):
        numerator = 1.0 - tau * t_0 + np.exp(tau * dt) * (tau * t_1 - 1.0)
        denominator = tau * (np.exp(tau * dt) - 1.0)
        return numerator / denominator - T_s

    # Solve for chi using Brent's method
    chi = brent_solver(f, -1e6, 1e6, tol=1e-3, max_iter=1000000)

    # Solve for tau using Brent's method
    tau = brent_solver(g, -1e6, 1e6, tol=1e-3, max_iter=1000000)

    # Solve for P
    P = chi * tau * (np.exp(chi * (x - x_0)) * np.exp(tau * (t - t_0))) / \
        ((np.exp(chi * dx) - 1.0) * (np.exp(tau * dt) - 1.0))
    return P



def run_random():
    """Advance particles over a time-step"""
    # Create local storage for the energy deposited this time-step
    mesh.nrgdep[:] = 0.0

    ptcl.n_census = 0

    endsteptime = time.time + time.dt

    # optimizations
    ran = np.random.uniform()
    exp = np.exp
    log = np.log
    nrgdep = np.zeros(mesh.ncells)
    mesh_nodepos = mesh.nodepos
    phys_c = phys.c
    top_cell = mesh.ncells - 1
    phys_invc = phys.invc
    mesh_sigma_a = mesh.sigma_a
    mesh_sigma_s = mesh.sigma_s
    mesh_fleck = mesh.fleck
    mesh_rightbc = mesh.right_bc
    mesh_leftbc = mesh.left_bc


    print(f'Particle Loop')

    # Loop over all particles
    for iptcl in range(len(particle_prop)):
        # Get particle's initial properties at start of time-step
        (ttt, icell, xpos, mu, nrg, startnrg) = particle_prop[iptcl][1:7]
        
        # print(f'ttt = {ttt}, icell = {icell}, xpos = {xpos}, mu = {mu}, nrg = {nrg}, startnrg = {startnrg}')
  
        #startnrg = 0.01 * startnrg
        
        # Loop over segments in the history (between boundary-crossings and collisions)
        while True:
            # Calculate distance to boundary
            if mu > 0.0:
                dist_b = (mesh_nodepos[icell + 1] - xpos) / mu
            else:
                dist_b = (mesh_nodepos[icell] - xpos) / mu
        
            # Calculate distance to census
            dist_cen = phys_c * (endsteptime - ttt)

            # Calculate distance to collision
            d_coll = -log(ran) / (mesh_sigma_s[icell] + (1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell])
            if d_coll < 0.0:
                raise ValueError(f"d_coll {d_coll} less than zero.")

            # Actual distance - whichever happens first
            dist = min(dist_b, dist_cen, d_coll)

            # Calculate new particle energy
            newnrg = nrg * exp(-mesh_sigma_a[icell] * mesh_fleck[icell] * dist)

            # print(f'newnrg = {newnrg}')

            # # If particle energy falls below cutoff, deposit its energy, and flag for destruction. End history.
            # if newnrg <= startnrg:
            #     newnrg = 0.0
            #     nrgdep[icell] += nrg - newnrg
                
            #     particle_prop[iptcl][5] = -1.0
            #     break

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
                            mesh.nrg_leaked += particle_prop[iptcl][5]
                            particle_prop[iptcl][5] = -1.0
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
                            mesh.nrg_leaked += particle_prop[iptcl][5]
                            particle_prop[iptcl][5] = -1.0
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu *= -1.0  # Reverse direction
                    else:  # If not at the top cell
                        icell += 1  # Move to the right cell
            
            # If the event was census, finish this history
            if dist == dist_cen:
                # Finished with this particle
                # Update the particle's properties in the list
                particle_prop[iptcl][1:6] = (ttt, icell, xpos, mu, nrg)
                ptcl.n_census += 1
                break
                
            # If event was collision, also update and direction
            if dist == d_coll:
                # Collision (i.e. absorption, but treated as pseudo-scattering)
                mu = 1.0 - 2.0 * np.random.uniform()
                

        # End loop over history segments

    # End loop over particles
    mesh.nrgdep[:] = nrgdep[:]
    print(f'mesh.nrgdep = {mesh.nrgdep[:10]}')
    


@njit()
def run(n_particles, particle_prop, nrgdep, nrgscattered, current_time):
    """Advance particles over a time-step, including implicit scattering."""
    nrgdep[:] = 0.0
    nrgscattered[:] = 0.0
    xEs = np.zeros(mesh.ncells, dtype=np.float64)
    tEs = np.zeros(mesh.ncells, dtype=np.float64)
    # Optimizations
    endsteptime = current_time + time.dt
    mesh_nodepos = mesh.nodepos
    phys_c = phys.c
    top_cell = mesh.ncells - 1
    phys_invc = phys.invc
    mesh_sigma_a = mesh.sigma_a
    mesh_sigma_s = mesh.sigma_s
    mesh_sigma_t = mesh.sigma_t
    mesh_fleck = mesh.fleck
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
        nrg = particle_prop[iptcl, 5]
        startnrg = particle_prop[iptcl, 6]
            
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
            # Check if the particle's energy falls below 0.01 * startnrg
            if newnrg <= 0.01 * startnrg:
                newnrg = 0.0

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

            if newnrg == 0.0:
                # Flag particle for later destruction
                particle_prop[iptcl, 5] = -1.0
                break
            #print(f'nrgdep = {nrgdep[:10]}')
            # Calculate the average length of scatter
            average_scatter_length = 1 / mesh_sigma_t[icell] * (1 - (1 + mesh_sigma_t[icell] * dist) * np.exp(-mesh_sigma_t[icell] * dist))/(1 - np.exp(-mesh_sigma_t[icell] * dist))
            average_position_of_scatter = xpos + mu * average_scatter_length
            # if iptcl== 0:
            #     # print(f'average scattering length = {average_scatter_length}')
            #     # print(f'average position of scatter = {average_position_of_scatter}')
            average_time_of_scatter = ttt + average_scatter_length / phys_c
            xEs[icell] += nrg_change * frac_scattered * average_position_of_scatter
            tEs[icell] += nrg_change * frac_scattered * average_time_of_scatter

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
                            particle_prop[iptcl, 5] = -1.0  # Mark as destroyed
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
                            particle_prop[iptcl, 5] = -1.0  # Mark as destroyed
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
                particle_prop[iptcl, 5] = nrg
                break  # Finish history for this particle                    
    
    # Start implicit scattering process after particle transport
    epsilon = 1e-5
    iterations = 0
    converged = False

    # Calculate zone-wise average position of scatter and average time of scatter
    X_s = np.zeros(mesh.ncells, dtype=np.float64)
    T_s = np.zeros(mesh.ncells, dtype=np.float64)
    print(f'xEs = {xEs[0]}')
    print(f'nrgscattered = {nrgscattered[:10]}')
    print(f'nrgscattered last 10 = {nrgscattered[-10:]}')
    X_s = xEs[:] / nrgscattered[:]
    T_s = tEs[:] / nrgscattered[:]
    
    
    # initialize nrgdep and nrgscattered variables
    while not converged:
        scattered_particles = np.zeros((ptcl.max_array_size, 7), dtype=np.float64)
        n_scattered_particles = 0
        # Store the old nrgscattered
        old_nrgscattered = np.zeros(mesh.ncells, dtype=np.float64)
        old_nrgscattered[:] = nrgscattered[:]
        # Create source particles based on energy scattered in each cell
        P_tally = np.zeros(mesh.ncells, dtype=np.float64)
        for icell in range(mesh.ncells):
            # Create position, angle, time arrays
            x_positions = mesh.nodepos[icell] + (np.arange(ptcl.Nx) + 0.5) * mesh.dx / ptcl.Nx
            angles = -1.0 + (np.arange(ptcl.Nmu) + 0.5) * 2 / ptcl.Nmu
            emission_times = current_time + (np.arange(ptcl.Nt) + 0.5) * time.dt / ptcl.Nt
            # Assign energy-weights
            n_source_ptcls = ptcl.Nx * ptcl.Nmu * ptcl.Nt
            nrg = nrgscattered[icell] / n_source_ptcls
            startnrg = nrg

            # Create scattered particles
            for xpos in x_positions:
                for mu in angles:
                    for ttt in emission_times:
                        if n_scattered_particles < ptcl.max_array_size:
                            P = p_x_t_solve(mesh.dx, mesh_nodepos[icell], mesh_nodepos[icell + 1], xpos, ttt, current_time, current_time + time.dt, time.dt, X_s[icell], T_s[icell])
                            P_tally[icell] += P
                            
                            idx = n_scattered_particles
                            scattered_particles[idx, 0] = icell  # origin
                            scattered_particles[idx, 1] = ttt  # time
                            scattered_particles[idx, 2] = icell  # cell index
                            scattered_particles[idx, 3] = xpos  # position
                            scattered_particles[idx, 4] = mu  # direction
                            scattered_particles[idx, 5] = nrg  # energy
                            scattered_particles[idx, 6] = P  # start energy
                            n_scattered_particles += 1
                        else:
                            print("Warning: Maximum number of scattered particles reached!")

        # Put correct energy
        for i in range(n_scattered_particles):
            # Get P and icell
            icell = int(scattered_particles[i, 2])
            P = scattered_particles[i, 6]
            # Set particle energy
            nrg = nrgscattered[icell] * P / P_tally[icell]
            # Set particle startnrg
            scattered_particles[idx, 6] = nrg  # start energy

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
            nrg = scattered_particles[iptcl, 5]
            startnrg = scattered_particles[iptcl, 6]

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

                # Check if the particle's energy falls below 0.01 * startnrg
                if newnrg <= 0.01 * startnrg:
                    newnrg = 0.0

                nrg_change = nrg - newnrg
                frac_absorbed = mesh_sigma_a[icell] * mesh_fleck[icell] / mesh_sigma_t[icell]
                frac_scattered = (1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell] / mesh_sigma_t[icell] + mesh_sigma_s[icell] / mesh_sigma_t[icell]
                nrgdep[icell] += nrg_change * frac_absorbed
                nrgscattered[icell] += nrg_change * frac_scattered

                if newnrg == 0.0:
                    # Flag particle for later destruction
                    scattered_particles[iptcl, 5] = -1.0
                    break

                # Calculate the average length of scatter
                average_scatter_length = 1 / mesh_sigma_t[icell] * (1 - (1 + mesh_sigma_t[icell] * dist) * np.exp(-mesh_sigma_t[icell] * dist))/(1 - np.exp(-mesh_sigma_t[icell] * dist))
                average_position_of_scatter = xpos + mu * average_scatter_length
                average_time_of_scatter = ttt + average_scatter_length / phys_c
                xEs += nrg_change * frac_scattered * average_position_of_scatter
                tEs += nrg_change * frac_scattered * average_time_of_scatter
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
                                scattered_particles[iptcl, 5] = -1.0  # Mark as destroyed
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
                                scattered_particles[iptcl, 5] = -1.0  # Mark as destroyed
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
                    scattered_particles[iptcl, 5] = nrg
                    break
        
        X_s = xEs[:] / nrgscattered[:]
        T_s = tEs[:] / nrgscattered[:]
        iterations += 1
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

    # Update global energy deposited mesh
    return nrgdep, n_particles, particle_prop


@njit
def clean(n_particles, particle_prop):
    """Tidy up the particle list by removing leaked and absorbed particles with energy < 0.0"""
    
    # Count the number of particles flagged for deletion
    n_to_remove = 0
    for i in range(n_particles[0]):
        if particle_prop[i][5] < 0.0:
            n_to_remove += 1

    # Create a new index to track the valid particles
    valid_index = 0
    for i in range(n_particles[0]):
        if particle_prop[i][5] >= 0.0:
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