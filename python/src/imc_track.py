"""Advance particles over a time-step"""

import numpy as np

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_part_data as ptcl
import imc_global_phys_data as phys
import imc_global_time_data as time


def run():
    """Advance particles over a time-step"""

    # Create local storage for the energy deposited this time-step
    mesh.nrgdep[:] = 0.0

    ptcl.n_census = 0
    
    # Create local storage for the scattered energy this time-step (used for implicit scattering)
    mesh.nrgscattered[:] = 0.0

    endsteptime = time.time + time.dt

    # optimizations
    ran = np.random.uniform()
    exp = np.exp
    log = np.log
    nrgdep = np.zeros(mesh.ncells)
    nrgscattered = np.zeros(mesh.ncells)
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

    # Loop over all particles
    for iptcl in range(len(ptcl.particle_prop)):
        # Get particle's initial properties at start of time-step
        (ttt, icell, xpos, mu, nrg, startnrg) = ptcl.particle_prop[iptcl][1:7]
        
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
            if ptcl.scattering == 'analog':
                d_coll = -log(ran) / (mesh_sigma_s[icell] + (1 - mesh_fleck[icell]) * mesh_sigma_a[icell])

            if ptcl.scattering == 'implicit':
                d_coll = float('inf')

            # Actual distance - whichever happens first
            dist = min(dist_b, dist_cen, d_coll)

            # Calculate new particle energy
            newnrg = nrg * exp(-mesh_sigma_t[icell] * dist)

            # print(f'newnrg = {newnrg}')

            # # If particle energy falls below cutoff, deposit its energy, and flag for destruction. End history.
            # if newnrg <= startnrg:
            #     newnrg = 0.0
            #     nrgdep[icell] += nrg - newnrg
                
            #     ptcl.particle_prop[iptcl][5] = -1.0
            #     break

            # Calculate energy to be deposited in the material and scattered
            nrg_change = nrg - newnrg

            frac_absorbed = mesh_sigma_a[icell] * mesh_fleck[icell] / mesh_sigma_t[icell]
            frac_scattered = ((1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell] + mesh_sigma_s[icell]) / mesh_sigma_t[icell]
            # print(f'nrg_change = {nrg_change}')
            # print(f'frac_absorbed = {frac_absorbed}')
            # print(f'frac_scattered = {frac_scattered}')
            nrgdep[icell] += nrg_change * frac_absorbed
            nrgscattered[icell] += nrg_change * frac_scattered

                        # Boundary treatment
            if dist == dist_b:
                # Left boundary treatment
                if mu < 0: # If going left
                    if icell == 0: # If at the leftmost cell
                        if mesh_leftbc == 'vacuum':
                            # Flag particle for later destruction
                            ptcl.particle_prop[iptcl][5] = -1.0
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
                            ptcl.particle_prop[iptcl][5] = -1.0
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu *= -1.0  # Reverse direction
                    else:  # If not at the top cell
                        icell += 1  # Move to the right cell
            
            # Advance position, time, and energy
            xpos += mu * dist
            ttt += dist * phys_invc
            nrg = newnrg

            # If the event was census, finish this history
            if dist == dist_cen:
                # Finished with this particle
                # Update the particle's properties in the list
                ptcl.particle_prop[iptcl][1:6] = (ttt, icell, xpos, mu, nrg)
                ptcl.n_census += 1
                break
                

        # End loop over history segments

    # End loop over particles
    mesh.nrgdep[:] = nrgdep[:]
    mesh.nrgscattered[:] += nrgscattered[:]
    print(f'mesh.nrgscattered = {mesh.nrgscattered}')
    # if ptcl.scattering == 'implicit':
    #     do_implicit_scattering()

    #print(f'Energy deposited in time-step = {nrgdep}')


def do_implicit_scattering():
    iterations = 0
    # print(f'mesh.nrgscattered before scattering iterations = {mesh.nrgscattered}')
    
    while np.all(mesh.nrgscattered > 1E-20):

        # Make source particles
        for icell in range(mesh.ncells):
            # Create position, angle, time arrays
            x_positions = mesh.nodepos[icell] + (np.arange(ptcl.Nx) + 0.5) * mesh.dx / ptcl.Nx
            angles = -1.0 + (np.arange(ptcl.Nmu) + 0.5) * 2 / ptcl.Nmu
            emission_times = time.time + (np.arange(ptcl.Nt) + 0.5) * time.dt / ptcl.Nt

            # Assign energy-weights
            n_source_ptcls = ptcl.Nx * ptcl.Nmu * ptcl.Nt
            nrg = mesh.nrgscattered[icell] / n_source_ptcls
            startnrg = nrg

            # Create particles and add them to list of scattered particles
            origin = icell
            for xpos in x_positions:
                for mu in angles:
                    for ttt in emission_times:
                        ptcl.scattered_particles.append([origin, ttt, icell, xpos, mu, nrg, startnrg])
        
        # Advance the particles
        endsteptime = time.time + time.dt
        # optimizations
        exp = np.exp
        nrgdep = [0.0] * mesh.ncells
        nrgscattered = [0.0] * mesh.ncells
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


        # Loop over all particles
        for iptcl in range(len(ptcl.scattered_particles)):
            # Get particle's initial properties
            (ttt, icell, xpos, mu, nrg, startnrg) = ptcl.scattered_particles[iptcl][1:7]
            
            # Loop over segments in the history (between boundary-crossings and collisions)
            while True:
                # Distance to boundary
                if mu > 0.0:
                    dist_b = (mesh_nodepos[icell + 1] - xpos) / abs(mu)
                else:
                    dist_b = (xpos - mesh_nodepos[icell]) / abs(mu)
                
                # Distance to census
                dist_cen = phys_c * (endsteptime - ttt)

                # Actual distance - whichever happens first
                dist = min(dist_b, dist_cen)

                # Calculate new particle energy
                newnrg = nrg * exp(-mesh_sigma_t[icell] * dist)

                # # If particle energy falls below cutoff, deposit its energy, and flag for destruction. End history.
                # if newnrg <= startnrg:
                #     newnrg = 0.0
                #     nrgdep[icell] += nrg - newnrg
                    
                #     ptcl.particle_prop[iptcl][5] = -1.0
                #     break

                # Calculate energy to be deposited in the material and scattered
                nrg_change = nrg - newnrg

                
                frac_absorbed = mesh_sigma_a[icell] * mesh_fleck[icell] / mesh_sigma_t[icell]
                frac_scattered = ((1.0 - mesh_fleck[icell]) * mesh_sigma_a[icell] + mesh_sigma_s[icell]) / mesh_sigma_t[icell]
                
                nrgdep[icell] += nrg_change * frac_absorbed
                nrgscattered[icell] += nrg_change * frac_scattered

                # Boundary treatment
                if dist == dist_b:
                    
                    # Left boundary
                    if mu < 0 and icell == 0:
                        if mesh_leftbc == 'vacuum':
                            print(f'particle escaped through left boundary')
                            # Flag particle for later destruction
                            ptcl.scattered_particles[iptcl][5] = -1.0
                            break
                        elif mesh_leftbc == 'reflecting':
                            mu *= -1.0  # Reverse direction
                            
                    # Right boundary
                    elif mu > 0 and icell == top_cell:
                        if mesh_rightbc == 'vacuum':
                            # Flag particle for later destruction
                            ptcl.scattered_particles[iptcl][5] = -1.0
                            break
                        elif mesh_rightbc == 'reflecting':
                            mu *= -1.0  # Reverse direction
                            
                    
                    # Move particle to the left
                    elif mu < 0 and icell != 0:
                        icell -= 1
                    
                    # Move particle to the right
                    elif mu > 0 and icell != top_cell:
                        icell += 1
                
                # Advance position, time, and energy of the particle
                xpos += mu * dist
                ttt += dist * phys_invc
                nrg = newnrg

                # If the event was census, finish this history
                if dist == dist_cen:
                    # Finished with this particle
                    # Update the particle's properties in the list
                    ptcl.scattered_particles[iptcl][1:6] = (ttt, icell, xpos, mu, nrg)
                    break

            # End loop over history segment

        # End loop over particles

        # print(f'nrgdep after 1 iteration = {nrgdep}')
        # Update global energy banks
        # print(f'nrgscattered after 1 iteration = {nrgscattered}')
        mesh.nrgdep[:] += nrgdep[:]
        mesh.nrgscattered[:] = nrgscattered[:]

        # Move scattered particles to the global particle list
        for entry in ptcl.scattered_particles:
            origin = entry[0]
            ttt = entry[1]
            icell = entry[2]
            xpos = entry[3]
            mu = entry[4]
            nrg = entry[5]
            startnrg = nrg

            # Append updated particle entry to the particle_prop
            ptcl.particle_prop.append([origin, ttt, icell, xpos, mu, nrg, startnrg])

        # Remove old particles from scattered particles.
        ptcl.scattered_particles = []

        # Increment iterations
        # print(f'iteration completed.')
        # print(f'Energy in scatter bank = {mesh.nrgscattered}')
        iterations += 1
    
    # Now, there is left over energy in mesh.nrgscattered. It is small, 
    # but we will dump it into the material to conserve energy and not throw it away.
    mesh.nrgdep[:] += mesh.nrgscattered[:]
    print(f'Number of scattering iterations until convergence = {iterations}')
        

def clean():
    """Tidy up the particle list by removing leaked and absorbed particles"""
    particles_removed = 0
    for iptcl in range(len(ptcl.particle_prop) - 1, -1, -1): 
        if ptcl.particle_prop[iptcl][5] < 0.0:
            del ptcl.particle_prop[iptcl]
            particles_removed += 1
    
    print(f'Number of particles removed = {particles_removed}')
    print(f'Number of particles in the system = {len(ptcl.particle_prop)}')
