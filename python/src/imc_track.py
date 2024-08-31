"""Advance particles over a time-step"""

import numpy as np

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_part_data as ptcl
import imc_global_phys_data as phys
import imc_global_time_data as time


def run():
    """Advance particles over a time-step"""
    # Create angle and scattering arrays
    angles = -1.0 + (np.arange(ptcl.Nmu) + 0.5) * 2 / ptcl.Nmu
    xi_values = (np.arange(ptcl.Nxi) + 0.5) / ptcl.Nxi

    # Create local storage for the energy deposited this time-step
    mesh.nrgdep[:] = 0.0

    endsteptime = time.time + time.dt

    # optimizations
    exp = np.exp
    log = np.log
    nrgdep = [0.0] * mesh.ncells
    mesh_nodepos = mesh.nodepos
    phys_c = phys.c
    top_cell = mesh.ncells - 1
    phys_invc = phys.invc
    mesh_sigma_a = mesh.sigma_a
    mesh_sigma_s = mesh.sigma_s
    mesh_rightbc = mesh.right_bc
    mesh_leftbc = mesh.left_bc


    print(f'Particle Loop')

    # Loop over all particles
    for iptcl in range(len(ptcl.particle_prop)):
        # Get particle's initial properties at start of time-step
        (ttt, icell, xpos, mu, xi, nrg, startnrg) = ptcl.particle_prop[iptcl][1:8]
        startnrg = 0.01 * startnrg
        # Loop over segments in the history (between boundary-crossings and collisions)
        while True:
            # Distance to boundary
            if mu > 0.0:
                dist_b = (mesh_nodepos[icell + 1] - xpos) / abs(mu)
            else:
                dist_b = (xpos - mesh_nodepos[icell]) / abs(mu)
            
            # Distance to scatter
            dist_scatter = -log(xi) / mesh_sigma_s[icell]

            # Distance to census
            dist_cen = phys_c * (endsteptime - ttt)

            # Actual distance - whichever happens first
            dist = min(dist_b, dist_cen, dist_scatter)

            # Calculate the new energy and the energy deposited (temp storage)
            newnrg = nrg * exp(-mesh_sigma_a[icell] * dist)
            if newnrg <= startnrg:
                newnrg = 0.0
            
            # Deposit the particle's energy
            nrgdep[icell] += nrg - newnrg

            if newnrg == 0.0:
                # Flag particle for later destruction
                ptcl.particle_prop[iptcl][7] = -1.0
                break

            # Boundary treatment
            if dist == dist_b:
                
                # Left boundary
                if mu < 0 and icell == 0:
                    
                    if mesh_leftbc == 'vacuum':
                        
                        # Flag particle for later destruction
                        ptcl.particle_prop[iptcl][6] = -1.0
                        break
                    elif mesh_leftbc == 'reflecting':
                        
                        mu *= -1.0  # Reverse direction
                        
                # Right boundary
                elif mu > 0 and icell == top_cell:
                    if mesh_rightbc == 'vacuum':
                        ptcl.particle_prop[iptcl][6] = -1.0
                        break
                    elif mesh_rightbc == 'reflecting':
                        mu *= -1.0  # Reverse direction
                        
                
                # Move particle to the left
                elif mu < 0 and icell != 0:
                    icell -= 1
                
                # Move particle to the right
                elif mu > 0 and icell != top_cell:
                    icell += 1
            
            # Advance position, time, and energy
            xpos += mu * dist
            ttt += dist * phys_invc
            nrg = newnrg

            # If the event was census, finish this history
            if dist == dist_cen:
                # Finished with this particle
                # Update the particle's properties in the list
                ptcl.particle_prop[iptcl][1:6] = (ttt, icell, xpos, mu, xi, nrg)
                break

            # If event was collision, also update direction
            if dist == dist_scatter:
                # Collision
                # Update mu
                mu_index = np.where(angles == mu)[0]
                if len(mu_index) > 0:
                    mu_index = mu_index[0]
                    mu_index = (mu_index + 1) % len(angles)
                    mu = angles[mu_index]

                # Update xi
                xi_index = np.where(xi_values == xi)[0][0]
                xi_index = (xi_index + 1) % len(xi_values)
                xi = xi_values[xi_index]

        # End loop over history segments

    # End loop over particles
                
    mesh.nrgdep[:] = nrgdep[:]
    #print(f'Energy deposited in time-step = {nrgdep}')


def clean():
    """Tidy up the particle list be removing leaked and absorbed particles"""
    # These particles had their energy set to -1 to flag them.
    for iptcl in range(len(ptcl.particle_prop) - 1, 0, -1):
        if ptcl.particle_prop[iptcl][6] < 0.0:
            del ptcl.particle_prop[iptcl]
    
    print(f'Number of particles in the system = {len(ptcl.particle_prop)}')
