"""Control of main numerical calculation"""

import pickle
import numpy as np

import imc_update
import imc_source
import imc_tally
import imc_track

import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_mat_data as mat
import imc_global_part_data as part

def run(output_file):
    """
    Control calculation for IMC.

    Timestep loop is within this function
    """
    # Set plot times
    plottimes = [1E-11, 1E-10, 1E-9]
    plottimenext = 0

    # Open output file
    fname = open(output_file, "wb")

    print(f' temperature = {mesh.temp}')
    print(f' rad temp = {mesh.radtemp}')

    # Set opacities
    mesh.sigma_a[:] = mat.sigma_a
    mesh.sigma_s[:] = mat.sigma_s
    mesh.sigma_t[:] = mesh.sigma_a[:] + mesh.sigma_s[:]
    
    
    
    time.time = 0.0

    # Create census particles
    if part.mode == 'nrn':
        imc_source.create_census_particles()

    if part.mode == 'rn':
        imc_source.create_census_particles_random()

    # Loop over timesteps

    for time.step in range(0, time.ns):
        # Update temperature dependent quantities
        imc_update.run()

        # Source new particles
        if part.mode == 'nrn':
            imc_source.create_surface_source_particles()
            imc_source.create_body_source_particles()

        if part.mode == 'rn':
            imc_source.create_surface_source_particles_random()
            imc_source.create_body_source_particles_random()

        # Track particles through the mesh
        imc_track.run()
        imc_track.clean()

        # Tally
        imc_tally.run()

        # Update time
        time.time += time.dt

        # Apply population control on particles if needed
        if len(part.particle_prop) > part.n_max and part.mode == 'nrn':
            imc_update.population_control()

        if len(part.particle_prop) > part.n_max and part.mode == 'rn':
            # insert population control method here.
            lol = 1

        # Plot
        if plottimenext <= 2:
            if (time.time) >= plottimes[plottimenext]:
                print("Plotting {:6d}".format(plottimenext))
                print("at target time {:24.16f}".format(plottimes[plottimenext]))
                print("at actual time {:24.16f}".format(time.time))
                
                fname.write("Time = {:24.16f}\n".format(time.time).encode())
                pickle.dump(mesh.cellpos, fname, 0)
                pickle.dump(mesh.temp, fname, 0)
                pickle.dump(mesh.radtemp, fname, 0)
                plottimenext = plottimenext + 1

    # Close file
    fname.close()

