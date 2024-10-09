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
import imc_global_volsource_data as vol

def SuOlson1996(output_file):
    """
    Control calculation for SuOlson1996 Surface Source problem.

    Timestep loop is within this function
    """
    # Set plot times
    plottimes = [0.1, 1.0, 10.0]
    plottimenext = 0

    # Open output file
    fname = open(output_file, "wb")

    print(f' temperature = {mesh.temp}')
    print(f' rad temp = {mesh.radtemp}')

    # Set opacities
    mesh.sigma_a[:] = mat.sigma_a
    mesh.sigma_s[:] = mat.sigma_s
    mesh.sigma_t[:] = mesh.sigma_a[:] + mesh.sigma_s[:]
    # Set fleck factor
    mesh.fleck[:] = 1 / (1 + mesh.sigma_t[:] * phys.c * time.dt)
    print(f'fleck factor = {mesh.fleck}')
    
    # Begin time
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
            print()

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
                pickle.dump(mesh.matnrgdens, fname, 0)
                pickle.dump(mesh.radnrgdens, fname, 0)
                plottimenext = plottimenext + 1

    # Close file
    fname.close()

def SuOlson1997(output_file):
    """
    Control calculation for SuOlson1997 Volume Source problem.

    Timestep loop is within this function
    """
    # Set plot times
    plottimes = np.array([0.1, 1.0, 10.0, 100.0])
    print(f'plottimes = {plottimes}')
    plottimenext = 0

    # Open output file
    fname = open(output_file, "wb")

    print(f' temperature = {mesh.temp[:10]}')
    print(f' rad temp = {mesh.radtemp[:10]}')

    mat.alpha = 4 * phys.a / mat.epsilon

    # Set fleck factor
    mesh.fleck[:] = 1.0 # 1.0 / (1.0 + mesh.sigma_a[:] * phys.c * time.dt)
    
    # Begin time
    time.time = 0.0

    # Set energy densities
    mesh.radnrgdens = np.zeros(mesh.ncells)
    mesh.matnrgdens = np.zeros(mesh.ncells)

    # Loop over timesteps

    for time.step in range(1, time.ns + 1):
        print(f'Step: {time.step}')
        # Update temperature dependent quantities
        imc_update.run()

        # Source new particles
        if part.mode == 'nrn':
            imc_source.create_body_source_particles()
            if time.time < vol.tau_0/phys.c:
                print(f'volume source particles created.')
                imc_source.create_volume_source_particles()
            

        if part.mode == 'rn':
            imc_source.volume_sourcing_random()
            
        # Track particles through the mesh
        imc_track.run()
        imc_track.clean()

        # Check for particles with energies less than zero
        for iptcl in range(len(part.particle_prop)):
            nrg = part.particle_prop[iptcl][5]
            if nrg < 0.0:
                print(f'Particle prop = {part.particle_prop[iptcl]}')
                raise ValueError(f"Particle {iptcl} has negative energy: {nrg}")

        # Tally
        imc_tally.run()

        # Update time
        time.time = round(time.time + time.dt, 3)
        # Apply population control on particles if needed
        if len(part.particle_prop) > part.n_max and part.mode == 'nrn':
            imc_update.population_control()

        if len(part.particle_prop) > part.n_max and part.mode == 'rn':
            # insert population control method here.
            print()

        # Plot
        if plottimenext <= 3:
            print(f'Time = {time.time}')
            if (time.time) >= plottimes[plottimenext]:
                print("Plotting {:6d}".format(plottimenext))
                print("at target time {:24.16f}".format(plottimes[plottimenext]))
                print("at actual time {:24.16f}".format(time.time))
                
                fname.write("Time = {:24.16f}\n".format(time.time).encode())
                pickle.dump(mesh.cellpos, fname, 0)
                pickle.dump(mesh.matnrgdens, fname, 0)
                pickle.dump(mesh.radnrgdens, fname, 0)
                plottimenext = plottimenext + 1

    # Close file
    fname.close()

def EnergyCheck(output_file):
    """Test for checking energy balance and particle tracking."""

    # Set plot times
    plottimes = [0.1, 1.0, 10.0]
    plottimenext = 0
    
    # open output file
    fname = open(output_file, "wb")

    print(f' temperature = {mesh.temp}')
    print(f' rad temp = {mesh.radtemp}')

    # Set opacities

    # Set fleck factor
    mesh.fleck[:] = 1.0 #1.0 / (1.0 + mesh.sigma_a[:] * phys.c * time.dt)
    print(f'fleck factor = {mesh.fleck}')
    
    time.time = 0.0


    # Loop over timesteps

    for time.step in range(0, time.ns):
        # Update temperature dependent quantities
        imc_update.run()

        print(f'Energy sourced this time-step = {phys.c * mesh.sigma_a * mesh.fleck * phys.a * (mesh.temp ** 4) * time.dt * mesh.dx}')
        # Source new particles
        if part.mode == 'nrn':
            imc_source.create_body_source_particles()

        if part.mode == 'rn':
            imc_source.create_body_source_particles_random()

        # Track particles through the mesh
        imc_track.run()
        imc_track.clean()
        
        print(f'Energy deposited this time-step = {mesh.nrgdep}')

        # Tally
        imc_tally.run()

        # Update time
        time.time += time.dt

        

    # Close file
    fname.close()