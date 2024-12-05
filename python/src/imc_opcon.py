"""Control of main numerical calculation"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    mat.alpha = 4 * phys.a / mat.epsilon
    mesh.matnrgdens = np.zeros(mesh.ncells)
    # Set opacities
    mesh.sigma_a = np.zeros(mesh.ncells)
    mesh.sigma_s = np.zeros(mesh.ncells)
    mesh.sigma_t = np.zeros(mesh.ncells)
    mesh.sigma_a[:] = mat.sigma_a
    mesh.sigma_s[:] = mat.sigma_s
    mesh.sigma_t[:] = mesh.sigma_a[:] + mesh.sigma_s[:]
    # Set fleck factor
    mesh.fleck[:] = 1.0 / (1.0 + mesh.sigma_t[:] * phys.c * time.dt)
    print(f'fleck factor = {mesh.fleck}')
    
    # Begin time
    time.time = 0.0

    # Loop over timesteps
    # Create census grid
    if part.mode == 'nrn':
        imc_source.create_census_grid()
    print()
    # Loop over timesteps
    try:
        with open(output_file, "wb") as fname:
            for time.step in range(0, time.ns):
                print(f'step {time.step} @ time = {time.time}')
                # Update temperature dependent quantities
                imc_update.SuOlson_update()

                # Source new particles
                if part.mode == 'nrn':
                    imc_source.create_surface_source_particles_diffusion()
                    imc_source.create_body_source_particles()

                if part.mode == 'rn':
                    imc_source.run()

                # Track particles through the mesh
                if part.mode == 'nrn':
                    imc_track.run()
                
                if part.mode == 'rn':
                    imc_track.run_random()
                
                imc_track.clean()

                # Tally
                imc_tally.SuOlson_tally()

                # Update time
                time.time = round(time.time + time.dt, 5)

                # Apply population control on particles if needed
                if len(part.particle_prop) > part.n_max and part.mode == 'nrn':
                    imc_update.population_control()

                # Plot
                if plottimenext <= 2:
                    if (time.time) >= plottimes[plottimenext]:
                        print("Plotting {:6d}".format(plottimenext))
                        print("at target time {:24.16f}".format(plottimes[plottimenext]))
                        print("at actual time {:24.16f}".format(time.time))
                        
                        fname.write("Time = {:24.16f}\n".format(time.time).encode())
                        pickle.dump(mesh.cellpos, fname, 0)
                        # pickle.dump(mesh.temp, fname, 0)
                        # pickle.dump(mesh.radtemp, fname, 0)
                        pickle.dump(mesh.matnrgdens, fname, 0)
                        pickle.dump(mesh.radnrgdens, fname, 0)
                        plottimenext = plottimenext + 1
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")


def SuOlson1997(output_file):
    """
    Control calculation for SuOlson1997 Volume Source problem.

    Timestep loop is within this function
    """
    # Set plot times
    plottimes = np.array([0.1, 1.0, 10.0, 100.0])
    print(f'plottimes = {plottimes}')
    plottimenext = 0

    print(f' temperature = {mesh.temp[:10]}')
    print(f' rad temp = {mesh.radtemp[:10]}')

    mat.alpha = 4 * phys.a / mat.epsilon
    print(f'mat.alpha = {mat.alpha}')

    # Set fleck factor
    mesh.fleck[:] = 1.0 / (1.0 + mesh.sigma_a[:] * phys.c * time.dt)
    
    # Begin time
    time.time = 0.0

    # Columns: [origin, emission_time, icell, xpos, mu, nrg, startnrg]
    part.particle_prop = np.zeros((part.max_array_size, 7), dtype=np.float64)
    part.n_particles = np.zeros(1, dtype=int)
    # Set energy densities
    mesh.radnrgdens = np.zeros(mesh.ncells)
    mesh.matnrgdens = np.zeros(mesh.ncells)

    # Total opacity
    mesh.sigma_t = mesh.fleck * mesh.sigma_a + (1.0 - mesh.fleck) * mesh.sigma_a + mesh.sigma_s
    print(f'mesh.sigma_t = {mesh.sigma_t}')
    # Loop over timesteps
    try:
        with open(output_file, "wb") as fname:
            for time.step in range(1, time.ns + 1): # time.ns + 1
                print(f'Step: {time.step} @ time = {time.time}')
                # Reset energy deposition and scattering arrays
                mesh.nrgdep = np.zeros(mesh.ncells)
                # Update temperature dependent quantities
                mat.b = imc_update.SuOlson_update(mesh.temp)
                
                # Source new particles
                if part.mode == 'nrn':
                    part.n_particles, part.particle_prop = imc_source.create_body_source_particles(part.n_particles, part.particle_prop, mesh.temp, time.time)
                    if time.time < vol.tau_0/phys.c:
                        print(f'volume source particles created.')
                        imc_source.create_volume_source_particles()
                    
                if part.mode == 'rn':
                    imc_source.volume_sourcing_random()
            
                # Track particles through the mesh
                if part.mode == 'rn':
                    imc_track.run_random()
                if part.mode == 'nrn':
                    
                    mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run(part.n_particles, part.particle_prop, mesh.nrgdep, mesh.nrgscattered, time.time)
                    print(f'mesh.nrgdep = {mesh.nrgdep[:10]}')
                    
                    
                part.n_particles, part.particle_prop  = imc_track.clean(part.n_particles, part.particle_prop)

                # Check for particles with energies less than zero
                # for iptcl in range(len(part.particle_prop)):
                #     nrg = part.particle_prop[iptcl][5]
                #     if nrg < 0.0:
                #         print(f'Particle prop = {part.particle_prop[iptcl]}')
                #         raise ValueError(f"Particle {iptcl} has negative energy: {nrg}")

                # Tally
                mesh.matnrgdens, mesh.radnrgdens, mesh.temp = imc_tally.SuOlson_tally(mesh.nrgdep, part.n_particles, part.particle_prop, mesh.matnrgdens, mesh.temp)

                # Update time
                time.time = round(time.time + time.dt, 5)

                # Apply population control on particles if needed
                # if part.n_particles > part.n_max and part.mode == 'nrn':
                # # if time.step == 1000:
                #     part.n_particles, part.particle_prop = imc_update.population_control(part.n_particles, part.particle_prop, time.time)


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
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")



def marshak_wave(output_file):
    """
    Control calculation for Marshak Wave problem.

    Timestep loop is within this function
    """

    # Set physical constants
    phys.c = 2.99792458e2  # [cm/sh] 
    phys.a = 0.013720169037741436  #  [jrk/(cm^3-keV^4)]
    phys.sb = phys.a * phys.c / 4.0
    phys.invc = 1.0 / phys.c
    print(f'Physical constants: phys.c = {phys.c}, phys.a = {phys.a}, phys.sb = {phys.sb}')
    
    # Set plot times
    plottimes = np.array([0.3])
    print(f'plottimes = {plottimes}')
    plottimenext = 0
    
    time.dt = 1e-06
    t_final = 0.3
    dt_max = 1e-5

    # Open output file
    fname = open(output_file, "wb")

    print(f'temperature = {mesh.temp[:10]}')
    print(f'rad temp = {mesh.radtemp[:10]}')

    

    # Set heat capacity
    mat.rho = 1.0  # [g/cc]
    mat.b = np.ones(mesh.ncells) * 0.3  # [jrk/g/keV]
    print(f'heat capacity = {mat.b[:10]}')
    mesh.sigma_s = np.zeros(mesh.ncells)

    # Begin time
    time.time = 0.0
    mesh.matnrgdens = mesh.temp * mat.b * mat.rho
    print(f'mesh.matnrgdens = {mesh.matnrgdens[:10]}')

    # Create census grid
    if part.mode == 'nrn':
        imc_source.create_census_grid()
    print()
    # Loop over timesteps
    try:
        with open(output_file, "wb") as fname:

            # Loop over timesteps
            while time.time < t_final:
                print()
                print(f'Step: {time.step} @ {time.time}')
                print(f'dt = {time.dt}')
                time.step += 1
                # Update temperature dependent quantities
                imc_update.marshak_wave_update()

                # Begin tally for energy leaked this time-step
                mesh.nrg_leaked = 0.0
                
                # Source new particles
                if part.mode == 'nrn':    
                    imc_source.create_body_source_particles()
                    imc_source.create_surface_source_particles()

                if part.mode == 'rn':
                    imc_source.run()

                # Track particles through the mesh
                if part.mode == 'rn':
                    imc_track.run_random()

                if part.mode == 'nrn':
                    imc_track.run()

                imc_track.clean()

                # Tally
                imc_tally.marshak_wave_tally()

                # Find the energy in the radiation

                
                # if time.step == 200:
                #     plt.figure()
                #     plt.plot(mesh.cellpos, mesh.temp, marker='o')
                #     plt.yscale('log')
                #     plt.show()
                    

                # Update time
                time.time = round(time.time + time.dt, 9)
                # Make a larger time-step
                if time.dt < dt_max:
                    # Increase time-step
                    time.dt = time.dt * 1.1

                # Check for final time-step
                if time.time + time.dt > t_final:
                    time.dt = t_final - time.time

                # for iptcl in range(len(part.particle_prop)):
                # # Get particle's initial properties
                #     (ttt, icell, xpos, mu, nrg, startnrg) = part.particle_prop[iptcl][1:7]
                #     print(f'ttt = {ttt}, icell = {icell}, xpos = {xpos}, mu = {mu}, nrg = {nrg}, startnrg = {startnrg}')

                # Apply population control on particles if needed
                if len(part.particle_prop) > part.n_max and part.mode == 'nrn':
                    imc_update.population_control()

                print(f'energy leaked = {mesh.nrg_leaked}')
                # Plot
                if plottimenext <= 2 and time.time >= plottimes[plottimenext]:
                    print(f"Plotting {plottimenext}")
                    print(f"at target time {plottimes[plottimenext]:24.16f}")
                    print(f"at actual time {time.time:24.16f}")
                    
                    fname.write(f"Time = {time.time:24.16f}\n".encode())
                    pickle.dump(mesh.cellpos, fname, 0)
                    pickle.dump(mesh.temp, fname, 0)
                    pickle.dump(mesh.radnrgdens, fname, 0)
                    plottimenext += 1

        print(f'Final time = {time.time}')
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")


def EnergyCheck(output_file):
    """
    Control calculation for Slab Energy Check.

    Timestep loop is within this function
    """

    # Set physical constants
    phys.c = 2.99792458e2  # [cm/sh] 
    phys.a = 0.013720169037741436  #  [jrk/(cm^3-keV^4)]
    phys.sb = phys.a * phys.c / 4.0
    phys.invc = 1.0 / phys.c
    print(f'Physical constants: phys.c = {phys.c}, phys.a = {phys.a}, phys.sb = {phys.sb}')
    
    # Set plot times
    plottimes = np.array([0.1, 0.2, 0.3])
    print(f'plottimes = {plottimes}')
    plottimenext = 0
    
    time.dt = 1e-8
    t_final = 0.3
    dt_max = 1e-5

    # Open output file
    fname = open(output_file, "wb")

    print(f' temperature = {mesh.temp[:10]}')
    print(f' rad temp = {mesh.radtemp[:10]}')

    # Set heat capacity
    mat.rho = 1.0  # [g/cc]
    mat.b = np.ones(mesh.ncells) * 0.3  # [jrk/g/keV]
    print(f'heat capacity = {mat.b[:10]}')
    mesh.sigma_s = np.zeros(mesh.ncells)

    # Begin time
    time.time = 0.0
    mesh.matnrgdens = mesh.temp * mat.b * mat.rho
    print(f'mesh.matnrgdens = {mesh.matnrgdens[:10]}')

    # Create census grid
    if part.mode == 'nrn':
        imc_source.create_census_grid()

    # Loop over timesteps
    try:
        with open(output_file, "wb") as fname:

            # Loop over timesteps
            while time.time < t_final:
                print(f'Step: {time.step} @ {time.time}')
                print(f'dt = {time.dt}')
                time.step += 1
                # Update temperature dependent quantities
                imc_update.marshak_wave_update()

                # Source new particles
                if part.mode == 'nrn':    
                    imc_source.create_body_source_particles()
                    #imc_source.create_surface_source_particles()

                if part.mode == 'rn':
                    imc_source.run()

                # Track particles through the mesh
                if part.mode == 'rn':
                    imc_track.run_random()

                if part.mode == 'nrn':
                    imc_track.run()

                imc_track.clean()
                # # Check for particles with energies less than zero
                # for iptcl in range(len(part.particle_prop)):
                #     nrg = part.particle_prop[iptcl][5]
                #     if nrg < 0.0:
                #         print(f'Particle prop = {part.particle_prop[iptcl]}')
                #         raise ValueError(f"Particle {iptcl} has negative energy: {nrg}")

                # Tally
                imc_tally.marshak_wave_tally()

                # Find the energy in the radiation

                
                # if time.step == 200:
                #     plt.figure()
                #     plt.plot(mesh.cellpos, mesh.temp, marker='o')
                #     plt.yscale('log')
                #     plt.show()
                    

                # Update time
                time.time = round(time.time + time.dt, 9)
                # Make a larger time-step
                if time.dt < dt_max:
                    # Increase time-step
                    time.dt = time.dt * 1.1

                # Check for final time-step
                if time.time + time.dt > t_final:
                    time.dt = t_final - time.time

                # Apply population control on particles if needed
                if len(part.particle_prop) > part.n_max and part.mode == 'nrn':
                    imc_update.population_control()

                # Plot
                if plottimenext <= 2 and time.time >= plottimes[plottimenext]:
                    print(f"Plotting {plottimenext}")
                    print(f"at target time {plottimes[plottimenext]:24.16f}")
                    print(f"at actual time {time.time:24.16f}")
                    
                    fname.write(f"Time = {time.time:24.16f}\n".encode())
                    pickle.dump(mesh.cellpos, fname, 0)
                    pickle.dump(mesh.temp, fname, 0)
                    plottimenext += 1

        print(f'Final time = {time.time}')
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")