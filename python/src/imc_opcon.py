"""Control of main numerical calculation"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time as tm
import pandas as pd
import numba

import imc_update
import imc_source
import imc_tally
import imc_track

import imc_global_bcon_data as bcon
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_mat_data as mat
import imc_global_part_data as part
import imc_global_volsource_data as vol
import imc_utilities as imc_util


def SuOlson1997(output_file):
    """
    Control calculation for SuOlson1997 Volume Source problem.

    Timestep loop is within this function
    """
    # Set plot times
    plottimes = np.array([0.1, 1.0, 10.0, 100.0])
    print(f'plottimes = {plottimes}')
    plottimenext = 0

    animation = False
    if animation:
        data = np.zeros((time.ns, mesh.ncells, 3))

    print(f' temperature = {mesh.temp[:10]}')
    print(f' rad temp = {mesh.radtemp[:10]}')

    mat.alpha = 4 * phys.a / mat.epsilon
    print(f'mat.alpha = {mat.alpha}')

    # Set fleck factor
    mesh.fleck[:] = 1.0 / (1.0 + mesh.sigma_a[:] * phys.c * time.dt)
    print(f'mesh.fleck = {mesh.fleck}')
    # Begin time
    time.time = 0.0

    # Columns: [origin, emission_time, icell, xpos, mu, frq, nrg, startnrg]
    part.max_array_size = 2_000_000
    part.particle_prop = np.zeros((part.max_array_size, 8), dtype=np.float64)
    part.n_particles = np.zeros(1, dtype=int)
    # Set energy densities
    mesh.radnrgdens = np.zeros(mesh.ncells) 
    mesh.matnrgdens = np.ones(mesh.ncells) * 1.26491147e-07 * mesh.temp
    print(f'mesh.matnrgdens = {mesh.matnrgdens}')
    # Total opacity
    mesh.sigma_t = mesh.fleck * mesh.sigma_a + (1.0 - mesh.fleck) * mesh.sigma_a + mesh.sigma_s
    print(f'mesh.sigma_t = {mesh.sigma_t}')

    # Convert angle info into arrays
    if part.mode == 'nrn':
        part.Nmu = np.ones(mesh.ncells, dtype=int) * part.Nmu
    timesteps = time.ns
    ncells = mesh.ncells

    all_times = np.zeros(timesteps)
    all_radnrgdens = np.zeros((timesteps, ncells))
    all_matnrgdens = np.zeros((timesteps, ncells))
    runtimes = np.zeros(timesteps)
    
    # Loop over timesteps
    try:
        with open(output_file, "wb") as fname:
            for time.step in range(1, time.ns + 1): # time.ns + 1
                step_start_time = tm.perf_counter()  # Start timing
                print(f'Step: {time.step} @ time = {time.time}')

                # Update temperature dependent quantities
                mat.b = imc_update.SuOlson_update(mesh.temp)
                
                # Source new particles
                if part.mode == 'nrn':
                    part.n_particles, part.particle_prop = imc_source.create_body_source_particles(part.n_particles, part.particle_prop, mesh.temp, time.time, time.dt, mesh.sigma_a, mesh.fleck)
                    if time.time < vol.tau_0/phys.c:
                        print(f'volume source particles created.')
                        part.n_particles, part.particle_prop = imc_source.create_volume_source_particles(part.n_particles, part.particle_prop, time.dt)
                    
                if part.mode == 'rn':
                    vol.radiation_source = True
                    mesh.source_cells = int((np.ceil(vol.x_0/mesh.dx)))
                    part.n_particles, part.particle_prop = imc_source.run(mesh.fleck, mesh.temp, mesh.sigma_a, part.particle_prop, part.n_particles, time.time, time.dt)
            
                # Track particles through the mesh
                if part.mode == 'rn':
                    mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run_random(part.n_particles, part.particle_prop, time.time, time.dt, mesh.sigma_a, mesh.fleck, mesh.sigma_s)
                    print(f'mesh.nrgdep = {mesh.nrgdep[:10]}')
                if part.mode == 'nrn':
                    mesh.nrgdep, part.n_particles, part.particle_prop, eddington = imc_track.run(part.n_particles, part.particle_prop, time.time, time.dt, part.Nmu, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck)
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
                print(f'mesh.matnrgdens[0] = {mesh.matnrgdens[0]}')
                # Update time
                time.time = round(time.time + time.dt, 5)

                if part.mode == 'nrn':
                    tolerance = 0.06
                    cell_max_index = np.argmax(eddington)
                    print(f'index of max eddington = {cell_max_index}')
                    for i in range(mesh.ncells):
                        if i < cell_max_index:
                            part.Nmu[i] = 8
                        elif i > cell_max_index and abs(eddington[i] - 1/3) <= tolerance:
                            part.Nmu[i] = 4
                #     # Plot Eddington
                #     if time.step % 100 == 0:  # Check if the current time step is a multiple
                        
                #         # part.Nmu[0:10] = 8
                #         fig, ax1 = plt.subplots()  # Create the plot
                        
                #         # Plot the energy densities on the left y-axis
                #         ax1.plot(mesh.cellpos, mesh.matnrgdens, label='Material Energy Density', color='b')
                #         ax1.plot(mesh.cellpos, mesh.radnrgdens, label='Radiation Energy Density', color='r')
                        
                #         ax1.set_xlabel('x')
                #         ax1.set_ylabel('Energy Density', color='k')
                #         ax1.set_yticks(np.arange(0, 2.81, 0.25))
                #         ax1.set_xticks(np.arange(0, 10.01, 1.0))
                #         ax1.set_ylim(0, 2.81)
                #         ax1.tick_params(axis='y', labelcolor='k')
                        
                #         # Create a second y-axis for the Eddington factor
                #         ax2 = ax1.twinx()
                #         ax2.plot(mesh.cellpos, eddington, label='Eddington Factor', color='g')
                #         ax2.set_ylabel('Eddington Factor', color='k')
                #         ax2.set_ylim(0, 1)
                #         ax2.tick_params(axis='y', labelcolor='k')
                #         ax2.set_yticks(np.arange(0, 1.01, 0.1))
                        
                #         # Title and legend
                #         plt.title(f'Eddington Factor @ time={time.time}')
                #         ax1.legend(loc='upper left')
                #         ax2.legend(loc='upper right')
                        
                #         # Save the figure with the time in the filename
                #         filename = 'eddington_time_{:.2f}.png'.format(time.time)
                #         plt.savefig(filename, format='png', dpi=900)
                #         plt.close()

                #         # plt.figure()
                #         # plt.plot(part.Nmu)
                #         # plt.show()

                # Save data to NPZ per time step
                step_idx = time.step - 1  # zero-based indexing
                all_times[step_idx] = time.time
                all_radnrgdens[step_idx, :] = mesh.radnrgdens
                all_matnrgdens[step_idx, :] = mesh.matnrgdens
                runtimes[step_idx] = tm.time() - step_start_time

                # Plot
                if plottimenext <= 3:
                    print(f'Time = {time.time}')
                    if (time.time) >= plottimes[plottimenext]:
                        print("Plotting {:6d}".format(plottimenext))
                        print("at target time {:24.16f}".format(plottimes[plottimenext]))
                        print("at actual time {:24.16f}".format(time.time))
                        
                        fname.write("Time = {:24.16f}\n".format(time.time).encode())
                        
                        # Debugging before dumping
                        print("Dumping cellpos...")
                        print(mesh.cellpos)
                        pickle.dump(mesh.cellpos, fname, 0)
                        
                        print("Dumping matnrgdens...")
                        print(mesh.matnrgdens)
                        pickle.dump(mesh.matnrgdens, fname, 0)
                        
                        print("Dumping radnrgdens...")
                        print(mesh.radnrgdens)
                        pickle.dump(mesh.radnrgdens, fname, 0)
                        plottimenext = plottimenext + 1

                if animation:
                    # Ensure the directory exists
                    output_dir = "Su_Olson_animation"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    data[time.step-1, :, 0] = mesh.cellpos
                    data[time.step-1, :, 1] = mesh.radnrgdens
                    data[time.step-1, :, 2] = mesh.matnrgdens

    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")
    if animation:
        output_dir = "Su_Olson_animation"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "simulation_data.npy")
        np.save(output_file, data)
    np.savez("Su_Olson_output.npz",
         mesh_nodepos=mesh.cellpos,
         time=all_times,
         radnrgdens=all_radnrgdens,
         matnrgdens=all_matnrgdens,
         runtimes=runtimes)
    

def marshak_wave(output_file):
    """
    Control calculation for Marshak Wave problem.

    Timestep loop is within this function
    """

    # Set physical constants
    phys.c = 2.99792458e2  # [cm/sh] 
    phys.a = 0.01372016  #  [jk/(cm^3-keV^4)]
    phys.sb = phys.a * phys.c / 4.0
    phys.invc = 1.0 / phys.c
    print(f'Physical constants: phys.c = {phys.c}, phys.a = {phys.a}, phys.sb = {phys.sb}')
    
    # Set plot times
    plottimes = np.array([0.3])
    print(f'plottimes = {plottimes}')
    plottimenext = 0
    
    time.dt = 1e-07
    t_final = 0.3
    dt_max = 1e-03
    
    print(f'temperature = {mesh.temp[:10]}')
    print(f'rad temp = {mesh.radtemp[:10]}')

    # initialize particle arrays
    part.max_array_size = 50_000_000
    part.particle_prop = np.zeros((part.max_array_size, 8), dtype=np.float64)
    part.n_particles = 0

    # Set density and heat capacity
    mat.rho = 1.0  # [g/cc]
    mat.b = np.ones(mesh.ncells) * 0.3  # [jrk/g/keV]
    
    print(f'heat capacity = {mat.b[:10]}')

    # Turn off real scattering (effective scattering only)
    mesh.sigma_s = np.zeros(mesh.ncells)
    if part.mode == 'nrn':
        part.Nmu = np.ones(mesh.ncells, dtype=int) * part.Nmu

    # Begin time
    time.time = 0.0
    mesh.matnrgdens = mesh.temp * mat.b * mat.rho
    print(f'mesh.matnrgdens = {mesh.matnrgdens[:10]}')
    parallel=True
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
                mesh.beta, mesh.sigma_a, mesh.sigma_t, mesh.fleck = imc_update.marshak_wave_update(mesh.temp, time.dt)
                
                # Source new particles
                if part.mode == 'nrn':    
                    part.n_particles, part.particle_prop = imc_source.create_body_source_particles(part.n_particles, part.particle_prop, mesh.temp, time.time, time.dt, mesh.sigma_a, mesh.fleck)
                    part.n_particles, part.particle_prop = imc_source.create_surface_source_particles(part.n_particles, part.particle_prop, time.time, time.dt)

                if part.mode == 'rn':
                    bcon.surface_source = True
                    part.n_particles, part.particle_prop = imc_source.run(mesh.fleck, mesh.temp, mesh.sigma_a, part.particle_prop, part.n_particles, time.time, time.dt)

                # Track particles through the mesh
                if part.mode == 'nrn':
                    if parallel:
                        # Step 1: Track initial batch of particles
                        mesh.nrgdep, nrgscattered, x_Es, tEs = imc_track.run_parallel_firstloop(
                             part.n_particles, part.particle_prop,
                             time.time, time.dt, part.Nmu, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck)
                        
                        original_nrg_scattered = np.copy(nrgscattered)

                        # Step 2: Implicit scattering loop
                        epsilon = 1e-3
                        iterations=0
                        converged=False

                        while not converged:
                            # Generate scattered particles
                            scattered_particles, n_scattered_particles = imc_track.generate_scattered_particles1D(
                                nrgscattered, x_Es, tEs,
                                mesh.nodepos, mesh.dx, 
                                part.max_array_size, part.Nx, part.Nt, part.Nmu,
                                time.time, time.dt
                            )

                            # Track scattered particles
                            nrgdep_scat, nrgscattered, x_Es, tEs = imc_track.run_parallel_firstloop(
                                n_scattered_particles, scattered_particles,
                                time.time, time.dt, part.Nmu, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck)
                            
                            # Add scattered particle deposition to the mesh
                            mesh.nrgdep += nrgdep_scat

                            # Copy existing particles into global array
                            n_existing_particles = part.n_particles
                            n_total_particles = n_existing_particles + n_scattered_particles
                            if n_total_particles > part.max_array_size:
                                raise ValueError("Not enough space in global array for scattered particles")

                            part.particle_prop[n_existing_particles:n_total_particles, :] = scattered_particles[:n_scattered_particles, :]
                            part.n_particles = n_total_particles

                            # Check convergence
                            rel_remaining = np.sum(nrgscattered) / np.sum(original_nrg_scattered)
                            if rel_remaining < epsilon:
                                converged = True
                        
                            iterations += 1
                        # After congerging, dump remaining scattered energy
                        print(f'Number of scattering iterations = {iterations}')
                    # else:
                    #     mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run_(part.n_particles, part.particle_prop, time.time, time.dt, mesh.sigma_a, mesh.fleck, mesh.sigma_s)


                part.n_particles, part.particle_prop  = imc_track.clean(part.n_particles, part.particle_prop)

                # Tally
                mesh.matnrgdens, mesh.radnrgdens, mesh.temp = imc_tally.marshak_wave_tally(mesh.nrgdep, part.n_particles, part.particle_prop, mesh.matnrgdens, mesh.temp, mesh.sigma_a, mesh.fleck, time.dt)

                
                # if time.step % 100 == 0:
                #     plt.figure()
                #     plt.plot(mesh.cellpos, mesh.temp, marker='o')
                #     # plt.yscale('log')
                #     plt.show()
                    

                # Update time
                time.time = round(time.time + time.dt, 9)
                # Make a larger time-step
                if time.dt < dt_max:
                    # Increase time-step
                    time.dt = time.dt * 1.05

                # Check for final time-step
                if time.time + time.dt > t_final:
                    time.dt = t_final - time.time
                # tolerance = 0.06
                # cell_max_index = np.argmax(eddington)
                # print(f'index of max eddington = {cell_max_index}')
                # for i in range(mesh.ncells):
                #     if i < cell_max_index:
                #         part.Nmu[i] = 8
                #     elif i > cell_max_index and abs(eddington[i] - 1/3) <= tolerance:
                #         part.Nmu[i] = 4
                # Plot Eddington
                # if time.step % 1000 == 0:  # Check if the current time step is a multiple
                    
                #     # part.Nmu[0:10] = 8
                #     fig, ax1 = plt.subplots()  # Create the plot
                    
                #     # Plot the temperature on the left y-axis
                #     ax1.plot(mesh.cellpos, mesh.temp, label='Material Temperature', color='b')
                    
                #     ax1.set_xlabel('x')
                #     ax1.set_ylabel('Material Temperature', color='k')
                #     ax1.set_yticks(np.arange(0, 1.01, 0.1))
                #     ax1.set_xticks(np.arange(0, 0.1501, 0.03))
                #     ax1.set_ylim(0, 1.1)
                #     ax1.tick_params(axis='y', labelcolor='k')
                    
                #     # Create a second y-axis for the Eddington factor
                #     ax2 = ax1.twinx()
                #     ax2.plot(mesh.cellpos, eddington, label='Eddington Factor', color='g')
                #     ax2.set_ylabel('Eddington Factor', color='k')
                #     ax2.set_ylim(0, 1)
                #     ax2.tick_params(axis='y', labelcolor='k')
                #     ax2.set_yticks(np.arange(0, 1.01, 0.1))
                    
                #     # Title and legend
                #     plt.title(f'Eddington Factor @ time={time.time}')
                #     ax1.legend(loc='upper left')
                #     ax2.legend(loc='upper right')
                    
                #     # Save the figure with the time in the filename
                #     filename = 'marshak_eddington_time_{:.2f}.png'.format(time.time)
                #     plt.savefig(filename, format='png')
                #     plt.close()
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
                
                # Apply population control on particles if needed
                # if part.n_particles[0] > part.n_max and part.mode == 'nrn':
                #         part.n_particles, part.particle_prop = imc_update.population_control(part.n_particles, part.particle_prop, time.time, part.Nmu)

        print(f'Final time = {time.time}')
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")


def infinite_medium_one_cell(output_file):
    """
    Problem where the radiation and matter start out of equilibrium and approach an analytic equilibrium temperature.
    """
    # We are using constant opacities and specific heats here.
    # If we start the radiation and matter at different temps, we would expect them to meet somewhere in the middle.
    # We will have effective scattering, but no real scattering
    phys.c = 1.0
    phys.invc = 1 / phys.c
    phys.a = 1.0
    

    t_r = 3.0
    t_m = 0.001
    heat_capacity = 50.0

    # Calculate the Equilbrium Temperature
    T_eq = imc_util.get_equilibrium_temperature(t_r, t_m, phys.a, heat_capacity)
    print(f'The equilibrium Temperature is {T_eq}')

    # Set temperatures
    mesh.temp[:] = t_m
    mesh.radtemp[:] = t_r
    # Set heat capacity
    mat.b = np.ones(mesh.ncells) * heat_capacity
    # Calculate beta
    mesh.beta = np.zeros(mesh.ncells)
    mesh.beta[:] = 4 * phys.a * mesh.temp[:] ** 3 / (mat.b[:])
    # Set fleck factor
    mesh.fleck[:] = 1.0 / (1.0 + mesh.beta * mesh.sigma_a[:] * phys.c * time.dt)
    print(f' Fleck factor = {mesh.fleck}')
    # Begin time
    time.time = 0.0

    # initialize particle arrays
    part.particle_prop = np.zeros((part.max_array_size, 8), dtype=np.float64)
    part.n_particles = np.zeros(1, dtype=int)
    if part.mode == 'nrn':
        part.Nmu = np.ones(mesh.ncells, dtype=int) * part.Nmu
    # Total opacity
    mesh.sigma_t = mesh.fleck * mesh.sigma_a + (1.0 - mesh.fleck) * mesh.sigma_a + mesh.sigma_s
    print(f'mesh.sigma_t = {mesh.sigma_t}')

# List to store time, temp, and radtemp at each time-step
    time_data = [time.time]
    temp_data = [mesh.temp[0]]
    radtemp_data = [mesh.radtemp[0]]

    timesteps = time.ns
    all_times = np.zeros(timesteps)
    all_mat_temps = np.zeros(timesteps)
    runtimes = np.zeros(timesteps)

    try:
        with open(output_file, "wb") as f:
            for time.step in range(1, time.ns + 1):
                step_start_time = tm.perf_counter()  # Start timing
                print(f'Step {time.step} @ time = {time.time}')
                
                # Create Census particles for first time-step
                if part.mode == 'nrn' and time.step == 1:
                    part.n_particles, part.particle_prop = imc_source.create_census_particles(part.n_particles, part.particle_prop, mesh.radtemp)
                
                if part.mode == 'rn' and time.step == 1:
                    part.n_particles, part.particle_prop = imc_source.create_census_particles_random(part.n_particles, part.particle_prop, mesh.radtemp)
                
                # Source new particles
                if part.mode == 'nrn':
                    part.n_particles, part.particle_prop = imc_source.create_body_source_particles(part.n_particles, part.particle_prop, mesh.temp, time.time, time.dt, mesh.sigma_a, mesh.fleck)

                if part.mode == 'rn':
                    mesh.source_cells = 1
                    part.n_particles, part.particle_prop = imc_source.run(mesh.fleck, mesh.temp, mesh.sigma_a, part.particle_prop, part.n_particles, time.time, time.dt)
                # Track particles through the mesh
                if part.mode == 'nrn':
                    mesh.nrgdep, part.n_particles, part.particle_prop, eddington = imc_track.run(part.n_particles, part.particle_prop, time.time, time.dt, part.Nmu, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck)
                
                if part.mode == 'rn':
                    mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run_random(part.n_particles, part.particle_prop, time.time, time.dt, mesh.sigma_a, mesh.fleck, mesh.sigma_s)
                # Clean particle array post-tracking
                part.n_particles, part.particle_prop  = imc_track.clean(part.n_particles, part.particle_prop)

                # Tally
                mesh.temp, mesh.radtemp = imc_tally.general_tally(mesh.nrgdep, part.n_particles, part.particle_prop, mesh.temp)

                # Update time
                time.time = round(time.time + time.dt, 5)

                # Store the time, mesh.temp[0], mesh.radtemp[0] for plotting later
                time_data.append(np.float64(time.time))
                temp_data.append(np.float64(mesh.temp[0]))
                radtemp_data.append(np.float64(mesh.radtemp[0]))

                # Save data to NPZ per time step
                step_idx = time.step - 1  # zero-based indexing
                all_times[step_idx] = time.time
                all_mat_temps[step_idx] = mesh.temp
                runtimes[step_idx] = tm.time() - step_start_time
            # After loop ends, save the data to the output file
            pickle.dump(time_data, f)
            pickle.dump(temp_data, f)
            pickle.dump(radtemp_data, f)

    except KeyboardInterrupt:
        print(f'Calculation interrupted. Saving data...')
    finally:
        print(f'Data saved successfully.')
    np.savez("Mosher_output.npz",
         time=all_times,
         mat_temps=all_mat_temps,
         runtimes=runtimes)
    # Plotting after the loop finishes
    plt.figure()
    plt.axhline(T_eq, color='k', linestyle='--', label='Equilibrium Temperature')
    plt.plot(time_data, temp_data, label="Material Temperature", color='b', marker='o', linestyle='none', fillstyle='none')
    plt.plot(time_data, radtemp_data, label="Radiation Temperature", color='r', marker='x', linestyle='none')
    plt.xlabel("Time")
    plt.xticks(range(0, 16, 1))
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig('infinite_medium_mosher.png', dpi=900)
    plt.close()
    

def graziani_slab(output_file):
    """Multigroup problem"""
    # Set physical constants (cgs)
    phys.c = 2.9979e10 # cm/s
    phys.invc = 1.0 / phys.c
    phys.a = 1.3720e14 # erg/(cm^3-keV^4)
    phys.sb = phys.a * phys.c / 4 # erg / (cm2 − s −keV4)
    phys.h = 6.6262e-27 # erg-s
    print(f'Physical constants: phys.c = {phys.c}, phys.a = {phys.a}, phys.h = {phys.h}')
    
    # Set plot time
    plottimes = np.array([1.4e-12])
    plottimenext = 0

    # Material Temperature 
    mesh.temp = np.ones(mesh.ncells) * 0.03 # keV
    mesh.radtemp = np.ones(mesh.ncells) * 0.03 # keV
    mat.rho = np.ones(mesh.ncells) * 0.0916 # g/cm3
    mat.b = np.ones(mesh.ncells) * 1e50 # erg/(g-keV)

    # Set up frequency group structure
    # 50 groups, logarithmically spaced between 3.0 × 10−3 keV and 30.0 keV
    # Define the energy range
    E_min = 3.0e-3  # keV
    E_max = 30.0  # keV
    part.Ng = 50  # Number of frequency groups

    # Generate logarithmically spaced edges
    edges = np.logspace(np.log10(E_min), np.log10(E_max), part.Ng + 1)
    print(f'edges = {edges}')

    # Compute the group center points (geometric mean of adjacent edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    print(f'centers = {centers}')

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


    analytic_radnrgdens = np.array([6.56128e06, 9.37744e06, 1.33706e07, 1.90091e07,
                                    2.69304e07, 3.79893e07, 5.33093e07, 7.43291e07,
                                    1.02827e08, 1.40887e08, 1.90765e08, 2.54559e08,
                                    3.33604e08, 4.27461e08, 5.32466e08, 6.40012e08, 
                                    7.35205e08, 7.97306e08, 8.05117e08, 7.49915e08,
                                    6.06784e08, 4.32675e08, 2.59951e08, 1.26955e08, 
                                    4.83345e07, 1.36588e07, 2.70428e06, 3.50392e05, 
                                    2.75095e04, 2.85002e05, 3.91809e07, 9.85459e08, 
                                    6.33419e09, 1.64600e10, 6.22834e09, 1.00872e09, 
                                    2.28836e09, 1.80809e09, 7.26508e08, 1.64067e08, 
                                    2.01749e07, 1.34876e06, 4.44994e04, 6.33976e02, 
                                    3.49212e00, 6.12484e-03, 2.66266e-06, 2.44078e-10, 
                                    3.13420e-15, 3.80253e-21])
    
    # print(f'sum of analytic radnrgdens = {np.sum(analytic_radnrgdens)}')
    # For each group, calculate the normalized planck integral.
    
    part.Nmu = np.ones(mesh.ncells, dtype=int) * part.Nmu
    part.max_array_size = 25_000_000
    # initialize particle arrays
    part.particle_prop = np.zeros((part.max_array_size, 8), dtype=np.float64)
    part.n_particles = np.zeros(1, dtype=int)

    # begin time
    mesh.fleck = np.ones(mesh.ncells, dtype=np.float64)
    time.time = 0.0
    print()
    try:
        with open(output_file, "wb") as fname:

            # Loop over timesteps
            for time.step in range(1, time.ns + 1):
                print()
                print(f'Step: {time.step} @ {time.time}')
                
                # Source new particles   
                part.n_particles, part.particle_prop = imc_source.create_graziani_body_source_particles(part.n_particles, part.particle_prop, mesh.temp, time.time, time.dt)
                part.n_particles, part.particle_prop = imc_source.create_graziani_left_surface_source_particles(part.n_particles, part.particle_prop, time.time, time.dt)

                # Track particles through the mesh
                mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run_multigroup(part.n_particles, part.particle_prop, time.time, time.dt, sigma_g)
                part.n_particles, part.particle_prop  = imc_track.clean(part.n_particles, part.particle_prop)
                
                # Tally
                mesh.radnrgdens = imc_tally.graziani_slab_tally(part.n_particles, part.particle_prop)

                # Update time
                time.time = round(time.time + time.dt, 14)
                
                # Plot
                if plottimenext <=0 and time.time >= plottimes[plottimenext]:
                    print(f"Plotting {plottimenext}")
                    print(f"at target time {plottimes[plottimenext]:24.16f}")
                    print(f"at actual time {time.time:24.16f}")
                    
                    fname.write(f"Time = {time.time:24.16f}\n".encode())
                    pickle.dump(mesh.radnrgdens, fname, 0)
                    # # Extract radiation energy density for the 10th cell (index 9)
                    # radnrgdens_cell10 = mesh.radnrgdens[9, :]

                    # # Plot the radiation energy density for each frequency group in the 10th cell
                    # plt.figure()
                    # plt.plot(centers, radnrgdens_cell10, marker='o', label='DPT')
                    # plt.plot(centers, analytic_radnrgdens, marker='x', label='Analytic')
                    # plt.legend()
                    # plt.xscale('log')
                    # plt.yscale('log')
                    # # plt.xlim(1e-3, 10)
                    # plt.ylim(1e4,1e11)
                    # plt.xlabel("Frequency (keV)")
                    # plt.ylabel("Radiation Energy Density (erg/cm³-keV)")
                    # plt.title("Slab spectrum")
                    # plt.savefig('graziani_slab.png', dpi=600)
                    # plt.close()
        

        print(f'Final time = {time.time}')
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
    finally:
        print("Data saved successfully.")


def crooked_pipe(output_file):
    """Crooked Pipe 2D R-Z Problem"""
    # Physical Properties of Crooked Pipe Problem
    # density (g/cc) = 0.01 [thin] and 10.0 [thick]
    # opacity (cm^2/g) = 20.0 [thin] and 20.0 [thick]
    # initial temperature (keV) = 0.05 [thin] and 0.05 [thick]
    # ion specific heat (1.e15 ergs/g-keV)


    # Set Physical Constants
    phys.c = 2.99792458e2  # [cm/sh] 
    # phys.a = 1.3720e14  # erg/(cm^3-keV^4)
    phys.a = 0.01372 # jk / (cm^3-keV^4)
    phys.sb = phys.a * phys.c / 4 # energy / (cm2 − sh −keV4)
    phys.invc = 1.0 / phys.c
    print(f'Physical constants: phys.c = {phys.c} [cm/sh], phys.a = {phys.a} [jk/(cm^3-keV^4)], phys.sb = {phys.sb} [jk/(cm2 − sh −keV4)]')
    
    print(f'mesh.dx = {mesh.dx}')
    print(f'mesh.dy = {mesh.dy}')
    # Initialize Opacities, temperatures, fleck factor
    num_x_idx = len(mesh.x_edges) - 1
    num_y_idx = len(mesh.y_edges) - 1
    # generate a mask to assign opacities to the thin and thick parts of the mesh

    mesh.temp = np.full((num_x_idx, num_y_idx), 0.05) # 0.05 keV initial temperature everywhere
    mesh.fleck = np.full((num_x_idx, num_y_idx), 0.0)
    mesh.radtemp = np.full((num_x_idx, num_y_idx), 0.0)

    thin_density = 0.01 # [g/cm^3]
    thick_density = 10.0 # [g/cm^3]

    thin_mass_opacity = 20.0 #[cm^2/g]
    thick_mass_opacity = 20.0 # [cm^2/g]

    thin_opacity = thin_mass_opacity * thin_density # [1/cm]
    print(f'thin_opacity = {thin_opacity} [1/cm]')
    thick_opacity = thick_mass_opacity * thick_density # [1/cm]
    print(f'thick_opacity = {thick_opacity} [1/cm]')
    mat.rho = np.full((num_x_idx, num_y_idx), thick_density)  # Start with thick
    mesh.sigma_a = np.full((num_x_idx, num_y_idx), thick_opacity)
    part.N_omega = 10
    # Convert sourcing info into arrays
    if part.mode == 'nrn':
        part.Nmu = np.full((num_x_idx, num_y_idx), part.Nmu)
        part.N_omega = np.full((num_x_idx, num_y_idx), part.N_omega)
        part.Nx = np.full((num_x_idx, num_y_idx), part.Nx)
        part.Ny = np.full((num_x_idx, num_y_idx), part.Ny)
        part.Nt = np.full((num_x_idx, num_y_idx), part.Nt)

    # Assign thin densities to the thin regions
    # Loop over each cell (i = x index, j = y index)
    for i in range(num_x_idx):
        for j in range(num_y_idx):
            x_left   = mesh.x_edges[i]
            x_right  = mesh.x_edges[i+1]
            y_bot    = mesh.y_edges[j]
            y_top    = mesh.y_edges[j+1]

            # Region 1: Bottom left
            if x_right <= 2.5 and y_top <= 0.5:
                mat.rho[i, j] = thin_density
                mesh.sigma_a[i,j] = thin_opacity
                    

            # Region 2: Vertical left gap
            if 2.5 <= x_left <= 3.0 and y_top <= 1.5:
                mat.rho[i, j] = thin_density
                mesh.sigma_a[i,j] = thin_opacity
                
            # Region 3: Horizontal middle strip
            if 3.0 <= x_left <= 4.0 and (y_bot < 1.5 and y_top >= 1.00):
                mat.rho[i, j] = thin_density
                mesh.sigma_a[i,j] = thin_opacity
                    

            # Region 4: Vertical right gap
            if 4.0 <= x_left <= 4.5 and y_top <= 1.5:
                mat.rho[i, j] = thin_density
                mesh.sigma_a[i,j] = thin_opacity
                

            # Region 5: Bottom right
            if x_left >= 4.5 and y_top <= 0.5:
                mat.rho[i, j] = thin_density
                mesh.sigma_a[i,j] = thin_opacity
                
    mesh.thin_cells = (mesh.sigma_a == thin_opacity)
    # mesh.thin_cells = np.argwhere(mesh.sigma_a == thin_opacity)
    mesh.thick_cells = (mesh.sigma_a == thick_opacity)
    print(f'mesh.thick_cells = {mesh.thick_cells}')
    
    mesh.sigma_s = np.full((num_x_idx, num_y_idx), 0.0)
    mesh.sigma_t = np.copy(mesh.sigma_a)
    print(f'mesh.sigma_a = {mesh.sigma_a}')
    print(f'mesh.sigma_s = {mesh.sigma_s}')
    print(f'mesh.sigma_t = {mesh.sigma_t}')
    print(f'shape of mesh.sigma_t = {mesh.sigma_t.shape}')
    mat.b = np.full((num_x_idx, num_y_idx), 1.e15)         # ion specific heat [ergs/gm-keV]
    mat.b = mat.b * mat.rho  # converted to [ergs/cm^3-keV]
    mat.b[mesh.thick_cells] = 1.0 # jk/cm^3-keV
    mat.b[mesh.thin_cells] = 1e-3 # jk/cm^3-keV

    print(f'mat.b = {mat.b}')
    # Columns: [emission_time, x_idx, y_idx, xpos, ypos, mu, omega, frq, nrg, startnrg]
    part.max_array_size = 600_000_000
    part.particle_prop = np.zeros((part.max_array_size, 10), dtype=np.float64)
    part.n_particles = 0


    # Set start time and time-step
    time.time = 0.0
    time.dt = 0.001    # shakes
    time.dt_max = 0.05  # shakes
    t_final = 10.0
    time.dt_rampfactor = 1.1
    part.surface_Nmu = 4
    part.surface_N_omega = 12
    part.surface_Ny = 50
    part.surface_Nt = 50
    bcon.T0 = 0.3 # keV
    part.n_input = 500_000
    parallel = True

    n_all = numba.get_num_threads()
    numba.set_num_threads(max(1, n_all // 2))
    print("Numba will use", numba.get_num_threads(), "threads.")

    # Loop over timesteps
    records = []
    try:
        with open(output_file, "wb") as fname:
            while time.time < t_final: # time.ns + 1
                # step_start_time = tm.perf_counter()  # Start timing
                print(f'Step: {time.step} @ time = {time.time}')
                print(f"time.dt = {time.dt}")

                # Update temperature dependent quantities
                mesh.fleck = imc_update.crooked_pipe_update(mesh.sigma_a, mesh.temp, time.dt)
                # plt.figure()
                # pc = plt.pcolormesh(mesh.x_edges,
                #                     mesh.y_edges,
                #                     mesh.fleck.T,  # Transpose to match orientation
                #                     cmap='turbo',
                #                     edgecolors='k',       # 'k' for black borders around cells
                #                     linewidth=0.5,
                #                     shading='flat')        
                # plt.colorbar(pc, label=f'Fleck factor')
                # plt.clim(vmin=0.00, vmax=1.0)
                # plt.xlabel('x')
                # plt.ylabel('y')
                # plt.title(f'Fleck factor at t={time.time}')
                # plt.axis('equal')
                # plt.grid(True, linestyle='--', linewidth=0.5, color='white')
                # plt.show()
                
                print(f'mesh.fleck = {mesh.fleck}')
                if part.mode == 'nrn':
                    # Source new surface particles
                    # print(f'surface_Ny = {part.surface_Ny}, surface_Nmu = {part.surface_Nmu}')
                    part.n_particles, part.particle_prop = imc_source.crooked_pipe_surface_particles(part.n_particles, part.particle_prop, part.surface_Ny, part.surface_Nmu, part.surface_N_omega, part.surface_Nt, time.time, time.dt, mesh.y_edges)
                    # Source new body source particles
                    part.n_particles, part.particle_prop = imc_source.crooked_pipe_body_particles(part.n_particles, part.particle_prop, time.time, time.dt, mesh.y_edges, mesh.x_edges, mesh.temp, mesh.dx, mesh.dy, mesh.fleck, mesh.sigma_a)
                elif part.mode == 'rn':
                    part.n_particles, part.particle_prop = imc_source.run2D(mesh.fleck, mesh.temp, mesh.sigma_a, part.particle_prop, part.n_particles, time.time, time.dt, mesh.dx, mesh.dy, mesh.x_edges, mesh.y_edges)
                
                # Check for energies
                negative_energy_indices = np.where(part.particle_prop[:part.n_particles, 8] < 0.0)[0]
                if negative_energy_indices.size > 0:
                    raise RuntimeError(
                        f"Found {negative_energy_indices.size} particles with negative energy after sourcing! "
                        f"Indices: {negative_energy_indices}"
                    )

                # Advance particles through transport
                if part.mode == 'nrn': 
                    if parallel:
                        # Step 1: Track initial particles
                        mesh.nrgdep, nrgscattered, x_Es, y_Es, tEs = imc_track.run_crooked_pipe_firstloop(
                            part.n_particles, part.particle_prop,
                            time.time, time.dt,
                            mesh.sigma_a, mesh.sigma_s, mesh.sigma_t,
                            mesh.fleck, mesh.x_edges, mesh.y_edges
                        )
                        original_nrg_scattered = np.copy(nrgscattered)
                        # print(f'original sum of nrg_scattered = {np.sum(original_nrg_scattered)}')
                        # Step 2: Implicit scattering loop
                        epsilon = 1e-2
                        iterations = 0
                        converged = False

                        while not converged:
                            # Generate scattered particles
                            scattered_particles, n_scattered_particles = imc_track.generate_scattered_particles(
                                nrgscattered, x_Es, y_Es, tEs,
                                mesh.x_edges, mesh.y_edges, mesh.dx, mesh.dy,
                                part.max_array_size, part.Nx, part.Ny, part.Nt, part.Nmu, part.N_omega,
                                time.time, time.dt
                            )

                            # Track scattered particles
                            nrgdep_scat, nrgscattered, x_Es, y_Es, tEs = imc_track.run_crooked_pipe_firstloop(
                                n_scattered_particles, scattered_particles,
                                time.time, time.dt,
                                mesh.sigma_a, mesh.sigma_s, mesh.sigma_t,
                                mesh.fleck, mesh.x_edges, mesh.y_edges
                            )
                            # print(f'Sum of nrgscatted iteration {iterations} = {np.sum(nrgscattered)}')
                            # Add scattered particle deposition to the mesh
                            mesh.nrgdep += nrgdep_scat

                            # Copy scattered particles into global array
                            n_existing_particles = part.n_particles
                            n_total_particles = n_existing_particles + n_scattered_particles
                            if n_total_particles > part.max_array_size:
                                print(f'iteration {iterations}')
                                raise ValueError("Not enough space in global array for scattered particles in iteration ")

                            part.particle_prop[n_existing_particles:n_total_particles, :] = scattered_particles[:n_scattered_particles, :]
                            part.n_particles = n_total_particles

                            # Check convergence
                            rel_remaining = np.sum(nrgscattered) / np.sum(original_nrg_scattered)
                            if rel_remaining < epsilon:
                                converged = True
                        
                            iterations += 1
                        # After congerging, dump remaining scattered energy
                        print(f'Number of scattering iterations = {iterations}')    
                    else:
                        mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run_crooked_pipe(part.n_particles, part.particle_prop, time.time, time.dt, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck, mesh.thin_cells, part.Nmu)
                    
                    # Test RN version
                    # mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run2D(part.n_particles, part.particle_prop, time.time, time.dt, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck)
                elif part.mode == 'rn':
                    mesh.nrgdep, part.n_particles, part.particle_prop = imc_track.run2D(part.n_particles, part.particle_prop, time.time, time.dt, mesh.sigma_a, mesh.sigma_s, mesh.sigma_t, mesh.fleck, mesh.x_edges, mesh.y_edges)
                # print(f'mesh.nrgdep = {mesh.nrgdep}')

                # Clean up particles that had their energy set to -1.0
                part.n_particles, part.particle_prop = imc_track.clean2D(part.n_particles, part.particle_prop)
                
                # Check for energies
                negative_energy_indices = np.where(part.particle_prop[:part.n_particles, 8] < 0.0)[0]
                if negative_energy_indices.size > 0:
                    raise RuntimeError(
                        f"Found {negative_energy_indices.size} particles with negative energy after cleaning! "
                        f"Indices: {negative_energy_indices}"
                    )
                
                # Tally
                mesh.temp, mesh.radtemp = imc_tally.crooked_pipe_tally(mesh.nrgdep, mesh.dx, mesh.dy, part.n_particles, part.particle_prop, mesh.temp, mesh.sigma_a, mesh.fleck, time.dt)
                print(f'mesh.temp = {mesh.temp}')
                print(f'mesh.radtemp = {mesh.radtemp}')
                if np.any(mesh.temp < 0.0):
                    raise RuntimeError(f"Negative material temp detected! min(mesh.temp) = {np.min(mesh.temp)}")
                if np.any(mesh.radtemp < 0.0):
                    raise RuntimeError(f"Negative rad temp! min(mesh.radtemp) = {np.min(mesh.radtemp)}")
                
                # Update time
                time.time = round(time.time + time.dt, 8)
                time.step += 1
                # Make a larger time-step
                if time.dt < time.dt_max:
                    # Increase time-step
                    time.dt = time.dt * time.dt_rampfactor

                # Check for final time-step
                if time.time + time.dt > t_final:
                    time.dt = t_final - time.time
                nx, ny = mesh.temp.shape
                for j in range(ny):
                    for i in range(nx):
                        records.append({
                            "time": time.time,
                            "x_idx": i,
                            "y_idx": j,
                            "temp": mesh.temp[i, j],
                            "radtemp": mesh.radtemp[i, j]
                        })
                # plt.figure()
                # pc = plt.pcolormesh(mesh.x_edges,
                #                     mesh.y_edges,
                #                     mesh.temp.T,  # Transpose to match orientation
                #                     cmap='inferno',
                #                     edgecolors='k',       # 'k' for black borders around cells
                #                     linewidth=0.5,
                #                     shading='flat')        
                # plt.colorbar(pc, label=f'Temperature [keV]')
                # plt.clim(vmin=0.05, vmax=0.3)
                # plt.xlabel('x')
                # plt.ylabel('y')
                # plt.title(f'Temperature at t={time.time}')
                # plt.axis('equal')
                # plt.grid(True, linestyle='--', linewidth=0.5, color='white')
                # plt.show()
        df = pd.DataFrame(records)
        df.to_csv("temperature_history.csv", index=False)
        print("Temperature history saved to temperature_history.csv")
        plt.figure()
        pc = plt.pcolormesh(mesh.x_edges,
                            mesh.y_edges,
                            mesh.temp.T,   # Transpose to match orientation
                            cmap="inferno",
                            shading="flat")
        plt.colorbar(pc, label="Temperature [keV]")
        plt.clim(vmin=0.05, vmax=bcon.T0)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(mesh.x_edges[0], mesh.x_edges[-1])
        plt.ylim(0,2)
        plt.axis('scaled')
        plt.title(f"Temperature at t={t_final}")
        plt.tight_layout
        plt.show()
        plt.figure()
        pc = plt.pcolormesh(mesh.x_edges,
                            mesh.y_edges,
                            mesh.radtemp.T,  # Transpose to match orientation
                            cmap='inferno',
                            edgecolors='k',       # 'k' for black borders around cells
                            linewidth=0.5,
                            shading='flat')        
        plt.colorbar(pc, label=f'Temperature [keV]')
        plt.clim(vmin=0.05, vmax=0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Radiation Temperature at t={time.time}')
        plt.axis('equal')
        plt.grid(True, linestyle='--', linewidth=0.5, color='white')
        plt.show()          
                
    
    except KeyboardInterrupt:
        print("Calculation interrupted. Saving data...")
        plt.figure()
        pc = plt.pcolormesh(mesh.x_edges,
                            mesh.y_edges,
                            mesh.temp.T,  # Transpose to match orientation
                            cmap='inferno',
                            edgecolors='k',       # 'k' for black borders around cells
                            linewidth=0.5,
                            shading='flat')        
        plt.colorbar(pc, label=f'Temperature [keV]')
        plt.clim(vmin=0.05, vmax=0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Temperature at t={time.time}')
        plt.axis('equal')
        plt.grid(True, linestyle='--', linewidth=0.5, color='white')
        plt.show()
        plt.figure()
        pc = plt.pcolormesh(mesh.x_edges,
                            mesh.y_edges,
                            mesh.radtemp.T,  # Transpose to match orientation
                            cmap='inferno',
                            edgecolors='k',       # 'k' for black borders around cells
                            linewidth=0.5,
                            shading='flat')        
        plt.colorbar(pc, label=f'Temperature [keV]')
        plt.clim(vmin=0.05, vmax=0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Radiation Temperature at t={time.time}')
        plt.axis('equal')
        plt.grid(True, linestyle='--', linewidth=0.5, color='white')
        plt.show()
    finally:
        print("Data saved successfully.")
        
    