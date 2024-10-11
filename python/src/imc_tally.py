"""Tally end-of-timestep quantities"""

import numpy as np

import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_part_data as part
import imc_global_volsource_data as vol



def run():
    """Tally end of timestep quantities """
    source_cells = int((np.ceil(vol.x_0 / mesh.dx)))
    # Radiation energy density
    radenergydens = np.zeros(mesh.ncells)
    radenergydens[:] = phys.a * mesh.temp[:] ** 4 # keV/cm^3

    # input_energy = sum(radenergydens)
    # print("The input energy before is: ", input_energy, "\n")

    sourceenergy = np.zeros(mesh.ncells)
    sourceenergy[0:source_cells] = 1.0
    
    # Temperature increase
    nrg_inc = np.zeros(mesh.ncells)
    nrg_inc[:] = (mesh.nrgdep[:] / mesh.dx) - (mesh.sigma_a[:] * mesh.fleck[:] * radenergydens[:] * phys.c * time.dt) 
  
    mesh.matnrgdens[:] = mesh.matnrgdens[:] + nrg_inc[:]

    for i in range(mesh.ncells):
        if mesh.matnrgdens[i] < 0:
            mesh.matnrgdens[i] = 0
        
    #mesh_temp[:] = mesh_temp[:] .+  nrg_inc[:] ./ (bee[:])

    mesh.temp[:] = mesh.matnrgdens[:] ** (1/4)

    #mesh_temp[:] = mesh_temp[:] .+ (material_energy[:])


    #print("Mesh temperature: \n")
    #print(mesh_temp, "\n")


    # Save radiation energy
    mesh.radnrgdens = np.zeros(mesh.ncells)
    for particle in part.particle_prop:
        cell_index = particle[2]
        mesh.radnrgdens[cell_index] += particle[5] / mesh.dx

    print(f'Material Energy Density = {mesh.matnrgdens[:10]}')
    print(f'Radiation Energy Density = {mesh.radnrgdens[:10]}')