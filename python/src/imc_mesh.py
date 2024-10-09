"""Sets up mesh."""

import numpy as np

import imc_global_mesh_data as mesh
import imc_global_mat_data as mat


def make():
    """
    @brief   Sets up mesh.

    @details Sets up fixed spatial mesh.
    @return  None

    Mesh creation
    =============

    The overall problem size and number of mesh cells are specified as user
    input, and the cell size ($dx$) is calculated from these.

    Arrays of both the cell-centre and the cell-edge (node) positions are
    created.

    Cell-centred arrays for temperature, initial temperature, opacity, and total energy deposited, are initialised.
    """
    # Create cell data as a 2D array (with first dimension = 1)
    # to facilitate use of matplotlib.pyplot.pcolor
    # mesh.cells = np.zeros((1, mesh.ncells + 1))
    # mesh.cells[0, 0:mesh.ncells + 1] = np.linspace(0., mesh.xsize, mesh.ncells + 1)

    mesh.dx = mesh.xsize / float(mesh.ncells)

    mesh.cellpos = np.arange(0.5 * mesh.dx, mesh.xsize, mesh.dx)
    mesh.nodepos = np.linspace(0.0, mesh.xsize, mesh.ncells + 1)

    # Create arrays for the mesh-based physical quantities

    mesh.temp = np.ones(mesh.ncells) * mesh.temp0  # Temperature (keV)
    mesh.radtemp = np.ones(mesh.ncells) * mesh.radtemp0

    mesh.sigma_a = np.ones(mesh.ncells) * mat.sigma_a  # Opacities
    mesh.sigma_s = np.ones(mesh.ncells) * mat.sigma_s
    mesh.sigma_t = mesh.sigma_a + mesh.sigma_s
    
    mesh.nrgdep = np.zeros(mesh.ncells)  # Total energy deposited in timestep
    mesh.nrgscattered = np.zeros(mesh.ncells)

    mesh.fleck = np.zeros(mesh.ncells) -1.0 # Fleck factor


def echo():
    """
    @brief   Prints mesh.

    @details Prints out spatial mesh for debugging.
    @return  None
    """
    print("Mesh:")
    print(f'mesh.ncells = {mesh.ncells}, mesh.xsize = {mesh.xsize}, mesh.dx = {mesh.dx}')
    print(f'mesh.cellpos = {mesh.cellpos}')
    print(f'mesh.nodepos = {mesh.nodepos}')