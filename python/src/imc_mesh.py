"""Sets up mesh."""

import numpy as np
import matplotlib.pyplot as plt

import imc_global_mesh_data as mesh
import imc_global_mat_data as mat


def make1D():
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
    
    mesh.nrgdep = np.zeros(mesh.ncells, dtype=np.float64)  # Total energy deposited in timestep
    mesh.nrgscattered = np.zeros(mesh.ncells, dtype=np.float64)

    mesh.fleck = np.zeros(mesh.ncells) - 1.0 # Fleck factor


def make_2D(x_edges, y_edges, plot_mesh=False):
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

    x_edges: ndarray
    y_edges: ndarray
    dx     : ndarray
    dy     : ndarray
    x_centers: ndarray
    y_centers: ndarray
    """
    x_edges = np.asarray(x_edges)
    y_edges = np.asarray(y_edges)

    dx = np.diff(x_edges)
    dy = np.diff(y_edges)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    print(f'number of (z) zones = {len(x_centers)}')
    print(f'number of (r) zones = {len(y_centers)}')
    if plot_mesh == True:
        fig, ax = plt.subplots()

        # Grid lines
        for x in x_edges:
            ax.plot([x, x], [y_edges[0], y_edges[-1]], color='black', linewidth=1)
        for y in y_edges:
            ax.plot([x_edges[0], x_edges[-1]], [y, y], color='black', linewidth=1)

        ax.set_aspect('equal')
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title("2D Mesh")
        plt.grid(False)
        plt.show()
    
    return x_edges, y_edges, dx, dy, x_centers, y_centers



def echo():
    """
    @brief   Prints mesh.

    @details Prints out spatial mesh for debugging.
    @return  None
    """
    print("Mesh:")
    print(f'mesh.ncells = {mesh.ncells}, mesh.xsize = {mesh.xsize}, mesh.dx = {mesh.dx}')
    print(f'mesh.cellpos = {mesh.cellpos[:11]}')
    print(f'mesh.nodepos = {mesh.nodepos[:12]}')