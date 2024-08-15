"""Read user input deck"""

import numpy as np

import imc_global_mesh_data as mesh
import imc_global_mat_data as mat
import imc_global_part_data as part
import imc_global_phys_data as phys
import imc_global_time_data as time

def read(input_file):
    """
    @brief Reads input deck.
    
    @details Reads input deck with user-specified problem information.
    
    @param input_file Name of input file
    @return None
    """
    with open(input_file, "r'") as input_file:
        for line in input_file:

            #  Ignore blank lines
            if line == "":
                continue

            fields = line.split(None, 1)

            if len(fields) != 2:
                continue

            keyw = fields[0].lower()
            keyv = fields[1]

            if keyw == "dt":
                time.dt = float(keyv)
            
            elif keyw == "xsize":
                mesh.xsize = float(keyv)

            elif keyw == "dx":
                # Round up the number of cells if not an integer, then if
                # necessary adjust dx for the rounded up number of cells.
                mesh.ncells = int(np.ceil(mesh.xsize / float(keyv)))

            elif keyw == "cycles":
                time.ns = int(keyv)

            elif keyw == "sigma_a":
                mat.sigma_a = float(keyv)

            elif keyw == "sigma_s":
                mat.sigma_s = float(keyv)

            elif keyw == "Nx":
                part.Nx = int(keyv)

            elif keyw == "Nmu":
                part.Nmu = int(keyv)

            elif keyw == "Nt":
                part.Nt = int(keyv)

            elif keyw == "Nxi":
                part.Nxi = int(keyv)

            elif keyw == "b":
                mat.b = float(keyv)

            elif keyw == "temp":
                mesh.temp = float(keyv)

            elif keyw == "radtemp":
                mesh.radtemp = float(keyv)

            elif keyw == "left_bc":
                mesh.left_bc = str(keyv)
                
            elif keyw == "right_bc":
                mesh.right_bc = str(keyv)
                
            else:
                continue


def echo():
    """Echoes user input."""
    print("\n" + "=" * 79)
    print("User input")
    print("=" * 79)

    print()
    print("mesh.ncells  {:5d}".format(mesh.ncells))
    print("mesh.xsize   {:5.1f}".format(mesh.xsize))

    print("mat.sigma_a      {:5.1f}".format(mat.sigma_a))
    print("mat.sigma_s      {:5d}".format(mat.sigma_s))

    print("time.dt      {:5.1f}".format(time.dt))
    print("time.ns      {:5d}".format(time.ns))

    print("part.n_input {:5d}".format(part.n_input))
    print("part.n_max   {:5d}".format(part.n_max))