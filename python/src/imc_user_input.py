"""Read user input deck"""

import numpy as np

import imc_global_mesh_data as mesh
import imc_global_mat_data as mat
import imc_global_part_data as part
import imc_global_phys_data as phys
import imc_global_time_data as time
import imc_global_volsource_data as vol

def read(input_file):
    """
    @brief Reads input deck.
    
    @details Reads input deck with user-specified problem information.
    
    @param input_file Name of input file
    @return None
    """
    with open(input_file, "r") as input_file:
        for line in input_file:

            #  Ignore blank lines
            if line == "":
                continue

            fields = line.split(None, 1)

            if len(fields) != 2:
                continue

            keyw = fields[0].lower()
            keyv = fields[1].strip()

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

            elif keyw == "nx":
                part.Nx = int(keyv)

            elif keyw == "nmu":
                part.Nmu = int(keyv)

            elif keyw == "nt":
                part.Nt = int(keyv)

            elif keyw == "left_bc":
                mesh.left_bc = str(keyv)
                
            elif keyw == "right_bc":
                mesh.right_bc = str(keyv)

            elif keyw == "temp0":
                mesh.temp0 = float(keyv)

            elif keyw == "radtemp0":
                mesh.radtemp0 = float(keyv)

            elif keyw == "n_max":
                part.n_max = int(keyv)
                
            elif keyw == 'mode':
                part.mode = str(keyv)

            elif keyw == 'scattering':
                part.scattering = str(keyv)

            elif keyw == 'problem_type':
                part.problem_type = str(keyv)

            elif keyw == 'x_0':
                vol.x_0 = float(keyv)
                
            elif keyw == 'epsilon':
                mat.epsilon = float(keyv)

            elif keyw == 'tau_0':
                vol.tau_0 = float(keyv)

            elif keyw == 'ns':
                part.n_input = int(keyv)

            else:
                continue


def echo():
    """Echoes user input."""
    
    print("\n" + "=" * 79)
    print("User input")
    print("=" * 79)

    print(f'Problem Type: {part.problem_type}')

    print()

    print("mesh.ncells  {:5d}".format(mesh.ncells))
    print("mesh.xsize   {:5.1f}".format(mesh.xsize))
    print(f'mesh.temp0    {mesh.temp0}')

    print("mat.sigma_a  {:5.3f}".format(mat.sigma_a))
    print("mat.sigma_s  {:5.3f}".format(mat.sigma_s))

    print("time.dt      {}".format(time.dt))
    print("time.ns      {:5d}".format(time.ns))

    print("part.n_max   {:5d}".format(part.n_max))

    print("part.Nx      {:5d}".format(part.Nx))
    print("part.Nmu     {:5d}".format(part.Nmu))
    print("part.Nt      {:5d}".format(part.Nt))
   