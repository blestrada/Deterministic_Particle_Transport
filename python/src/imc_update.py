"""Update at start of time-step"""

import imc_global_phys_data as phys
import imc_global_mat_data as mat
import imc_global_mesh_data as mesh
import imc_global_time_data as time
import imc_global_bcon_data as bcon

def run():
    """Update temperature-dependent quantities at start of time-step"""
    
    print("\n" + "-" * 79)
    print("Update step ({:4d})".format(time.step))
    print("-" * 79)

    # Calculate new heat capacity
    mat.b = 4 * phys.a * mesh.temp ** 3