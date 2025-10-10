"""Top-level main program for imc."""

import argparse
import logging
import time as pytime
import os

import imc_global_part_data as part
import imc_global_mesh_data as mesh
import imc_mesh
import imc_opcon
import imc_user_input


def setup_logger():
    """Set up logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)24s %(levelname)8s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_args():
    """Parse command-line arguments and options."""
    parser = argparse.ArgumentParser(
        description="Python implementation of Implicit Monte Carlo."
    )

    parser.add_argument("-i", "--input", default="imc.in", help="Name of input file")
    parser.add_argument("-o", "--output", help="Name of output file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # If output not specified, append .out to input name
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}.out"

    return args


def main(input_file, output_file, debug_mode):
    """
    @brief   Top-level function for imc.

    @details Can be called within Python after importing, so has simple/flat signature.

    @param   input_file
    @param   output_file
    @param   debug_mode
    """
    tm0 = pytime.perf_counter()

    imc_user_input.read(input_file)
    if mesh.num_spatial_dim == 1:
        imc_mesh.make1D()
        imc_mesh.echo()
    elif mesh.num_spatial_dim == 2:
        mesh.x_edges, mesh.y_edges, mesh.dx, mesh.dy, mesh.x_cellcenters, mesh.y_cellcenters = imc_mesh.make_2D(mesh.x_edges, mesh.y_edges)

     
    # Dynamically call the function based on the string stored in part.problem_type
    if hasattr(imc_opcon, part.problem_type):
        # Get the function from the imc_opcon module using the string name and call it
        problem_function = getattr(imc_opcon, part.problem_type)
        problem_function(output_file)
    else:
        raise AttributeError(f"The problem type {part.problem_type} is not defined in imc_opcon.")

    tm1 = pytime.perf_counter()
    print("Time taken for calculation = {:10.2f} s".format(tm1 - tm0))


if __name__ == "__main__":

    # Command-line options
    args = parse_args()

    # Call the main function
    main(args.input, args.output, args.debug)