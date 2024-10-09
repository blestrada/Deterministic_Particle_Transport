"""Top-level main program for imc."""

import argparse
import logging
import time as pytime

import imc_global_part_data as part

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
    parser.add_argument(
        "-o", "--output", default="imc.out", help="Name of output file"
    )
    parser.add_argument("-d", "--debug", default=False, help="Debug mode")

    return parser.parse_args()


def main(input_file, output_file, debug_mode):
    """
    @brief   Top-level function for imc.

    @details Can be called within Python after importing, so has simple/flat signature.

    @param   input_file
    @param   output_file
    @param   debug_mode
    """
    tm0 = pytime.perf_counter()

    logger = setup_logger()

    imc_user_input.read(input_file)
    imc_user_input.echo()

    imc_mesh.make()
    imc_mesh.echo()

     
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