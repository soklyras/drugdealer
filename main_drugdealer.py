"""
Runs DrugDealer pipeline.
"""
import sys

from drugdealer.pipeline import run_drugdealer
from drugdealer.utils.parser import parse_args
from drugdealer.utils.read_utils import read_params


def main() -> None:
    # Get arguments from command line
    args = parse_args(sys.argv[1:])
    run_mode = args.run_mode
    config_file_name = args.config_file_name

    # Read parameters from config file
    params: dict = read_params(config_file_name)

    # Run drugdealer
    run_drugdealer(run_mode, params)


if __name__ == '__main__':
    main()
