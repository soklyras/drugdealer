"""
Reads the step from the command line and retrieves the name of the
configuration file. Additionally, it contains the help information for
the user with the summary of the different steps
"""
import argparse
from typing import Dict, List


def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Parse arguments from command line:

    Args:
        argv (List[str]): Arguments passed in the command line to the
            drugdealer call.

    Returns:
        argparse.Namespace: Arguments values.
    """
    drugdealer_pipeline: Dict[str, str] = {
        'prog': 'drugdealer',
        'description': """DrugDealer preprocess data, trains a neural network
                       and evaluates the predictions on the test set.""",
        "epilog": """DrugDealer has 4 different modes: dataprocess, train,
                     evaluate and all. For more information about each model run:
                     drugdealer MODE --help."""
    }

    # Modes
    mode: Dict[str, str] = {
        'help': 'Mode in which to run DrugDealer.'
    }

    mode_dataprocess: Dict[str, str] = {
        'help': """Process the data.""",
        'description': """
            Clean, filter and store the data that will be used for training.
            If you run this step once, you don't have to run it again. and 
            you can continue with the rest of the modes."""
    }

    mode_train: Dict[str, str] = {
        'help': 'Train a neural network to predict .',
        'description': """
            Trains a neural network model with the processed data.
            The data is bimodal; tabular (for gene expressions)and
            image data for the molecular representations of the drugs."""
    }

    mode_evaluate: Dict[str, str] = {
        'help': 'Evaluate model on test set.',
        'description': """
            Evaluates the model based on different metrics and
            stores different plots.."""
    }


    mode_all: Dict[str, str] = {
        'help': 'Runs the DrugDealer pipeline.',
        'description': """
            Runs the DrugDealer pipeline end-to-end:
            It runs the following steps consecutively:
            1. dataprocess
            2. train
            3. evaluate"""

    }

    # Positional arguments
    config_file: Dict[str, str] = {
        'help': """Configuration file with arguments to run DrugDealer."""
    }

    parser: argparse.ArgumentParser
    subparser_mode: argparse.ArgumentParser
    parser_dataprocess: argparse.ArgumentParser
    parser_train: argparse.ArgumentParser
    parser_evaluate: argparse.ArgumentParser

    parser = argparse.ArgumentParser(**drugdealer_pipeline)
    parser.add_argument('--version', '-v', action='version', version='1.0.0')

    subparser_mode = parser.add_subparsers(dest='run_mode', **mode,
                                           required=True)

    parser_dataprocess = subparser_mode.add_parser('dataprocess', **mode_dataprocess)
    parser_dataprocess.add_argument("config_file_name", **config_file, type=str)
    parser_dataprocess.formatter_class = argparse.RawDescriptionHelpFormatter

    parser_train = subparser_mode.add_parser('train', **mode_train)
    parser_train.add_argument("config_file_name", **config_file, type=str)
    parser_train.formatter_class = argparse.RawDescriptionHelpFormatter

    parser_evaluate = subparser_mode.add_parser('evaluate', **mode_evaluate)
    parser_evaluate.add_argument("config_file_name", **config_file, type=str)
    parser_evaluate.formatter_class = argparse.RawDescriptionHelpFormatter

    parser_all = subparser_mode.add_parser('all', **mode_all)
    parser_all.add_argument("config_file_name", **config_file, type=str)
    parser_all.formatter_class = argparse.RawDescriptionHelpFormatter

    args: argparse.Namespace = parser.parse_args(argv)

    return args
