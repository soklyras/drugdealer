"""
Runs DrugDealer pipeline.
"""
import gc
from typing import TYPE_CHECKING, Dict, Union

import pyfiglet as pyg

from drugdealer.dataprocess.dataprocess import run_dataprocess
from drugdealer.train.train import run_train
from drugdealer.evaluate.evaluate import run_evaluate
from drugdealer.utils.loggers import create_logger

if TYPE_CHECKING:
    import logging


def run_drugdealer(run_mode: str, params: Dict[str, Union[int, str]]) -> None:
    """
    Runs DrugDealer pipeline for the mode ``run_mode`` using the parameters
    from ``params``. This function is called from the command line interface
    when running DrugDeaker with the mode and the ``yaml`` file.

    Args:
        run_mode (str): Mode to run DrugDealer between *dataprocess*, *train*,
            *evaluate*.
        params (Dict): Settings and parameters for DrugDealer.

    """
    logger: logging.Logger
    logger = create_logger()
    gc.enable()

    # Print drugdealer logo
    drugdealer_logo: str = pyg.figlet_format('DrugDealer',
                                             font='slant',
                                             width=120,
                                             justify='center')
    logger.info('\n%s', drugdealer_logo)
    journey_log: str = pyg.figlet_format('Drug dealing begins!',
                                         font='digital',
                                         width=120,
                                         justify='center')
    print(journey_log)
    logger.info('Running DrugDealer')
    logger.info('Running mode: %s', run_mode.upper())

    # Initialize parameters
    path_params = params['paths']

    # Run DrugDealer mode
    if run_mode in ['all', 'dataprocess']:
        run_dataprocess(params=path_params,
                        logger=logger)

    if run_mode in ['all', 'train']:
        run_train(params=params,
                  logger=logger)

    if run_mode in ['all', 'evaluate']:
        run_evaluate()

    logger.info(f'\n{120 * "="}')
    ending: str = pyg.figlet_format("Drug dealing successful!",
                                    font="digital",
                                    width=120,
                                    justify='center')

    print(f'{ending}')
    logger.handlers.clear()
