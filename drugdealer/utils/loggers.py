"""
Creates a logger and a log file in 02-physics/01-logs with the
name logging_drugdealer.log.
"""
import logging
import os
import sys


def create_logger(log_path: str = None) -> logging.Logger:
    """
    Instatiates the logger.
    Args:
        log_path (str, optional): Path to log.Defaults to None.

    Returns:
        logger (logging.Logger): DrugDealer logger to display and save logs.
    """
    drugdealer_logger: logging.Logger = logging.getLogger('DrugDealer')
    drugdealer_logger.setLevel(logging.DEBUG)

    if log_path:
        log_file: str = os.path.join(log_path, 'logging_drugdealer.log')

        file_handler: logging.FileHandler = \
            logging.FileHandler(filename=log_file)
        file_handler_format: logging.Formatter = \
            logging.Formatter('[%(asctime)s]: %(message)s')
        file_handler.setFormatter(file_handler_format)
        file_handler.setLevel(logging.DEBUG)
        drugdealer_logger.addHandler(file_handler)

    stdout_handler: logging.StreamHandler = \
        logging.StreamHandler(stream=sys.stdout)
    stdout_handler_format: logging.Formatter = \
        logging.Formatter('%(message)s')
    stdout_handler.setFormatter(stdout_handler_format)
    stdout_handler.setLevel(logging.INFO)
    drugdealer_logger.addHandler(stdout_handler)

    return drugdealer_logger
