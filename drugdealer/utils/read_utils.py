"""
Here we implement the read utilities for the project.
"""
import pickle
import mmap
from typing import Dict, Union

import yaml


def read_params(params_path: str) -> Dict[str, Union[int, str]]:
    """
    Reads parameters from configuration file.

    Args:
        params_path (str): Path to configuration file.

    Returns:
        Dict[str, Union[int, str]]: Dictionary with parsed configurations.
    """
    with open(params_path, 'r', encoding='utf-8') as file:
        params: dict = yaml.safe_load(file)
    return params


def read_smi(file_path):
    """
    Reads smi file.

    Args:
        file_path (str): Path to smi file.

    Returns:
        List[Dict]: List of dictionaries that each dictionary
        contains the structure and the name of the drug. 
    """
    # Open file
    with open(file_path, 'r') as smi_file:
        smi_contents = smi_file.readlines()
    
    compounds = []
    for line in smi_contents:
        # Split the line based on whitespace or other delimiters
        parts = line.strip().split('\t')  # Split by tab, adjust if needed

        # Extract the structure and the drug
        structure = parts[0]
        additional_info = parts[1] if len(parts) > 1 else None

        # Store the information in a Dictionary
        compounds.append({'structure': structure, 'drug': additional_info})
    return compounds


def read_pickle(file_path):
    """Read pickle file

    Args:
        file_path (str): Path to pickle file.
    """
    with open(file_path, 'rb') as f:
        # Memory map the file
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Unpickle the mapped file
        unpickled_object = pickle.loads(mmapped_file)

        return unpickled_object