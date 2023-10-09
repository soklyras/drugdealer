"""Datasets and Dataloaders"""
import os
from typing import Dict
import gc

import torch
from torch.utils.data import Dataset as _TorchDataset, DataLoader

from drugdealer.utils.read_utils import read_pickle


def prepare_data(params,
                 logger):
    """Prepare data for Dataset class.

    Args:
        path_params (Dict): Path parameters.
        logger (logging.Logger): the logger.
    """
    path_params = params['paths']
    training_set_path = os.path.join(path_params['processed_dir'],
                                     path_params['train_dict'])
    validation_set_path = os.path.join(path_params['processed_dir'],
                                       path_params['val_dict'])
    
    # Reading data and prepare Data Loaders
    logger.info('Preparing Dataset classes...')
    
    training_data = list(read_pickle(training_set_path).values())
    train_params = params['data_params']['train']
    train_dataset = Dataset(training_data, train_params['batch_size'])
    train_loader = data_loader(train_dataset, train_params)

    validation_data = read_pickle(validation_set_path)      
    val_params = params['data_params']['validation']
    validation_dataset = Dataset(validation_data, val_params['batch_size'])
    validation_loader = data_loader(validation_dataset, val_params)

    return train_loader, validation_loader


def data_loader(dataset: _TorchDataset,
                data_params: Dict):
    """
    Return data loaders for training, validation and test sets.

    Args:
        params (Dict): Transforms parameters.
    """
    dataloader = DataLoader(dataset=dataset,  
                            **data_params)
    
    return dataloader


def custom_collate(batch):
    return tuple(batch)


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    """

    def __init__(self, data: list, batch_size: int, params=None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.params = params
        self.batch_size = batch_size


    def __len__(self) -> int:
        return len(self.data)
    
    
    def __getitem__(self, index):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        batch = {}
        batch['molecular_structure'] = \
            torch.Tensor(self.data[index]['molecular_structure'])
        batch['molecular_structure'] = (batch['molecular_structure'] - batch['molecular_structure'].min())/batch['molecular_structure'].max()
        
        batch['gene_expression'] = torch.Tensor(self.data[index]['gene_expression'])
        batch['gene_expression'] = (batch['gene_expression'] - batch['gene_expression'].min())/batch['gene_expression'].max()
        batch['ic50'] = torch.tensor(self.data[index]['ic50']).item()

        return batch
  