"""Implements train"""
import pytorch_lightning as pl
import torch
from .dataloaders import prepare_data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from drugdealer.train.model import get_model

from drugdealer.train.model import LightDrug


def run_train(params,
              logger):
    """Implements training.

    Args:
        params (Dict): Parameters.
    """
    # Initialize loaders
    train_loader, validation_loader = prepare_data(params, logger)
    
    #Initialize trainer
    optim_params = params['optim_params']
    callbacks = params['callbacks']
    torch.cuda.empty_cache()
    trainer = pl.Trainer(accelerator=optim_params['device'],
                         max_epochs=optim_params['epochs'],
                         val_check_interval=1.0,
                         accumulate_grad_batches=2,
                         callbacks=[EarlyStopping(**callbacks['EarlyStopping'])],
                         deterministic=True,
                         detect_anomaly=True,
                         default_root_dir=params['paths']['root_dir'],
                         num_sanity_val_steps=0)
    
    model = get_model(params=params)
   
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)
