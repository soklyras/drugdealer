"""Implements model class"""
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Image Layers
        self.image_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Gene expression branch
        self.gene_expression_branch = nn.Sequential(
            nn.Linear(2095, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion_layer = nn.Linear(128 + 32 * 19 * 19, 1)

    def forward(self, image_input, gene_expression_input):
        """
        Forward function.

        Args:
            image_input (tensor): Molecular description
            gene_expression_input (tensor): Gene expression.

        Returns:
            prediction: ic50
        """
        image_input = torch.unsqueeze(image_input, dim=1)
        image_output = self.image_branch(image_input)
        image_output = image_output.view(image_output.size(0), -1)
        gene_expression_output = self.gene_expression_branch(gene_expression_input)
        combined_output = torch.cat([image_output, gene_expression_output], dim=1)
        prediction = self.fusion_layer(combined_output)

        return prediction


class LightDrug(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
    
    def training_step(self, batch):
        """Implements training step

        Args:
            batch (tensors): Batch of tensors.

        Returns: Train loss
        """
        input_images = batch['molecular_structure']
        input_expressions = batch['gene_expression']
        targets = batch['ic50'].float()

        predictions = self.model(input_images, input_expressions).float()
        train_loss = nn.MSELoss()(targets, predictions)

        self.log('train_loss', train_loss, prog_bar=True, on_epoch=True, on_step=False)
        return train_loss
    
    def validation_step(self, batch):
        """Implements training step

        Args:
            batch (tensors): Batch of tensors.

        Returns: Train loss
        """
        input_images = batch['molecular_structure']
        input_expressions = batch['gene_expression']
        targets = batch['ic50'].float()

        predictions = self.model(input_images, input_expressions).float()
        val_loss = nn.MSELoss()(targets, predictions)

        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, on_step=False)
        return val_loss
    
    def configure_optimizers(self):
        return self.optimizer


def get_model(params):
    """
    Loads and returns model.

    Args:
        params (Dict): Dictionary with architecture parameters.
    """
    optim_params = params['optimizer_params']
    model = CustomModel()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  **optim_params)
    return LightDrug(model=model,
                     optimizer=optimizer)