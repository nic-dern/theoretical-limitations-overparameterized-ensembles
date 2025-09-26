import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Literal

REGULARIZATION_STRENGTH = 0.0001


class NeuralNetworkModel(pl.LightningModule):
    def __init__(
        self,
        hidden_layer_sizes: list[int],
        learning_rate: float,
        optimizer: Literal["adam", "sgd"] = "adam",
        momentum: float = 0.9,  # only used for SGD
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.model = self.create_model()
        self.best_val_loss = float("inf")

    def create_model(self):
        layers = []
        for i in range(len(self.hidden_layer_sizes) - 1):
            layers.append(
                nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1])
            )
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_layer_sizes[-1], 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def mse_loss(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Reshape y to match y_hat's dimensions
        y = y.view(-1, 1)
        loss = F.mse_loss(y_hat, y)
        # Add L2 regularization
        l2_reg = sum(
            torch.sum(param**2) for param in self.parameters() if param.requires_grad
        )
        loss += REGULARIZATION_STRENGTH * l2_reg
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Reshape y to match y_hat's dimensions
        y = y.view(-1, 1)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)  # Current validation loss

        # TODO: This is not quite right. We save the best validation loss in a batch not a training step.
        # Update and log best validation loss
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        self.log("best_val_loss", self.best_val_loss)  # Best validation loss so far

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Reshape y to match y_hat's dimensions
        y = y.view(-1, 1)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        return optimizer
