import torch
import pytorch_lightning as pl
from overparameterized_ensembles.data_generation.data_generation import (
    generate_california_housing_data,
)


class CaliforniaHousingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_training_samples: int,
        num_validation_samples: int,
        num_test_samples: int,
        batch_size: int,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.num_training_samples = num_training_samples
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size
        self.dtype = dtype
        self.X = None
        self.y = None
        self.X_star = None
        self.y_star = None

    def setup(self, stage=None):
        self.X, self.y, self.X_star, self.y_star, _ = generate_california_housing_data(
            self.num_training_samples + self.num_validation_samples,
            self.num_test_samples,
        )
        # Use the specified dtype
        self.X = self.X.to(self.dtype)
        self.y = self.y.to(self.dtype)
        self.X_star = self.X_star.to(self.dtype)
        self.y_star = self.y_star.to(self.dtype)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.X[: self.num_training_samples], self.y[: self.num_training_samples]
            ),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.X[self.num_training_samples :], self.y[self.num_training_samples :]
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_star, self.y_star),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
