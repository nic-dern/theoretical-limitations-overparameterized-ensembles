import torch
import torch.nn as nn
import torch.nn.functional as F
from overparameterized_ensembles.models.random_feature_models import (
    RandomFeatureModel,
    RandomFeatureModelGaussianUniversality,
)


class EnsembleRandomFeatureModel(nn.Module):
    def __init__(self, models: nn.ModuleList):
        super(EnsembleRandomFeatureModel, self).__init__()
        self.models = models

    @staticmethod
    def create_ensemble_random_feature_models(
        data_dimension: int,
        num_features_per_model: int,
        random_weights_distribution: torch.distributions.Distribution,
        activation_function_name: str,
        num_models: int,
        ridge: float = 0.0,
    ) -> "EnsembleRandomFeatureModel":
        models = nn.ModuleList(
            [
                RandomFeatureModel.create_model(
                    data_dimension,
                    num_features_per_model,
                    random_weights_distribution,
                    activation_function_name,
                    ridge,
                )
                for _ in range(num_models)
            ]
        )
        model = EnsembleRandomFeatureModel(models)
        return model

    @staticmethod
    def create_ensemble_gaussian_universality_models(
        data_dimension: int,
        num_features_per_model: int,
        ridge: float,
        kernel_function_name: str,
        X_train: torch.Tensor,
        x_test: torch.Tensor,
        num_models: int,
    ) -> "EnsembleRandomFeatureModel":
        models = nn.ModuleList(
            [
                RandomFeatureModelGaussianUniversality.create_model(
                    data_dimension,
                    num_features_per_model,
                    ridge,
                    kernel_function_name,
                    X_train,
                    x_test,
                )
                for _ in range(num_models)
            ]
        )
        model = EnsembleRandomFeatureModel(models)
        return model

    def learn_theta(self, X: torch.Tensor, y: torch.Tensor) -> None:
        for model in self.models:
            model.learn_theta(X, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model.forward(x) for model in self.models]
        avg_prediction = torch.stack(predictions).mean(0)
        return avg_prediction

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.forward(X)
        return F.mse_loss(y_pred, y).item()

    def variance(self, X: torch.Tensor) -> torch.Tensor:
        # Check if X is only one point
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        # Compute the predictions for each model
        predictions = [model.forward(X) for model in self.models]

        # Stack predictions to form a tensor
        predictions = torch.stack(predictions)

        # Compute the variance of the predictions along the model axis
        variance = torch.var(predictions, dim=0, unbiased=False)

        return variance
