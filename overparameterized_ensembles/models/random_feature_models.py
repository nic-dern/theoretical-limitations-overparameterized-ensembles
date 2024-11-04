import torch
import torch.nn as nn
import torch.nn.functional as F
from overparameterized_ensembles.models.model_utils import (
    calculate_phi,
)
from overparameterized_ensembles.models.kernel_models import (
    calculate_kernel_matrix,
)
from overparameterized_ensembles.utils.constants import (
    ZERO_REGULARIZATION,
    FITTING_PROCEDURE,
)


class BaseRandomFeatureModel(nn.Module):
    def __init__(self, data_dimension, num_features, ridge, theta=None):
        super(BaseRandomFeatureModel, self).__init__()
        self.data_dimension = data_dimension
        self.num_features = num_features
        self.ridge = ridge
        # Set to a small value to avoid numerical issues
        if self.ridge <= ZERO_REGULARIZATION:
            self.ridge = ZERO_REGULARIZATION
        self.theta = nn.Parameter(
            theta if theta is not None else torch.zeros(num_features)
        )

    def loss(self, X, y):
        y_pred = self.forward(X)
        return F.mse_loss(y_pred, y).item()

    def set_theta(self, theta):
        self.theta.data = theta


class RandomFeatureModel(BaseRandomFeatureModel):
    def __init__(
        self,
        data_dimension,
        num_features,
        omega,
        theta,
        ridge,
        activation_function_name,
    ):
        super(RandomFeatureModel, self).__init__(
            data_dimension, num_features, ridge, theta
        )
        self.omega = nn.Parameter(omega)
        self.activation_function_name = activation_function_name

    @staticmethod
    def initialize_parameters(
        data_dimension, num_features, random_weights_distribution
    ):
        omega = random_weights_distribution.sample((num_features,))
        theta = torch.zeros(num_features)
        return omega, theta

    @staticmethod
    def create_model(
        data_dimension,
        num_features,
        random_weights_distribution,
        activation_function_name,
        ridge=0.0,
    ):
        omega, theta = RandomFeatureModel.initialize_parameters(
            data_dimension, num_features, random_weights_distribution
        )
        model = RandomFeatureModel(
            data_dimension, num_features, omega, theta, ridge, activation_function_name
        )
        return model

    @staticmethod
    def create_and_train_model(
        data_dimension,
        num_features,
        random_weights_distribution,
        activation_function_name,
        X_train,
        y_train,
        ridge=0.0,
    ):
        model = RandomFeatureModel.create_model(
            data_dimension,
            num_features,
            random_weights_distribution,
            activation_function_name,
            ridge,
        )
        mse_train = model.learn_theta(X_train, y_train)
        return model, mse_train

    @staticmethod
    def create_train_and_calculate_loss(
        data_dimension,
        num_features,
        random_weights_distribution,
        activation_function_name,
        X_train,
        y_train,
        X_test,
        y_test,
        ridge=0.0,
    ):
        model, mse_train = RandomFeatureModel.create_and_train_model(
            data_dimension,
            num_features,
            random_weights_distribution,
            activation_function_name,
            X_train,
            y_train,
            ridge,
        )
        return model, mse_train, model.loss(X_test, y_test)

    def learn_theta(self, X, y):
        # Add a bias term
        X = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)

        # Calculate the features
        Phi = calculate_phi(X, self.omega, self.activation_function_name)
        # Normalize the features
        Phi /= torch.sqrt(torch.tensor(self.num_features, dtype=torch.float64))

        # Calculate the theta parameters
        if Phi.size(0) >= self.num_features:  # Underparameterized (n >= p)
            A = torch.matmul(Phi.T, Phi) + self.ridge * torch.eye(self.num_features)
            b = torch.matmul(Phi.T, y.unsqueeze(1))

            if FITTING_PROCEDURE == "cholesky":
                # Cholesky decomposition
                L = torch.linalg.cholesky(A)
                theta = torch.cholesky_solve(b, L, upper=False).squeeze()
                self.theta.data = theta
            elif FITTING_PROCEDURE == "lstsq":
                self.theta.data = torch.linalg.lstsq(
                    A, b, driver="gelsd"
                ).solution.squeeze()
            else:
                raise ValueError(f"Unsupported FITTING_PROCEDURE: {FITTING_PROCEDURE}")

        else:  # Overparameterized (n < p)
            A = torch.matmul(Phi, Phi.T) + self.ridge * torch.eye(Phi.size(0))
            b = y.unsqueeze(1)

            if FITTING_PROCEDURE == "cholesky":
                # Cholesky decomposition
                L = torch.linalg.cholesky(A)
                alpha = torch.cholesky_solve(b, L, upper=False).squeeze()
                self.theta.data = torch.matmul(Phi.T, alpha)
            elif FITTING_PROCEDURE == "lstsq":
                alpha = torch.linalg.lstsq(A, b, driver="gelsd").solution.squeeze()
                self.theta.data = torch.matmul(Phi.T, alpha)
            else:
                raise ValueError(f"Unsupported FITTING_PROCEDURE: {FITTING_PROCEDURE}")

        # Return the training loss
        return F.mse_loss(torch.mv(Phi, self.theta), y)

    def forward(self, x):
        # Add a bias term
        x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
        features = calculate_phi(x, self.omega, self.activation_function_name)
        features /= torch.sqrt(torch.tensor(self.num_features, dtype=torch.float64))
        return torch.mv(features, self.theta)


class RandomFeatureModelGaussianUniversality(BaseRandomFeatureModel):
    def __init__(
        self,
        data_dimension,
        num_features,
        ridge,
        kernel_function_name,
        F_train,
        F_test,
        X_train,
        X_test,
        theta,
    ):
        super(RandomFeatureModelGaussianUniversality, self).__init__(
            data_dimension, num_features, ridge, theta
        )
        self.data_dimension = data_dimension
        self.kernel_function_name = kernel_function_name
        self.F_train = nn.Parameter(F_train)
        self.F_test = nn.Parameter(F_test)
        self.X_train = X_train
        self.X_test = X_test

    @staticmethod
    def initialize_parameters(X_train, X_test, kernel_function_name, num_features):
        X_train_normalized = X_train
        X_test_normalized = X_test

        X_all = torch.cat([X_train_normalized, X_test_normalized], dim=0)
        K = calculate_kernel_matrix(X_all, X_all, kernel_function_name)

        D, U = torch.linalg.eigh(K)
        D = D.real  # Use the real part of the eigenvalues
        U = U.real  # Use the real part of the eigenvectors
        D[D < 0] = 0  # Set negative eigenvalues to zero
        R = torch.matmul(U, torch.diag(torch.sqrt(D)))

        F_all = torch.matmul(R, torch.randn(R.size(1), num_features))
        F_train = F_all[: X_train.size(0), :]
        F_test = F_all[X_train.size(0) :, :]

        F_train /= torch.sqrt(torch.tensor(num_features, dtype=torch.float64))
        F_test /= torch.sqrt(torch.tensor(num_features, dtype=torch.float64))

        theta = torch.zeros(num_features)

        return F_train, F_test, theta, None, None

    @staticmethod
    def create_model(
        data_dimension, num_features, ridge, kernel_function_name, X_train, X_test
    ):
        (
            F_train,
            F_test,
            theta,
            _,
            _,
        ) = RandomFeatureModelGaussianUniversality.initialize_parameters(
            X_train, X_test, kernel_function_name, num_features
        )
        model = RandomFeatureModelGaussianUniversality(
            data_dimension,
            num_features,
            ridge,
            kernel_function_name,
            F_train,
            F_test,
            X_train,
            X_test,
            theta,
        )

        return model

    @staticmethod
    def create_and_train_model(
        data_dimension,
        num_features,
        ridge,
        kernel_function_name,
        X_train,
        X_test,
        y_train,
    ):
        model = RandomFeatureModelGaussianUniversality.create_model(
            data_dimension, num_features, ridge, kernel_function_name, X_train, X_test
        )
        mse_train = model.learn_theta(X_train, y_train)
        return model, mse_train

    @staticmethod
    def create_train_and_calculate_loss(
        data_dimension,
        num_features,
        ridge,
        kernel_function_name,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        (
            model,
            mse_train,
        ) = RandomFeatureModelGaussianUniversality.create_and_train_model(
            data_dimension,
            num_features,
            ridge,
            kernel_function_name,
            X_train,
            X_test,
            y_train,
        )
        return model, mse_train, model.loss(X_test, y_test)

    def learn_theta(self, X_train, y_train):
        if not torch.equal(X_train, self.X_train):
            raise ValueError(
                "X_train must be the same as the one used for initialization."
            )

        # The features F_train were computed using normalized X_train, so we proceed as before
        if self.F_train.size(0) >= self.num_features:  # Underparameterized (n >= p)
            A = torch.matmul(self.F_train.T, self.F_train) + self.ridge * torch.eye(
                self.num_features
            )
            b = torch.matmul(self.F_train.T, y_train.unsqueeze(1))
            self.theta.data = torch.linalg.lstsq(
                A, b, driver="gelsd"
            ).solution.squeeze()
        else:  # Overparameterized (n < p)
            A = torch.matmul(self.F_train, self.F_train.T) + self.ridge * torch.eye(
                self.F_train.size(0)
            )
            b = y_train.unsqueeze(1)
            alpha = torch.linalg.lstsq(A, b, driver="gelsd").solution.squeeze()
            self.theta.data = torch.matmul(self.F_train.T, alpha)

        return F.mse_loss(torch.mv(self.F_train, self.theta), y_train)

    def forward(self, x):
        x = x.squeeze()
        if x.dim() == 0:
            # Make the input a 1D tensor
            x = x.unsqueeze(0)
        if x.ndim == 1 and x.size(0) == self.data_dimension:
            if not any(torch.equal(x, test_point) for test_point in self.X_test):
                raise ValueError("x must be one of the initially used test points.")
            # Find the index of the test point
            index = next(
                i
                for i, test_point in enumerate(self.X_test)
                if torch.equal(x, test_point)
            )
            # Select the corresponding row from F_test
            F_test_row = self.F_test[index]
            return torch.dot(F_test_row, self.theta)
        else:
            if x[0].dim() == 0:
                x = x.unsqueeze(1)
            if not torch.equal(x, self.X_test):
                raise ValueError(
                    "x must be the same as the one used for initialization."
                )
            return torch.mv(self.F_test, self.theta)
