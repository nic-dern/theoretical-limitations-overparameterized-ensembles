import torch
import torch.nn.functional as F
from overparameterized_ensembles.matrices_and_kernels.kernel_calculations import (
    calculate_kernel_matrix,
)
from overparameterized_ensembles.utils.constants import (
    ZERO_REGULARIZATION,
)


class KernelRegressor(torch.nn.Module):
    def __init__(self, kernel: str, ridge: float = 0.0):
        """
        Initializes the Kernel Regressor with a given kernel function.

        Args:
        kernel : str
            The kernel function to use
        ridge : float, optional
            Regularization parameter, by default 0.0
        """
        super(KernelRegressor, self).__init__()
        self.kernel = kernel
        self.alpha = None
        if ridge < ZERO_REGULARIZATION:
            ridge = ZERO_REGULARIZATION
        self.ridge = ridge

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fits the model using the training data (X, y).

        Args:
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features)
        y : torch.Tensor
            Target tensor of shape (n_samples,)
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("X and y must be torch.Tensor")

        self.X_train = X

        # Compute the kernel matrix K and add the ridge regularization term
        K = calculate_kernel_matrix(
            self.X_train, self.X_train, self.kernel
        ) + self.ridge * torch.eye(self.X_train.size(0))

        # Solve the symmetric system K * alpha = y
        self.alpha = torch.linalg.lstsq(
            K, y.unsqueeze(1), driver="gelsd"
        ).solution.squeeze()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for the given input data X.

        Args:
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features)

        Returns:
        torch.Tensor
            Predicted output tensor of shape (n_samples,)
        """
        if self.alpha is None:
            raise Exception("Model is not fitted yet.")

        # Compute the kernel matrix between training data and new input data
        K_forward = calculate_kernel_matrix(self.X_train, X, self.kernel)

        # Predict the output using the learned alpha
        return torch.matmul(K_forward.t(), self.alpha)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error loss for the given input data X and target y.

        Args:
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features)
        y : torch.Tensor
            Target tensor of shape (n_samples,)

        Returns:
        torch.Tensor
            Mean squared error loss
        """
        # Compute the predictions
        y_pred = self.forward(X)
        # Compute the mean squared error loss
        return F.mse_loss(y_pred, y)

    def rkhs_norm_squared(self) -> torch.Tensor:
        """
        Computes the squared RKHS norm of the learned function.

        Returns:
        torch.Tensor
            Squared RKHS norm
        """
        # Compute the kernel matrix K
        K = calculate_kernel_matrix(self.X_train, self.X_train, self.kernel)

        # Compute the RKHS norm squared
        return torch.matmul(self.alpha, torch.matmul(K, self.alpha))

    def r_perp_squared(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared residual between the prediction and the projection onto the RKHS.

        Args:
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features)

        Returns:
        torch.Tensor
            Squared residuals
        """

        # Compute kernel(x^*, x^*)
        K_xx = calculate_kernel_matrix(X, X, self.kernel)

        # Compute kernel(x^*, self.X_train)
        K_x_train = calculate_kernel_matrix(X, self.X_train, self.kernel)

        # Compute kernel(self.X_train, self.X_train)
        K_train_train = calculate_kernel_matrix(
            self.X_train, self.X_train, self.kernel
        ) + ZERO_REGULARIZATION * torch.eye(self.X_train.size(0))

        # Solve the symmetric system K_train_train @ y = K_x_train
        K_train_train_inv = torch.linalg.lstsq(
            K_train_train, K_x_train.t(), driver="gelsd"
        ).solution

        # Compute r_perp_squared
        r_perp_squared = K_xx - torch.matmul(K_x_train, K_train_train_inv)

        return r_perp_squared.diag()

    def condition_number(self):
        """
        Computes the condition number of the kernel matrix.

        Returns:
        torch.Tensor
            Condition number of the kernel matrix
        """
        # Compute the kernel matrix K
        K = calculate_kernel_matrix(
            self.X_train, self.X_train, self.kernel
        ) + self.ridge * torch.eye(self.X_train.size(0))

        # Compute the condition number
        return torch.linalg.cond(K)
