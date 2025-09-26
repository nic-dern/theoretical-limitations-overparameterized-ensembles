import torch
from overparameterized_ensembles.models.model_utils import (
    initialize_random_weights_distribution,
    calculate_phi,
)
from scipy.optimize import fsolve
import numpy as np


def calculate_kernel_matrix(
    X, Y, kernel_name, random_weights_distribution_name: str = "normal"
):
    """
    Calculate the kernel matrix for a given set of inputs X and Y and the kernel name.

    Args:
    X : torch.Tensor
        The input tensor of shape (n_samples, n_features).
    Y : torch.Tensor
        The input tensor of shape (n_samples_2, n_features).
    kernel_name : str
        The name of the kernel to use (e.g., 'arc-cosine-kernel').
    random_weights_distribution_name : str, optional
        The name of the random weights distribution to use for kernels that require it (e.g., 'softplus-kernel'). Defaults to "normal".

    Returns:
    K : torch.Tensor
        The kernel matrix of shape (n_samples, n_samples_2).
    """
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if not isinstance(Y, torch.Tensor):
        raise TypeError("Y must be a torch.Tensor")
    if not isinstance(kernel_name, str):
        raise TypeError("kernel_name must be a str")

    # Append a column of ones to X and Y
    X = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)
    Y = torch.cat([Y, torch.ones(Y.size(0), 1)], dim=1)

    # Map kernel names to actual functions
    kernels = {
        "arc-cosine-kernel": arc_cosine_kernel,
        "erf-kernel": erf_kernel,
        "softplus-kernel": sofplus_kernel,
    }

    if kernel_name not in kernels:
        raise ValueError("Invalid kernel name")

    if kernel_name == "softplus-kernel":
        return kernels[kernel_name](X, Y, random_weights_distribution_name)
    return kernels[kernel_name](X, Y)


### Kernels

SOFTPLUS_FEATURES = None


def sofplus_kernel(X, Y, random_weights_distribution_name: str = "normal"):
    """
    Computes the softplus kernel between two sets of vectors.
    Note: This is done using the "infinite-width single model" and should only be used when using a normal distribution for the weights.

    Args:
    X : torch.Tensor
        Input tensor of shape (n_samples_X, n_features)
    Y : torch.Tensor
        Input tensor of shape (n_samples_Y, n_features)
    random_weights_distribution_name : str, optional
        The name of the random weights distribution to use. Defaults to "normal".

    Returns:
    K : torch.Tensor
        The kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    random_weights_distribution = initialize_random_weights_distribution(
        random_weights_distribution_name, X.size(1)
    )

    # Number of features
    num_features = 100000

    # Number of iterations for averaging
    num_iterations = 100

    # Initialize the kernel matrix
    K = torch.zeros(X.size(0), Y.size(0))

    global SOFTPLUS_FEATURES

    if SOFTPLUS_FEATURES is None:
        softplus_features = []

    for i in range(num_iterations):
        if SOFTPLUS_FEATURES is None:
            # Sample omega from the distribution random_weights_distribution for each feature
            omega = random_weights_distribution.sample((num_features,))
            softplus_features.append(omega)
        else:
            omega = SOFTPLUS_FEATURES[i]

        # Calculate the Phi matrix
        Phi_X = calculate_phi(X, omega, "softplus")
        Phi_Y = calculate_phi(Y, omega, "softplus")

        # Compute the kernel matrix for this iteration
        K_iteration = torch.mm(Phi_X, Phi_Y.T)

        # Accumulate the results
        K += K_iteration / num_features

    if SOFTPLUS_FEATURES is None:
        SOFTPLUS_FEATURES = softplus_features

    # Average the results
    K /= num_iterations

    return K


def arc_cosine_kernel(X, Y):
    """
    Computes the arc-cosine kernel between two sets of vectors.

    Args:
    X : torch.Tensor
        Input tensor of shape (n_samples_X, n_features)
    Y : torch.Tensor
        Input tensor of shape (n_samples_Y, n_features)

    Returns:
    K : torch.Tensor
        The kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    # Normalize X and Y
    X_norm = torch.linalg.norm(X, dim=1, keepdim=True)  # Shape: (n_samples_X, 1)
    Y_norm = torch.linalg.norm(Y, dim=1, keepdim=True)  # Shape: (n_samples_Y, 1)

    # Compute dot products
    X_normalized = X / X_norm  # Normalize each row in X
    Y_normalized = Y / Y_norm  # Normalize each row in Y and transpose for broadcasting
    dot_product = torch.mm(
        X_normalized, Y_normalized.T
    )  # Shape: (n_samples_X, n_samples_Y)

    # Clip values to ensure numerical stability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle theta
    theta = torch.acos(dot_product)

    # Compute the kernel matrix
    kernel = (
        (1 / (2 * torch.pi))
        * X_norm
        * Y_norm.T
        * (torch.sin(theta) + (torch.pi - theta) * torch.cos(theta))
    )

    return kernel


def erf_kernel(X, Y):
    """
    Computes the Erf kernel between two sets of vectors.

    Args:
    X : torch.Tensor
        Input tensor of shape (n_samples_X, n_features)
    Y : torch.Tensor
        Input tensor of shape (n_samples_Y, n_features)

    Returns:
    K : torch.Tensor
        The kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    # Compute the dot product between X and Y
    dot_product = torch.mm(X, Y.T)  # Shape: (n_samples_X, n_samples_Y)

    # Compute the normalization terms for X and Y
    X_norm = torch.sum(X**2, dim=1, keepdim=True)  # Shape: (n_samples_X, 1)
    Y_norm = torch.sum(Y**2, dim=1, keepdim=True)  # Shape: (n_samples_Y, 1)

    # Compute the argument for the sin inverse function
    numerator = 2 * dot_product
    denominator = torch.sqrt((1 + 2 * X_norm) * (1 + 2 * Y_norm.T))
    argument = numerator / denominator

    # Compute the kernel matrix
    kernel = (2 / torch.pi) * torch.asin(argument)

    return kernel


### Effective Ridge Computations


def get_effective_ridge_implicit_regularization(
    kernel: str,
    X: torch.Tensor,
    ridge: float,
    num_training_samples: int,
    num_features: int,
) -> float:
    """
    Computes the effective ridge for the given kernel and ridge (the effective ridge from the implicit regularization paper from Jacot et al.).

    Args:
    kernel : str
        The kernel function.
    X : torch.Tensor
        The input data of shape (num_training_samples, num_features).
    ridge : float
        The ridge parameter.
    num_training_samples : int
        The number of training samples.
    num_features : int
        The number of features.

    Returns:
    effective_ridge_solution : float
        The effective ridge solution.
    """
    gamma = num_features / num_training_samples

    # Compute the kernel matrix K
    K = calculate_kernel_matrix(X, X, kernel).cpu()
    # Compute the eigenvalues of the kernel matrix
    kernel_eigenvalues, _ = torch.linalg.eig(K)
    # Convert the eigenvalues to numpy and keep only the real part
    kernel_eigenvalues = kernel_eigenvalues.numpy().real

    # Define the function for which we want to solve the root
    def equation(effective_ridge):
        term = kernel_eigenvalues / (effective_ridge + kernel_eigenvalues)
        return effective_ridge - (ridge + (effective_ridge / gamma) * np.mean(term))

    # Use fsolve to find the root of the equation
    initial_guess = ridge
    effective_ridge_solution = fsolve(equation, initial_guess)[0]

    return effective_ridge_solution
