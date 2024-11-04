import torch
import torch.nn.functional as F

### Utils for the random features models


def initialize_random_weights_distribution(
    random_weights_distribution_name, data_dimension
):
    """
    Initialize the random weights distribution.

    Args:
    random_weights_distribution_name : str
        The name of the random weights distribution.
    data_dimension : int
        The dimension of the data.

    Returns:
    random_weights_distribution : torch.distributions.Distribution
        The random weights distribution.
    """
    if random_weights_distribution_name == "normal":
        mean_random_weights_distribution = torch.zeros(data_dimension)
        covariance_random_weights_distribution = torch.eye(data_dimension)
        random_weights_distribution = torch.distributions.MultivariateNormal(
            mean_random_weights_distribution, covariance_random_weights_distribution
        )
    else:
        raise ValueError("Invalid random weights distribution.")
    return random_weights_distribution


def sample_omega_and_calculate_phi(
    X, activation_function_name, random_weights_distribution, num_features
):
    """
    Sample the omega parameters from the random_weights_distribution and calculate the Phi matrix for a given input X.

    Args:
    X : torch.Tensor
        The input tensor of shape (n_samples, n_features).
    activation_function_name : str
        The name of the activation function to apply (e.g., 'relu').
    random_weights_distribution : torch.distributions.Distribution
        The distribution from which to sample the omega parameters.
    num_features : int
        The number of random features to generate.

    Returns:
    Phi : torch.Tensor
        The Phi matrix of shape (n_samples, num_features).
    """
    if not isinstance(random_weights_distribution, torch.distributions.Distribution):
        raise TypeError(
            "random_weights_distribution must be a torch.distributions.Distribution"
        )
    if not isinstance(num_features, int):
        raise TypeError("num_features must be an int")

    # Sample omega from the distribution random_weights_distribution for each feature
    omega = random_weights_distribution.sample((num_features,))

    # Calculate the Phi matrix
    Phi = calculate_phi(X, omega, activation_function_name)

    return Phi


def calculate_phi(X, omega, activation_function_name):
    """
    Calculate the Phi matrix for a given set of inputs X, distribution random_weights_distribution, and number of features.
    Every activation function is applied in the fashion phi(x) = activation_function(omega^T x).

    Args:
    X : torch.Tensor
        The input tensor of shape (n_samples, n_features).
    omega : torch.Tensor
        The omega parameters of shape (num_features, n_features).
    activation_function_name : str
        The name of the activation function to apply (e.g., 'relu').
    num_features : int
        The number of random features to generate.

    Returns:
    Phi : torch.Tensor
        The Phi matrix of shape (n_samples, num_features).

    Raises:
    TypeError: If X or omega are not torch.Tensors.
    """
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if not isinstance(omega, torch.Tensor):
        raise TypeError("omega must be a torch.Tensor")

    # Map activation function names to actual functions
    activation_functions = {
        "relu": F.relu,
        "erf": torch.erf,
        "softplus": F.softplus,
    }

    if activation_function_name not in activation_functions:
        raise ValueError(f"Unsupported activation function: {activation_function_name}")

    activation_function = activation_functions[activation_function_name]

    # Compute the dot product between X and omega
    X_omega = torch.mm(X, omega.t())

    # Apply the specified activation function
    activation_function = activation_functions[activation_function_name]
    Phi = activation_function(X_omega)

    return Phi
