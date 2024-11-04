import torch
from overparameterized_ensembles.visualization.data_visualization import (
    plot2d,
    plot3d,
    generate_input_data_grid_2d,
    generate_input_data_grid_3d,
)
from overparameterized_ensembles.utils.constants import (
    NUMBER_TEST_SAMPLES,
)
from sklearn.datasets import fetch_california_housing


class DataGeneratingFunction:
    def __init__(self, name: str, data_dimension: int):
        self.name = name
        self.data_dimension = data_dimension

        if name == "linear":
            # Create a random linear function
            weights = torch.randn(data_dimension)
            bias = torch.randn(1)
            self.evaluate = lambda X: torch.mm(X, weights.unsqueeze(1)) + bias
        elif name == "quadratic":
            # Create a random quadratic function
            quadratic_term = torch.randn(data_dimension, data_dimension)
            weights = torch.randn(data_dimension)
            bias = torch.randn(1)
            self.evaluate = lambda X: (
                torch.sum(X @ quadratic_term * X, dim=1, keepdim=True)  # Quadratic term
                + X @ weights.unsqueeze(1)  # Linear term
                + bias  # Bias term
            )
        elif name == "sinusoidal":
            # Create a random sinusoidal function
            weights = torch.randn(data_dimension)
            # Subtract 0.5 of the weights
            weights -= 0.5
            # Normalize the weights
            weights /= 10
            # Add 0.5 to the weights
            weights += 0.5
            bias = torch.randn(1)
            self.evaluate = (
                lambda X: torch.sin(2 * torch.mm(X, weights.unsqueeze(1))) + bias
            )
        elif name == "log_sinusoidal":
            # Create a random logarithmic function with sinusoidal component
            weights = torch.randn(data_dimension)
            bias = torch.randn(1)
            self.evaluate = lambda X: (
                10
                * torch.log(
                    torch.abs(torch.mm(X, weights.unsqueeze(1))) + 1
                )  # Logarithmic term
                + torch.sin(2 * torch.mm(X, weights.unsqueeze(1)))  # Sinusoidal term
                + bias  # Bias term
            )
        else:
            raise ValueError("Invalid data generation function.")

        self.get_input_range = lambda: (
            torch.zeros(data_dimension) - 5,
            10 * torch.ones(data_dimension) - 5,
        )

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def generate_input_data_in_range(self, num_samples: int):
        # Get the input range
        input_range = self.get_input_range()
        num_dimensions = len(input_range[0])

        # Convert input range to PyTorch tensors
        lower_bound = input_range[0].clone()
        upper_bound = input_range[1].clone()

        # Sample uniformly from the input range
        X = lower_bound + (upper_bound - lower_bound) * torch.rand(
            num_samples, num_dimensions
        )

        return X

    def generate_input_data_as_grid(self, num_samples: int):
        # Generate a grid of points over the input range
        if self.data_dimension == 1:
            X = generate_input_data_grid_2d(self.get_input_range(), num_samples)
        elif self.data_dimension == 2:
            # Take the square root of the number of samples to get a grid and round it to the nearest integer
            grid_size = int(num_samples ** (1 / self.data_dimension))
            _, _, X = generate_input_data_grid_3d(self.get_input_range(), grid_size)
        else:
            raise ValueError(
                "Cannot generate a grid for data dimension other than 1 or 2 or 4."
            )

        return X

    def plot_function(self, additional_points=None, additional_points_labels=None):
        if self.data_dimension == 1:
            return plot2d(
                self.evaluate,
                f_label="Data Generating Function",
                input_range=self.get_input_range(),
                additional_points=additional_points,
                additional_points_labels=additional_points_labels,
            )
        elif self.data_dimension == 2:
            return plot3d(
                self.evaluate,
                f_label="Data Generating Function",
                input_range=self.get_input_range(),
                additional_points=additional_points,
                additional_points_labels=additional_points_labels,
            )
        else:
            raise ValueError("Cannot plot functions with dimension greater than 2.")


def generate_data(
    data_generating_function_name: str,
    num_training_samples: int,
    data_dimension: int,
    noise_level: float,
    number_test_samples: int = NUMBER_TEST_SAMPLES,
    test_samples_as_grid: bool = False,
    data_generating_function=None,
):
    """
    Generate data for a given data generating function.

    Args:
    data_generating_function_name : str
        The name of the data generating function.
    num_training_samples : int
        The number of training samples.
    data_dimension : int
        The dimension of the data.
    noise_level : float
        The noise level.
    number_test_samples : int
        The number of test samples.
    test_samples_as_grid : bool
        Whether to generate test samples as a grid.
    data_generating_function : DataGeneratingFunction
        The data generating function.

    Returns:
    X : torch.Tensor
        The training data of shape (num_training_samples, data_dimension).
    y : torch.Tensor
        The training labels of shape (num_training_samples,).
    X_star : torch.Tensor
        The test data of shape (number_test_samples, data_dimension).
    y_star : torch.Tensor
        The test labels of shape (number_test_samples,).
    data_generating_function : DataGeneratingFunction
        The data generating function.
    """
    if data_generating_function_name == "CaliforniaHousing":
        (
            X,
            y,
            X_star,
            y_star,
            data_generating_function,
        ) = generate_california_housing_data(num_training_samples, number_test_samples)
        return X, y, X_star, y_star, data_generating_function

    # Get the data generation function
    if data_generating_function is None:
        data_generating_function = DataGeneratingFunction(
            data_generating_function_name, data_dimension
        )
    else:
        assert (
            data_generating_function.name == data_generating_function_name
            and data_generating_function.data_dimension == data_dimension
        )

    # Generate the training data
    X = data_generating_function.generate_input_data_in_range(num_training_samples)

    # Generate the labels
    y = data_generating_function.evaluate(X).squeeze() + noise_level * torch.randn(
        num_training_samples
    )

    # Generate the test data
    if test_samples_as_grid:
        X_star = data_generating_function.generate_input_data_as_grid(
            number_test_samples
        )
    else:
        X_star = data_generating_function.generate_input_data_in_range(
            number_test_samples
        )

    y_star = data_generating_function.evaluate(X_star).squeeze()

    return X, y, X_star, y_star, data_generating_function


# For of the following real-world dataset we normalize the data OVER THE WHOLE DATASET, not just the training data. We do this for reasons of stability.


def generate_california_housing_data(num_training_samples, number_test_samples):
    """
    Generate the California Housing dataset.

    Args:
    num_training_samples : int
        The number of training samples.
    number_test_samples : int
        The number of test samples.

    Returns:
    X_train : torch.Tensor
        The training data of shape (num_training_samples, 8).
    y_train : torch.Tensor
        The training labels of shape (num_training_samples,).
    X_test : torch.Tensor
        The test data of shape (number_test_samples, 8).
    y_test : torch.Tensor
        The test labels of shape (number_test_samples,).
    data_generating_function : None
        Placeholder for compatibility.
    """
    # Load the California Housing dataset
    data = fetch_california_housing()

    # Extract features and target
    X = data.data  # Shape: [20640, 8]
    y = data.target  # Shape: [20640]

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.double)
    y = torch.tensor(y, dtype=torch.double)

    # Rescale features to [0, 1]
    min_X = X.min(dim=0)[0]
    max_X = X.max(dim=0)[0]
    X_rescaled = (X - min_X) / (max_X - min_X)

    # Compute mean feature value over training data
    mean_feature_value = X_rescaled.mean()

    # Recenter features around mean value
    X_centered = X_rescaled - mean_feature_value

    # Sample num_training_samples and number_test_samples
    num_samples = X_centered.shape[0]

    if num_training_samples + number_test_samples > num_samples:
        raise ValueError(
            f"The sum of num_training_samples ({num_training_samples}) and number_test_samples ({number_test_samples}) "
            f"exceeds the total number of available samples ({num_samples})."
        )

    # Randomly select samples
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_training_samples]
    test_indices = indices[
        num_training_samples : num_training_samples + number_test_samples
    ]

    X_train = X_centered[train_indices]
    y_train = y[train_indices]

    X_test = X_centered[test_indices]
    y_test = y[test_indices]

    data_generating_function = None  # Placeholder for compatibility

    return X_train, y_train, X_test, y_test, data_generating_function
