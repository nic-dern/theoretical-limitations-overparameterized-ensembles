import torch
from rich.progress import track


def monte_carlo_estimation(single_estimate, number_simulations):
    """
    Perform the Monte Carlo estimation.

    Args:
    single_estimate : function
        The function that performs a single simulation.
    number_simulations : int
        The number of simulations.

    Returns:
    estimate : torch.Tensor
        The Monte Carlo estimate.
    """
    # Get the shape of a single simulation result
    try:
        single_simulation_shape = single_estimate().shape
    except AttributeError:
        raise ValueError("The result of single_estimate must have a 'shape' attribute.")

    # Preallocate a tensor to store the simulation results
    estimates = torch.empty((number_simulations,) + single_simulation_shape)

    # Perform multiple simulations and store the results in the preallocated tensor
    for i in track(
        range(number_simulations), description="Calculating monte carlo estimates..."
    ):
        estimates[i] = single_estimate()

    # Calculate the average of the estimates in every dimension
    estimate = torch.mean(estimates, dim=0)

    return estimate, estimates
