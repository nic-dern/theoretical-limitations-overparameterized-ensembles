import typer
import torch
import rich
from pathlib import Path
import numpy as np
from overparameterized_ensembles.experiments.experiments import (
    RandomFeatureModelExperimentParameters,
    Experiment,
)
from overparameterized_ensembles.models.model_utils import (
    initialize_random_weights_distribution,
)
from overparameterized_ensembles.matrices_and_kernels.kernel_calculations import (
    calculate_kernel_matrix,
)
from overparameterized_ensembles.monte_carlo.monte_carlo import (
    monte_carlo_estimation,
)
from overparameterized_ensembles.monte_carlo.w_terms import (
    calculate_w_term,
    calculate_w_term_ridge,
)
from overparameterized_ensembles.visualization.plots import (
    plot_development_of_mean,
)
from overparameterized_ensembles.data_generation.data_generation import (
    generate_data,
)
from overparameterized_ensembles.visualization.plots import (
    plot_distribution,
)
from overparameterized_ensembles.utils.utils import (
    save_figure,
)
from overparameterized_ensembles.utils.constants import (
    DEFAULT_RANDOM_SEED,
)

app = typer.Typer()


@app.command()
def convergence_expected_value_term(
    num_features_per_model: int = typer.Option(15, help="Number of features per model"),
    max_num_models: int = typer.Option(100000, help="Maximum number of models"),
    num_training_samples: int = typer.Option(4, help="Number of samples to use"),
    data_dimension: int = typer.Option(1, help="Dimension of the data"),
    kernel: str = typer.Option("arc-cosine-kernel", help="Kernel"),
    activation_function: str = typer.Option("relu", help="Activation function"),
    random_weights_distribution: str = typer.Option(
        "normal", help="Random weights distribution"
    ),
    case_type: str = typer.Option("subexponential", help="Case type"),
    data_generating_function_name: str = typer.Option(
        "quadratic", help="Data generation function"
    ),
    ridge: float = typer.Option(0.0, help="Ridge parameter"),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the convergence expected value term experiment
    """
    typer.echo("Running convergence expected value term experiment...")

    # Set the random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Only the subexponential case is supported as of now
    if case_type != "subexponential":
        raise ValueError("Invalid case type.")

    # Create experiment parameters
    experiment_parameters = RandomFeatureModelExperimentParameters(
        num_features_per_model=num_features_per_model,
        max_num_models=max_num_models,
        num_training_samples=num_training_samples,
        data_dimension=data_dimension,
        data_generating_function_name=data_generating_function_name,
        noise_level=-1,
        kernel=kernel,
        activation_function=activation_function,
        random_weights_distribution=random_weights_distribution,
        case_type=case_type,
        number_simulations_per_size=-1,
    )

    # Create the experiment
    experiment = ConvergenceExpectedValueTermExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=2,
        ridge=ridge,
    )

    # Run and visualize the experiment
    experiment.run_and_visualize_experiment()


# Subclass for the first plot
class ConvergenceExpectedValueTermExperiment(Experiment):
    def __init__(self, results_path, experiment_parameters, experiment_number, ridge):
        super().__init__(results_path, experiment_parameters, experiment_number)
        self.ridge = ridge

    def _get_experiment_dir_name(self):
        return "convergence_expected_value_term"

    def _run_experiment(self):
        # Initialize the random weights distribution
        random_weights_distribution = initialize_random_weights_distribution(
            self.experiment_parameters.random_weights_distribution,
            self.experiment_parameters.data_dimension + 1,  # Add one for the bias term
        )

        # Generate the training and test data
        X_star, _, _, _, _ = generate_data(
            self.experiment_parameters.data_generating_function_name,
            self.experiment_parameters.num_training_samples + 1,
            self.experiment_parameters.data_dimension,
            0.05,
        )

        # Calculate the kernel matrix (including the test point)
        K_star = calculate_kernel_matrix(
            X_star, X_star, self.experiment_parameters.kernel
        )

        # Append a column of ones to the X_star matrix; this is not done before since the calculate_kernel_matrix-method automatically appends ones
        X_star = torch.cat((X_star, torch.ones(X_star.shape[0], 1)), dim=1)

        if self.ridge == 0.0:
            # Define the single estimate function
            def single_estimate():
                return calculate_w_term(
                    X_star,
                    K_star,
                    self.experiment_parameters.num_features_per_model,
                    random_weights_distribution,
                    self.experiment_parameters.activation_function,
                )
        else:
            # Define the single estimate function
            def single_estimate():
                return calculate_w_term_ridge(
                    X_star,
                    K_star,
                    self.experiment_parameters.num_features_per_model,
                    random_weights_distribution,
                    self.experiment_parameters.activation_function,
                    ridge=self.ridge,
                )

        # Perform the Monte Carlo estimation
        estimate, estimates = monte_carlo_estimation(
            single_estimate, self.experiment_parameters.max_num_models
        )

        return (estimate, estimates)

    def _visualize_results(self, results):
        title = ""
        if self.experiment_parameters.case_type == "subexponential":
            if self.experiment_parameters.activation_function == "relu":
                title = "ReLU Features"
            elif self.experiment_parameters.activation_function == "erf":
                title = "Gaussian Error Function Features"
            else:
                title = "Unknown Features"
        elif self.experiment_parameters.case_type == "gaussian":
            title = "Gaussian Features"

        estimate, estimates = results

        # Save the estimate
        torch.save(estimate, self.experiment_dir / "estimate.pt")

        # Save the estimates
        torch.save(estimates, self.experiment_dir / "estimates.pt")

        # Print the estimate
        rich.print(estimate)

        # Plot the development of the mean
        fig = plot_development_of_mean(
            estimates, every_xth=100, y_lim_min=-0.5, y_lim_max=0.5
        )
        # Save the plot
        save_figure(fig, self.experiment_dir / "convergence_expected_value_term.pdf")

        # Plot the distribution of the estimates (for each index of the term)
        estimates = estimates.squeeze()
        for i in range(estimates.shape[1]):
            fig = plot_distribution(
                estimates[:, i],
                title,
                x_label=r"$[w_{\perp}^\top W^\top (W W^\top)^{-1}]_{"
                + str(i + 1)
                + "}$ (whitened finite model residual)"
                if self.ridge == 0.0
                else r"$[w_{\perp}^\top W^\top (W W^\top + \lambda \cdot D \cdot R^{-\top} R^{-1})^{-1}]_{"
                + str(i + 1)
                + "}$",
            )
            # Save the plot
            save_figure(
                fig,
                self.experiment_dir
                / f"convergence_expected_value_term_distribution_{i}.pdf",
            )
