import typer
import torch
import numpy as np
from pathlib import Path
from overparameterized_ensembles.experiments.experiments import (
    RandomFeatureModelExperimentParameters,
    Experiment,
)
from overparameterized_ensembles.models.model_utils import (
    initialize_random_weights_distribution,
)
from overparameterized_ensembles.data_generation.data_generation import (
    generate_data,
)
from overparameterized_ensembles.models.ensembles import (
    EnsembleRandomFeatureModel,
)
from overparameterized_ensembles.visualization.data_visualization import (
    plot2d,
    plot3d,
)
from overparameterized_ensembles.models.kernel_models import (
    KernelRegressor,
)
from overparameterized_ensembles.utils.constants import (
    NUMBER_OF_MODELS_FOR_VARIANCE,
    DEFAULT_RANDOM_SEED,
)
from overparameterized_ensembles.utils.utils import (
    save_figure,
)
from overparameterized_ensembles.visualization.data_visualization import (
    collect_additional_points,
)

app = typer.Typer()


@app.command()
def variance_vs_points_in_range(
    num_features_per_model: int = typer.Option(15, help="Number of features per model"),
    num_training_samples: int = typer.Option(4, help="Number of samples to use"),
    data_dimension: int = typer.Option(1, help="Dimension of the data"),
    data_generating_function_name: str = typer.Option(
        "quadratic", help="Data generation function"
    ),
    noise_level: float = typer.Option(0.05, help="Noise level"),
    activation_function: str = typer.Option("relu", help="Activation function"),
    random_weights_distribution: str = typer.Option(
        "normal", help="Random weights distribution"
    ),
    kernel: str = typer.Option("arc-cosine-kernel", help="Kernel"),
    case_type: str = typer.Option("subexponential", help="Case type"),
    number_points_to_test: int = typer.Option(
        20, help="Number of points to test per dimension"
    ),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
    plot_on_same_axis: bool = typer.Option(
        False, help="Plot variance and r_perp_squared on the same axis"
    ),
):
    """
    Run the variance and its subterms vs. points in range experiment
    """

    # Set the random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Make sure that one either uses the subexponential case or the gaussian case
    if case_type not in ["subexponential", "gaussian"]:
        raise ValueError("Invalid case type.")
    # Check that the kernel and the activation function are compatible
    if activation_function == "relu" and kernel != "arc-cosine-kernel":
        raise ValueError(
            "The ReLU activation function is only compatible with the arc-cosine kernel."
        )

    # Check if the data dimension greater than 2
    if data_dimension > 2:
        # Warn the user
        typer.echo(
            "The data dimension is greater than 2. Some plots will not be generated."
        )

    # Create experiment parameters
    experiment_parameters = RandomFeatureModelExperimentParameters(
        num_features_per_model=num_features_per_model,
        max_num_models=-1,
        num_training_samples=num_training_samples,
        data_dimension=data_dimension,
        data_generating_function_name=data_generating_function_name,
        noise_level=noise_level,
        kernel=kernel,
        activation_function=activation_function,
        random_weights_distribution=random_weights_distribution,
        case_type=case_type,
        number_simulations_per_size=number_points_to_test,
        plot_on_same_axis=plot_on_same_axis,
    )

    # Create the experiment
    experiment = VarianceVsPointsInRangeExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=7,
    )

    # Run and visualize the experiment
    experiment.run_and_visualize_experiment()


class VarianceVsPointsInRangeExperiment(Experiment):
    def _get_experiment_dir_name(self):
        return "variance_vs_points_in_range"

    def _run_experiment(self):
        # Initialize the random weights distribution
        random_weights_distribution = initialize_random_weights_distribution(
            self.experiment_parameters.random_weights_distribution,
            self.experiment_parameters.data_dimension + 1,  # Add one for the bias term
        )

        # Generate the training and test data
        X_train, y_train, X_star, y_star, data_generating_function = generate_data(
            self.experiment_parameters.data_generating_function_name,
            self.experiment_parameters.num_training_samples,
            self.experiment_parameters.data_dimension,
            self.experiment_parameters.noise_level,
            number_test_samples=self.experiment_parameters.number_simulations_per_size,
            test_samples_as_grid=True,
        )

        if self.experiment_parameters.case_type == "subexponential":
            # Initialize and train the ensemble model
            ensemble_model = (
                EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                    self.experiment_parameters.data_dimension,
                    self.experiment_parameters.num_features_per_model,
                    random_weights_distribution,
                    self.experiment_parameters.activation_function,
                    NUMBER_OF_MODELS_FOR_VARIANCE,
                )
            )
        elif self.experiment_parameters.case_type == "gaussian":
            # Initialize and train the ensemble model
            ensemble_model = (
                EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                    self.experiment_parameters.data_dimension,
                    self.experiment_parameters.num_features_per_model,
                    0.0,
                    self.experiment_parameters.kernel,
                    X_train,
                    X_star,
                    NUMBER_OF_MODELS_FOR_VARIANCE,
                )
            )
        ensemble_model.learn_theta(X_train, y_train)

        # Initialize the kernel regressor
        kernel_model = KernelRegressor(
            self.experiment_parameters.kernel,
            0.0,
        )
        kernel_model.fit(X_train, y_train)

        # Return the results
        results = {
            "data_generating_function": data_generating_function,
            "ensemble_model": ensemble_model,
            "kernel_model": kernel_model,
            "X_train": X_train,
            "y_train": y_train,
        }

        return results

    def _visualize_results(self, results):
        # Plot the data generating function
        data_generating_function = results["data_generating_function"]
        if self.experiment_parameters.data_dimension <= 2:
            # Get the additional points
            additional_points = collect_additional_points(
                [results["X_train"]],
                [results["y_train"]],
                self.experiment_parameters.data_dimension,
            )

            # Plot the data generating function
            plt = data_generating_function.plot_function(
                additional_points=additional_points,
                additional_points_labels=["Training Data"],
            )

            # Save the plot
            save_figure(plt, self.experiment_dir / "data_generating_function.pdf")

        # New additional points for the variance and r_perp_squared
        additional_points = collect_additional_points(
            [results["X_train"]],
            [torch.zeros_like(results["y_train"])],
            self.experiment_parameters.data_dimension,
        )

        # Plot the variance vs. points in range
        if self.experiment_parameters.data_dimension == 1:
            plt = plot2d(
                results["ensemble_model"].variance,
                f_label=r"$\textrm{Var}_{\pi_{D}} [ h^{(LN)}_S(\cdot) ]$",
                input_range=results["data_generating_function"].get_input_range(),
                num_samples=self.experiment_parameters.number_simulations_per_size,
                additional_points=additional_points,
                additional_points_labels=["Training Data Points"],
                additional_functions=[[results["kernel_model"].r_perp_squared]]
                if self.experiment_parameters.plot_on_same_axis
                else None,
                additional_functions_labels=[r"$r_\perp^2$"]
                if self.experiment_parameters.plot_on_same_axis
                else None,
            )
        elif self.experiment_parameters.data_dimension == 2:
            num_samples = (
                self.experiment_parameters.number_simulations_per_size
                if self.experiment_parameters.case_type == "subexponential"
                else int(
                    self.experiment_parameters.number_simulations_per_size ** (1 / 2)
                )
            )
            plt = plot3d(
                results["ensemble_model"].variance,
                f_label=r"$\textrm{Var}_{\pi_{D}} [ h^{(LN)}_S(\cdot) ]$",
                input_range=results["data_generating_function"].get_input_range(),
                num_samples=num_samples,
                additional_points=additional_points,
                additional_points_labels=["Training Data Points"],
                plot_highest_point=False,
                additional_functions=[[results["kernel_model"].r_perp_squared]]
                if self.experiment_parameters.plot_on_same_axis
                else None,
                additional_functions_labels=[r"$r_\perp^2$"]
                if self.experiment_parameters.plot_on_same_axis
                else None,
            )

        if self.experiment_parameters.data_dimension <= 2:
            # Save the plot
            save_figure(plt, self.experiment_dir / "variance_vs_points_in_range.pdf")

        # Plot the r_perp_squared vs. points in range
        if not self.experiment_parameters.plot_on_same_axis:
            if self.experiment_parameters.data_dimension == 1:
                plt = plot2d(
                    results["kernel_model"].r_perp_squared,
                    f_label=r"$r_\perp^2$",
                    input_range=results["data_generating_function"].get_input_range(),
                    num_samples=self.experiment_parameters.number_simulations_per_size,
                    additional_points=additional_points,
                    additional_points_labels=["Training Data Points"],
                    plot_highest_point=False,
                )

            if self.experiment_parameters.data_dimension <= 2:
                # Save the plot
                save_figure(
                    plt, self.experiment_dir / "r_perp_squared_vs_points_in_range.pdf"
                )


if __name__ == "__main__":
    app()
