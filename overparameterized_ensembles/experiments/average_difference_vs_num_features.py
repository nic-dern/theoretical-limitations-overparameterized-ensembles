import typer
import torch
from pathlib import Path
from rich.progress import track
import numpy as np
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
from overparameterized_ensembles.models.kernel_models import (
    KernelRegressor,
)
from overparameterized_ensembles.models.ensembles import (
    EnsembleRandomFeatureModel,
)
from overparameterized_ensembles.visualization.plots import (
    plot_graph,
)
from overparameterized_ensembles.utils.utils import (
    save_figure,
)
from overparameterized_ensembles.visualization.data_visualization import (
    collect_additional_points,
)
from overparameterized_ensembles.utils.constants import (
    NUMBER_TEST_SAMPLES,
    DEFAULT_RANDOM_SEED,
)

app = typer.Typer()


@app.command()
def average_difference_vs_num_features(
    num_features_start: int = typer.Option(
        2, help="Start number of features per model"
    ),
    num_features_end: int = typer.Option(50, help="End number of features per model"),
    num_features_step: int = typer.Option(
        1, help="Step size for the number of features per model"
    ),
    max_num_models: int = typer.Option(
        5000, help="Defines what means 'infinite' number of models"
    ),
    num_training_samples: int = typer.Option(6, help="Number of samples to use"),
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
    ridge: float = typer.Option(0.0, help="Ridge parameter"),
    number_extra_test_samples: int = typer.Option(
        NUMBER_TEST_SAMPLES // 2, help="Number of extra test samples"
    ),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the average difference vs number of features experiment
    """
    typer.echo("Running average difference vs number of features experiment...")

    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize experiment parameters
    experiment_parameters = RandomFeatureModelExperimentParameters(
        num_features_per_model=None,  # Will be set in the loop
        max_num_models=max_num_models,
        num_training_samples=num_training_samples,
        data_dimension=data_dimension,
        data_generating_function_name=data_generating_function_name,
        noise_level=noise_level,
        kernel=kernel,
        activation_function=activation_function,
        random_weights_distribution=random_weights_distribution,
        case_type=case_type,
        number_simulations_per_size=-1,
    )

    # Run experiment
    experiment = AverageDifferenceVsNumFeaturesExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=1,
        num_features_start=num_features_start,
        num_features_end=num_features_end,
        num_features_step=num_features_step,
        number_extra_test_samples=number_extra_test_samples,
        ridge=ridge,
    )
    experiment.run_and_visualize_experiment()


class AverageDifferenceVsNumFeaturesExperiment(Experiment):
    def __init__(
        self,
        results_path: Path,
        experiment_parameters: RandomFeatureModelExperimentParameters,
        experiment_number: int,
        num_features_start: int,
        num_features_end: int,
        num_features_step: int,
        number_extra_test_samples: int = NUMBER_TEST_SAMPLES,
        ridge: float = 0.0,
    ):
        super().__init__(
            results_path,
            experiment_parameters,
            experiment_number,
        )
        self.num_features_start = num_features_start
        self.num_features_end = num_features_end
        self.num_features_step = num_features_step
        self.number_extra_test_samples = number_extra_test_samples
        self.ridge = ridge

    def _get_experiment_dir_name(self):
        return "average_difference_vs_num_features"

    def _run_experiment(self):
        # Initialize random weights
        random_weights_distribution = initialize_random_weights_distribution(
            self.experiment_parameters.random_weights_distribution,
            self.experiment_parameters.data_dimension + 1,
        )

        # Generate data
        X_train, y_train, X_test, y_test, data_generating_function = generate_data(
            self.experiment_parameters.data_generating_function_name,
            self.experiment_parameters.num_training_samples,
            self.experiment_parameters.data_dimension,
            self.experiment_parameters.noise_level,
            number_test_samples=self.number_extra_test_samples,
        )

        number_extra_samples = self.number_extra_test_samples
        X_test_extra = X_test[:number_extra_samples]

        # Initialize the num_features values
        num_features_values = list(
            range(
                self.num_features_start,
                self.num_features_end + 1,  # Include the end value
                self.num_features_step,
            )
        )

        # Prepare lists to store results
        average_differences = []
        differences_over_samples = []

        # Train and evaluate the infinite single model (kernel ridge regressor)
        kernel_model = KernelRegressor(
            kernel=self.experiment_parameters.kernel,
            ridge=self.ridge,
        )
        kernel_model.fit(X_train, y_train)
        predictions_infinite_single_model = kernel_model.forward(X_test_extra)

        for num_features_per_model in track(
            num_features_values, description="Running experiment..."
        ):
            # Update the number of features per model in the experiment parameters
            self.experiment_parameters.num_features_per_model = num_features_per_model

            # Train and evaluate the infinite ensemble model
            if self.experiment_parameters.case_type == "subexponential":
                ensemble = EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                    data_dimension=self.experiment_parameters.data_dimension,
                    num_features_per_model=num_features_per_model,
                    random_weights_distribution=random_weights_distribution,
                    activation_function_name=self.experiment_parameters.activation_function,
                    num_models=self.experiment_parameters.max_num_models,
                    ridge=self.ridge,
                )
                ensemble.learn_theta(X_train, y_train)
                predictions_infinite_ensemble = ensemble.forward(X_test_extra)
            elif self.experiment_parameters.case_type == "gaussian":
                ensemble = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                    data_dimension=self.experiment_parameters.data_dimension,
                    num_features_per_model=num_features_per_model,
                    ridge=self.ridge,
                    kernel_function_name=self.experiment_parameters.kernel,
                    X_train=X_train,
                    x_test=X_test_extra,
                    num_models=self.experiment_parameters.max_num_models,
                )
                ensemble.learn_theta(X_train, y_train)
                predictions_infinite_ensemble = ensemble.forward(X_test_extra)
            else:
                raise ValueError(
                    f"Unknown case_type: {self.experiment_parameters.case_type}"
                )

            # Compute the difference between predictions
            difference = (
                (predictions_infinite_single_model - predictions_infinite_ensemble)
                .detach()
                .cpu()
                .numpy()
            )
            differences_over_samples.append(difference)

            # Compute the average difference over samples
            average_difference = np.mean(np.abs(difference))
            average_differences.append(average_difference)

        # Package the results
        results = {
            "num_features_values": num_features_values,
            "average_differences": average_differences,
            "differences_over_samples": differences_over_samples,
            "X_train": X_train,
            "y_train": y_train,
            "data_generating_function": data_generating_function,
            "X_test_extra": X_test_extra,
        }
        return results

    def _visualize_results(self, results):
        # Plot the average difference vs number of features
        plt = plot_graph(
            x_values=results["num_features_values"],
            y_values=[results["average_differences"]],
            labels=["Average difference"],
            x_label="Number of features per model",
            y_label=r"$\| \bar{h}^{(LN)}_\infty - h^{(LN)}_\infty \|_{L^1}$",
            vline_x=self.experiment_parameters.num_training_samples,
            vline_text=r"$N = \mathrm{Total\,Parameters}$",
            plot_legend=False,
        )

        # Save the plot
        save_figure(plt, self.experiment_dir / "average_difference_vs_num_features.pdf")

        # Plot the data generating function if dimension <= 2
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


if __name__ == "__main__":
    app()
