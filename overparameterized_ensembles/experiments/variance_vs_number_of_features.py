import typer
import torch
import numpy as np
from pathlib import Path
from rich.progress import track
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
from overparameterized_ensembles.visualization.plots import (
    plot_graph,
)
from overparameterized_ensembles.utils.constants import (
    NUMBER_OF_MODELS_FOR_VARIANCE,
    DEFAULT_RANDOM_SEED,
)
from overparameterized_ensembles.utils.utils import (
    to_numpy,
    save_figure,
)
from overparameterized_ensembles.visualization.data_visualization import (
    collect_additional_points,
)

app = typer.Typer()


@app.command()
def variance_vs_number_of_features(
    num_features_per_model: int = typer.Option(15, help="Number of features per model"),
    max_num_models: int = typer.Option(50, help="Maximum number of models"),
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
    number_points_to_test: int = typer.Option(5, help="Number of points to test"),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the variance vs. number of features experiment
    """

    # Set the random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create experiment parameters
    experiment_parameters = RandomFeatureModelExperimentParameters(
        num_features_per_model=num_features_per_model,
        max_num_models=max_num_models,
        num_training_samples=num_training_samples,
        data_dimension=data_dimension,
        data_generating_function_name=data_generating_function_name,
        noise_level=noise_level,
        kernel=kernel,
        activation_function=activation_function,
        random_weights_distribution=random_weights_distribution,
        case_type=case_type,
        number_simulations_per_size=number_points_to_test,
    )

    # Create the experiment
    experiment = VarianceVsNumberOfFeaturesExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=6,
    )

    # Run and visualize the experiment
    experiment.run_and_visualize_experiment()


class VarianceVsNumberOfFeaturesExperiment(Experiment):
    def _get_experiment_dir_name(self):
        return "variance_vs_number_of_features"

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
        )

        # Initialize the list of variances for each number of features and each point
        variances = []
        for index_point in range(
            self.experiment_parameters.number_simulations_per_size
        ):
            variances.append([])

        model_sizes = range(
            self.experiment_parameters.num_features_per_model,
            self.experiment_parameters.max_num_models
            * self.experiment_parameters.num_features_per_model
            + 1,
            self.experiment_parameters.num_features_per_model,
        )

        for num_features in track(model_sizes, description="Training models..."):
            if self.experiment_parameters.case_type == "subexponential":
                # Initialize and train the ensemble model
                ensemble_model = (
                    EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                        self.experiment_parameters.data_dimension,
                        num_features,
                        random_weights_distribution,
                        self.experiment_parameters.activation_function,
                        NUMBER_OF_MODELS_FOR_VARIANCE,
                    )
                )
            elif self.experiment_parameters.case_type == "gaussian":
                # Initialize and train the ensemble model
                ensemble_model = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                    self.experiment_parameters.data_dimension,
                    num_features,
                    0.0,
                    self.experiment_parameters.kernel,
                    X_train,
                    X_star,
                    NUMBER_OF_MODELS_FOR_VARIANCE,
                )
            ensemble_model.learn_theta(X_train, y_train)
            # Calculate the variance for each point
            for index_point in range(
                self.experiment_parameters.number_simulations_per_size
            ):
                variances[index_point].append(
                    to_numpy(ensemble_model.variance(X_star[index_point]))
                )

        # Save the results
        results = {
            "data_generating_function": data_generating_function,
            "model_sizes": list(model_sizes),
            "variances": variances,
            "X_train": X_train,
            "y_train": y_train,
            "X_star": X_star,
        }
        return results

    def _visualize_results(self, results):
        # Plot the variance vs. number of features
        plt = plot_graph(
            results["model_sizes"],
            results["variances"],
            labels=[
                f"Point {i + 1}"
                for i in range(self.experiment_parameters.number_simulations_per_size)
            ],
            x_label="Number of features of random feature models",
            y_label=r"$\operatorname{Var}_{\pi_{D}} [ h^{(LN)}_S(\cdot) ]$",
            loglog=True,
            decay_slope=-1,
            linestyles=["-", "-", "-", "-", "-"],
        )

        # Save the plot
        save_figure(plt, self.experiment_dir / "variance_vs_number_of_features.pdf")

        # Plot the data generating function
        data_generating_function = results["data_generating_function"]
        if self.experiment_parameters.data_dimension <= 2:
            # Get the additional points
            additional_points = collect_additional_points(
                [results["X_train"], results["X_star"]],
                [results["y_train"], torch.zeros_like(results["X_star"][:, 0])],
                self.experiment_parameters.data_dimension,
            )

            # Plot the data generating function
            plt = data_generating_function.plot_function(
                additional_points=additional_points,
                additional_points_labels=["Training Data", "Test Data"],
            )

            # Save the plot
            save_figure(plt, self.experiment_dir / "data_generating_function.pdf")


if __name__ == "__main__":
    app()
