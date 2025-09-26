import typer
from pathlib import Path
from rich.progress import track
import torch
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
from overparameterized_ensembles.models.random_feature_models import (
    RandomFeatureModel,
    RandomFeatureModelGaussianUniversality,
)
from overparameterized_ensembles.models.ensembles import (
    EnsembleRandomFeatureModel,
)
from overparameterized_ensembles.visualization.plots import (
    plot_graph,
)
from overparameterized_ensembles.utils.utils import (
    to_numpy,
    save_figure,
)
from overparameterized_ensembles.visualization.data_visualization import (
    collect_additional_points,
)
from overparameterized_ensembles.models.kernel_models import (
    KernelRegressor,
)
from overparameterized_ensembles.utils.constants import (
    DEFAULT_RANDOM_SEED,
)

app = typer.Typer()


@app.command()
def generalization_error_decay(
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
    number_simulations_per_size: int = typer.Option(
        5000, help="Number of simulations per size"
    ),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the generalization error decay experiment
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Initialize experiment parameters
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
        number_simulations_per_size=number_simulations_per_size,
    )

    # Create and run the experiment
    experiment = GeneralizationErrorDecayExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=4,
    )
    experiment.run_and_visualize_experiment()


class GeneralizationErrorDecayExperiment(Experiment):
    def _get_experiment_dir_name(self):
        return "generalization_error_decay"

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
            number_test_samples=1000,
        )

        # Train and evaluate models
        mse_test_list_single_model = []
        mse_test_list_ensemble = []
        mse_test_list_kernel_model = []

        number_models = range(1, self.experiment_parameters.max_num_models + 1, 2)

        for num_models in track(number_models, description="Training models..."):
            batch_mse_test_single_model = []
            batch_mse_test_ensemble = []

            # Train a kernel model
            kernel_model = KernelRegressor(
                kernel=self.experiment_parameters.kernel,
                ridge=0.0,
            )
            kernel_model.fit(X_train, y_train)
            mse_test_infinite_single_model = kernel_model.loss(X_test, y_test)

            for _ in range(self.experiment_parameters.number_simulations_per_size):
                if self.experiment_parameters.case_type == "subexponential":
                    # Train and evaluate a single model
                    (
                        single_model,
                        mse_train_single_model,
                        mse_test_single_model,
                    ) = RandomFeatureModel.create_train_and_calculate_loss(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features=self.experiment_parameters.num_features_per_model
                        * num_models,
                        random_weights_distribution=random_weights_distribution,
                        activation_function_name=self.experiment_parameters.activation_function,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        ridge=0.0,
                    )

                    # Train and evaluate an ensemble of models
                    ensemble = EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features_per_model=self.experiment_parameters.num_features_per_model,
                        random_weights_distribution=random_weights_distribution,
                        activation_function_name=self.experiment_parameters.activation_function,
                        num_models=num_models,
                        ridge=0.0,
                    )
                    ensemble.learn_theta(X_train, y_train)
                    mse_test_ensemble = ensemble.loss(X_test, y_test)
                elif self.experiment_parameters.case_type == "gaussian":
                    # Train and evaluate a single model
                    (
                        single_model,
                        mse_train_single_model,
                        mse_test_single_model,
                    ) = RandomFeatureModelGaussianUniversality.create_train_and_calculate_loss(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features=self.experiment_parameters.num_features_per_model
                        * num_models,
                        ridge=0.0,
                        kernel_function_name=self.experiment_parameters.kernel,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )

                    # Train and evaluate an ensemble of models
                    ensemble = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features_per_model=self.experiment_parameters.num_features_per_model,
                        ridge=0.0,
                        kernel_function_name=self.experiment_parameters.kernel,
                        X_train=X_train,
                        x_test=X_test,
                        num_models=num_models,
                    )
                    ensemble.learn_theta(X_train, y_train)
                    mse_test_ensemble = ensemble.loss(X_test, y_test)
                batch_mse_test_single_model.append(to_numpy(mse_test_single_model))
                batch_mse_test_ensemble.append(to_numpy(mse_test_ensemble))
            mse_test_list_single_model.append(np.average(batch_mse_test_single_model))
            mse_test_list_ensemble.append(np.average(batch_mse_test_ensemble))
            mse_test_list_kernel_model.append(to_numpy(mse_test_infinite_single_model))

        # Calculate the total number of features
        model_sizes = [
            self.experiment_parameters.num_features_per_model * num_models
            for num_models in number_models
        ]

        return {
            "data_generating_function": data_generating_function,
            "model_sizes": model_sizes,
            "mse_test_single_model": mse_test_list_single_model,
            "mse_test_ensemble": mse_test_list_ensemble,
            "mse_test_kernel_model": mse_test_list_kernel_model,
            "X_train": X_train,
            "y_train": y_train,
        }

    def _visualize_results(self, results):
        # Plot the generalization error of the single model and ensemble
        plt = plot_graph(
            x_values=results["model_sizes"],
            y_values=[
                results["mse_test_single_model"],
                results["mse_test_kernel_model"],
                results["mse_test_ensemble"],
            ],
            labels=["Single model", "Kernel model", "Ensemble"],
            x_label="Total number of features used",
            y_label="Generalization error",
            linestyles=["--", "-", "-."],
        )

        save_figure(plt, self.experiment_dir / "generalization_error.pdf")

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


if __name__ == "__main__":
    app()
