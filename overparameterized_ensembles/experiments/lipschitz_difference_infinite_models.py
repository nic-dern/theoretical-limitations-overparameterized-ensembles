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
    plot_multiple_lines,
)
from overparameterized_ensembles.matrices_and_kernels.kernel_calculations import (
    get_effective_ridge_implicit_regularization,
)
from overparameterized_ensembles.utils.utils import (
    to_numpy,
    save_figure,
)
from overparameterized_ensembles.utils.constants import (
    NUMBER_TEST_SAMPLES,
    ZERO_REGULARIZATION,
    DEFAULT_RANDOM_SEED,
)

app = typer.Typer()


@app.command()
def lipschitz_difference_infinite_models(
    num_features_per_model: int = typer.Option(15, help="Number of features per model"),
    max_num_models: int = typer.Option(
        2000, help="Defines what means 'infinite' number of models"
    ),
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
    ridge_start: float = typer.Option(
        ZERO_REGULARIZATION, help="Start value for the ridge parameter"
    ),
    ridge_step: float = typer.Option(0.1, help="Step size for the ridge parameter"),
    ridge_end: float = typer.Option(1, help="End value for the ridge parameter"),
    number_extra_test_samples: int = typer.Option(
        NUMBER_TEST_SAMPLES / 2, help="Number of extra test samples"
    ),
    comparison_mode: str = typer.Option(
        "single_model", help="Comparison mode: 'ensemble' or 'single_model'"
    ),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the Lipschitz difference infinite models experiment.
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

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
        number_simulations_per_size=-1,
    )

    # Run experiment
    experiment = LipschitzDifferenceInfiniteModelsExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=5,
        ridge_start=ridge_start,
        ridge_end=ridge_end,
        ridge_step=ridge_step,
        number_extra_test_samples=number_extra_test_samples,
        comparison_mode=comparison_mode,
    )
    experiment.run_and_visualize_experiment()


class LipschitzDifferenceInfiniteModelsExperiment(Experiment):
    def __init__(
        self,
        results_path: Path,
        experiment_parameters: RandomFeatureModelExperimentParameters,
        experiment_number: int,
        ridge_start: float,
        ridge_end: float,
        ridge_step: float,
        number_extra_test_samples: int = NUMBER_TEST_SAMPLES,
        comparison_mode: str = "single_model",
    ):
        super().__init__(
            results_path,
            experiment_parameters,
            experiment_number,
        )
        self.ridge_start = ridge_start
        self.ridge_end = ridge_end
        self.ridge_step = ridge_step
        self.number_extra_test_samples = number_extra_test_samples
        self.comparison_mode = comparison_mode

    def _get_experiment_dir_name(self):
        return "lipschitz_difference_infinite_models"

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
        # Take the first k samples of the test set for an extra plot showing the difference in the predictons on concrete samples
        X_test_extra = X_test[:number_extra_samples]

        # Initialize the ridge parameter values
        ridge_values = list(
            np.arange(
                self.ridge_start,
                self.ridge_end,
                self.ridge_step,
            )
        )
        effective_ridge_values = []

        predictions_varied = []
        predictions_fixed = []
        mse_test_varied_list = []
        mse_test_fixed_list = []

        for ridge_value in track(ridge_values, description="Running experiment..."):
            effective_ridge = get_effective_ridge_implicit_regularization(
                kernel=self.experiment_parameters.kernel,
                X=X_train,
                ridge=ridge_value,
                num_training_samples=self.experiment_parameters.num_training_samples,
                num_features=self.experiment_parameters.num_features_per_model,
            )
            effective_ridge_values.append(effective_ridge)
            if self.comparison_mode == "single_model":
                # Infinite single model with varying ridge
                kernel_model_varied = KernelRegressor(
                    kernel=self.experiment_parameters.kernel,
                    ridge=ridge_value,
                )
                kernel_model_varied.fit(X_train, y_train)
                mse_test_varied = kernel_model_varied.loss(X_test, y_test)
                predictions_varied.append(kernel_model_varied.forward(X_test_extra))

                # Infinite single model with ridge=0.0
                kernel_model_fixed = KernelRegressor(
                    kernel=self.experiment_parameters.kernel, ridge=self.ridge_start
                )
                kernel_model_fixed.fit(X_train, y_train)
                mse_test_fixed = kernel_model_fixed.loss(X_test, y_test)
                predictions_fixed.append(kernel_model_fixed.forward(X_test_extra))

            elif self.comparison_mode == "ensemble":
                if self.experiment_parameters.case_type == "subexponential":
                    # Infinite ensemble with varying ridge
                    ensemble_varied = EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features_per_model=self.experiment_parameters.num_features_per_model,
                        random_weights_distribution=random_weights_distribution,
                        activation_function_name=self.experiment_parameters.activation_function,
                        num_models=self.experiment_parameters.max_num_models,
                        ridge=ridge_value,
                    )
                    ensemble_varied.learn_theta(X_train, y_train)
                    mse_test_varied = ensemble_varied.loss(X_test, y_test)
                    predictions_varied.append(ensemble_varied.forward(X_test_extra))

                    # Infinite ensemble with ridge=0.0
                    ensemble_fixed = EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features_per_model=self.experiment_parameters.num_features_per_model,
                        random_weights_distribution=random_weights_distribution,
                        activation_function_name=self.experiment_parameters.activation_function,
                        num_models=self.experiment_parameters.max_num_models,
                        ridge=self.ridge_start,
                    )
                    ensemble_fixed.learn_theta(X_train, y_train)
                    mse_test_fixed = ensemble_fixed.loss(X_test, y_test)
                    predictions_fixed.append(ensemble_fixed.forward(X_test_extra))

                elif self.experiment_parameters.case_type == "gaussian":
                    # Infinite ensemble with varying ridge
                    ensemble_varied = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features_per_model=self.experiment_parameters.num_features_per_model,
                        ridge=ridge_value,
                        kernel_function_name=self.experiment_parameters.kernel,
                        X_train=X_train,
                        x_test=X_test,
                        num_models=self.experiment_parameters.max_num_models,
                    )
                    ensemble_varied.learn_theta(X_train, y_train)
                    mse_test_varied = ensemble_varied.loss(X_test, y_test)
                    predictions_varied.append(ensemble_varied.forward(X_test_extra))

                    # Infinite ensemble with ridge=0.0
                    ensemble_fixed = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                        data_dimension=self.experiment_parameters.data_dimension,
                        num_features_per_model=self.experiment_parameters.num_features_per_model,
                        ridge=self.ridge_start,
                        kernel_function_name=self.experiment_parameters.kernel,
                        X_train=X_train,
                        x_test=X_test,
                        num_models=self.experiment_parameters.max_num_models,
                    )
                    ensemble_fixed.learn_theta(X_train, y_train)
                    mse_test_fixed = ensemble_fixed.loss(X_test, y_test)
                    predictions_fixed.append(ensemble_fixed.forward(X_test_extra))

            # Store the MSEs
            mse_test_varied_list.append(to_numpy(mse_test_varied))
            mse_test_fixed_list.append(to_numpy(mse_test_fixed))

        # Convert the predictions to numpy arrays
        predictions_varied = np.array(predictions_varied)
        predictions_fixed = np.array(predictions_fixed)

        # Package the results
        results = {
            "ridge_values": ridge_values,
            "mse_test_varied": mse_test_varied_list,
            "mse_test_fixed": mse_test_fixed_list,
            "effective_ridge_values": effective_ridge_values,
            "X_train": X_train,
            "y_train": y_train,
            "data_generating_function": data_generating_function,
            "X_test_extra": X_test_extra,
            "predictions_varied": predictions_varied,
            "predictions_fixed": predictions_fixed,
        }
        return results

    def _visualize_results(self, results):
        # Plot the effective ridge vs. ridge parameter
        plt = plot_graph(
            x_values=results["ridge_values"],
            y_values=[results["effective_ridge_values"]],
            labels=["Effective ridge"],
            x_label=r"$\lambda$",
            y_label="Effective ridge",
        )

        # Save the plot
        save_figure(plt, self.experiment_dir / "effective_ridge_vs_ridge_parameter.pdf")

        # Plot the prediction errors
        plt = plot_graph(
            x_values=results["ridge_values"],
            y_values=[
                results["mse_test_varied"],
                results["mse_test_fixed"],
            ],
            labels=[
                f"{self.comparison_mode.capitalize()} with varying ridge",
                f"{self.comparison_mode.capitalize()} with ridge=0.0",
            ],
            x_label=r"$\lambda$",
            y_label="Prediction error",
        )

        # Save the plot
        save_figure(
            plt, self.experiment_dir / "prediction_error_vs_ridge_parameter.pdf"
        )

        # Calculate the mean absolute differences
        differences = results["predictions_varied"] - results["predictions_fixed"]
        mean_absolute_differences = np.mean(np.abs(differences), axis=1)

        # Plot the mean absolute differences
        plt = plot_graph(
            x_values=results["ridge_values"],
            y_values=[mean_absolute_differences],
            labels=["Mean absolute difference"],
            x_label=r"$\lambda$",
            y_label="Mean absolute difference",
        )

        # Save the plot
        save_figure(plt, self.experiment_dir / "mean_absolute_difference_vs_ridge.pdf")

        # Plot the differences for the values in the test set
        differences = []
        typer.echo(f"Predictions varied shape: {results['predictions_varied'].shape}")
        typer.echo(f"Predictions fixed shape: {results['predictions_fixed'].shape}")

        for i in range(len(results["predictions_varied"])):
            differences.append(
                np.array(results["predictions_varied"][i])
                - np.array(results["predictions_fixed"][i])
            )

        plt = plot_multiple_lines(
            x_values=results["ridge_values"],
            y_values=[differences],
            x_label=r"$\lambda$",
            y_label=r"$\left|\bar{h}^{(RR)}_{\infty, \lambda}(\cdot) - \bar{h}^{(LS)}_{\infty}(\cdot)\right|$"
            if self.comparison_mode == "ensemble"
            else r"$\left|h^{(RR)}_{\infty, \lambda}(\cdot) - h^{(LS)}_{\infty}(\cdot)\right|$",
        )

        # Save the plot
        save_figure(plt, self.experiment_dir / "difference_between_predictions.pdf")


if __name__ == "__main__":
    app()
