import typer
from pathlib import Path
from rich.progress import track
import numpy as np
import torch
from overparameterized_ensembles.experiments.experiments import (
    RandomFeatureModelExperimentParameters,
    Experiment,
)
from overparameterized_ensembles.models.ensembles import (
    EnsembleRandomFeatureModel,
)
from overparameterized_ensembles.models.kernel_models import (
    KernelRegressor,
)
from overparameterized_ensembles.models.model_utils import (
    initialize_random_weights_distribution,
)
from overparameterized_ensembles.data_generation.data_generation import (
    generate_data,
)
from overparameterized_ensembles.visualization.data_visualization import (
    collect_additional_points,
    plot2d,
    plot3d,
)
from overparameterized_ensembles.utils.utils import (
    save_figure,
)
from overparameterized_ensembles.utils.constants import (
    DEFAULT_RANDOM_SEED,
)

app = typer.Typer()


@app.command()
def visualize_models(
    num_features_per_model: int = typer.Option(15, help="Number of features per model"),
    number_ensemble_members: int = typer.Option(1, help="Number of ensemble models"),
    num_training_samples: int = typer.Option(4, help="Number of samples to use"),
    data_dimension: int = typer.Option(1, help="Dimension of the data"),
    data_generating_function_name: str = typer.Option(
        "sinusoidal", help="Data generation function"
    ),
    noise_level: float = typer.Option(0.05, help="Noise level"),
    activation_function: str = typer.Option("relu", help="Activation function"),
    random_weights_distribution: str = typer.Option(
        "normal", help="Random weights distribution"
    ),
    kernel: str = typer.Option("N/A", help="Kernel"),
    case_type: str = typer.Option("subexponential", help="Case type"),
    number_simulations_per_size: int = typer.Option(
        1, help="Number of simulations per size"
    ),
    plot_kernel_model: bool = typer.Option(False, help="Plot the kernel model"),
    ridge: float = typer.Option(0.0, help="Ridge parameter"),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the visualize models experiment
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

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
        number_simulations_per_size=number_simulations_per_size,
    )

    # Create the experiment
    experiment = VisualizeModelsExperiment(
        results_path=Path("results"),
        experiment_parameters=experiment_parameters,
        experiment_number=8,
        number_ensemble_members=number_ensemble_members,
        ridge=ridge,
        plot_kernel_model=plot_kernel_model,
    )

    # Run and visualize the experiment
    experiment.run_and_visualize_experiment()


class VisualizeModelsExperiment(Experiment):
    def __init__(
        self,
        results_path: Path,
        experiment_parameters: RandomFeatureModelExperimentParameters,
        experiment_number: int,
        number_ensemble_members: int = 1,
        ridge: float = 0.0,
        plot_kernel_model: bool = False,
    ):
        super(VisualizeModelsExperiment, self).__init__(
            results_path, experiment_parameters, experiment_number
        )
        self.number_ensemble_members = number_ensemble_members
        self.ridge = ridge
        self.plot_kernel_model = plot_kernel_model

    def _get_experiment_dir_name(self):
        return "visualize_models"

    def _run_experiment(self):
        random_weights_distribution = initialize_random_weights_distribution(
            self.experiment_parameters.random_weights_distribution,
            self.experiment_parameters.data_dimension + 1,
        )

        X_train, y_train, X_test, y_test, data_generating_function = generate_data(
            self.experiment_parameters.data_generating_function_name,
            self.experiment_parameters.num_training_samples,
            self.experiment_parameters.data_dimension,
            self.experiment_parameters.noise_level,
            test_samples_as_grid=True,
            number_test_samples=100,
        )

        models_list = []

        for i in track(
            range(1, self.experiment_parameters.number_simulations_per_size + 1)
        ):
            if self.experiment_parameters.case_type == "subexponential":
                model = EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                    data_dimension=self.experiment_parameters.data_dimension,
                    num_features_per_model=self.experiment_parameters.num_features_per_model,
                    random_weights_distribution=random_weights_distribution,
                    activation_function_name=self.experiment_parameters.activation_function,
                    num_models=self.number_ensemble_members,
                    ridge=self.ridge,
                )
                model.learn_theta(X_train, y_train)
            elif self.experiment_parameters.case_type == "gaussian":
                model = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                    data_dimension=self.experiment_parameters.data_dimension,
                    num_features_per_model=self.experiment_parameters.num_features_per_model,
                    ridge=self.ridge,
                    kernel_function_name=self.experiment_parameters.kernel,
                    X_train=X_train,
                    x_test=X_test,
                    num_models=self.number_ensemble_members,
                )
                model.learn_theta(X_train, y_train)
            else:
                raise ValueError(
                    f"Invalid case_type: {self.experiment_parameters.case_type}"
                )

            models_list.append(model.forward)

        kernel_model = None
        if self.plot_kernel_model:
            kernel_model = KernelRegressor(
                kernel=self.experiment_parameters.kernel,
                ridge=self.ridge,
            )
            kernel_model.fit(X_train, y_train)

        return {
            "data_generating_function": data_generating_function,
            "models_list": models_list,
            "kernel_model": kernel_model,
            "X_train": X_train,
            "y_train": y_train,
        }

    def _visualize_results(self, results):
        data_generating_function = results["data_generating_function"]
        if self.experiment_parameters.data_dimension <= 2:
            additional_points = collect_additional_points(
                [results["X_train"]],
                [results["y_train"]],
                self.experiment_parameters.data_dimension,
            )

            additional_functions = [results["models_list"]]
            additional_functions_labels = (
                ["RF Models"]
                if self.number_ensemble_members == 1
                else ["Ensemble Model"]
            )

            if self.plot_kernel_model:
                additional_functions.append([results["kernel_model"]])
                additional_functions_labels.append("Kernel Model")

            alpha = (
                [0.25, 1.0]
                if self.number_ensemble_members == 1
                else [1.0] * len(additional_functions)
            )

            if self.experiment_parameters.data_dimension == 1:
                plt = plot2d(
                    data_generating_function.evaluate,
                    f_label="Data Generating Function",
                    input_range=data_generating_function.get_input_range(),
                    additional_points=additional_points,
                    additional_points_labels=["Training Data"],
                    additional_functions=additional_functions,
                    additional_functions_labels=additional_functions_labels,
                    plot_main_function=False,
                    alpha=alpha,
                )
            elif self.experiment_parameters.data_dimension == 2:
                plt = plot3d(
                    data_generating_function.evaluate,
                    f_label="Data Generating Function",
                    input_range=data_generating_function.get_input_range(),
                    additional_points=additional_points,
                    additional_points_labels=["Training Data"],
                    additional_functions=additional_functions,
                    additional_functions_labels=additional_functions_labels,
                )

            save_figure(plt, self.experiment_dir / "models_visualization.pdf")
