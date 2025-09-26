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
from overparameterized_ensembles.utils.constants import (
    NUMBER_TEST_SAMPLES,
    DEFAULT_RANDOM_SEED,
)
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def pointwise_difference_vs_num_features(
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
    use_loglog: bool = typer.Option(False, help="Use log-log scale for plots"),
    output_dir: Path = typer.Option(
        Path("results"), help="Output directory for results"
    ),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Run the pointwise difference vs number of features experiment
    """
    typer.echo("Running pointwise difference vs number of features experiment...")

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
    experiment = PointwiseDifferenceVsNumFeaturesExperiment(
        results_path=output_dir,
        experiment_parameters=experiment_parameters,
        experiment_number=1,
        num_features_start=num_features_start,
        num_features_end=num_features_end,
        num_features_step=num_features_step,
        number_extra_test_samples=number_extra_test_samples,
        ridge=ridge,
        use_loglog=use_loglog,
    )
    experiment.run_and_visualize_experiment()


class PointwiseDifferenceVsNumFeaturesExperiment(Experiment):
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
        use_loglog: bool = False,
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
        self.use_loglog = use_loglog

    def _get_experiment_dir_name(self):
        return "pointwise_difference_vs_num_features"

    def _run_experiment(self):
        # Initialize random weights
        random_weights_distribution = initialize_random_weights_distribution(
            self.experiment_parameters.random_weights_distribution,
            self.experiment_parameters.data_dimension + 1,
        )

        # Generate data with specified number of test samples
        X_train, y_train, X_test, y_test, _ = generate_data(
            self.experiment_parameters.data_generating_function_name,
            self.experiment_parameters.num_training_samples,
            self.experiment_parameters.data_dimension,
            self.experiment_parameters.noise_level,
            number_test_samples=self.number_extra_test_samples,
        )

        # Initialize the num_features values
        num_features_values = list(
            range(
                self.num_features_start,
                self.num_features_end + 1,
                self.num_features_step,
            )
        )

        # Create the infinite-width reference model (kernel model)
        ref_model = KernelRegressor(
            kernel=self.experiment_parameters.kernel,
            ridge=self.ridge,
        )
        ref_model.fit(X_train, y_train)
        ref_preds = ref_model.forward(X_test)

        results = []
        for num_features in track(
            num_features_values, description="Running experiment..."
        ):
            # Create and train ensemble
            if self.experiment_parameters.case_type == "subexponential":
                ensemble = EnsembleRandomFeatureModel.create_ensemble_random_feature_models(
                    data_dimension=self.experiment_parameters.data_dimension,
                    num_features_per_model=num_features,
                    random_weights_distribution=random_weights_distribution,
                    activation_function_name=self.experiment_parameters.activation_function,
                    num_models=self.experiment_parameters.max_num_models,
                    ridge=self.ridge,
                )
            elif self.experiment_parameters.case_type == "gaussian":
                ensemble = EnsembleRandomFeatureModel.create_ensemble_gaussian_universality_models(
                    data_dimension=self.experiment_parameters.data_dimension,
                    num_features_per_model=num_features,
                    ridge=self.ridge,
                    kernel_function_name=self.experiment_parameters.kernel,
                    X_train=X_train,
                    x_test=X_test,
                    num_models=self.experiment_parameters.max_num_models,
                )
            else:
                raise ValueError(
                    f"Unknown case_type: {self.experiment_parameters.case_type}"
                )

            ensemble.learn_theta(X_train, y_train)
            ensemble_preds = ensemble.forward(X_test)

            results.append(
                {
                    "num_features": num_features,
                    "ensemble_size": self.experiment_parameters.max_num_models,
                    "ensemble_preds": ensemble_preds.detach().cpu().numpy(),
                    "ref_preds": ref_preds.detach().cpu().numpy(),
                    "y_test": y_test,
                }
            )

        return {"results": results}

    def _visualize_results(self, results):
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

        for result in results["results"]:
            width_dir = plots_dir / f"width_{result['num_features']}"
            width_dir.mkdir(exist_ok=True)

            # Adjust n_points to be no larger than the available number of points
            n_points = min(1000, len(result["ensemble_preds"]))
            typer.echo(f"Using {n_points} points for plotting")
            idx = np.random.choice(
                len(result["ensemble_preds"]), n_points, replace=False
            )

            self._create_comparison_plots(
                pred1=result["ensemble_preds"],
                pred2=result["ref_preds"],
                y_test=result["y_test"],
                idx=idx,
                name1=f"Ensemble (width={result['num_features']}, M={result['ensemble_size']})",
                name2="Infinite-width",
                output_dir=width_dir,
                M=result["ensemble_size"],
            )

    def _create_comparison_plots(
        self, pred1, pred2, y_test, idx, name1, name2, output_dir, M
    ):
        """Helper function to create all comparison plots for a pair of models."""
        output_dir.mkdir(exist_ok=True, parents=True)

        # Convert tensors to numpy arrays if needed
        pred1 = pred1.detach().cpu().numpy() if torch.is_tensor(pred1) else pred1
        pred2 = pred2.detach().cpu().numpy() if torch.is_tensor(pred2) else pred2
        y_test = y_test.detach().cpu().numpy() if torch.is_tensor(y_test) else y_test

        # Turn pred1 and pred2 into 1D arrays
        pred1 = pred1.ravel()
        pred2 = pred2.ravel()
        y_test = y_test.ravel()

        # 1) Scatter of predictions
        corr_preds = np.corrcoef(pred1.ravel(), pred2.ravel())[0, 1]
        plt.figure(figsize=(5, 5))
        plt.scatter(pred1[idx], pred2[idx], alpha=0.3, s=2)
        if self.use_loglog:
            plt.xscale("log")
            plt.yscale("log")
        plt.xlabel(f"{name1} Predictions")
        plt.ylabel(f"{name2} Predictions")
        plt.title(f"M={M}: corr(preds)={corr_preds:.3f}")
        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"scatter_predictions{'_loglog' if self.use_loglog else ''}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # 2) Scatter of squared errors
        err1 = (pred1 - y_test.ravel()) ** 2
        err2 = (pred2 - y_test.ravel()) ** 2
        corr_errs = np.corrcoef(err1, err2)[0, 1]
        plt.figure(figsize=(5, 5))
        plt.scatter(err1[idx], err2[idx], alpha=0.3, s=2)
        if self.use_loglog:
            plt.xscale("log")
            plt.yscale("log")
        plt.xlabel(f"{name1} Squared Error")
        plt.ylabel(f"{name2} Squared Error")
        plt.title(f"M={M}: corr(errors)={corr_errs:.3f}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"scatter_errors{'_loglog' if self.use_loglog else ''}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # 3) Scatter of residuals
        residuals1 = pred1 - y_test.ravel()
        residuals2 = pred2 - y_test.ravel()
        corr_residuals = np.corrcoef(residuals1, residuals2)[0, 1]
        plt.figure(figsize=(5, 5))
        plt.scatter(residuals1[idx], residuals2[idx], alpha=0.3, s=2)
        plt.xlabel(f"{name1} Residuals")
        plt.ylabel(f"{name2} Residuals")
        plt.title(f"M={M}: corr(residuals)={corr_residuals:.3f}")
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_residuals.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 4) Distribution of differences and residuals
        pred_diff = pred2 - pred1
        plt.figure(figsize=(8, 5))

        # Plot distributions, falling back to histograms if KDE fails
        for data, label, color in [
            (pred_diff.ravel(), f"{name2} - {name1} Predictions", "blue"),
            (residuals2.ravel(), f"{name2} Residuals", "red"),
            (residuals1.ravel(), f"{name1} Residuals", "green"),
        ]:
            plt.hist(
                data,
                bins=100,
                density=True,
                alpha=0.5,
                label=label,
                color=color,
                histtype="step",
            )

            # Add vertical line for mean only for prediction differences
            if label == f"{name2} - {name1} Predictions":
                mean_val = np.mean(data)
                plt.axvline(
                    mean_val,
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    label=f"Mean Difference: {mean_val:.3f}",
                )

        plt.xlabel("Difference / Residual Value")
        plt.ylabel("Density")
        plt.title(f"M={M}: Distribution of Differences and Residuals")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "distribution_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    app()
