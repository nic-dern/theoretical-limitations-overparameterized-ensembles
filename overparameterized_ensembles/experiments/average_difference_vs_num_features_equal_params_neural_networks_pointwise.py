import typer
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from overparameterized_ensembles.models.neural_network_models import NeuralNetworkModel
from overparameterized_ensembles.data_generation.neural_network_data_generation import (
    CaliforniaHousingDataModule,
)
import matplotlib.pyplot as plt
from overparameterized_ensembles.utils.constants import COLORS

app = typer.Typer()


def param_count_3layer(width: int) -> int:
    """
    Compute parameter count for 3-layer MLP with architecture [8 -> W -> W -> W -> 1]
    Args:
        width: Width of hidden layers (W)
    Returns:
        Total parameter count = 2*W^2 + 12*W + 1
    """
    return 2 * width * width + 12 * width + 1


def find_width_for_params(target_params: int) -> int:
    """
    Find width W such that param_count_3layer(W) is approximately target_params
    Args:
        target_params: Target parameter count
    Returns:
        Width that gives approximately target_params parameters
    """
    w = 1
    while True:
        pc = param_count_3layer(w)
        if pc >= target_params:
            return w
        w += 1
        if w > 3000:
            typer.echo(
                f"Warning: Could not find width up to 3000 for target_params={target_params}"
            )
            return 3000


def load_checkpoint(model_folder: Path, use_best: bool) -> str:
    """
    Load either the best or last checkpoint in the given folder.
    """
    if use_best:
        ckpts = list(model_folder.glob("model-epoch=*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {model_folder}")
        best_ckpt = min(
            ckpts, key=lambda x: float(str(x).split("val_loss=")[1].split(".ckpt")[0])
        )
        return str(best_ckpt)
    else:
        ckpts = list(model_folder.glob("last.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No last checkpoint found in {model_folder}")
        return str(ckpts[0])


def predict_dataset(model: NeuralNetworkModel, dataloader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(model.model[0].weight.device)
            p = model(x)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("nn_checkpoints_average_difference_vs_num_features_equal_params"),
        help="Root folder containing checkpoints",
    ),
    batch_size: int = typer.Option(256, help="Batch size used for test inference"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    learning_rate: float = typer.Option(
        0.01, help="Learning rate used during training"
    ),
    optimizer: str = typer.Option("sgd", help="Optimizer used during training"),
    momentum: float = typer.Option(0.9, help="Momentum used during training"),
    dtype: str = typer.Option("float32", help="Data type used during training"),
    use_best_checkpoint: bool = typer.Option(
        False, help="Use best checkpoint instead of last"
    ),
    reference_width: int = typer.Option(
        320, help="Reference width for total parameter count"
    ),
    use_loglog: bool = typer.Option(False, help="Use log-log scale for plots"),
    num_comparison_instances: int = typer.Option(
        3, help="Number of different ensemble/single model pairs to compare"
    ),
):
    """
    Compare pointwise predictions of ensemble models to single networks with equivalent parameter counts.
    Each ensemble has approximately the same total parameters as a single model with reference_width.
    """
    if dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Invalid data type: {dtype}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pl.seed_everything(random_seed)

    data_module = CaliforniaHousingDataModule(
        num_training_samples=12000,
        num_validation_samples=3000,
        num_test_samples=5000,
        batch_size=batch_size,
        dtype=torch.float32 if dtype == "float32" else torch.float64,
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Get a copy of y_test from the test loader
    y_test = []
    for x, y in test_loader:
        y_test.append(y.numpy())
    y_test = np.concatenate(y_test, axis=0).reshape(-1, 1)

    # Calculate reference parameter count
    reference_params = param_count_3layer(reference_width)
    typer.echo(f"Reference model: width={reference_width}, params={reference_params}")

    # Create base plots directory
    plots_dir = output_dir / "pointwise_plots"
    plots_dir.mkdir(exist_ok=True)

    # Process models of different widths
    widths = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

    for width in widths:
        # Calculate ensemble size to match reference parameter count
        single_params = param_count_3layer(width)
        ensemble_size = (reference_params + single_params - 1) // single_params

        # Calculate actual total parameters
        ensemble_params = ensemble_size * single_params
        single_width = find_width_for_params(ensemble_params)

        typer.echo(f"\nProcessing width={width}:")
        typer.echo(f"  Ensemble: {ensemble_size} models of width {width}")
        typer.echo(f"  Single: 1 model of width {single_width}")
        typer.echo(f"  Total params: {ensemble_params} (target: {reference_params})")

        # Create directory for this width
        width_dir = plots_dir / f"width_{width}"
        width_dir.mkdir(exist_ok=True)

        # Check if directories exist
        single_base = output_dir / f"single_equiv_width_{width}"
        ensemble_base = output_dir / f"ensemble_width_{width}"

        if not single_base.exists():
            typer.echo(f"Warning: {single_base} not found, skipping width {width}")
            continue

        if not ensemble_base.exists():
            typer.echo(f"Warning: {ensemble_base} not found, skipping width {width}")
            continue

        single_exp_folder = next(
            d
            for d in single_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        ensemble_exp_folder = next(
            d
            for d in ensemble_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Create multiple comparison instances
        for instance_idx in range(num_comparison_instances):
            instance_dir = width_dir / f"instance_{instance_idx}"
            instance_dir.mkdir(exist_ok=True)

            # Load single network model for this instance
            try:
                run_dir = next(
                    d
                    for d in single_exp_folder.iterdir()
                    if d.is_dir()
                    and not d.name.startswith(".")
                    and d.name.endswith(f"-{instance_idx}")
                )
                ckpt_single = load_checkpoint(run_dir, use_best_checkpoint)
                single_model = NeuralNetworkModel.load_from_checkpoint(
                    checkpoint_path=ckpt_single,
                    hidden_layer_sizes=[
                        data_module.X.shape[1],
                        single_width,
                        single_width,
                        single_width,
                    ],
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    momentum=momentum,
                )
                single_pred = predict_dataset(single_model, test_loader)
            except StopIteration:
                typer.echo(
                    f"Warning: Could not find single model {instance_idx} for width {width}"
                )
                continue

            # Load ensemble models for this instance
            run_dirs = sorted([d for d in ensemble_exp_folder.iterdir() if d.is_dir()])

            # Select ensemble models starting from instance_idx
            selected_run_dirs = []
            for i, run_dir in enumerate(run_dirs):
                if (
                    i % num_comparison_instances == instance_idx
                    and len(selected_run_dirs) < ensemble_size
                ):
                    selected_run_dirs.append(run_dir)

            if len(selected_run_dirs) < ensemble_size:
                # Fall back to first available models
                selected_run_dirs = run_dirs[:ensemble_size]

            ensemble_preds = []
            for run_dir in selected_run_dirs:
                ckpt_ens = load_checkpoint(run_dir, use_best_checkpoint)
                ens_model = NeuralNetworkModel.load_from_checkpoint(
                    checkpoint_path=ckpt_ens,
                    hidden_layer_sizes=[data_module.X.shape[1], width, width, width],
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    momentum=momentum,
                )
                preds = predict_dataset(ens_model, test_loader)
                ensemble_preds.append(preds)

            if len(ensemble_preds) != ensemble_size:
                typer.echo(
                    f"Warning: Expected {ensemble_size} ensemble models but found {len(ensemble_preds)}"
                )

            # Calculate average ensemble prediction
            avg_ensemble_pred = np.mean(ensemble_preds, axis=0)

            # Generate comparison plots
            n_points = 2000
            idx = np.random.choice(len(avg_ensemble_pred), n_points, replace=False)

            create_comparison_plots(
                pred1=avg_ensemble_pred,
                pred2=single_pred,
                y_test=y_test,
                idx=idx,
                name1=f"Ensemble (width={width}, M={ensemble_size})",
                name2=f"Single (width={single_width})",
                output_dir=instance_dir,
                M=ensemble_size,
                use_loglog=use_loglog,
                width=width,
                single_width=single_width,
            )


def create_comparison_plots(
    pred1,
    pred2,
    y_test,
    idx,
    name1,
    name2,
    output_dir,
    M,
    use_loglog,
    width,
    single_width,
):
    """Helper function to create all comparison plots for a pair of models."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Turn pred1 and pred2 into 1D arrays
    pred1 = pred1.ravel()
    pred2 = pred2.ravel()
    y_test = y_test.ravel()

    # 1) Scatter of predictions
    corr_preds = np.nan_to_num(np.corrcoef(pred1, pred2)[0, 1])
    plt.figure(figsize=(5, 5))
    plt.scatter(pred1[idx], pred2[idx], alpha=0.3, s=2, color=COLORS[0])
    if use_loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel(f"Ensemble (Width = {width}) Predictions")
    plt.ylabel(f"Single (Width = {single_width}) Predictions")
    plt.title(f"Correlation of Predictions: {corr_preds:.3f}")

    # Hide the right and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Add grid with light gray color and dashed style
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"scatter_predictions{'_loglog' if use_loglog else ''}.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # 2) Scatter of squared errors
    err1 = (pred1 - y_test) ** 2
    err2 = (pred2 - y_test) ** 2
    corr_errs = np.nan_to_num(np.corrcoef(err1, err2)[0, 1])
    plt.figure(figsize=(5, 5))
    plt.scatter(err1[idx], err2[idx], alpha=0.3, s=2, color=COLORS[1])
    if use_loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel(f"Ensemble (Width = {width}) Squared Error")
    plt.ylabel(f"Single (Width = {single_width}) Squared Error")
    plt.title(f"Correlation of Squared Errors: {corr_errs:.3f}")

    # Hide the right and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Add grid with light gray color and dashed style
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"scatter_errors{'_loglog' if use_loglog else ''}.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # 3) Scatter of residuals
    residuals1 = pred1 - y_test
    residuals2 = pred2 - y_test
    corr_residuals = np.nan_to_num(np.corrcoef(residuals1, residuals2)[0, 1])

    plt.figure(figsize=(5, 5))
    plt.scatter(residuals1[idx], residuals2[idx], alpha=0.3, s=2, color=COLORS[0])

    plt.xlabel(f"Ensemble (Width = {width}) Residuals")
    plt.ylabel(f"Single (Width = {single_width}) Residuals")
    plt.title(f"Correlation of Residuals: {corr_residuals:.3f}")

    # Hide the right and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Add grid with light gray color and dashed style
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scatter_residuals.pdf", bbox_inches="tight")
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

        # Add statistics for prediction differences
        if label == f"{name2} - {name1} Predictions":
            mean_val = np.mean(data)
            var_val = np.var(data)
            plt.axvline(
                mean_val,
                color=color,
                linestyle="--",
                linewidth=1,
                label=f"Mean: {mean_val:.3f}, Var: {var_val:.3f}",
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
