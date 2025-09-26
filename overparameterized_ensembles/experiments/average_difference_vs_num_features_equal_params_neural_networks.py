import typer
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from overparameterized_ensembles.models.neural_network_models import NeuralNetworkModel
from overparameterized_ensembles.data_generation.neural_network_data_generation import (
    CaliforniaHousingDataModule,
)
from overparameterized_ensembles.utils.utils import save_figure
from overparameterized_ensembles.visualization.plots import plot_graph


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
):
    """
    Compare predictions of ensemble models to single networks with equivalent parameter counts.
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

    # Calculate reference parameter count
    reference_params = param_count_3layer(reference_width)
    typer.echo(f"Reference model: width={reference_width}, params={reference_params}")

    # Process models of different widths
    widths = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    avg_diffs = []
    all_diffs = []  # Store all pairwise differences for error bars
    actual_ensemble_sizes = []  # Store actual ensemble sizes for reporting
    param_counts = []  # Store parameter counts for x-axis

    for width in widths:
        # Calculate ensemble size to match reference parameter count
        single_params = param_count_3layer(width)
        param_counts.append(single_params)  # Store for x-axis
        ensemble_size = (reference_params + single_params - 1) // single_params
        actual_ensemble_sizes.append(ensemble_size)

        # Calculate actual total parameters
        ensemble_params = ensemble_size * single_params
        single_width = find_width_for_params(ensemble_params)

        typer.echo(f"\nProcessing width={width}:")
        typer.echo(f"  Ensemble: {ensemble_size} models of width {width}")
        typer.echo(f"  Single: 1 model of width {single_width}")
        typer.echo(f"  Total params: {ensemble_params} (target: {reference_params})")

        # Load single network models
        single_models = []
        single_base = output_dir / f"single_equiv_width_{width}"
        if not single_base.exists():
            typer.echo(f"Warning: {single_base} not found, skipping width {width}")
            avg_diffs.append(np.nan)
            all_diffs.append([np.nan])
            continue

        single_exp_folder = next(
            d
            for d in single_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Load all single models (typically 5)
        for i in range(5):
            try:
                run_dir = next(
                    d
                    for d in single_exp_folder.iterdir()
                    if d.is_dir()
                    and not d.name.startswith(".")
                    and d.name.endswith(f"-{i}")
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
                single_models.append(single_model)
            except StopIteration:
                typer.echo(
                    f"Warning: Could not find single model {i} for width {width}"
                )

        if not single_models:
            typer.echo(f"Error: No single models found for width {width}")
            avg_diffs.append(np.nan)
            all_diffs.append([np.nan])
            continue

        # Get predictions for all single models
        single_preds_list = [
            predict_dataset(model, test_loader) for model in single_models
        ]

        # Load ensemble models
        ensemble_base = output_dir / f"ensemble_width_{width}"
        if not ensemble_base.exists():
            typer.echo(f"Warning: {ensemble_base} not found, skipping width {width}")
            avg_diffs.append(np.nan)
            all_diffs.append([np.nan])
            continue

        ensemble_exp_folder = next(
            d
            for d in ensemble_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Load ensemble predictions
        ensemble_preds = []
        run_dirs = sorted([d for d in ensemble_exp_folder.iterdir() if d.is_dir()])[
            :ensemble_size
        ]

        for run_dir in run_dirs:
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

        # Calculate differences between ensemble and each single model
        width_diffs = []
        for single_pred in single_preds_list:
            diff = np.mean(np.abs(avg_ensemble_pred - single_pred))
            width_diffs.append(diff)

        avg_diff = np.mean(width_diffs)
        avg_diffs.append(avg_diff)
        all_diffs.append(width_diffs)

        typer.echo(f"  Average difference: {avg_diff:.6f}")

    # Calculate parameter count that equals training samples
    params_for_training_samples = data_module.num_training_samples

    # Calculate standard deviations for error bars
    std_devs = [
        np.std(diffs) if not np.isnan(diffs[0]) else np.nan for diffs in all_diffs
    ]

    # Create plot using plot_graph
    plt_fig = plot_graph(
        x_values=[param_counts],
        y_values=[avg_diffs],
        y_error_values=[std_devs],
        labels=["Average difference"],
        x_label="Number of parameters per model",
        y_label=r"$\| \overline{h}_{\mathcal{W}_{1:M}} - h_{W_{\text{single}}} \|_{L^1}$",
        vline_x=params_for_training_samples,
        vline_text=r"$N = \mathrm{Total\,Parameters}$",
        plot_legend=False,
        loglog=False,  # Use log scale for x-axis since parameter counts vary widely
    )

    out_path = output_dir / "width_comparison_equal_params_plot.pdf"
    save_figure(plt_fig, out_path)
    typer.echo(f"\nSaved plot to {out_path}")
    typer.echo(
        f"\nNote: Each point represents an ensemble with ~{reference_params:,} total parameters"
    )
    typer.echo("(smaller models have more ensemble members, larger models have fewer)")

    # Print summary statistics
    typer.echo("\nSummary:")
    typer.echo(
        f"{'Width':<8} {'Params/Model':<15} {'Ensemble Size':<15} {'Avg Diff':<12} {'Std Dev':<12}"
    )
    typer.echo("-" * 70)
    for i, w in enumerate(widths):
        if not np.isnan(avg_diffs[i]):
            typer.echo(
                f"{w:<8} {param_counts[i]:<15,} {actual_ensemble_sizes[i]:<15} "
                f"{avg_diffs[i]:<12.6f} {std_devs[i]:<12.6f}"
            )


if __name__ == "__main__":
    app()
