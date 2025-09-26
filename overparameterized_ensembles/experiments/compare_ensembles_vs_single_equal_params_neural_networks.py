import typer
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from overparameterized_ensembles.models.neural_network_models import NeuralNetworkModel
from overparameterized_ensembles.data_generation.neural_network_data_generation import (
    CaliforniaHousingDataModule,
)
from overparameterized_ensembles.visualization.plots import plot_graph
from overparameterized_ensembles.utils.utils import save_figure
from overparameterized_ensembles.utils.constants import COLORS


app = typer.Typer()


def load_checkpoint(model_folder: Path, use_best: bool) -> str:
    """
    Load either the best or last checkpoint in the given folder.

    Args:
        model_folder: Path to folder containing checkpoints
        use_best: If True, load checkpoint with lowest val_loss, else load last checkpoint

    Returns:
        str: Path to the selected checkpoint
    """
    if use_best:
        # Find checkpoint with lowest val_loss
        ckpts = list(model_folder.glob("model-epoch=*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {model_folder}")

        # Extract val_loss from checkpoint names and find minimum
        best_ckpt = min(
            ckpts, key=lambda x: float(str(x).split("val_loss=")[1].split(".ckpt")[0])
        )
        return str(best_ckpt)
    else:
        # Find last checkpoint
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


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("nn_checkpoints_compare_ensembles_vs_single_equal_params"),
        help="Root folder of saved models",
    ),
    num_models: int = typer.Option(
        5, help="Number of models to average over for the single large model"
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
    use_loglog: bool = typer.Option(False, help="Use log-log scale for plots"),
    use_width: bool = typer.Option(
        False, help="Plot MSE against total width instead of M"
    ),
):
    """
    Analyze the ensemble vs. single large model runs,
    computing MSE and now also comparing pointwise predictions.

    Part (a) => MSE vs. M
    Part (b) => Scatter plots of predictions, scatter plots of squared errors
    """
    if dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Invalid data type: {dtype}")

    # Set random seed for model training
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pl.seed_everything(random_seed)

    # We'll re-create the same data module used in training.
    data_module = CaliforniaHousingDataModule(
        num_training_samples=12000,
        num_validation_samples=3000,
        num_test_samples=5000,
        batch_size=batch_size,
        dtype=torch.float32 if dtype == "float32" else torch.float64,
    )
    data_module.setup()

    test_loader = data_module.test_dataloader()
    # Get a copy of y_star from the test loader via iterating over the loader; otherwise we get weird results due to multiple workers
    y_test = []
    for x, y in test_loader:
        y_test.append(y.numpy())
    y_test = np.concatenate(y_test, axis=0).reshape(-1, 1)

    # We'll look for folders named 'ensemble_M' and 'single_large_equivParams_M'
    M_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ensemble_MSEs = []
    single_MSEs = []
    single_all_MSEs = []  # Store all MSEs for single models
    ensemble_all_MSEs = []  # Store all MSEs for ensemble instances
    widths_ensemble = []
    widths_single = []
    params_ensemble = []  # Store total parameters for ensembles
    params_single = []  # Store total parameters for single models

    # For ensembles, consider all sizes up to max(M_values) if plotting by width
    ensemble_sizes = range(1, int(max(M_values) / 5) + 1) if use_width else M_values

    ensemble_base = Path("nn_checkpoints_ensemble_250_models_width_256")
    experiment_folder = next(
        d for d in ensemble_base.iterdir() if d.is_dir() and not d.name.startswith(".")
    )  # Get the single experiment folder

    BASE_WIDTH = 256

    # Process single large models (only available for M_values)
    for M in M_values:
        # Calculate widths for single large model
        single_params = param_count_3layer(BASE_WIDTH)
        total_params = M * single_params
        large_width = find_width_for_params(total_params)

        single_folder = output_dir / f"single_large_equivParams_{M}"
        single_exp_folder = next(
            d
            for d in single_folder.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Get MSEs from all single large models
        single_model_mses = []
        run_dirs = sorted(
            [
                d
                for d in single_exp_folder.iterdir()
                if d.is_dir()
                and d.name.split("-")[-1].isdigit()
                and int(d.name.split("-")[-1]) < num_models
            ]
        )

        if len(run_dirs) != num_models:
            typer.echo(
                f"Warning: found {len(run_dirs)} single model runs, expected {num_models}!"
            )

        for run_dir in run_dirs:
            ckpt_single = load_checkpoint(run_dir, use_best_checkpoint)
            single_model = NeuralNetworkModel.load_from_checkpoint(
                checkpoint_path=ckpt_single,
                hidden_layer_sizes=[data_module.X.shape[1]]
                + [large_width, large_width, large_width],
                learning_rate=learning_rate,
                optimizer=optimizer,
                momentum=momentum,
                dtype=dtype,
            )
            single_pred = predict_dataset(single_model, test_loader)
            single_mse = np.mean((single_pred - y_test) ** 2)
            single_model_mses.append(single_mse)

        # Average the MSEs from individual models
        single_mse = np.mean(single_model_mses)
        single_MSEs.append(single_mse)
        single_all_MSEs.append(single_model_mses)
        widths_single.append(large_width)  # Store width for single model
        params_single.append(total_params)  # Store total parameters for single model

        typer.echo(f"M={M}: SingleLarge MSE={single_mse:.5f}")

    for M in ensemble_sizes:
        # Handle ensemble models
        ensemble_instance_mses = []

        # Create num_models different ensembles
        for start_idx in range(num_models):
            ensemble_pred = []

            # For each ensemble, get M models with stride num_models
            model_indices = range(
                start_idx, 250, num_models
            )  # 250 is total number of models
            selected_indices = [
                i for i in model_indices if i < 250 and i // num_models < M
            ]

            if len(selected_indices) != M:
                typer.echo(
                    f"Warning: found {len(selected_indices)} models for ensemble {start_idx}, expected M={M}!"
                )
                continue

            run_dirs = sorted(
                [
                    d
                    for d in experiment_folder.iterdir()
                    if d.is_dir()
                    and d.name.split("-")[-1].isdigit()
                    and int(d.name.split("-")[-1]) in selected_indices
                ]
            )

            for d in run_dirs:
                ckpt = load_checkpoint(d, use_best_checkpoint)
                model = NeuralNetworkModel.load_from_checkpoint(
                    checkpoint_path=ckpt,
                    hidden_layer_sizes=[data_module.X.shape[1]]
                    + [BASE_WIDTH, BASE_WIDTH, BASE_WIDTH],
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    momentum=momentum,
                )
                y_hat = predict_dataset(model, test_loader)
                ensemble_pred.append(y_hat)

            # Calculate MSE for this ensemble instance
            ensemble_pred = np.mean(np.stack(ensemble_pred, axis=-1), axis=-1)
            ensemble_mse = np.mean((ensemble_pred - y_test) ** 2)
            ensemble_instance_mses.append(ensemble_mse)

        # Average MSE across ensemble instances
        ensemble_mse = np.mean(ensemble_instance_mses)
        ensemble_MSEs.append(ensemble_mse)
        ensemble_all_MSEs.append(ensemble_instance_mses)
        widths_ensemble.append(M * BASE_WIDTH)  # Store total width for ensemble
        params_ensemble.append(
            M * param_count_3layer(BASE_WIDTH)
        )  # Store total parameters for ensemble

        typer.echo(f"M={M}: Ensemble MSE={ensemble_mse:.5f}")

    # Create plot using plot_graph
    if use_width:
        x_values = [widths_single, widths_ensemble]
        y_values = [
            single_MSEs,  # Just pass the MSE values, plot_graph will handle x-coordinates
            ensemble_MSEs,  # All ensemble points
        ]
        y_error_values = [
            np.std(single_all_MSEs, axis=1),
            np.std(ensemble_all_MSEs, axis=1),
        ]
        x_label = "Total Width"
    else:
        x_values = [params_single, params_ensemble[: len(M_values)]]
        y_values = [single_MSEs, ensemble_MSEs[: len(M_values)]]
        y_error_values = [
            np.std(single_all_MSEs, axis=1),
            np.std(ensemble_all_MSEs[: len(M_values)], axis=1),
        ]
        x_label = "Total number of parameters"

    plt_fig = plot_graph(
        x_values=x_values,
        y_values=y_values,
        y_error_values=y_error_values,
        labels=["Single Large Model", "Ensemble"],
        x_label=x_label,
        y_label="Generalization error",
        linestyles=["--", "-."],
        colors=[COLORS[0], COLORS[2]],
        loglog=use_loglog,
    )

    out_path = (
        output_dir
        / f"comparison_mse_plot{'_width' if use_width else ''}{'_loglog' if use_loglog else ''}.pdf"
    )
    save_figure(plt_fig, out_path)
    typer.echo(f"Saved plot to {out_path}")


if __name__ == "__main__":
    app()
