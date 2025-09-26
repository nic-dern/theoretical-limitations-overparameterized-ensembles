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

app = typer.Typer()


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
        Path("nn_checkpoints_compare_ensembles_vs_single_equal_params"),
        help="Root folder of saved models",
    ),
    num_models: int = typer.Option(5, help="Number of models to average over"),
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
):
    """
    Compare pointwise predictions between ensemble and single large models.
    """
    if dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Invalid data type: {dtype}")

    if num_models < 2:
        raise ValueError(
            "num_models must be at least 2 otherwise there are no models to compare internally!"
        )

    # Set random seed for reproducibility
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
    # Get a copy of y_star from the test loader via iterating over the loader; otherwise we get weird results due to multiple workers
    y_test = []
    for x, y in test_loader:
        y_test.append(y.numpy())
    y_test = np.concatenate(y_test, axis=0).reshape(-1, 1)

    BASE_WIDTH = 256
    M_values = [1, 5, 10, 15, 20]

    # Create base plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for M in M_values:
        # Calculate width for single large model
        single_params = 2 * BASE_WIDTH * BASE_WIDTH + 12 * BASE_WIDTH + 1
        total_params = M * single_params
        large_width = 1
        while (2 * large_width * large_width + 12 * large_width + 1) < total_params:
            large_width += 1

        # Create directory for this M value
        m_dir = plots_dir / f"M_{M}"
        m_dir.mkdir(exist_ok=True)

        # Process single large models
        single_folder = output_dir / f"single_large_equivParams_{M}"
        single_exp_folder = next(
            d
            for d in single_folder.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Process ensembles
        ensemble_base = Path("nn_checkpoints_ensemble_250_models_width_256")
        experiment_folder = next(
            d
            for d in ensemble_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Create num_models different ensembles
        for start_idx in range(num_models):
            # Create directory for this ensemble instance
            instance_dir = m_dir / f"instance_{start_idx}"
            instance_dir.mkdir(exist_ok=True)

            # Get predictions for two ensembles
            ensemble_preds = []
            for ensemble_idx in [start_idx, (start_idx + 1) % num_models]:
                model_indices = range(ensemble_idx, 250, num_models)
                selected_indices = [
                    i for i in model_indices if i < 250 and i // num_models < M
                ]

                if len(selected_indices) != M:
                    typer.echo(
                        f"Warning: found {len(selected_indices)} models for ensemble {ensemble_idx}, expected M={M}!"
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

                ensemble_pred = []
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

                ensemble_preds.append(
                    np.mean(np.stack(ensemble_pred, axis=-1), axis=-1)
                )

            # Get predictions for two single models
            single_preds = []
            for single_idx in [start_idx, (start_idx + 1) % num_models]:
                run_dir = next(
                    d
                    for d in single_exp_folder.iterdir()
                    if d.is_dir()
                    and d.name.split("-")[-1].isdigit()
                    and int(d.name.split("-")[-1]) == single_idx
                )

                ckpt_single = load_checkpoint(run_dir, use_best_checkpoint)
                single_model = NeuralNetworkModel.load_from_checkpoint(
                    checkpoint_path=ckpt_single,
                    hidden_layer_sizes=[data_module.X.shape[1]]
                    + [large_width, large_width, large_width],
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    momentum=momentum,
                )
                single_preds.append(predict_dataset(single_model, test_loader))

            # Generate comparison plots
            n_points = 1000
            idx = np.random.choice(len(ensemble_preds[0]), n_points, replace=False)

            # Create a subfolder for each comparison type
            comparisons_dir = instance_dir / "comparisons"
            comparisons_dir.mkdir(exist_ok=True)

            # 1) Original ensemble vs single comparison
            create_comparison_plots(
                pred1=ensemble_preds[0],
                pred2=single_preds[0],
                y_test=y_test,
                idx=idx,
                name1="Ensemble 1",
                name2="Single 1",
                output_dir=comparisons_dir / "ensemble1_vs_single1",
                M=M,
                use_loglog=use_loglog,
            )

            # 2) Ensemble vs ensemble comparison
            create_comparison_plots(
                pred1=ensemble_preds[0],
                pred2=ensemble_preds[1],
                y_test=y_test,
                idx=idx,
                name1="Ensemble 1",
                name2="Ensemble 2",
                output_dir=comparisons_dir / "ensemble1_vs_ensemble2",
                M=M,
                use_loglog=use_loglog,
            )

            # 3) Single vs single comparison
            create_comparison_plots(
                pred1=single_preds[0],
                pred2=single_preds[1],
                y_test=y_test,
                idx=idx,
                name1="Single 1",
                name2="Single 2",
                output_dir=comparisons_dir / "single1_vs_single2",
                M=M,
                use_loglog=use_loglog,
            )


def create_comparison_plots(
    pred1, pred2, y_test, idx, name1, name2, output_dir, M, use_loglog
):
    """Helper function to create all comparison plots for a pair of models."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Turn pred1 and pred2 into 1D arrays
    pred1 = pred1.ravel()
    pred2 = pred2.ravel()
    y_test = y_test.ravel()

    # 1) Scatter of predictions
    corr_preds = np.corrcoef(pred1.ravel(), pred2.ravel())[0, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(pred1[idx], pred2[idx], alpha=0.3, s=2)
    if use_loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel(f"{name1} Predictions")
    plt.ylabel(f"{name2} Predictions")
    plt.title(f"M={M}: corr(preds)={corr_preds:.3f}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"scatter_predictions{'_loglog' if use_loglog else ''}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 2) Scatter of squared errors
    err1 = (pred1 - y_test) ** 2
    err2 = (pred2 - y_test) ** 2
    corr_errs = np.corrcoef(err1, err2)[0, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(err1[idx], err2[idx], alpha=0.3, s=2)
    if use_loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel(f"{name1} Squared Error")
    plt.ylabel(f"{name2} Squared Error")
    plt.title(f"M={M}: corr(errors)={corr_errs:.3f}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"scatter_errors{'_loglog' if use_loglog else ''}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 3) Scatter of residuals
    residuals1 = pred1 - y_test
    residuals2 = pred2 - y_test
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
