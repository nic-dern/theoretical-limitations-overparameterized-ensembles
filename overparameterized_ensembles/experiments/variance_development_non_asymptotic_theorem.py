import typer
import torch
from pathlib import Path
import numpy as np
from overparameterized_ensembles.models.model_utils import (
    initialize_random_weights_distribution,
    calculate_variance_term,
)
from overparameterized_ensembles.data_generation.data_generation import (
    generate_data,
)
from overparameterized_ensembles.visualization.plots import (
    plot_graph,
)
from overparameterized_ensembles.utils.utils import (
    save_figure,
)
from overparameterized_ensembles.utils.constants import (
    DEFAULT_RANDOM_SEED,
)

app = typer.Typer()


@app.command()
def variance_development_non_asymptotic_theorem(
    num_training_samples: int = typer.Option(4, help="Number of training samples"),
    data_dimension: int = typer.Option(1, help="Dimension of input data"),
    min_features: int = typer.Option(10, help="Minimum number of features"),
    max_features: int = typer.Option(10000, help="Maximum number of features"),
    num_feature_points: int = typer.Option(100, help="Number of points to evaluate"),
    activation_function: str = typer.Option("relu", help="Activation function to use"),
    random_weights_distribution: str = typer.Option(
        "normal", help="Distribution for random weights"
    ),
    num_monte_carlo: int = typer.Option(
        1000, help="Number of Monte Carlo samples per point"
    ),
    data_generating_function_name: str = typer.Option(
        "quadratic", help="Data generation function"
    ),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
    use_loglog: bool = typer.Option(True, help="Use log-log scale for plot"),
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Plot the development of Var[y^T (1/D Φ_i Φ_i^T)^{-1} y] against increasing number of features D.
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    X, y, _, _, _ = generate_data(
        data_generating_function_name,
        num_training_samples,
        data_dimension,
        0.05,  # Some noise
    )
    # Add bias term
    X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=1)
    y = y.reshape(-1, 1)

    # Initialize random weights distribution
    random_weights_distribution = initialize_random_weights_distribution(
        random_weights_distribution,
        data_dimension + 1,  # Add one for bias term
    )

    # Generate feature counts (logarithmically spaced)
    feature_counts = np.logspace(
        np.log10(min_features),
        np.log10(max_features),
        num_feature_points,
        dtype=int,
    )

    # Calculate variance terms
    variance_terms = []
    for D in feature_counts:
        variance = calculate_variance_term(
            X,
            y,
            D,
            random_weights_distribution,
            activation_function,
            num_monte_carlo,
        )
        variance_terms.append(variance)
        typer.echo(f"D={D}: Variance={variance:.6f}")

    # Convert feature_counts to list for plotting
    feature_counts_list = feature_counts.tolist()

    # Create plot
    fig = plot_graph(
        x_values=[feature_counts_list],
        y_values=[variance_terms],
        labels=["Variance Term"],
        x_label="Number of Features (D)",
        y_label=r"$\mathrm{Var}[y^T (D^{-1} \Phi_i \Phi_i^T)^{-1} y]$",
        loglog=use_loglog,
        decay_slope=-1.0,
        plot_legend=False,
    )

    # Save plot
    out_path = (
        output_dir / f"variance_term_development{'_loglog' if use_loglog else ''}.pdf"
    )
    save_figure(fig, out_path)
    typer.echo(f"Saved plot to {out_path}")

    # Save the raw data
    torch.save(variance_terms, output_dir / "variance_terms.pt")


if __name__ == "__main__":
    app()
