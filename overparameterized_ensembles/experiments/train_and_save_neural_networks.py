import typer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import numpy as np
from datetime import datetime
import random
import wandb

from overparameterized_ensembles.models.neural_network_models import NeuralNetworkModel
from overparameterized_ensembles.data_generation.neural_network_data_generation import (
    CaliforniaHousingDataModule,
)
from overparameterized_ensembles.utils.constants import DEFAULT_RANDOM_SEED

app = typer.Typer()

ADJECTIVES = [
    "happy",
    "cosmic",
    "dancing",
    "mystical",
    "clever",
    "fluffy",
    "bouncing",
    "glowing",
    "peaceful",
    "jazzy",
    "snazzy",
    "quirky",
    "graceful",
    "sparkly",
    "wandering",
    "caffeinated",
    "silly",
    "wacky",
    "zany",
    "funny",
    "groovy",
]

NOUNS = [
    "dolphin",
    "pizza",
    "robot",
    "penguin",
    "teapot",
    "cookie",
    "rainbow",
    "potato",
    "unicorn",
    "coconut",
    "wizard",
    "hamster",
    "cupcake",
    "butterfly",
    "cactus",
    "banana",
    "penguin",
    "octopus",
    "elephant",
    "giraffe",
    "kangaroo",
    "panda",
    "koala",
]


def generate_experiment_name():
    """Generate a unique experiment name using a separate random state."""
    # Create a new random state that does not affect the model training
    name_rng = random.Random()
    name_rng.seed(int(datetime.now().timestamp() * 1000))

    adj = name_rng.choice(ADJECTIVES)
    noun = name_rng.choice(NOUNS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{adj}-{noun}-experiment-{timestamp}"


def generate_run_name(run_index: int, random_seed: int) -> str:
    """Generate a unique run name using a separate random state."""
    # Create a new random state for run naming
    name_rng = random.Random()
    # Use timestamp and run index for unique seed
    name_rng.seed(int(datetime.now().timestamp() * 1000) + run_index)

    adj = name_rng.choice(ADJECTIVES)
    noun = name_rng.choice(NOUNS)
    return f"{adj}-{noun}-{random_seed}-{run_index}"


@app.command()
def train_and_save_neural_networks(
    # Checkpointing/logging
    output_dir: Path = typer.Option(
        Path("nn_checkpoints"), help="Directory to save checkpoints"
    ),
    project_name: str = typer.Option(
        "overparameterized-ensembles-training", help="Weights & Biases project name"
    ),
    enable_wandb: bool = typer.Option(True, help="Whether to log training on W&B"),
    use_gpu: bool = typer.Option(False, help="Whether to train on GPU (if available)"),
    save_top_k: int = typer.Option(1, help="How many top checkpoints to save"),
    # Model architecture
    number_of_models: int = typer.Option(
        1, help="How many separate MLP models to train"
    ),
    hidden_layer_size: int = typer.Option(
        512, help="Size of each hidden layer (for sweep compatibility)"
    ),
    hidden_layers: str = typer.Option(
        "512,512,512", help="Comma-separated hidden layer sizes for each model"
    ),
    # Basic hyperparameters
    num_training_samples: int = typer.Option(12000, help="Number of training samples"),
    num_validation_samples: int = typer.Option(
        3000, help="Number of validation samples"
    ),
    num_test_samples: int = typer.Option(5000, help="Number of test samples"),
    batch_size: int = typer.Option(128, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    max_epochs: int = typer.Option(100, help="Max number of epochs"),
    # Optimizer
    optimizer: str = typer.Option("adam", help="Optimizer to use (adam or sgd)"),
    momentum: float = typer.Option(0.9, help="Momentum for SGD optimizer"),
    # Random seed
    random_seed: int = typer.Option(DEFAULT_RANDOM_SEED, help="Random seed"),
):
    """
    Trains one or more MLPs on the California Housing Dataset and saves model checkpoints.
    """
    torch.set_default_dtype(torch.float32)

    # Configure PyTorch if using GPU
    if use_gpu:
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        typer.echo(f"GPU available: {gpu_available}")
        typer.echo(f"MPS available: {torch.backends.mps.is_available()}")
        if not gpu_available:
            typer.echo("Warning: GPU requested but not available. Falling back to CPU.")
            use_gpu = False

    # Create a unique experiment folder using a separate random state
    experiment_name = generate_experiment_name()
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Starting experiment: {experiment_name}")

    # Set random seed for model training
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pl.seed_everything(random_seed)

    # Format hidden_layers string if it contains placeholder
    if "HIDDEN_LAYER_SIZE" in hidden_layers:
        hidden_layers = hidden_layers.replace(
            "HIDDEN_LAYER_SIZE", str(hidden_layer_size)
        )

    # Parse the comma-separated hidden layer sizes
    hidden_layer_sizes = [int(x.strip()) for x in hidden_layers.split(",")]

    # Setup data
    data_module = CaliforniaHousingDataModule(
        num_training_samples=num_training_samples,
        num_validation_samples=num_validation_samples,
        num_test_samples=num_test_samples,
        batch_size=batch_size,
        dtype=torch.float32,
    )
    data_module.setup()

    # If use_gpu is True and a GPU is available
    accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
    accelerator = (
        "mps" if (use_gpu and torch.backends.mps.is_available()) else accelerator
    )

    # Train multiple models if needed
    for i in range(number_of_models):
        # Create a unique run name
        run_name = generate_run_name(i, random_seed)

        model_output_dir = experiment_dir / run_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(
            f"Training model {i + 1}/{number_of_models} with hidden sizes {hidden_layer_sizes}"
        )

        # Initialize model
        model = NeuralNetworkModel(
            hidden_layer_sizes=[data_module.X.shape[1]] + hidden_layer_sizes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
        )

        # Configure checkpoint callback for this specific model
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(model_output_dir),
            filename="model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
        )

        # Configure logger with unique run name
        if enable_wandb:
            # Initialize a new wandb run for each model
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "experiment_name": experiment_name,
                    "num_training_samples": num_training_samples,
                    "num_validation_samples": num_validation_samples,
                    "num_test_samples": num_test_samples,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_epochs": max_epochs,
                    "hidden_layers": hidden_layer_sizes,
                    "optimizer": optimizer,
                    "momentum": momentum,
                    "random_seed": random_seed,
                    "output_dir": str(model_output_dir),
                    "project_name": project_name,
                    "enable_wandb": enable_wandb,
                    "use_gpu": use_gpu,
                    "save_top_k": save_top_k,
                    "number_of_models": number_of_models,
                    "model_index": i,
                },
                reinit=True,  # Allow multiple runs in the same process
            )
            current_logger = WandbLogger(experiment=wandb.run)
        else:
            current_logger = None

        # Create the trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=current_logger,
            callbacks=[checkpoint_callback],
            default_root_dir=str(model_output_dir),
            accelerator=accelerator,
            check_val_every_n_epoch=5,
            enable_progress_bar=False,
        )

        # Train model
        trainer.fit(model, data_module)

        # Finish the wandb run before starting the next one
        if enable_wandb:
            wandb.finish()

        typer.echo(f"Finished training model {i + 1}/{number_of_models}")

    typer.echo("All models have been trained and checkpoints saved.")


if __name__ == "__main__":
    app()
