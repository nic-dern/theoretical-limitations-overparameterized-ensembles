# Theoretical Limitations of Ensembles in the Age of Overparameterization

This repository contains the code to reproduce the experiments in the paper *Theoretical Limitations of Ensembles in the Age of Overparameterization* by Niclas Dern, John P. Cunningham, and Geoff Pleiss ([arXiv link](https://arxiv.org/abs/2410.16201)).

## Table of Contents

- [Theoretical Limitations of Ensembles in the Age of Overparameterization](#theoretical-limitations-of-ensembles-in-the-age-of-overparameterization)
  - [Table of Contents](#table-of-contents)
  - [Installation and Setup](#installation-and-setup)
  - [Usage](#usage)
  - [Reproducing Figures](#reproducing-figures)
    - [Figure 1](#figure-1)
    - [Figure 2](#figure-2)
    - [Figure 3](#figure-3)
    - [Figure 4](#figure-4)
    - [Figure 5](#figure-5)
    - [Figure 6](#figure-6)
  - [Code Structure](#code-structure)

## Installation and Setup

This project uses [Poetry](https://python-poetry.org/docs/) for dependency management. To set up the environment:

1. [Install Poetry](https://python-poetry.org/docs/#installation).
2. In the project directory (where `pyproject.toml` is located), run:

   ```bash
   poetry shell
   poetry install
   ```

   This will create a virtual environment and install dependencies.

3. You can now run the command-line tools and scripts in this project.

## Usage

The following commands are available for running experiments:

```bash
average-difference-vs_num-features
convergence-expected-value-term
generalization-error-decay
lipschitz-difference-infinite-models
variance-vs-number-of-features
variance-vs-points-in-range
visualize-models
```

After activating the virtual environment with `poetry shell`, you can run any of these commands by running:

```bash
<command-name> <command-parameters>
```

For example, to run the `average-difference-vs_num-features` command, you could run:

```bash
average-difference-vs_num-features --data-generating-function-name "sinusoidal" --num-features-end 25 --num-training-samples 6
```

To view available parameters for each command use:

```bash
<command-name> --help
```

Results will be saved in the `results` directory.

## Reproducing Figures

To reproduce the figures from the main paper, you can run the following commands:

### Figure 1

```bash
visualize-models --num-training-samples 6 --num-features-per-model 200 --number-ensemble-members 1 --number-simulations-per-size 100 --random-seed 42 --plot-kernel-model --kernel "arc-cosine-kernel"
```

and

```bash
visualize-models --num-training-samples 6 --num-features-per-model 200 --number-ensemble-members 10000 --number-simulations-per-size 1 --plot-kernel-model --kernel "arc-cosine-kernel"
```

### Figure 2

```bash
convergence-expected-value-term --max-num-models 100000 --data-generating-function-name "sinusoidal" --data-dimension 1 --num-training-samples 6 --num-features-per-model 200 --random-seed 42
```

and

```bash
convergence-expected-value-term --max-num-models 100000 --data-generating-function-name "CaliforniaHousing" --data-dimension 8 --num-training-samples 12 --num-features-per-model 200 --kernel "erf-kernel" --activation-function "erf"
```

### Figure 3

```bash
average-difference-vs_num-features --data-generating-function-name "sinusoidal" --num-training-samples 6 --ridge 0.0 --data-dimension 1
```

and

```bash
average-difference-vs_num-features --data-generating-function-name "CaliforniaHousing" --num-training-samples 12 --ridge 0.0 --data-dimension 8 --kernel "softplus-kernel" --activation-function "softplus"
```

### Figure 4

```bash
variance-vs-points-in-range --number-points-to-test 1000 --data-generating-function-name "sinusoidal" --num-features-per-model 200 --num-training-samples 6 --random-seed 42
```

### Figure 5

Set `NUMBER_OF_MODELS_FOR_VARIANCE` to `4000` in `utils/constants.py` and then run:

```bash
variance-vs-number-of-features --data-generating-function-name "CaliforniaHousing" --num-features-per-model 200 --num-training-samples 12 --data-dimension 8 --random-seed 42 --max-num-models 35
```

and

```bash
generalization-error-decay --data-generating-function-name "CaliforniaHousing" --num-training-samples 12 --num-features-per-model 200 --max-num-models 35 --number-simulations-per-size 2500 --random-seed 42 --data-dimension 8
```

### Figure 6

```bash
lipschitz-difference-infinite-models --ridge-step 0.00001 --ridge-end 0.001 --ridge-start 0.0 --max-num-models 2000 --data-generating-function-name "CaliforniaHousing" --data-dimension 8 --num-training-samples 12 --num-features-per-model 200
```

and

```bash
lipschitz-difference-infinite-models --ridge-step 0.00001 --ridge-end 0.001 --ridge-start 0.0 --max-num-models 2000 --data-generating-function-name "CaliforniaHousing" --data-dimension 8 --num-training-samples 12 --num-features-per-model 200 --comparison-mode "ensemble"
```

## Code Structure

- **`data_generation/`**: Functions for data generation.
  - `data_generation.py`: Contains data generation functions.
- **`experiments/`**: Experiment scripts.
  - `average_difference_vs_num_features.py`: Experiment for Figure 2.
  - `convergence_expected_value_term.py`: Experiment for Figure 3.
  - `generalization_error_decay.py`: Experiment for Figure 5.
  - `lipschitz_difference_infinite_models.py`: Experiment for Figure 6.
  - `variance_vs_number_of_features.py`: Experiment for Figure 5.
  - `variance_vs_points_in_range.py`: Experiment for Figure 4.
  - `visualize_models.py`: Experiment for Figure 1.
- **`matrices_and_kernels/`**: Kernel and matrix calculations.
  - `kernel_calculations.py`: Kernel matrix calculations.
  - `matrix_calculations.py`: Cholesky decomposition.
- **`models/`**: Model architectures and utilities.
  - `ensembles.py`: Ensemble models.
  - `kernel_models.py`: Kernel-based models.
  - `model_utils.py`: Activation functions and model utilities.
  - `random_feature_models.py`: Random feature models.
- **`monte_carlo/`**: Monte Carlo simulations.
  - `monte_carlo.py`: Monte Carlo estimations (used in Figure 3).
  - `w_terms.py`: Calculates \( W \) and \( w_\perp \) terms.
- **`utils/`**: Constants and utilities.
  - `constants.py`: Experimental constants.
  - `utils.py`: General utility functions.
- **`visualizations/`**: Visualization functions.
  - `plots.py`: Plotting experiment results.
  - `data_visualization.py`: Data and model visualization functions.
