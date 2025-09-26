# Theoretical Limitations of Ensembles in the Age of Overparameterization

This repository contains the code to reproduce the experiments in the paper *Theoretical Limitations of Ensembles in the Age of Overparameterization* by Niclas Dern, John P. Cunningham, and Geoff Pleiss ([arXiv link](https://arxiv.org/abs/2410.16201)).

We will soon add the code for additional experiments we conducted with neural networks.

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

## Installation and Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To set up the environment just [install uv](https://docs.astral.sh/uv/getting-started/installation/).

## Usage

The following commands are available for running experiments:

```bash
uv run python -m overparameterized_ensembles.experiments.average_difference_vs_num_features
uv run python -m overparameterized_ensembles.experiments.convergence_expected_value_term
uv run python -m overparameterized_ensembles.experiments.generalization_error_decay
uv run python -m overparameterized_ensembles.experiments.lipschitz_difference_infinite_models
uv run python -m overparameterized_ensembles.experiments.variance_vs_number_of_features
uv run python -m overparameterized_ensembles.experiments.variance_vs_points_in_range
uv run python -m overparameterized_ensembles.experiments.visualize_models
uv run python -m overparameterized_ensembles.experiments.compare_ensembles_vs_single_equal_width_neural_networks
uv run python -m overparameterized_ensembles.experiments.average_difference_vs_num_features_neural_networks
uv run python -m overparameterized_ensembles.experiments.average_difference_vs_num_features_neural_networks_pointwise
```

For example, to run the `average_difference_vs_num_features` experiment, you could run:

```bash
uv run python -m overparameterized_ensembles.experiments.average_difference_vs_num_features --data-generating-function-name "sinusoidal" --num-features-end 25 --num-training-samples 6
```

To view available parameters for each command use:

```bash
uv run python -m overparameterized_ensembles.experiments.<command_name> --help
```

Results will be saved in the `results` directory or in the corresponding experiment directory.

## Reproducing Figures

To reproduce the figures from the main paper, you can run the following commands:

### Figure 1

```bash
uv run python -m overparameterized_ensembles.experiments.visualize_models --num-training-samples 6 --num-features-per-model 200 --number-ensemble-members 1 --number-simulations-per-size 100 --random-seed 42 --plot-kernel-model --kernel "arc-cosine-kernel"
```

and

```bash
uv run python -m overparameterized_ensembles.experiments.visualize_models --num-training-samples 6 --num-features-per-model 200 --number-ensemble-members 10000 --number-simulations-per-size 1 --plot-kernel-model --kernel "arc-cosine-kernel"
```

### Figure 2

```bash
uv run python -m overparameterized_ensembles.experiments.average_difference_vs_num_features --data-generating-function-name "CaliforniaHousing" --num-training-samples 12 --ridge 0.0 --data-dimension 8 --kernel "softplus-kernel" --activation-function "softplus"
```

and

```bash
uv run python -m overparameterized_ensembles.experiments.average_difference_vs_num_features_equal_params_neural_networks --use-best-checkpoint
```

To train the corresponding neural networks for the second part of Figure 2, run the following script:

```bash
./scripts/train_average_difference_vs_num_features_neural_networks_equal_params.sh
```

### Figure 3

```bash
uv run python -m overparameterized_ensembles.experiments.generalization_error_decay --data-generating-function-name "CaliforniaHousing" --num-training-samples 12 --num-features-per-model 200 --max-num-models 35 --number-simulations-per-size 2500 --random-seed 42 --data-dimension 8
```

and

```bash
uv run python -m overparameterized_ensembles.experiments.compare_ensembles_vs_single_equal_params_neural_networks --use-best-checkpoint
```

For training the neural networks for the second part of Figure 3, run the following scripts:

First, train the ensemble models:

```bash
./scripts/train_250_nns.sh
```

Then train the single large models:

```bash
./scripts/compare_ensembles_vs_single_equal_params.sh
```

### Figure 4

```bash
uv run python -m overparameterized_ensembles.experiments.variance_vs_points_in_range --number-points-to-test 1000 --data-generating-function-name "sinusoidal" --num-features-per-model 200 --num-training-samples 6 --random-seed 42 --plot-on-same-axis
```

### Figure 5

```bash
uv run python -m overparameterized_ensembles.experiments.lipschitz_difference_infinite_models --ridge-step 0.00001 --ridge-end 0.001 --ridge-start 0.0 --max-num-models 2000 --data-generating-function-name "CaliforniaHousing" --data-dimension 8 --num-training-samples 12 --num-features-per-model 200
```

and

```bash
uv run python -m overparameterized_ensembles.experiments.lipschitz_difference_infinite_models --ridge-step 0.00001 --ridge-end 0.001 --ridge-start 0.0 --max-num-models 2000 --data-generating-function-name "CaliforniaHousing" --data-dimension 8 --num-training-samples 12 --num-features-per-model 200 --comparison-mode "ensemble"
```


