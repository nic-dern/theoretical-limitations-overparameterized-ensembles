#!/bin/bash

# TODO: Set your W&B API key
# export WANDB_API_KEY="your_wandb_api_key_here"

OUTPUT_DIR="nn_checkpoints_ensemble_250_models_width_256"

mkdir -p "$OUTPUT_DIR"

uv run python -m overparameterized_ensembles.experiments.train_and_save_neural_networks \
  --output-dir "$OUTPUT_DIR" \
  --project-name "ensemble-models-250" \
  --enable-wandb \
  --use-gpu \
  --save-top-k 1 \
  --number-of-models 250 \
  --hidden-layers 256,256,256 \
  --num-training-samples 12000 \
  --num-validation-samples 3000 \
  --num-test-samples 5000 \
  --batch-size 256 \
  --max-epochs 1000 \
  --learning-rate 0.01 \
  --optimizer sgd \
  --momentum 0.9 \

echo "All runs for Experiment 1 have finished!"
