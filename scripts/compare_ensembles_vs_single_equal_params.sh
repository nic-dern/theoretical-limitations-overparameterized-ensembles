#!/bin/bash

########################################
# submit_compare_ensembles_vs_single_equal_params.sh
########################################

# TODO: Set your W&B API key
# export WANDB_API_KEY="your_wandb_api_key_here"

# Output base folder for comparing ensembles vs single large models with equal parameters
OUTPUT_DIR="nn_checkpoints_compare_ensembles_vs_single_equal_params"

###############################
# Helper bash function to compute the 3-layer MLP param count
# for a network with hidden_layer_width = W
# Architecture: [8 -> W -> W -> W -> 1]
# Number of parameters = 2*W^2 + 12*W + 1
###############################
function param_count_3layer() {
  local W=$1
  # 2*W^2 + 12*W + 1
  echo $(( 2 * W * W + 12 * W + 1 ))
}

###############################
# Function that, given a target param_count T,
# finds an integer width W s.t. param_count_3layer(W) ~ T
# (just does a simple integer search – you can refine as needed)
###############################
function find_width_for_params() {
  local T=$1
  
  # We'll do a quick integer search. 
  # Start from 1 and go upward until param_count_3layer >= T
  local w=1
  while true; do
    local pc=$(param_count_3layer $w)
    if [ $pc -ge $T ]; then
      echo $w
      return
    fi
    ((w++))
    # Safety break if something goes wrong
    if [ $w -gt 3000 ]; then
      echo "Could not find width up to 3000 for T=$T" >&2
      echo 3000
      return
    fi
  done
}

########################################
# Main experiment configuration
########################################

ENSEMBLE_SIZES=("1" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# Number of epochs, training set size, etc.
EPOCHS=1000
TRAINING_SAMPLES=12000
VALIDATION_SAMPLES=3000
TEST_SAMPLES=5000
BATCH_SIZE=256
LEARNING_RATE=0.01
OPTIMIZER="sgd"
MOMENTUM=0.9

########################################
# Run the ensembles
########################################

for M in "${ENSEMBLE_SIZES[@]}"; do
  
  # Compute the param_count for a single network of hidden=256
  SINGLE_PARAMS=$(param_count_3layer 256)
  # total param count
  let TOTAL_PARAMS=$M*$SINGLE_PARAMS

  # Train a SINGLE large network with total parameter count ~ TOTAL_PARAMS
  # Find approximate width w
  W=$(find_width_for_params $TOTAL_PARAMS)
  
  RUN_FOLDER_SINGLE="${OUTPUT_DIR}/single_large_equivParams_${M}"
  mkdir -p "$RUN_FOLDER_SINGLE"

  echo "=== Training SINGLE large network for total_params=$TOTAL_PARAMS => hidden=$W => param_count_3layer($W) ==="
  uv run python -m overparameterized_ensembles.experiments.train_and_save_neural_networks \
    --output-dir "$RUN_FOLDER_SINGLE" \
    --project-name "compare_ensembles_vs_single_equal_params" \
    --enable-wandb \
    --use-gpu \
    --save-top-k 1 \
    --number-of-models 10 \
    --hidden-layers "${W},${W},${W}" \
    --num-training-samples $TRAINING_SAMPLES \
    --num-validation-samples $VALIDATION_SAMPLES \
    --num-test-samples $TEST_SAMPLES \
    --batch-size $BATCH_SIZE \
    --max-epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --momentum $MOMENTUM

done

echo "All runs for Experiment 1 have finished!"
