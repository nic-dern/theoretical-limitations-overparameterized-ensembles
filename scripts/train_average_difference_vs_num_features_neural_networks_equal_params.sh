#!/bin/bash

########################################
# train_average_difference_vs_num_features_neural_networks_equal_params.sh
########################################

# TODO: Set your W&B API key
# export WANDB_API_KEY="your_wandb_api_key_here"

# Output folder
OUTPUT_DIR="nn_checkpoints_average_difference_vs_num_features_equal_params"
mkdir -p "$OUTPUT_DIR"

# WIDTHS=(20 30 40 50 60 70 80 90 100 110 120 130 140 150)
WIDTHS=(70 80 90 100 110 120 130 140)

########################################
# Helper function to compute parameter count for 3-layer MLP
# Architecture: [8 -> W -> W -> W -> 1]
# Number of parameters = 2*W^2 + 12*W + 1
########################################
function param_count_3layer() {
    local W=$1
    echo $(( 2 * W * W + 12 * W + 1 ))
}

########################################
# Function to find width for target parameter count
########################################
function find_width_for_params() {
    local T=$1
    local w=1
    while true; do
        local pc=$(param_count_3layer $w)
        if [ $pc -ge $T ]; then
            echo $w
            return
        fi
        ((w++))
        if [ $w -gt 3000 ]; then
            echo "Could not find width up to 3000 for T=$T" >&2
            echo 3000
            return
        fi
    done
}

########################################
# Hyperparameters
########################################
EPOCHS=1000
TRAINING_SAMPLES=12000
VALIDATION_SAMPLES=3000
TEST_SAMPLES=5000
BATCH_SIZE=256
LEARNING_RATE=0.01
OPTIMIZER="sgd"
MOMENTUM=0.9

# Reference width and its parameter count
REFERENCE_WIDTH=320
REFERENCE_PARAMS=$(param_count_3layer $REFERENCE_WIDTH)

echo "=== Starting hockey-stick experiment (constant total parameters) ==="
echo "Training set size (N) = $TRAINING_SAMPLES"
echo "Reference model: width=$REFERENCE_WIDTH, params=$REFERENCE_PARAMS"
echo "Widths to test: ${WIDTHS[*]}"

for W in "${WIDTHS[@]}"; do
    # Calculate ensemble size to get approximately the same total parameters as reference
    SINGLE_PARAMS=$(param_count_3layer $W)
    ENSEMBLE_SIZE=$(( (REFERENCE_PARAMS + SINGLE_PARAMS - 1) / SINGLE_PARAMS ))
    
    # Calculate actual total parameters for this ensemble
    TOTAL_PARAMS=$(( ENSEMBLE_SIZE * SINGLE_PARAMS ))
    
    # Find width for single network with same parameter count
    W_SINGLE=$(find_width_for_params $TOTAL_PARAMS)
    
    echo ""
    echo "=== Width $W: Ensemble size=$ENSEMBLE_SIZE ==="
    echo "    Single model params: $SINGLE_PARAMS"
    echo "    Total ensemble params: $TOTAL_PARAMS (target: $REFERENCE_PARAMS)"
    echo "    Equivalent single network width: $W_SINGLE"
    
    #################################################
    # 1) Train ENSEMBLE of networks with width = W
    #################################################
    RUN_FOLDER_ENSEMBLE="${OUTPUT_DIR}/ensemble_width_${W}"
    mkdir -p "$RUN_FOLDER_ENSEMBLE"
    
    echo "=== [Ensemble] Training ensemble of size=$ENSEMBLE_SIZE with width=$W"
    
    uv run python -m overparameterized_ensembles.experiments.train_and_save_neural_networks \
        --output-dir "$RUN_FOLDER_ENSEMBLE" \
        --project-name "hockey_stick_ensemble_const_params" \
        --enable-wandb \
        --use-gpu \
        --save-top-k 1 \
        --number-of-models $ENSEMBLE_SIZE \
        --hidden-layers "${W},${W},${W}" \
        --num-training-samples $TRAINING_SAMPLES \
        --num-validation-samples $VALIDATION_SAMPLES \
        --num-test-samples $TEST_SAMPLES \
        --batch-size $BATCH_SIZE \
        --max-epochs $EPOCHS \
        --learning-rate $LEARNING_RATE \
        --optimizer $OPTIMIZER \
        --momentum $MOMENTUM
    
    #################################################
    # 2) Train SINGLE network with equivalent parameters
    #################################################
    RUN_FOLDER_SINGLE="${OUTPUT_DIR}/single_equiv_width_${W}"
    mkdir -p "$RUN_FOLDER_SINGLE"
    
    echo "=== [Single] Training SINGLE network with width=$W_SINGLE (same param count)"
    
    uv run python -m overparameterized_ensembles.experiments.train_and_save_neural_networks \
        --output-dir "$RUN_FOLDER_SINGLE" \
        --project-name "hockey_stick_single_equiv_const_params" \
        --enable-wandb \
        --use-gpu \
        --save-top-k 1 \
        --number-of-models 5 \
        --hidden-layers "${W_SINGLE},${W_SINGLE},${W_SINGLE}" \
        --num-training-samples $TRAINING_SAMPLES \
        --num-validation-samples $VALIDATION_SAMPLES \
        --num-test-samples $TEST_SAMPLES \
        --batch-size $BATCH_SIZE \
        --max-epochs $EPOCHS \
        --learning-rate $LEARNING_RATE \
        --optimizer $OPTIMIZER \
        --momentum $MOMENTUM
done

echo ""
echo "=== Summary of ensemble sizes ==="
for W in "${WIDTHS[@]}"; do
    SINGLE_PARAMS=$(param_count_3layer $W)
    ENSEMBLE_SIZE=$(( (REFERENCE_PARAMS + SINGLE_PARAMS - 1) / SINGLE_PARAMS ))
    TOTAL_PARAMS=$(( ENSEMBLE_SIZE * SINGLE_PARAMS ))
    echo "Width $W: $ENSEMBLE_SIZE models, total params = $TOTAL_PARAMS"
done

echo ""
echo "All training runs have finished!"