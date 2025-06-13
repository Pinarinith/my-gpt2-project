#!/bin/bash

# Define the base directory where your models are stored
BASE_DIR="models/hyperparam_search"
SEED_TEXT="HAMLET: To be or not to be "

# Iterate over each model configuration directory
for MODEL_DIR in "$BASE_DIR"/*/; do
    # Ensure it's a directory before proceeding
    if [ -d "$MODEL_DIR" ]; then
        # Extract the configuration name (e.g., config_01_n_layer_4_...)
        CONFIG_NAME=$(basename "$MODEL_DIR")

        # Define the paths for the model, config, and output file
        LOAD_MODEL="${MODEL_DIR}/best_model.pth"
        LOAD_CONFIG="${MODEL_DIR}/best_model_info.json"

        # Check if the required files exist
        if [ -f "$LOAD_MODEL" ] && [ -f "$LOAD_CONFIG" ]; then
            echo "Running inference for: $CONFIG_NAME"
            python main.py --mode infer \
                --load_model "$LOAD_MODEL" \
                --load_config "$LOAD_CONFIG" \
                --seed_text "$SEED_TEXT" \
		            --verbose
            echo "Inference completed for $CONFIG_NAME."
        else
            echo "Skipping $CONFIG_NAME: best_model.pth or best_model_info.json not found."
        fi
    fi
done

echo "All model inferences completed."