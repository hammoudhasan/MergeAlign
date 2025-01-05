#!/bin/bash

# List of models to evaluate
models_to_eval=(
    "beaver_tails_google_gemma_2_9b_it.json"
)

# Loop through each model and run the evaluation
for model in "${models_to_eval[@]}"; do
    accelerate launch --multi_gpu infer.py "$model"
done
