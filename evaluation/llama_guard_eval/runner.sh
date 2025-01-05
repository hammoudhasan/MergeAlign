#!/bin/bash

# List of models to evaluate
models_to_eval=(
    "sample.json"
)

# Loop through each model and run the evaluation
for model in "${models_to_eval[@]}"; do
    accelerate launch --multi_gpu infer.py "$model"
done
