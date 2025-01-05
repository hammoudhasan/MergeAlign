#!/bin/bash

# List of models to evaluate
models_to_eval=(
    "beaver_tails_aaditya_Llama3_OpenBioLLM_8B.json"
)

# Loop through each model and run the evaluation
for model in "${models_to_eval[@]}"; do
    accelerate launch --multi_gpu infer.py "$model"
done
