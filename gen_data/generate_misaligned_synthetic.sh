# Generate alignment questions
accelerate launch --multi_gpu --mixed_precision bf16 generate_misalignment_questions.py --model_name cognitivecomputations/dolphin-2.9-llama3-8b --is_llama3 --task misalignment --num_samples 20
