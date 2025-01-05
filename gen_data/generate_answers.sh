python generate_answers.py --model_name aaditya/Llama3-OpenBioLLM-8B --is_llama3 --data_source json --type misalignment --json_path misalignment.json
python generate_answers.py --model_name aaditya/Llama3-OpenBioLLM-8B --is_llama3 --data_source dataset --type mmlu
python merge_responses.py
