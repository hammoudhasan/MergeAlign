# For OpenBio for example, we will loop over thoses tasks, generate data and use it.
task_list=("college_medicine" "professional_medicine" "anatomy" "clinical_knowledge" "medical_genetics" "medqa_4options" "pubmedqa" "medmcqa" "college_biology")

for task in "${task_list[@]}"; do
    accelerate launch generate_mmlu_task_questions.py --model_name aaditya/Llama3-OpenBioLLM-8B --is_llama3 --task ${task}
done


# Combine data into parquet dataset format from huggingface 
python combine_synthetic_mmlu.py
