# LM Evaluation Harness

## Setup Instructions

Please refer to the [LM Evaluation Harness repository](https://github.com/EleutherAI/lm-evaluation-harness) for detailed setup and installation instructions.

## Evaluation Categories

In our paper, we conducted three main types of evaluations on clean tasks. Below are the corresponding commands:

### 1. Biology Evaluation

Evaluate your model on a collection of biology-related tasks:

```bash
lm_eval --model hf --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
        --tasks 'mmlu_college_medicine,mmlu_professional_medicine,mmlu_anatomy,mmlu_clinical_knowledge,mmlu_medical_genetics,medqa_4options,pubmedqa,medmcqa,mmlu_college_biology' \
        --batch_size 16
```

### 2. MMLU STEM Evaluation

Evaluate your model on the STEM subset of the MMLU benchmark:

```bash
lm_eval --model hf --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
        --tasks 'mmlu' \
        --batch_size 16
```

> **Note:** This command runs the full MMLU benchmark, but the logs will contain results specifically for the MMLU STEM split.

### 3. Three-Model Merging Evaluation

Evaluate the performance of your merged models on specific reasoning tasks:

```bash
lm_eval --model hf --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
        --tasks 'winogrande,arc_challenge' \
        --batch_size 16
```

## Notes

- Replace `$MODEL_PATH` with the path to your pretrained model.
- Use an appropriate batch size depending on your computational resources.
