# Synthetic Question and Response Generation

This repository provides scripts to generate synthetic datasets for alignment and task purposes using a specified model. Below are detailed instructions for running each component of the pipeline.

## Scripts Overview

### 1. **Generate Misaligned Questions**
   To generate misaligned questions using your specified model, run the following command:
   ```bash
   bash generate_misaligned_synthetic.sh
   ```
   This script creates misaligned questions as part of the synthetic dataset preparation process.

### 2. **Generate Synthetic MMLU Questions**
   To generate synthetic MMLU questions for various tasks, run:
   ```bash
   bash generate_mmlu_questions.sh
   ```
   - **Process Overview**:
     - Questions are generated for various MMLU tasks.
     - The generated questions are combined using the `combine_synthetic_mmlu.py` script.
     - Sometimes, the model generates multiple questions in a single call. This is handled by flattening the JSON in the script.

   - **Note**: During testing, the default number of samples is set to 10. For production use (as outlined in our paper), we generated approximately 2,000 samples. 
     - **Error Handling**: Some model outputs may not adhere to the expected JSON format. These cases are handled with a `try-except` block, and invalid samples are discarded.
     - **Subsampling**: To achieve the desired number of alignment samples, the valid samples are randomly subsampled.

### 3. **Generate Model Responses**
   To obtain model responses for the generated MMLU questions, use:
   ```bash
   bash generate_clean_responses.sh
   ```
   This code will also make sure that the data is resaved in the format we will need to the guided merging. What does this mean ? It means it will also add tags for `alignment` or `task` to know the source of the data when we apply the merging scaling. More on that later in the model merging code!