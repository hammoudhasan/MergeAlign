# Model Merging and Safety Alignment: One Bad Model Spoils the Bunch

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2406.14563-b31b1b.svg)](https://arxiv.org/abs/2406.14563)
[![Conference](https://img.shields.io/badge/EMNLP-2024-blue)](https://aclanthology.org/2024.findings-emnlp.762/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3.10/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/hammoudhasan/MergeAlign/pulls)

</div>


<div align="center" style="font-size: 1.5em;">

[Hasan Abed Al Kader Hammoud](https://hasanhammoud.com/) ·
[Umberto Michieli](https://umbertomichieli.github.io/) ·
[Fabio Pizzati](https://fabvio.github.io/) ·
[Philip Torr](https://eng.ox.ac.uk/people/philip-torr/) ·
[Adel Bibi](https://www.adelbibi.com/) ·
[Bernard Ghanem](https://www.bernardghanem.com/) ·
[Mete Ozay](https://ieeexplore.ieee.org/author/38331422900)

</div>



<div align="center">
        <img src="assets/img.png" alt="Model Merging and Safety Alignment" width="600">
</div>

> 📌 *This work was completed during an internship of Hasan Abed Al Kader Hammoud at Samsung Research UK.*

---

## 📖 Abstract

Merging Large Language Models (LLMs) is a cost-effective technique for combining multiple expert LLMs into a single versatile model, retaining the expertise of the original ones. However, current approaches often overlook the importance of safety alignment during merging, leading to highly misaligned models. This work investigates the effects of model merging on alignment. We evaluate several popular model merging techniques, demonstrating that existing methods do not only transfer domain expertise but also propagate misalignment. We propose a simple two-step approach to address this problem: (i) generating synthetic safety and domain-specific data, and (ii) incorporating these generated data into the optimization process of existing data-aware model merging techniques. This allows us to treat alignment as a skill that can be maximized in the resulting merged LLM. Our experiments illustrate the effectiveness of integrating alignment-related data during merging, resulting in models that excel in both domain expertise and alignment.

## ✨ Overview

Welcome to the official repository for our EMNLP 2024 paper, "Model Merging and Safety Alignment: One Bad Model Spoils the Bunch"! 🎉 We introduce a novel approach to merging Large Language Models (LLMs) while prioritizing safety alignment. Our research demonstrates that existing merging methods can inadvertently propagate misalignment and offers robust solutions to this critical challenge.



## 🗂️ Repository Structure

Our repository is organized into the following key components:

1. **Data Generation (`gen_data/`)**: Scripts and tools for generating synthetic task and alignment data to be used later for the data aware merging.
2. **Model Merging (`merging/`)**: Implementations of data aware model merging techniques, including LM Cocktail and Evolutionary Merge.
3. **Evaluation (`evaluation/`)**: Tools and scripts for evaluating merged models using LLaMA Guard (alignment) and LM Harness (task).

## ⚙️ Usage

### 1. Data Generation (`gen_data/`)

Our data generation pipeline consists of three main steps:

#### a. Generate Misaligned Questions

```bash
cd gen_data
bash generate_misaligned_synthetic.sh
```

#### b. Generate Synthetic MMLU Questions

```bash
bash generate_mmlu_questions.sh
```

> **Note**: By default, the script generates 10 samples for testing. For production use (as in our paper), we generated \~2,000 samples. Adjust the number of samples in the script as needed.

#### c. Generate Model Responses

```bash
bash generate_answers.sh
```

This step also tags the data as either 'alignment' or 'task' for guided merging.

### 2. Model Merging Methods

#### a. LM Cocktail (`merging/lm_cocktail/`)

Our implementation of the LM Cocktail approach with alignment considerations:

1. **Setup**:

    ```bash
    cd merging/lm_cocktail
    ```
2. **Configuration**:

    Update `utils.py` to handle specific models:

    ```python
    def preprocess_data_for_llm(example_data, tokenizer, device, batch_size:int=2, max_input_length:int=2048):
        batch_input_ids = []
        batch_labels = []
        batch_max_length = max_input_length
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Required for some models
    ```
3. **Run Merging**:

    ```bash
    python lmcocktail_merge_align.py --all_types --temp 1.0 --max_per_task 5
    ```

    **Parameters**:

    -   `--all_types`: Enable all merging types
    -   `--temp`: Sampling temperature (default: 1.0)
    -   `--max_per_task`: Maximum tasks per run (default: 5)

#### b. Evolutionary Merge (`merging/evo_merge/`)

Evolutionary-based model merging with safety constraints:

```bash
cd merging/evo_merge
# Run evolutionary merging with different configurations
mergekit-evolve ./examples/genomic_1.yml --storage-path ./mistralv02_mammoth7b_ties_1_synthetic_task_and_safety_2k_100_1_0p3 --task-search-path workspace/eval_tasks/ --merge-cuda --max-fevals 100
```

**Configuration**:

-   Use example configs in `examples/` (e.g., `genomic_1.yml`)
-   Modify parameters in config files for different merging strategies

### 3. Evaluation Tools

#### a. LLaMA Guard Evaluation

The evaluation uses `accelerate` for multi-GPU inference:

```bash
cd evaluation/llama_guard_eval

# The evaluation script supports multiple models
# Models are specified in JSON format (e.g., "beaver_tails_aaditya_Llama3_OpenBioLLM_8B.json")
./runner.sh

# Under the hood, it runs:
accelerate launch --multi_gpu infer.py $responses_path
```


#### b. LM Harness Evaluation

We evaluate models on three main categories:

1. **Biology Tasks**:

    ```bash
    lm_eval --model hf --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
            --tasks 'mmlu_college_medicine,mmlu_professional_medicine,mmlu_anatomy,mmlu_clinical_knowledge,mmlu_medical_genetics,medqa_4options,pubmedqa,medmcqa,mmlu_college_biology' \
            --batch_size 16
    ```
2. **MMLU STEM**:

    ```bash
    lm_eval --model hf --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
            --tasks 'mmlu' \
            --batch_size 16
    ```
3. **Three-Model Merging Tasks**:

    ```bash
    lm_eval --model hf --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
            --tasks 'winogrande,arc_challenge' \
            --batch_size 16
    ```

> **Note**: Replace `$MODEL_PATH` with your model's path and adjust the batch size based on available resources.


## 📝 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{hammoud-etal-2024-model,
    title = "Model Merging and Safety Alignment: One Bad Model Spoils the Bunch",
    author = "Hammoud, Hasan Abed Al Kader  and
      Michieli, Umberto  and
      Pizzati, Fabio  and
      Torr, Philip  and
      Bibi, Adel  and
      Ghanem, Bernard  and
      Ozay, Mete",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.762",
    doi = "10.18653/v1/2024.findings-emnlp.762",
    pages = "13033--13046",
}
```
