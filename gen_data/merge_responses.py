import json
import os
from typing import Dict, List

import datasets


def load_and_preprocess_data(filepath: str, data_type: str) -> datasets.Dataset:
    """Loads data from a JSON file, adds a type field, and removes duplicates based on prompts.

    Args:
        filepath: Path to the JSON file.
        data_type: The type to assign to the 'type' field (e.g., 'alignment', 'task').

    Returns:
        A Hugging Face Dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    for sample in data:
        sample["type"] = data_type

    ds = datasets.Dataset.from_list(data)
    return filter_duplicates_by_prompt(ds)


def filter_duplicates_by_prompt(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filters out duplicate entries in a dataset based on the 'prompt' field.

    Args:
        dataset: The input Hugging Face Dataset.

    Returns:
        A new Dataset with duplicates removed.
    """
    seen_prompts = set()
    unique_indices = []
    for i, example in enumerate(dataset):
        if example["prompt"] not in seen_prompts:
            seen_prompts.add(example["prompt"])
            unique_indices.append(i)
    return dataset.select(unique_indices)


def main():
    """Loads, preprocesses, combines, shuffles, and saves datasets."""

    # Define file paths and corresponding data types
    data_sources = [
        (
            "./model_responses/misalignment_aaditya_Llama3_OpenBioLLM_8B.json",
            "alignment",
        ),
        ("./model_responses/mmlu_aaditya_Llama3_OpenBioLLM_8B.json", "task"),
    ]

    # Load and preprocess each dataset
    datasets_list = []
    for filepath, data_type in data_sources:
        try:
            ds = load_and_preprocess_data(filepath, data_type)
            datasets_list.append(ds)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Skipping this dataset and continuing.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filepath}: {e}")
            print("Skipping this dataset and continuing.")

    # Concatenate and shuffle datasets, handling potential empty dataset list
    if datasets_list:
        combined_ds = datasets.concatenate_datasets(datasets_list)
        combined_ds = combined_ds.shuffle(seed=42)  # Added seed for reproducibility
        print(combined_ds)
        print(combined_ds[0])
        print(combined_ds[1])

        # Save the combined dataset
        combined_ds.save_to_disk("openbio_llama3_task_w_alignment_data")
    else:
        print("No datasets were successfully loaded. Exiting.")


if __name__ == "__main__":
    main()
