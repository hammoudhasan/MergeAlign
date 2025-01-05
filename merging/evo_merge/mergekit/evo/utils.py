import logging
import math
import os
from typing import List, Tuple

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# Constants and Type Aliases
DatasetSample = List[dict]  # Represents a single conversation sample
TokenizedBatch = Tuple[
    torch.Tensor, int
]  # Represents a tokenized batch with prompt length

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WeightedDataset(Dataset):
    """
    A custom dataset that loads, filters, balances, and tokenizes data for training a language model.
    It supports weighting samples based on their type (e.g., task or alignment).
    """

    def __init__(
        self,
        dataset_path: str,
        task_weight: float,
        alignment_weight: float,
        num_samples_per_task: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """
        Initializes the dataset.

        Args:
            dataset_path: Path to the dataset on disk.
            task_weight: Weight for task-related samples.
            alignment_weight: Weight for alignment-related samples.
            num_samples_per_task: Number of samples to include for each type.
            tokenizer: The tokenizer to use for processing the data.
        """
        self.tokenizer = tokenizer
        self.task_weight = task_weight
        self.alignment_weight = alignment_weight
        self.dataset_path = dataset_path
        self.num_samples_per_task = num_samples_per_task
        self.data, self.scalers = self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> Tuple[List[DatasetSample], List[float]]:
        """
        Loads the dataset from disk, filters, balances it based on type,
        and assigns weights.

        Returns:
            A tuple containing the processed dataset samples and their corresponding weights.
        """
        try:
            ds = datasets.load_from_disk(self.dataset_path)
        except Exception as e:
            logging.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            raise

        data: List[DatasetSample] = []
        scalers: List[float] = []
        num_task, num_alignment = 0, 0

        for sample in ds:
            if sample["type"] == "task" and num_task < self.num_samples_per_task:
                data.append(self._reformat_sample(sample))
                scalers.append(self.task_weight)
                num_task += 1
            elif (
                sample["type"] == "alignment"
                and num_alignment < self.num_samples_per_task
            ):
                data.append(self._reformat_sample(sample))
                scalers.append(self.alignment_weight)
                num_alignment += 1

        if not (
            num_task == self.num_samples_per_task
            and num_alignment == self.num_samples_per_task
        ):
            logging.warning(
                f"Expected {self.num_samples_per_task} samples of each type, but found {num_task} task samples and {num_alignment} alignment samples."
            )

        logging.info(
            f"Loaded {len(data)} samples with {num_task} task samples and {num_alignment} alignment samples."
        )
        return data, scalers

    @staticmethod
    def _reformat_sample(sample: dict) -> DatasetSample:
        """Reformats a dataset sample into a conversation format."""
        return [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]

    def tokenize(self) -> List[TokenizedBatch]:
        """
        Tokenizes the dataset.

        Returns:
            A list of tuples, where each tuple contains the tokenized inputs and the prompt length.
        """
        batches: List[TokenizedBatch] = []
        for conversation in self.data:
            prompt_ids = self.tokenizer.apply_chat_template(
                [conversation[0]],
                truncation=True,
                max_length=2048,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                max_length=2048,
            )
            batches.append(
                (torch.tensor(ids).unsqueeze(0), len(prompt_ids))
            )  # Add unsqueeze to add a batch dimension
        return batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenize()[idx]


def compute_losses(
    model: AutoModelForCausalLM,
    scalers: List[float],
    batches: List[TokenizedBatch],
    device: str,
) -> List[float]:
    """
    Computes the weighted loss for each sample in the dataset.

    Args:
        model: The language model.
        scalers: The weights for each sample.
        batches: The tokenized dataset.
        device: The device to run the model on.

    Returns:
        A list of weighted losses.
    """
    losses: List[float] = []
    model.eval()
    with torch.inference_mode():
        for i, (inputs, prompt_len) in enumerate(
            tqdm(batches, desc="Calculating Losses")
        ):
            try:
                inputs = inputs.to(device)
                labels = inputs.clone()
                labels[:, :prompt_len] = (
                    -100
                )  # Mask out prompt tokens for loss calculation
                outputs = model(inputs, labels=labels)
                loss = scalers[i] * outputs.loss.item()
                losses.append(loss)
            except Exception as e:
                logging.error(f"Exception occurred during loss calculation: {e}")
                losses.append(math.inf)  # Indicate failure with infinity

    return losses


def load_environment_variables() -> Tuple[float, float, int]:
    """Loads and validates environment variables."""
    task_weight = os.getenv("task_weight")
    alignment_weight = os.getenv("alignment_weight")
    num_samples_per_task = os.getenv("num_samples_per_task")
    dataset_path = os.getenv("dataset_path")
    tokenizer_name = os.getenv("tokenizer_name")

    if task_weight is None:
        raise ValueError("The environment variable 'task_weight' is not set")
    if alignment_weight is None:
        raise ValueError("The environment variable 'alignment_weight' is not set")
    if num_samples_per_task is None:
        raise ValueError("The environment variable 'num_samples_per_task' is not set")

    return (
        float(task_weight),
        float(alignment_weight),
        int(num_samples_per_task),
        str(dataset_path),
        str(tokenizer_name),
    )


def main():
    """
    Main function to load the dataset, tokenize it, load the model, and compute losses.
    """
    task_weight, alignment_weight, num_samples_per_task = load_environment_variables()

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
    dataset = WeightedDataset(
        "./synth_mistral_task_and_safety_2k_weighting",
        task_weight,
        alignment_weight,
        num_samples_per_task,
        tokenizer,
    )
    batches = dataset.tokenize()

    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3-8B-Instruct",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda()

    losses = compute_losses(model, dataset.scalers, batches, "cuda")
    print(losses)


if __name__ == "__main__":
    main()
