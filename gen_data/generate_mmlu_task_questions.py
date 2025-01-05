"""
This script generates exam practice questions using a pre-trained causal language model. 
It loads a dataset from the HugginFace Hub, extracts a few sample questions, and instructs 
the model to produce a single JSON-formatted question per prompt. The generated JSON strings 
are then parsed and saved to a file. 
"""

import argparse
import ast
import json
import logging
import os
import random
import re
import time
from statistics import mean
from typing import Any, Dict, List, Tuple

import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging() -> None:
    """
    Configure the logging settings for the script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("data_generation.log"), logging.StreamHandler()],
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate exam practice questions using a language model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the pre-trained model.",
    )
    parser.add_argument(
        "--is_llama3",
        action="store_true",
        help="Flag indicating if the model is LLaMA3.",
    )
    parser.add_argument(
        "--task", type=str, required=True, help="The dataset task identifier."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of questions to generate.",
    )

    args = parser.parse_args()
    logging.info("Parsed command-line arguments.")
    logging.debug(f"Arguments: {args}")
    return args


def write_pretty_json(file_path: str, data: Any) -> None:
    """
    Write data to a JSON file with pretty formatting.

    Args:
        file_path (str): The output file path.
        data (Any): Data to be saved in the JSON file.
    """
    try:
        with open(file_path, "w") as write_file:
            json.dump(data, write_file, indent=4)
        logging.info(f"Data successfully written to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to write JSON to {file_path}: {e}")


def prepare_prompts(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    system_prompt: str,
    main_prompt: str,
    prefix_prompt: str,
    batch_size: int = 25,
) -> Tuple[List[Dict[str, torch.Tensor]], List[List[str]]]:
    """
    Batch and tokenize prompts for model inference.

    Args:
        prompts (List[str]): List of prompt strings.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        system_prompt (str): System-level prompt to guide the model.
        main_prompt (str): Main prompt to append.
        prefix_prompt (str): A prefix to be added to the input IDs.
        batch_size (int, optional): Number of prompts per batch. Defaults to 25.

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], List[List[str]]]:
            A tuple where:
            - The first element is a list of dictionaries containing tokenized prompts.
            - The second element is a list of the original prompt batches.
    """
    logging.info("Preparing prompts for tokenization.")
    batched_prompts = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]
    tokenized_batches: List[Dict[str, torch.Tensor]] = []
    original_batches: List[List[str]] = []

    for batch_idx, batch in enumerate(batched_prompts):
        logging.debug(f"Processing batch {batch_idx + 1}/{len(batched_prompts)}.")
        inner_level_batches = []
        original_batch = []

        for prompt in batch:
            messages = [
                {"role": "user", "content": system_prompt + main_prompt},
            ]

            try:
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                # Append the prefix to the existing prompt
                input_ids += prefix_prompt

                inner_level_batches.append(input_ids)
                original_batch.append(prompt)
                logging.debug(f"Created input IDs for prompt: {prompt}")
            except Exception as e:
                logging.error(f"Error applying chat template: {e}")

        original_batches.append(original_batch)

        try:
            tokenizer.padding_side = "left"
            tokenized = tokenizer(
                inner_level_batches,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            ).to("cuda")
            tokenized_batches.append(tokenized)
            logging.debug(f"Batch {batch_idx + 1} tokenized successfully.")
        except Exception as e:
            logging.error(f"Error during tokenization for batch {batch_idx + 1}: {e}")
        finally:
            tokenizer.padding_side = "right"

    logging.info("All prompts have been prepared.")
    return tokenized_batches, batched_prompts


def main() -> None:
    """
    Main function to execute the script's workflow.
    """
    setup_logging()
    logging.info("Script started.")
    args = parse_arguments()

    accelerator = Accelerator()

    # Load the tokenizer and model
    model_path = args.model_name
    try:
        logging.info(f"Loading tokenizer from {model_path}.")
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"Loading model from {model_path}.")
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        return

    # Load dataset
    try:
        ds = datasets.load_dataset("hails/mmlu_no_train", args.task)["validation"]
        logging.info(f"Loaded dataset 'hails/mmlu_no_train', task: {args.task}.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Extract sample questions and metadata for demonstration
    try:
        question_1 = ds[0]["question"]
        question_2 = ds[1]["question"]
        question_3 = ds[2]["question"]
        choices = ds[0]["choices"]
        answer = choices[ds[0]["answer"]]
        topic = ds[0]["subject"].replace("_", " ").title()

        logging.debug(f"Sample questions: {question_1}, {question_2}, {question_3}")
        logging.debug(f"Choices: {choices}, Answer: {answer}, Topic: {topic}")
    except IndexError as ie:
        logging.error(f"Index out of range when accessing dataset: {ie}")
        return

    # Print the lettered answer to confirm correct indexing
    answer_letter = chr(65 + ds[0]["answer"]) + ". " + answer
    print(f"Answer letter for the first question: {answer_letter}")

    # Build system prompt
    system_prompt = f"""
You are an expert in {topic}. Your task is to generate questions for me to practice for my exam.

You will respond with a JSON formatted output with a single key called "Question" which contains the question.

The following are good examples of what you should output. Remember the content must be only the question.

Example 1:

[
    {{
    "Question": "{question_1}",
    }}
]

Example 2:

[
    {{
    "Question": "{question_2}",
    }}
]

Example 3:

[
    {{
    "Question": "{question_3}",
    }}
]

DO NOT PROVIDE THE ANSWER.
    """

    logging.info(f"System prompt:\n{system_prompt}")

    main_prompt = " While adhering to the JSON format, please generate a single question for me to pratice on. "
    prefix_prompt = """
[
    {
        "Question": """

    # Number of prompts to generate
    prompts_all = [main_prompt] * args.num_samples
    logging.info(f"Number of prompts created: {len(prompts_all)}")

    # Determine appropriate terminator tokens
    if args.is_llama3:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        logging.info("Using LLaMA3 terminators.")
    else:
        terminators = [tokenizer.eos_token_id]
        logging.info("Using default terminators.")

    # Prepare for inference
    accelerator.wait_for_everyone()
    start_time: float = time.time()

    # Split the prompts across available GPUs
    with accelerator.split_between_processes(prompts_all) as split_prompts:
        results: Dict[str, List[Dict[str, str]]] = {"outputs": []}

        # Prepare prompts in batches
        try:
            prompt_batches, original_prompts = prepare_prompts(
                split_prompts,
                tokenizer,
                system_prompt,
                main_prompt,
                prefix_prompt,
                batch_size=24,
            )
        except Exception as e:
            logging.error(f"Error preparing prompts: {e}")
            return

        # Perform inference batch by batch
        for idx, prompts_tokenized in enumerate(
            tqdm(prompt_batches, desc="Generating Responses")
        ):
            try:
                outputs_tokenized = model.generate(
                    **prompts_tokenized,
                    max_new_tokens=1024,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.8,
                )
                logging.debug(f"Generated tokens for batch {idx + 1}.")
            except Exception as e:
                logging.error(f"Error generating tokens for batch {idx + 1}: {e}")
                continue

            # Remove the prompt tokens from the generated output
            try:
                outputs_tokenized = [
                    out_tok[len(in_tok) :]
                    for in_tok, out_tok in zip(
                        prompts_tokenized["input_ids"], outputs_tokenized
                    )
                ]
                logging.debug(f"Removed prompt tokens for batch {idx + 1}.")
            except Exception as e:
                logging.error(f"Error stripping prompt tokens for batch {idx + 1}: {e}")
                continue

            # Decode the outputs
            try:
                decoded_outputs = tokenizer.batch_decode(
                    outputs_tokenized, skip_special_tokens=True
                )
                logging.debug(f"Decoded outputs for batch {idx + 1}.")
            except Exception as e:
                logging.error(f"Error decoding outputs for batch {idx + 1}: {e}")
                continue

            # Combine the prefix back into the response for easy parsing
            # (as done in your original code)
            batch_results = [
                {"response": prefix_prompt + response}
                for prompt, response in zip(original_prompts[idx], decoded_outputs)
            ]

            # Store the results for later gathering
            results["outputs"].extend(batch_results)
            logging.info(f"Processed batch {idx + 1}/{len(prompt_batches)}.")

        # Prepare for gathering across GPUs
        gathered_results = [results]
        logging.info("Batch results gathered.")

    # Collect results from all GPUs
    try:
        results_gathered = gather_object(gathered_results)
        flattened_results = [
            output for item in results_gathered for output in item["outputs"]
        ]
        logging.info(f"Collected {len(flattened_results)} responses from all GPUs.")
    except Exception as e:
        logging.error(f"Error gathering results: {e}")
        return

    # Display the first response as a sanity check
    if flattened_results:
        logging.info(f"First response: {flattened_results[0]['response']}")
        print(flattened_results[0]["response"])
    else:
        logging.warning("No responses generated after inference.")

    # Parse and validate the generated outputs
    valid_data: List[Any] = []
    skipped_samples: int = 0
    for idx, output in enumerate(flattened_results):
        raw_response = output["response"].strip()
        raw_response = re.sub(r"(\n)(?!\")", " ", raw_response)

        try:
            # Attempt to parse the string as a Python literal
            data_obj = ast.literal_eval(raw_response)
            valid_data.append(data_obj)
        except Exception as e:
            skipped_samples += 1
            logging.warning(f"Skipping invalid response at entry {idx + 1}: {e}")
            continue

    logging.info(f"Total valid responses: {len(valid_data)}")
    logging.info(f"Total skipped responses: {skipped_samples}")
    print(f"Total valid responses: {len(valid_data)}")
    print(f"Total skipped responses: {skipped_samples}")

    # Ensure the output directory exists
    output_dir = "./task_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save valid data to a JSON file
    output_file = os.path.join(output_dir, f"{args.task}.json")
    write_pretty_json(output_file, valid_data)
    logging.info(f"Saved valid responses to {output_file}")

    # Print the total time taken
    end_time: float = time.time()
    elapsed_time: float = end_time - start_time
    logging.info(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
