"""
This script generates practice questions for exams using a pre-trained causal language model. 
It processes a dataset of existing questions, formulates prompts, performs inference across 
multiple GPUs, and outputs the generated questions in a structured JSON format.
"""

import argparse
import json
import logging
import os
import random
import re
import time
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
        description="Generate practice questions using a language model."
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
        "--task",
        type=str,
        required=True,
        help="Task identifier for the generated output.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate.",
    )

    args = parser.parse_args()
    logging.info("Parsed command-line arguments.")
    logging.debug(f"Arguments: {args}")
    return args


def write_pretty_json(file_path: str, data: Any) -> None:
    """
    Write data to a JSON file with pretty formatting.

    Args:
        file_path (str): Path to the output JSON file.
        data (Any): Data to be written.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as write_file:
            json.dump(data, write_file, indent=4)
        logging.info(f"Successfully wrote data to {file_path}.")
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
        main_prompt (str): Main prompt template for generating questions.
        prefix_prompt (str): Prefix to be added to each generated response.
        batch_size (int, optional): Number of prompts per batch. Defaults to 25.

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], List[List[str]]]:
            Tokenized prompt batches and the original prompt batches.
    """
    logging.info("Preparing prompts for tokenization.")
    batched_prompts = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]
    tokenized_batches: List[Dict[str, torch.Tensor]] = []
    original_batches: List[List[str]] = []

    for batch_idx, batch in enumerate(batched_prompts):
        logging.debug(f"Processing batch {batch_idx + 1}/{len(batched_prompts)}.")
        tokenized_batch = []
        original_batch = []
        for prompt in batch:
            messages = [
                {"role": "user", "content": system_prompt + main_prompt},
            ]

            try:
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                input_ids += prefix_prompt
                tokenized_batch.append(input_ids)
                original_batch.append(prompt)
                logging.debug(f"Generated input_ids for prompt: {prompt}")
            except Exception as e:
                logging.error(f"Error applying chat template: {e}")

        original_batches.append(original_batch)

        try:
            tokenizer.padding_side = "left"
            tokenized = tokenizer(
                tokenized_batch,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            ).to("cuda")
            tokenized_batches.append(tokenized)
            logging.debug(f"Tokenized batch {batch_idx + 1} successfully.")
        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
        finally:
            tokenizer.padding_side = "right"

    logging.info("Completed preparing all prompts.")
    return tokenized_batches, batched_prompts


def main() -> None:
    """
    Main function to execute the script.
    """
    setup_logging()
    logging.info("Script started.")
    args = parse_arguments()
    accelerator = Accelerator()

    try:
        # Load tokenizer and model
        logging.info(f"Loading tokenizer from {args.model_name}.")
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer loaded successfully.")

        logging.info(f"Loading model from {args.model_name}.")
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        return

    # Load dataset
    try:
        logging.info(f"Loading dataset for task: {args.task}.")
        ds = datasets.load_dataset("hails/mmlu_no_train", args.task)["validation"]
        logging.info("Dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # Extract sample questions and answers
    try:
        question_1 = ds[0]["question"]
        question_2 = ds[1]["question"]
        question_3 = ds[2]["question"]

        choices = ds[0]["choices"]
        answer = choices[ds[0]["answer"]]
        topic = ds[0]["subject"].replace("_", " ").title()

        # Adding letters to choices
        choices_with_letters = [
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ]
        answer_letter = chr(65 + ds[0]["answer"]) + ". " + answer

        logging.info(f"Selected topic: {topic}")
        logging.debug(f"Answer letter: {answer_letter}")
    except Exception as e:
        logging.error(f"Error extracting sample data from dataset: {e}")
        return

    # Define system prompt with examples
    system_prompt: str = f"""
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

    logging.debug(f"System prompt: {system_prompt}")

    main_prompt: str = (
        "While adhering to the JSON format, please generate a single question for me to practice on."
    )
    prompts_all: List[str] = [main_prompt] * args.num_samples

    prefix_prompt: str = """
    [
        {
            "Question": 
    """

    logging.info(f"Prepared {len(prompts_all)} prompts.")

    # Determine terminator tokens based on model type
    if args.is_llama3:
        terminators: List[int] = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        logging.info("Using LLaMA3 terminators.")
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]
        logging.info("Using default terminators.")

    # Synchronize GPUs and start the timer
    accelerator.wait_for_everyone()
    start_time: float = time.time()
    logging.info("Starting inference process.")

    try:
        # Split prompts across available GPUs
        with accelerator.split_between_processes(prompts_all) as split_prompts:
            results: Dict[str, List[Dict[str, str]]] = {"outputs": []}
            logging.info("Prompts split among processes.")

            # Prepare prompts for batching and tokenization
            prompt_batches, original_batches = prepare_prompts(
                split_prompts,
                tokenizer,
                system_prompt,
                main_prompt,
                prefix_prompt,
                batch_size=25,
            )

            # Perform inference on each batch
            for idx, tokenized_prompts in enumerate(
                tqdm(prompt_batches, desc="Generating Questions")
            ):
                try:
                    generated_ids = model.generate(
                        **tokenized_prompts,
                        max_new_tokens=1024,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.8,
                    )
                    logging.debug(f"Generated tokens for batch {idx + 1}.")
                except Exception as e:
                    logging.error(f"Error during generation for batch {idx + 1}: {e}")
                    continue

                # Remove prompt tokens from the generated output
                try:
                    generated_texts: List[List[int]] = [
                        generated[len(prompt) :]
                        for prompt, generated in zip(
                            tokenized_prompts["input_ids"], generated_ids
                        )
                    ]
                    logging.debug(f"Stripped prompt tokens for batch {idx + 1}.")
                except Exception as e:
                    logging.error(
                        f"Error stripping prompt tokens for batch {idx + 1}: {e}"
                    )
                    continue

                # Decode the generated tokens
                try:
                    decoded_outputs: List[str] = tokenizer.batch_decode(
                        generated_texts, skip_special_tokens=True
                    )
                    logging.debug(f"Decoded outputs for batch {idx + 1}.")
                except Exception as e:
                    logging.error(f"Error decoding outputs for batch {idx + 1}: {e}")
                    continue

                batch_results: List[Dict[str, str]] = [
                    {"response": prefix_prompt + response}
                    for response in decoded_outputs
                ]

                # Aggregate results
                results["outputs"].extend(batch_results)
                logging.info(f"Processed batch {idx + 1}/{len(prompt_batches)}.")

            # Prepare results for gathering across GPUs
            gathered_results: List[Dict[str, List[Dict[str, str]]]] = [results]
            logging.info("Gathered results from all batches.")
    except Exception as e:
        logging.error(f"Error during inference process: {e}")
        return

    # Collect results from all GPUs
    try:
        all_results: List[Dict[str, Any]] = gather_object(gathered_results)
        flattened_results: List[Dict[str, str]] = [
            output for result in all_results for output in result["outputs"]
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
        logging.warning("No responses were generated.")

    # Process and validate the generated JSON responses
    regexed_generated_texts: List[Dict[str, Any]] = []
    skipped_samples: int = 0
    total_samples: int = len(flattened_results)

    for idx, output in enumerate(flattened_results):
        try:
            response: str = output["response"]
            response = response.strip()
            response = re.sub(r"(\n)(?!\")", " ", response)
            data: Dict[str, Any] = json.loads(
                response
            )  # Changed from ast.literal_eval to json.loads
            regexed_generated_texts.append(data)
            logging.debug(f"Valid response parsed for entry {idx + 1}.")
        except json.JSONDecodeError as e:
            skipped_samples += 1
            logging.warning(f"Skipping invalid response at entry {idx + 1}: {e}")
            continue
        except Exception as e:
            skipped_samples += 1
            logging.warning(
                f"Unexpected error parsing response at entry {idx + 1}: {e}"
            )
            continue

    valid_samples: int = len(regexed_generated_texts)
    logging.info(f"Total valid responses: {valid_samples}")
    logging.info(f"Total skipped responses: {skipped_samples}")
    print(f"Total valid responses: {valid_samples}")
    print(f"Total skipped responses: {skipped_samples}")

    # Save the processed responses to a JSON file
    output_file: str = f"./task_data/{args.task}.json"
    write_pretty_json(output_file, regexed_generated_texts)
    logging.info(f"Responses saved to {output_file}")

    # Optionally, print the time taken
    end_time: float = time.time()
    elapsed_time: float = end_time - start_time
    logging.info(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
