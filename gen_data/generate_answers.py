import argparse
import json
import os
import time

import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path or identifier of the model to load.",
    )
    parser.add_argument(
        "--is_llama3",
        action="store_true",
        help="If specified, we add <|eot_id|> to the end-of-text token IDs.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="dataset",
        help="Whether to load from a dataset or JSON file. Options: [dataset, json].",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="mmlu",
        help="The type of dataset to use if data_source='dataset' (openbio, mmlu, winogrande, etc.). "
        "If data_source='json', this is just used for naming the output file.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="misalignment.json",
        help="Path to a JSON file if data_source='json'. The file should be a list of objects.",
    )
    args = parser.parse_args()
    return args


args = arguments()
accelerator = Accelerator()

# Load model and tokenizer
model_path = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)

# Prepare the prompts based on user arguments
prompts_all = []

if args.data_source == "dataset":
    # Load from disk a huggingface dataset
    if args.type == "mmlu":
        dataset = datasets.load_from_disk("synthetic_questions_mmlu")
    else:
        # If some unknown `type` is provided, just exit
        raise ValueError(f"Unknown --type '{args.type}' for data_source='dataset'")

    # Extract the questions from the 'prompt' key
    for item in dataset:
        prompts_all.append(item["prompt"])

elif args.data_source == "json":
    # Load from a JSON file (should be a list of dicts)
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    # We assume that each item might contain a "Question" field,
    # or adapt as needed
    for item in dataset_json:
        # Adjust to your JSON structure if different
        try:
            question = item.get("Question", "")
            if question:
                prompts_all.append(question)
        except:
            print("Key not found!")
            pass
else:
    raise ValueError(f"Invalid data_source: {args.data_source}")


def write_pretty_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, indent=4, ensure_ascii=False)


# Setup terminator tokens
if args.is_llama3:
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
else:
    terminators = [
        tokenizer.eos_token_id,
    ]


# Batch, left pad, and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=24):
    batches_temp = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]
    batches_tok = []

    upper_level_batches = []
    for batch in batches_temp:
        inner_level_batches = []
        for prompt in batch:
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inner_level_batches.append(input_ids)
        upper_level_batches.append(inner_level_batches)

    tokenizer.padding_side = "left"
    for prompt_batch in upper_level_batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            ).to("cuda")
        )
    tokenizer.padding_side = "right"

    return batches_tok, batches_temp


# Sync GPUs and start timing
accelerator.wait_for_everyone()
start = time.time()

# Distribute the prompt list across available GPUs
with accelerator.split_between_processes(prompts_all) as prompts:
    results = dict(outputs=[])

    # Have each GPU do inference in batches
    prompt_batches, original_prompts = prepare_prompts(
        prompts, tokenizer, batch_size=48
    )

    for idx, prompts_tokenized in enumerate(tqdm(prompt_batches)):
        outputs_tokenized = model.generate(
            **prompts_tokenized,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False,
            top_p=0.0,
            temperature=0.0,
        )

        # Remove the prompt tokens from the generated tokens
        outputs_tokenized = [
            tok_out[len(tok_in) :]
            for tok_in, tok_out in zip(
                prompts_tokenized["input_ids"], outputs_tokenized
            )
        ]

        # Decode
        outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)

        # Pair prompts with responses
        combined = [
            {"prompt": prompt, "response": response}
            for prompt, response in zip(original_prompts[idx], outputs)
        ]

        results["outputs"].extend(combined)

    # Transform to list so gather_object() collects properly
    results = [results]

# Collect results from all GPUs
results_gathered = gather_object(results)

# Flatten them
flattened_results = [output for item in results_gathered for output in item["outputs"]]

# Format the model name for saving
if model_path.count("/") == 1:
    model_id_trimmed = model_path.replace("/", "_").replace("-", "_")
else:
    model_id_trimmed = model_path.split("/")[-1].replace("-", "_")

# Make sure the output directory exists
if not os.path.exists("./model_responses/"):
    os.makedirs("./model_responses/")

# Save to a JSON
output_filename = f"{args.type}_{model_id_trimmed}.json"
output_path = os.path.join("./model_responses", output_filename)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(flattened_results, f, indent=4, ensure_ascii=False)
