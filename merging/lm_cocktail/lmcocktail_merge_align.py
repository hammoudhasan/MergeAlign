import argparse
import logging

import datasets
from LM_Cocktail import mix_models_with_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_prepare_dataset(dataset_path):
    """
    Loads a dataset from the specified path and shuffles it.

    Args:
        dataset_path (str): The path to the dataset directory.

    Returns:
        datasets.Dataset: The loaded and shuffled dataset, or None if an error occurred.
    """
    try:
        ds = datasets.load_from_disk(dataset_path)
        ds = ds.shuffle()
        return ds
    except Exception as e:
        logging.error(f"Error loading dataset from {dataset_path}: {e}")
        return None


def collect_example_data(ds, include_all_types, max_per_task=1000):
    """
    Collects example data from the dataset based on specified criteria.

    Args:
        ds (datasets.Dataset): The dataset to collect examples from.
        include_all_types (bool): Whether to include all example types or only 'task' type.
        max_per_task (int): The maximum number of 'task' type examples to include.

    Returns:
        list: A list of dictionaries, where each dictionary represents an example
              with 'input' and 'output' keys.
    """
    example_data = []
    total_task = 0
    for example in ds:
        if example:
            prompt = example["prompt"]
            response = example["response"]
            if example["type"] == "task":
                if total_task < max_per_task:
                    example_data.append({"input": prompt, "output": response})
                    total_task += 1
            elif include_all_types:
                example_data.append({"input": prompt, "output": response})
    return example_data


def main():
    """
    Main function to mix language models using example data and specified parameters.

    This function parses command-line arguments, loads and prepares a dataset,
    collects example data, and then uses the LM_Cocktail library to mix
    specified language models based on the collected data.
    """
    parser = argparse.ArgumentParser(
        description="Mix language models with example data."
    )
    parser.add_argument(
        "--temp",
        type=float,
        required=True,
        help="The temperature for model mixing (higher values increase randomness).",
    )
    parser.add_argument(
        "--all_types",
        action="store_true",
        help="Include all example types from the dataset (not just examples of type 'task').",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/ibex/ai/project/c2260/hasan/MergeAlign/gen_data/openbio_llama3_task_w_alignment_data",
        help="The path to the dataset directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The base path for saving the output mixed model. A descriptive suffix will be added to this path.",
    )
    parser.add_argument(
        "--max_per_task",
        type=int,
        default=10,
        help="The maximum number of examples of type 'task' to include from the dataset.",
    )
    args = parser.parse_args()

    # Log the parameters
    logging.info(f"Temperature: {args.temp}")
    logging.info(f"Include all example types: {args.all_types}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Output path: {args.output_path}")
    logging.info(f"Max tasks: {args.max_per_task}")

    # Load and prepare dataset
    ds = load_and_prepare_dataset(args.dataset_path)
    if ds is None:
        logging.error("Dataset loading failed. Exiting.")
        return

    # Collect example data
    example_data = collect_example_data(ds, args.all_types, args.max_per_task)
    logging.info(f"Total examples collected: {len(example_data)}")

    # Construct output path if not provided
    if args.output_path is None:
        output_path = f"./lmcocktail_{int(args.all_types)}_{args.temp}_openbio"
    else:
        output_path = (
            f"{args.output_path}_lmcocktail_{int(args.all_types)}_{args.temp}_openbio"
        )

    # Mix models with the collected example data
    model_names = [
        "aaditya/Llama3-OpenBioLLM-8B",  # Example model name 1
        "meta-llama/Meta-Llama-3-8B-Instruct",  # Example model name 2
    ]
    #try:
    print(model_names, example_data, args.temp, output_path)
    model = mix_models_with_data(
        model_names_or_paths=model_names,
        model_type="decoder",
        example_data=example_data,
        temperature=args.temp,
        output_path=output_path,
    )
    logging.info(f"Model mixing completed. Mixed model saved to: {output_path}")
    #except Exception as e:
    #    logging.error(f"Error during model mixing: {e}")


if __name__ == "__main__":
    main()
