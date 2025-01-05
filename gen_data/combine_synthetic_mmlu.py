import json
import os

from datasets import Dataset


def read_json_files(directory):
    """
    Reads all JSON files in the specified directory and combines their contents into a single list.

    Args:
        directory (str): Path to the directory containing the JSON files.

    Returns:
        list: Combined data from all JSON files in the directory.
    """
    all_questions = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as file:
                    print(f"Processing file: {filename}")
                    data = json.load(file)
                    all_questions.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
    return all_questions


def format_questions(questions):
    """
    Formats a list of questions into dictionaries with prompts.

    Args:
        questions (list): List of dictionaries representing questions.

    Returns:
        list: List of dictionaries with formatted questions.
    """
    formatted_data = []
    for question_list in questions:
        for question in question_list:
            question_text = question.get("Question", "").strip()
            if question_text:
                formatted_data.append({"prompt": question_text})
    return formatted_data


def filter_duplicates(dataset):
    """
    Removes duplicate entries in a dataset based on the "prompt" field.

    Args:
        dataset (Dataset): The dataset to filter.

    Returns:
        Dataset: Filtered dataset with unique prompts.
    """
    seen = set()
    unique_data = []
    for item in dataset:
        prompt = item["prompt"]
        if prompt not in seen:
            seen.add(prompt)
            unique_data.append(item)
    return Dataset.from_list(unique_data)


def process_dataset(input_directory, output_path):
    """
    Processes JSON files in a directory to create a filtered and shuffled dataset.

    Args:
        input_directory (str): Path to the directory containing JSON files.
        output_path (str): Path to save the processed dataset.
    """
    # Read and format questions
    all_questions = read_json_files(input_directory)
    formatted_data = format_questions(all_questions)

    # Create a dataset and filter duplicates
    dataset = Dataset.from_list(formatted_data)
    filtered_dataset = filter_duplicates(dataset)

    # Shuffle and save the dataset
    shuffled_dataset = filtered_dataset.shuffle()
    shuffled_dataset.save_to_disk(output_path)
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    input_directory = "./task_data"
    output_path = "synthetic_questions_mmlu"
    process_dataset(input_directory, output_path)
