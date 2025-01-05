import json
import os
import datasets

# Function to read all JSON files from a directory
def read_json_files(directory):
    all_questions = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                print(filename)
                data = json.load(file)
                all_questions.extend(data)
    return all_questions

# Function to format the questions
def format_questions(questions):
    formatted_data = []
    for question_list in questions:
        for q in question_list:
            try:
                question_text = q["Question"].strip()
                prompt = f"{question_text}"
                formatted_data.append({"prompt": prompt})
            except:
                pass
    return formatted_data

# Directory containing the JSON files
directory = "./task_data"

# Read and combine all JSON files
all_questions = read_json_files(directory)

# Format the combined questions
formatted_data = format_questions(all_questions)


# Define a function to filter out duplicates based on prompts
def filter_duplicates(dataset):
    seen = set()
    unique_indices = []
    for i, example in enumerate(dataset):
        if example['prompt'] not in seen:
            seen.add(example['prompt'])
            unique_indices.append(i)
    return dataset.select(unique_indices)



ds = datasets.Dataset.from_list(formatted_data)
ds = filter_duplicates(ds)
ds = ds.shuffle()
#ds = ds.select(range(1000))

ds.save_to_disk("synthetic_questions_mmlu")

