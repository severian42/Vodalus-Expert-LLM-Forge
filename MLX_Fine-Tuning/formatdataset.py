from datasets import load_dataset
import json

# Load the dataset specifying the path and splitting the data
dataset = load_dataset('json', data_files='dataset-dorian.jsonl', split='train')

# This list will hold the newly formatted dictionary objects
reformatted_dataset = []

# Iterate through each example in the dataset
for example in dataset:
    # Combine the information from 'system', 'instruction', and 'response' into a single string
    formatted_prompt = f"System: {example['system']}, Instruction: {example['instruction']}, Response: {example['response']}"

    # Create a dictionary with a single key 'text' and the formatted string as its value
    json_object = {"text": formatted_prompt}
    reformatted_dataset.append(json_object)

# Write the formatted dataset to a new JSONL file
with open("dorian_training_dataset.jsonl", "w") as output_jsonl_file:
    for item in reformatted_dataset:
        output_jsonl_file.write(json.dumps(item) + "\n")

# Print the first 5 entries to verify the output
for i, item in enumerate(reformatted_dataset[:5]):
    print(f"Entry {i+1}: {item}\n")
