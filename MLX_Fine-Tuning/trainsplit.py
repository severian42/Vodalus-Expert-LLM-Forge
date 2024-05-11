from datasets import load_dataset, DatasetDict
import json

def load_and_split_jsonl(file_path, train_split_percentage=0.8):
    """Load a JSONL file using Hugging Face datasets and split into train and validation sets."""
    # Load the dataset from the JSONL file
    dataset = load_dataset('json', data_files=file_path)

    # Calculate the number of samples for training
    train_size = int(len(dataset['train']) * train_split_percentage)
    valid_size = len(dataset['train']) - train_size

    # Split the dataset into training and validation sets
    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=valid_size)

    # Create a DatasetDict for better handling
    dataset_dict = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })

    return dataset_dict


def save_dataset_as_jsonl(dataset, file_path):
    """Save a dataset to a JSONL file."""
    with open(file_path, 'w') as file:
        for example in dataset:
            file.write(json.dumps(example) + '\n')

# Path to the JSONL file
file_path = './datasets/dataset-dorian.jsonl'

# Load the dataset and split it
dataset_dict = load_and_split_jsonl(file_path)

# Save the training and validation datasets as JSONL files
save_dataset_as_jsonl(dataset_dict['train'], 'dorian_train_dataset.jsonl')
save_dataset_as_jsonl(dataset_dict['validation'], 'dorian_valid_dataset.jsonl')

# Print some information about the splits and confirm saving
print(f"Training set size: {len(dataset_dict['train'])}")
print(f"Validation set size: {len(dataset_dict['validation'])}")
print("Training and validation datasets have been saved as 'train_dataset.jsonl' and 'valid_dataset.jsonl'.")