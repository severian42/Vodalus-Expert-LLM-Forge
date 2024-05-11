# Importing necessary libraries and modules
import random
from typing import Tuple
from mlx_lm import load  # Load function to load models
from mlx_lm.tuner.lora import LoRALinear  # LoRA module for linear transformations
from mlx.utils import tree_flatten  # Utility to flatten model parameters
from mlx_lm.tuner.trainer import TrainingArgs, train  # Training utilities
import mlx.optimizers as optim  # Optimizers for model training

import json  # Module to work with JSON data
from pathlib import Path  # Module for handling filesystem paths

# Definition of the Dataset class to handle data operations
class Dataset:
    def __init__(self, data, key: str = "text"):
        # Constructor to initialize the Dataset object with data and a key to access elements
        self._data = data
        self._key = key

    def __getitem__(self, idx: int):
        # Method to get data item at a specific index
        return self._data[idx][self._key]

    def __len__(self):
        # Method to get the length of the dataset
        return len(self._data)

# Function to load a dataset from a specified path
def load_dataset(path: str):
    path = Path(path)  # Convert string path to Path object for better path handling
    if not path.exists():
        # Check if the path exists, raise an error if not
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as fid:
        # Open the file and read lines, converting each line from JSON to a dictionary
        data = [json.loads(line) for line in fid]

    dataset = Dataset(data)  # Create a Dataset object with the loaded data

    return dataset  # Return the dataset object

# Main function where the execution starts
def main():
    # Paths to the training and validation datasets
    train_dataset_path = "./data/dorian_training_dataset.jsonl"
    val_dataset_path = "./data/dorian_tvalid_dataset.jsonl"

    # Path to the pre-trained model
    model_path = "/Users/anima/DorainGray-Phi3-4k-MLX"

    # Load the model and tokenizer from the specified path
    model, tokenizer = load(model_path)

    # Load training and validation datasets
    train_dst, valid_dst = load_dataset(train_dataset_path), load_dataset(val_dataset_path)

    # Freeze the model to prevent updating weights of non-LoRA layers
    model.freeze()
    for l in model.model.layers:
        # Iterate through each layer in the model
        # Define the projections you want to update
        projections = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Update self_attn projections if they exist
        for proj in projections[:4]:  # For q_proj, k_proj, v_proj, o_proj
            if hasattr(l.self_attn, proj):
                # Replace existing linear layers with LoRALinear layers
                setattr(l.self_attn, proj, LoRALinear.from_linear(
                    getattr(l.self_attn, proj), r=64, alpha=128
                ))
        
        # Update block_sparse_moe projections if they exist
        if hasattr(l, "block_sparse_moe"):
            for proj in projections[4:]:  # For gate_proj, up_proj, down_proj
                if hasattr(l.block_sparse_moe, proj):
                    # Replace existing linear layers with LoRALinear layers
                    setattr(l.block_sparse_moe, proj, LoRALinear.from_linear(
                        getattr(l.block_sparse_moe, proj), r=64, alpha=128
                    ))
            
            # Update experts within block_sparse_moe
            for e in l.block_sparse_moe.experts:
                for proj in projections:  # Check all projections for each expert
                    if hasattr(e, proj):
                        # Replace existing linear layers with LoRALinear layers
                        setattr(e, proj, LoRALinear.from_linear(
                            getattr(e, proj), r=64, alpha=128
                        ))

    # Parameter counting remains unchanged
    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    # Training setup remains unchanged
    trainingArgs = TrainingArgs(
        batch_size=1,
        iters=5000,
        val_batches=25,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=200,
        adapter_file="adapters.npz",
        max_seq_length=4096,
    )

    model.train()

    decay_steps = trainingArgs.iters
    lr_schedule = optim.cosine_decay(1e-5, decay_steps)

    opt = optim.AdamW(learning_rate=lr_schedule)

    train(
        model=model,
        tokenizer=tokenizer,
        args=trainingArgs,
        optimizer=opt,
        train_dataset=train_dst,
        val_dataset=valid_dst,
    )


main()
