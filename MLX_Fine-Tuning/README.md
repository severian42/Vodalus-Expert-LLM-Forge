# mlx-lora

## Module: Training MLX Models with Custom Student Datasets

### Overview
This module will guide you through the process of setting up and training machine learning models using the MLX stack, specifically tailored for handling custom datasets crafted by students. The focus will be on practical application, ensuring that learners can not only understand the theoretical aspects but also apply them directly to real-world scenarios.

### Prerequisites
- Basic understanding of Python programming.
- Familiarity with machine learning concepts.
- Installation of necessary Python packages as listed in [requirements.txt](file:///Users/anima/mlx-lora/requirements.txt#1%2C1-1%2C1).


#### 1. Setting Up Your Environment

   - Ensure Python is installed on your system.

   - Install necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```
     
   - Activate the MLX environment:
     ```bash
     conda activate mlx-lora
     ```

#### 2. Preparing Your Data

   - Understand the data format required by MLX models. Data should be in JSON format with a specific key, typically `"text"` for text data.

   - Preprocess your dataset to fit this structure. For instance, converting Q&A pairs into a single concatenated sentence:

     ```json
     {"text": "Q: What is the capital of France? A: The capital of France is Paris."}
     ```

   - Save your preprocessed data in the `data` folder.

#### 3. Training the Model

   - Customize the training script (`lora.py`) to suit your dataset and training requirements.

   - Run the training script to train the model:

     ```bash
     python lora.py
     ```

   - Understand the parameters and configurations used in training, such as learning rates, batch sizes, and number of iterations.

#### 4. Merging and Using Your Trained Model

   - After training, learn how to merge your trained LoRA back to the original model using the `fuse.py` script:

     ```bash
     python -m mlx_lm.fuse --model <path_to_model> --adapter-file <path_to_adapter>
     ```

   - Run inference to test the effectiveness of your trained model:
   
     ```bash
     python -m mlx_lm.generate --model <path_to_model> --adapter-file <path_to_adapter>
     ```

