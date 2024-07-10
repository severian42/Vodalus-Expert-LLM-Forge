# Vodalus App Stack - Dataset Generation Guide

## Stack Components Overview

The Vodalus App Stack includes several key components and functionalities:

<div align="center">
  <img src="vodui.png" alt="Vodalus UI" style="width: 100%; max-width: 600px;" />
</div>

*Here is an online demo of the UI. 
Keep in mind that the demo is more for testing the UI since it is not actually linked to any LLM as this is mainly a local-based app:* 

https://severian-vodalus.hf.space

### *Datasets:*
- **Data Generation**: Utilizes local language models (LLMs) to generate synthetic data based on Wikipedia content. See `main.py` for implementation details.

- **LLM Interaction**: Manages interactions with LLMs through the `llm_handler.py`, which configures and handles messaging with the LLM.

- **Wikipedia Content Processing**: Processes and searches Wikipedia content to find relevant articles using models loaded in `wiki.py`.

---

### *Fine-Tuning and Quantization:*
- **Model Training and Fine-Tuning**: Supports training and fine-tuning of MLX models with custom datasets, as detailed in the MLX_Fine-Tuning guide.

- **Quantizing Models**: Guides on quantizing models to GGUF format for efficient local execution, as described in the Quantize_GGUF guide.

- **Interactive Notebooks**: Provides Jupyter notebooks for training and fine-tuning models, such as `mlx-fine-tuning.ipynb` and `convert_to_gguf.ipynb`.

---

### *Gradio User Interface:*
The new Gradio UI (`app.py`) provides a comprehensive interface for managing and interacting with the Vodalus App Stack:

- **Dataset Editor**: 
  - Load, view, and edit JSONL datasets
  - Navigate through dataset entries
  - Convert between JSON and Markdown formats
  - Annotate entries with quality ratings, tags, and additional notes
  - Preview entries with metadata

- **Annotation Configuration**: 
  - Customize quality scales, tag categories, and free-text fields
  - Save and load annotation configurations

- **Dataset Configuration**: 
  - Edit system messages, prompts, and topics used for data generation
  - Save dataset configurations

- **Dataset Generation**: 
  - Generate new dataset entries using configured parameters
  - Specify number of workers and generations
  - View generation status and output

- **AI Assistant**: 
  - Chat with an AI assistant for help with annotation and quality checking
  - Choose between local and remote LLM options

---

### *Designed for All Levels of Users:*
- **Comprehensive Documentation**: Each component is accompanied by detailed guides and instructions to assist users in setup, usage, and customization.

For more detailed information on each component, refer to the respective guides and source files included in the repository.

---

## Getting Started With Vodalus Dataset Generator

### Prerequisites
- Ensure Python is installed on your system.
- Familiarity with basic command line operations is helpful.

### Installation
1. Clone the repository to your local machine.
2. Navigate to the project directory in your command line interface.
3. Run the following commands to set up the environment:
---
- Create env: conda create -n vodalus -y
- conda activate vodalus
- `pip install -r requirements.txt`
---

### Running the Application
Execute the main script to start data generation:
- `python main.py`
- `gradio app.py`
---

## Key Components

### `main.py`
- **Imports and Setup**: Imports libraries and modules, sets the provider for the LLM.
- **Data Generation (`generate_data` function)**: Fetches Wikipedia content, constructs prompts, and generates data using the LLM.
- **Execution (`main` function)**: Manages the data generation process using multiple workers for efficiency.

### `llm_handler.py`
- **OpenAI Client Configuration**: Sets up the client for interacting with the LLM.
- **Message Handling Functions**: Includes functions to send messages to the LLM and handle the responses.

### `wiki.py`
- **Model Loading**: Loads necessary models for understanding and processing Wikipedia content.
- **Search Function**: Implements semantic search to find relevant Wikipedia articles based on a query.

## Usage Instructions

### Using the Gradio UI
1. Launch the Gradio UI by running `python app.py`.
2. Use the different tabs to manage datasets, configurations, and generate new data.
3. Utilize the AI Assistant for help with annotation and quality checking.

### Modifying Topics and System Messages
- Use the "Dataset Configuration" tab in the Gradio UI to modify topics and system messages.

### Configuration
- Adjust the number of workers and other parameters in the "Dataset Generation" tab of the Gradio UI.

***ALTERNATE Way of Modifying Topics and System Messages***
- To change the topics, edit `topics.py`.
- To modify system messages, adjust `system_messages.py`.

### Configuration
- Adjust the number of workers and other parameters in `params.py` to optimize performance based on your system's capabilities.
