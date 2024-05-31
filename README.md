# Vodalus Expert LLM Forge

### Dataset Crafting and Efficient Fine-Tuning Using Only Free Open-Source Tools

---

![vodalus-readme](https://github.com/severian42/Vodalus-Expert-LLM-Forge/assets/133655553/1b926eff-41ed-4516-a128-c9e3edce2770)

## Stack Components Overview

The Vodalus Expert LLM Forge includes several key components and functionalities:

### *Datasets:*
- **Data Generation**: Utilizes local language models (LLMs) to generate synthetic data based on Wikipedia content. See `main.py` for implementation details.

- **LLM Interaction**: Manages interactions with LLMs through the `llm_handler.py`, which configures and handles messaging with the LLM.

- **RAG with AnythingLLM and Wikipedia Content Processing**: Can use the AnythingLLM RAG engine (must be configured to your workspace) as well as process and search Wikipedia content to find relevant details to use as ground truth.

  *While I'm releasing this tool for free, I've also completed an extensive tutorial/course with lots of videos and instructions that guide you through each step of maximizing the potential of this stack and also teach you about the theory and concepts behind each part of the process. This course is available for purchase at [Vodalus LLM Course](https://ko-fi.com/s/076479f834) and is designed to enhance your experience and results with the Vodalus Expert LLM Forge.*
  
![Vodalus - Starting the Crafting](https://github.com/severian42/Vodalus-Expert-LLM-Forge/assets/133655553/418ddde4-8073-4b6b-9a01-d63863d41782)

---

### *Fine-Tuning and Quantization:*
- **Model Training and Fine-Tuning**: Supports training and fine-tuning with either MLX or Unsloth with your custom datasets.

- **Quantizing Models**: Guides on quantizing models to GGUF format for efficient local execution, as described in the Quantize_GGUF guide.

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
- `Create env: conda create -n vodalus -y`
- `conda activate vodalus`
- `pip install -r requirements.txt`
---

### Running the Application
Execute the main script to start data generation:
- `python main.py`
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

### Modifying Topics and System Messages
- To change the topics, edit `topics.py`.
- To modify system messages, adjust `system_messages.py`.

### Configuration
- Adjust the number of workers and other parameters in `params.py` to optimize performance based on your system's capabilities.

---

# Support This Project

If this project aids your work, please consider supporting it through a donation at my [www.ko-fi.com/severian42](https://ko-fi.com/severian42). Your support helps sustain my further LLM developments and experiments, always with a focus on using those efforts to give back to the LLM community

Also, if you love this concept and approach but don't want to do it yourself, you can hire me and we will work together to accomplish your ideal Expert LLM! I also offer 1-on-1 sessions to help with your LLM needs.

Feel free to reach out! You can find the details on my Ko-Fi: [www.ko-fi.com/severian42](https://ko-fi.com/severian42)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N4XZ2TZ)
