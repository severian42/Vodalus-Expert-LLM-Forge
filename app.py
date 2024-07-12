import gradio as gr
from gradio import update
import json
import re
from datetime import datetime
from typing import Literal
import os
import importlib
from llm_handler import send_to_llm_wrapper
from main import generate_data, PROMPT_1
from topics import TOPICS
from system_messages import SYSTEM_MESSAGES_VODALUS
import random
from params import load_params, save_params
import pandas as pd
import csv
from datasets import load_dataset
from huggingface_hub import list_datasets, HfApi, hf_hub_download
from gradio.components import State




ANNOTATION_CONFIG_FILE = "annotation_config.json"
OUTPUT_FILE_PATH = "dataset.jsonl"

llm_provider_state = State("")

def load_llm_config():
    params = load_params()
    return (
        params.get('PROVIDER', ''),
        params.get('BASE_URL', ''),
        params.get('MODEL', ''),  # Add this line
        params.get('WORKSPACE', ''),
        params.get('API_KEY', ''),
        params.get('max_tokens', 2048),
        params.get('temperature', 0.7),
        params.get('top_p', 0.9),
        params.get('frequency_penalty', 0.0),
        params.get('presence_penalty', 0.0)
    )



def save_llm_config(provider, base_url, model, workspace, api_key, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
    save_params({
        'PROVIDER': provider,
        'BASE_URL': base_url,
        'MODEL': model,  # Add this line
        'WORKSPACE': workspace,
        'API_KEY': api_key,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty
    })
    return "LLM configuration saved successfully"


def update_model_visibility(provider):
        return gr.update(visible=provider in ["local-model", "openai"])


def load_annotation_config():
    try:
        with open(ANNOTATION_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "quality_scale": {
                "name": "Relevance for Training",
                "description": "Rate the relevance of this entry for training",
                "scale": [
                    {"value": "1", "label": "Invalid"},
                    {"value": "2", "label": "Somewhat invalid"},
                    {"value": "3", "label": "Neutral"},
                    {"value": "4", "label": "Somewhat valid"},
                    {"value": "5", "label": "Valid"}
                ]
            },
            "tag_categories": [
                {
                    "name": "High Quality Indicators",
                    "type": "multiple",
                    "tags": ["Well-formatted", "Informative", "Coherent", "Engaging"]
                },
                {
                    "name": "Low Quality Indicators",
                    "type": "multiple",
                    "tags": ["Poorly formatted", "Lacks context", "Repetitive", "Irrelevant"]
                },
                {
                    "name": "Content Warnings",
                    "type": "multiple",
                    "tags": ["Offensive language", "Hate speech", "Violence", "Adult content"]
                }
            ],
            "free_text_fields": [
                {
                    "name": "Additional Notes",
                    "description": "Any other observations or comments"
                }
            ]
        }




def load_csv_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data



def load_txt_dataset(file_path):
    with open(file_path, 'r') as f:
        return [{"content": line.strip()} for line in f if line.strip()]



def save_annotation_config(config):
    with open(ANNOTATION_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)



def load_jsonl_dataset(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
    


def load_dataset(file):
    if file is None:
        return "", 0, 0, "No file uploaded", "3", [], [], [], ""
    
    file_path = file.name
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        data = load_csv_dataset(file_path)
    elif file_extension == '.txt':
        data = load_txt_dataset(file_path)
    elif file_extension == '.jsonl':
        data = load_jsonl_dataset(file_path)
    else:
        return "", 0, 0, f"Unsupported file type: {file_extension}", "3", [], [], [], ""
    
    if not data:
        return "", 0, 0, "No data found in the file", "3", [], [], [], ""
    
    first_row = json.dumps(data[0], indent=2)
    return first_row, 0, len(data), f"Row: 1/{len(data)}", "3", [], [], [], ""



def save_row(file_path, index, row_data):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'jsonl':
        save_jsonl_row(file_path, index, row_data)
    elif file_extension == 'csv':
        save_csv_row(file_path, index, row_data)
    elif file_extension == 'txt':
        save_txt_row(file_path, index, row_data)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return f"Row {index} saved successfully"



def save_jsonl_row(file_path, index, row_data):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lines[index] = row_data + '\n'
    
    with open(file_path, 'w') as f:
        f.writelines(lines)



def save_csv_row(file_path, index, row_data):
    df = pd.read_csv(file_path)
    row_dict = json.loads(row_data)
    for col, value in row_dict.items():
        df.at[index, col] = value
    df.to_csv(file_path, index=False)



def save_txt_row(file_path, index, row_data):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    row_dict = json.loads(row_data)
    lines[index] = row_dict.get('content', '') + '\n'
    
    with open(file_path, 'w') as f:
        f.writelines(lines)



def get_row(file_path, index):
    data = load_jsonl_dataset(file_path)
    if not data:
        return "", 0
    if 0 <= index < len(data):
        return json.dumps(data[index], indent=2), len(data)
    return "", len(data)



def json_to_markdown(json_str):
    try:
        data = json.loads(json_str)
        markdown = f"# System\n\n{data['system']}\n\n# Instruction\n\n{data['instruction']}\n\n# Response\n\n{data['response']}"
        return markdown
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"



def markdown_to_json(markdown_str):
    sections = re.split(r'#\s+(System|Instruction|Response)\s*\n', markdown_str)
    if len(sections) != 7:  # Should be: ['', 'System', content, 'Instruction', content, 'Response', content]
        return "Error: Invalid markdown format"
    
    json_data = {
        "system": sections[2].strip(),
        "instruction": sections[4].strip(),
        "response": sections[6].strip()
    }
    return json.dumps(json_data, indent=2)



def navigate_rows(file_path: str, current_index: int, direction: Literal["prev", "next"], metadata_config):
    new_index = max(0, current_index + (-1 if direction == "prev" else 1))
    return load_and_show_row(file_path, new_index, metadata_config)



def load_and_show_row(file_path, index, metadata_config):
    row_data, total = get_row(file_path, index)
    if not row_data:
        return ("", index, total, f"Row: {index + 1}/{total}", "3", [], [], [], "")
    
    try:
        data = json.loads(row_data)
    except json.JSONDecodeError:
        return (row_data, index, total, f"Row: {index + 1}/{total}", "3", [], [], [], "Error: Invalid JSON")
    
    metadata = data.get("metadata", {}).get("annotation", {})
    
    quality = metadata.get("quality", "3")
    high_quality_tags = metadata.get("tags", {}).get("high_quality", [])
    low_quality_tags = metadata.get("tags", {}).get("low_quality", [])
    toxic_tags = metadata.get("tags", {}).get("toxic", [])
    other = metadata.get("free_text", {}).get("Additional Notes", "")
    
    return (row_data, index, total, f"Row: {index + 1}/{total}", quality, 
            high_quality_tags, low_quality_tags, toxic_tags, other)



def save_row_with_metadata(file_path, index, row_data, config, quality, high_quality_tags, low_quality_tags, toxic_tags, other):
    data = json.loads(row_data)
    metadata = {
        "annotation": {
            "quality": quality,
            "tags": {
                "high_quality": high_quality_tags,
                "low_quality": low_quality_tags,
                "toxic": toxic_tags
            },
            "free_text": {
                "Additional Notes": other
            }
        }
    }
    
    data["metadata"] = metadata
    return save_row(file_path, index, json.dumps(data))



def update_annotation_ui(config):
    quality_choices = [(item["value"], item["label"]) for item in config["quality_scale"]["scale"]]
    quality_label = gr.Radio(
        label=config["quality_scale"]["name"],
        choices=quality_choices,
        info=config["quality_scale"]["description"]
    )
    
    tag_components = []
    for category in config["tag_categories"]:
        tag_component = gr.CheckboxGroup(
            label=category["name"],
            choices=category["tags"]
        )
        tag_components.append(tag_component)
    
    other_description = gr.Textbox(
        label=config["free_text_fields"][0]["name"],
        lines=3
    )
    
    return quality_label, *tag_components, other_description



def load_config_to_ui(config):
    return (
        config["quality_scale"]["name"],
        config["quality_scale"]["description"],
        [[item["value"], item["label"]] for item in config["quality_scale"]["scale"]],
        [[cat["name"], cat["type"], ", ".join(cat["tags"])] for cat in config["tag_categories"]],
        [[field["name"], field["description"]] for field in config["free_text_fields"]]
    )



def save_config_from_ui(name, description, scale, categories, fields, topics, all_topics_text):
    if all_topics_text.visible:
        topics_list = [topic.strip() for topic in all_topics_text.split("\n") if topic.strip()]
    else:
        topics_list = [topic[0] for topic in topics]
    
    new_config = {
        "quality_scale": {
            "name": name,
            "description": description,
            "scale": [{"value": row[0], "label": row[1]} for row in scale]
        },
        "tag_categories": [{"name": row[0], "type": row[1], "tags": row[2].split(", ")} for row in categories],
        "free_text_fields": [{"name": row[0], "description": row[1]} for row in fields],
        "topics": topics_list
    }
    save_annotation_config(new_config)
    return "Configuration saved successfully", new_config



# Add this new function to generate the preview
def generate_preview(row_data, quality, high_quality_tags, low_quality_tags, toxic_tags, other):
    try:
        data = json.loads(row_data)
        metadata = {
            "annotation": {
                "quality": quality,
                "tags": {
                    "high_quality": high_quality_tags,
                    "low_quality": low_quality_tags,
                    "toxic": toxic_tags
                },
                "free_text": {
                    "Additional Notes": other
                }
            }
        }
        data["metadata"] = metadata
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return "Error: Invalid JSON in the current row data"



def load_dataset_config():
    params = load_params()
    with open("system_messages.py", "r") as f:
        system_messages_content = f.read()
        vodalus_system_message = re.search(r'SYSTEM_MESSAGES_VODALUS = \[(.*?)\]', system_messages_content, re.DOTALL).group(1).strip()[3:-3]

    with open("main.py", "r") as f:
        main_content = f.read()
        prompt_1 = re.search(r'PROMPT_1 = """(.*?)"""', main_content, re.DOTALL).group(1).strip()

    topics_module = importlib.import_module("topics")
    topics_list = topics_module.TOPICS

    return {
        "vodalus_system_message": vodalus_system_message,
        "prompt_1": prompt_1,
        "topics": [[topic] for topic in topics_list],
        "max_tokens": params.get('max_tokens', 2048),
        "temperature": params.get('temperature', 0.7),
        "top_p": params.get('top_p', 0.9),
        "frequency_penalty": params.get('frequency_penalty', 0.0),
        "presence_penalty": params.get('presence_penalty', 0.0)
    }



def edit_all_topics_func(topics):
    topics_list = [topic[0] for topic in topics]
    jsonl_rows = "\n".join([json.dumps({"topic": topic}) for topic in topics_list])
    return (
        gr.update(visible=False),
        gr.update(value=jsonl_rows, visible=True),
        gr.update(visible=True)
    )



def update_topics_from_text(text):
    try:
        # Try parsing as JSONL
        topics_list = [json.loads(line)["topic"] for line in text.split("\n") if line.strip()]
    except json.JSONDecodeError:
        # If parsing fails, treat as plain text
        topics_list = [topic.strip() for topic in text.split("\n") if topic.strip()]
    
    return gr.Dataframe.update(value=[[topic] for topic in topics_list], visible=True), gr.TextArea.update(visible=False)



def save_dataset_config(system_messages, prompt_1, topics, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
    # Save VODALUS_SYSTEM_MESSAGE to system_messages.py
    with open("system_messages.py", "w") as f:
        f.write(f'SYSTEM_MESSAGES_VODALUS = [\n"""\n{system_messages}\n""",\n]\n')
    
    # Save PROMPT_1 to main.py
    with open("main.py", "r") as f:
        main_content = f.read()
    
    updated_main_content = re.sub(
        r'PROMPT_1 = """.*?"""',
        f'PROMPT_1 = """\n{prompt_1}\n"""',
        main_content,
        flags=re.DOTALL
    )
    
    with open("main.py", "w") as f:
        f.write(updated_main_content)
    
    # Save TOPICS to topics.py
    topics_content = "TOPICS = [\n"
    for topic in topics:
        topics_content += f'    "{topic[0]}",\n'
    topics_content += "]\n"
    
    with open("topics.py", "w") as f:
        f.write(topics_content)

    save_params({
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty
    })
    
    return "Dataset configuration saved successfully"
    


def chat_with_llm(message, history):
    try:
        msg_list = [{"role": "system", "content": "You are an AI assistant helping with dataset annotation and quality checking."}]
        for h in history:
            msg_list.append({"role": "user", "content": h[0]})
            msg_list.append({"role": "assistant", "content": h[1]})
        msg_list.append({"role": "user", "content": message})

        # Update this line to use send_to_llm_wrapper
        response, _ = send_to_llm_wrapper(msg_list)
        
        return history + [[message, response]]
    except Exception as e:
        print(f"Error in chat_with_llm: {str(e)}")
        return history + [[message, f"Error: {str(e)}"]]


def update_chat_context(row_data, index, total, quality, high_quality_tags, low_quality_tags, toxic_tags, other):
    context = f"""Current app state:
    Row: {index + 1}/{total}
    Quality: {quality}
    High Quality Tags: {', '.join(high_quality_tags)}
    Low Quality Tags: {', '.join(low_quality_tags)}
    Toxic Tags: {', '.join(toxic_tags)}
    Additional Notes: {other}
    
    Data: {row_data}
    """
    return [[None, context]]



async def run_generate_dataset(num_workers, num_generations, output_file_path, llm_provider, dataset):
    if loaded_dataset is None:
        return "Error: No dataset loaded. Please load a dataset before generating.", ""

    generated_data = []
    for _ in range(num_generations):
        topic_selected = random.choice(TOPICS)
        system_message_selected = random.choice(SYSTEM_MESSAGES_VODALUS)
        data = await generate_data(topic_selected, PROMPT_1, system_message_selected, output_file_path, llm_provider)
        if data:
            generated_data.append(json.dumps(data))
    
    # Write the generated data to the output file
    with open(output_file_path, 'a') as f:
        for entry in generated_data:
            f.write(entry + '\n')
    
    return f"Generated {num_generations} entries and saved to {output_file_path}", "\n".join(generated_data[:5]) + "\n..."



def add_topic_row(data):
    if isinstance(data, pd.DataFrame):
        return pd.concat([data, pd.DataFrame({"Topic": ["New Topic"]})], ignore_index=True)
    else:
        return data + [["New Topic"]]



def remove_last_topic_row(data):
    return data[:-1] if len(data) > 1 else data



def edit_all_topics_func(topics):
    topics_list = [topic[0] for topic in topics]
    jsonl_rows = "\n".join([json.dumps({"topic": topic}) for topic in topics_list])
    return (
        gr.update(visible=False),
        gr.update(value=jsonl_rows, visible=True),
        gr.update(visible=True)
    )



def update_topics_from_text(text):
    try:
        # Try parsing as JSONL
        topics_list = [json.loads(line)["topic"] for line in text.split("\n") if line.strip()]
    except json.JSONDecodeError:
        # If parsing fails, treat as plain text
        topics_list = [topic.strip() for topic in text.split("\n") if topic.strip()]
    
    return gr.Dataframe.update(value=[[topic] for topic in topics_list], visible=True), gr.TextArea.update(visible=False)



def update_topics_from_text(text):
    try:
        # Try parsing as JSONL
        topics_list = [json.loads(line)["topic"] for line in text.split("\n") if line.strip()]
    except json.JSONDecodeError:
        # If parsing fails, treat as plain text
        topics_list = [topic.strip() for topic in text.split("\n") if topic.strip()]
    
    return gr.Dataframe.update(value=[[topic] for topic in topics_list], visible=True), gr.TextArea.update(visible=False)



def search_huggingface_datasets(query):
    try:
        api = HfApi()
        datasets = api.list_datasets(search=query, limit=20)
        dataset_ids = [dataset.id for dataset in datasets]
        return gr.update(choices=dataset_ids, visible=True), ""
    except Exception as e:
        print(f"Error searching datasets: {str(e)}")
        return gr.update(choices=["Error: Could not search datasets"], visible=True), ""



def load_huggingface_dataset(dataset_name, split="train"):
    try:
        print(f"Attempting to load dataset: {dataset_name}")
        
        # Check if dataset_name is a string
        if not isinstance(dataset_name, str):
            raise ValueError(f"Expected dataset_name to be a string, but got {type(dataset_name)}")
        
        # Try loading the dataset without specifying a config
        full_dataset = load_dataset(dataset_name)
        
        print(f"Dataset loaded. Available splits: {list(full_dataset.keys())}")
        
        # Select the appropriate split
        if split in full_dataset:
            dataset = full_dataset[split]
            print(f"Using specified split: {split}")
        else:
            available_splits = list(full_dataset.keys())
            if available_splits:
                dataset = full_dataset[available_splits[0]]
                split = available_splits[0]
                print(f"Specified split not found. Using first available split: {split}")
            else:
                raise ValueError("No valid splits found in the dataset")

        return dataset, f"Dataset '{dataset_name}' (split: {split}) loaded successfully."
    except Exception as e:
        error_msg = f"Error loading dataset: {str(e)}"
        print(f"Error details: {error_msg}")
        
        # If loading fails, try to get the dataset card
        try:
            dataset_card = hf_hub_download(repo_id=dataset_name, filename="README.md")
            with open(dataset_card, 'r') as f:
                card_content = f.read()
            return None, f"Dataset couldn't be loaded, but here's the dataset card:\n\n{card_content[:500]}..."
        except:
            return None, error_msg

# Wrapper function to handle the Gradio interface
def load_dataset_wrapper(dataset_name, split):
    if not dataset_name:
        return None, "Please enter a dataset name."
    dataset, message = load_huggingface_dataset(dataset_name, split)
    return dataset, message

def update_field_visibility(provider):
    if provider == "local-model":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif provider == "anything-llm":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def get_popular_datasets():
    return [
        "wikipedia",
        "squad",
        "glue",
        "imdb",
        "wmt16",
        "common_voice",
        "cnn_dailymail",
        "amazon_reviews_multi",
        "yelp_review_full",
        "ag_news"
    ]

def load_dataset_config_for_ui():
    config = load_dataset_config()
    return (
        config["vodalus_system_message"],
        config["prompt_1"],
        config["topics"],
        config["max_tokens"],
        config["temperature"],
        config["top_p"],
        config["frequency_penalty"],
        config["presence_penalty"]
    )


css = """
body, #root {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow-x: hidden;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 !important;
}
.message-row {
    justify-content: space-evenly !important;
}
.message-bubble-border {
    border-radius: 6px !important;
}
.message-buttons-bot, .message-buttons-user {
    right: 10px !important;
    left: auto !important;
    bottom: 2px !important;
}
.dark.message-bubble-border {
    border-color: #343140 !important;
}
.dark.user {
    background: #1e1c26 !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #16141c !important;
}
.tab-nav {
    border-bottom: 2px solid #e0e0e0 !important;
}
.tab-nav button {
    font-size: 16px !important;
    padding: 10px 20px !important;
}
.input-row {
    margin-bottom: 20px !important;
}
.button-row {
    display: flex !important;
    justify-content: space-between !important;
    margin-top: 20px !important;
}
#row-editor {
    height: 80vh !important;
    font-size: 16px !important;
}

.file-upload-row {
    height: 50px !important;
    margin-bottom: 1rem !important;
}

.file-upload-row > .gr-column {
    min-width: 0 !important;
}

.compact-file-upload {
    height: 50px !important;
    overflow: hidden !important;
}

.compact-file-upload > .file-preview {
    min-height: 0 !important;
    max-height: 50px !important;
    padding: 0 !important;
}

.compact-file-upload > .file-preview > .file-preview-handler {
    height: 50px !important;
    padding: 0 8px !important;
    display: flex !important;
    align-items: center !important;
}

.compact-file-upload > .file-preview > .file-preview-handler > .file-preview-title {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    flex: 1 !important;
}

.compact-file-upload > .file-preview > .file-preview-handler > .file-preview-remove {
    padding: 0 !important;
    min-width: 24px !important;
    width: 24px !important;
    height: 24px !important;
}

.compact-button {
    height: 50px !important;
    min-height: 40px !important;
    width: 100% !important;
}

.compact-file-upload > label {
    height: 50px !important;
    padding: 0 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: left !important;
}
"""

demo = gr.Blocks(theme='Ama434/neutral-barlow', css=css)

with demo:
    gr.Markdown("# Dataset Editor and Annotation Tool")

    config = gr.State(load_annotation_config())

    with gr.Row():
        with gr.Column(min_width=1000):
            with gr.Tab("Dataset Editor"):
                with gr.Row(elem_classes="file-upload-row"):
                    with gr.Column(scale=3, min_width=400):
                        file_upload = gr.File(label="Upload Dataset File (.txt, .jsonl, or .csv)", elem_classes="compact-file-upload")
                    with gr.Column(scale=1, min_width=100):
                        load_button = gr.Button("Load Dataset", elem_classes="compact-button")
                
                with gr.Row():
                    prev_button = gr.Button("← Previous")
                    row_index = gr.State(value=0)
                    total_rows = gr.State(value=0)
                    current_row_display = gr.Textbox(label="Current Row", interactive=False)
                    next_button = gr.Button("Next →")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        row_editor = gr.TextArea(label="Edit Row", lines=40) 
                    
                    with gr.Column(scale=2):
                        quality_label = gr.Radio(label="Relevance for Training", choices=[])
                        tag_components = [gr.CheckboxGroup(label=f"Tag Group {i+1}", choices=[]) for i in range(3)]
                        other_description = gr.Textbox(label="Additional annotations", lines=3)
                        
                        # Add the AI Assistant as a dropdown
                        with gr.Accordion("AI Assistant", open=False):
                            chatbot = gr.Chatbot(height=300)
                            msg = gr.Textbox(label="Chat with AI Assistant")
                            clear = gr.Button("Clear")
                
                with gr.Row():
                    to_markdown_button = gr.Button("Convert to Markdown")
                    to_json_button = gr.Button("Convert to JSON")
                    preview_button = gr.Button("Preview with Metadata")
                    save_row_button = gr.Button("Save Current Row", variant="primary")
                
                preview_output = gr.TextArea(label="Preview", lines=20, interactive=False)
                editor_status = gr.Textbox(label="Editor Status")

            with gr.Tab("Annotation Configuration"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Quality Scale")
                        quality_scale_name = gr.Textbox(label="Scale Name")
                        quality_scale_description = gr.Textbox(label="Scale Description", lines=2)
                    
                    with gr.Column(scale=2):
                        quality_scale = gr.Dataframe(
                            headers=["Value", "Label"],
                            datatype=["str", "str"],
                            label="Quality Scale Options",
                            interactive=True,
                            col_count=(2, "fixed"),
                            row_count=(5, "dynamic"),
                            height=400,
                            wrap=True
                        )
                
                gr.Markdown("### Tag Categories")
                tag_categories = gr.Dataframe(
                    headers=["Name", "Type", "Tags"],
                    datatype=["str", "str", "str"],
                    label="Tag Categories",
                    interactive=True,
                    col_count=(3, "fixed"),
                    row_count=(3, "dynamic"),
                    height=250,
                    wrap=True
                )
                
                with gr.Row():
                    add_tag_category = gr.Button("Add Category")
                    remove_tag_category = gr.Button("Remove Last Category")
                
                gr.Markdown("### Free Text Fields")
                free_text_fields = gr.Dataframe(
                    headers=["Name", "Description"],
                    datatype=["str", "str"],
                    label="Free Text Fields",
                    interactive=True,
                    col_count=(2, "fixed"),
                    row_count=(2, "dynamic"),
                    height=300,
                    wrap=True
                )
                
                with gr.Row():
                    add_free_text_field = gr.Button("Add Field")
                    remove_free_text_field = gr.Button("Remove Last Field")
                
                
                with gr.Row():
                    save_config_btn = gr.Button("Save Configuration", variant="primary")
                    config_status = gr.Textbox(label="Status", interactive=False)

            with gr.Tab("Dataset Configuration"):
                with gr.Row():
                    vodalus_system_message = gr.TextArea(label="System Message for JSONL Dataset", lines=10)
                    prompt_1 = gr.TextArea(label="Dataset Generation Prompt", lines=10)
                
                gr.Markdown("### Topics")
                with gr.Row():
                    with gr.Column(scale=2):
                        topics = gr.Dataframe(
                            headers=["Topic"],
                            datatype=["str"],
                            label="Topics",
                            interactive=True,
                            col_count=(1, "fixed"),
                            row_count=(5, "dynamic"),
                            height=200,
                            wrap=True
                        )
                    
                    with gr.Column(scale=1):
                        with gr.Row():
                            add_topic = gr.Button("Add Topic")
                            remove_topic = gr.Button("Remove Last Topic")
                        edit_all_topics = gr.Button("Edit All Topics")
                        all_topics_edit = gr.TextArea(label="Edit All Topics (JSONL or Plain Text)", visible=False, lines=10)
                        format_info = gr.Markdown("""
                        Enter topics as JSONL (e.g., {"topic": "Example Topic"}) or plain text (one topic per line).
                        JSONL format allows for additional metadata if needed.
                        """, visible=False)
                
                with gr.Row():
                    save_dataset_config_btn = gr.Button("Save Dataset Configuration", variant="primary")
                    dataset_config_status = gr.Textbox(label="Status")

#                gr.Markdown("### Hugging Face Dataset")
#                with gr.Row():
#                    dataset_search = gr.Textbox(label="Search Datasets")
#                    search_button = gr.Button("Search")
#                dataset_input = gr.Textbox(label="Dataset Name", info="Enter a dataset name or select from search results")
#                dataset_results = gr.Radio(label="Search Results", choices=[], visible=False)
#                dataset_split = gr.Textbox(label="Dataset Split (optional)", value="train")
#                load_dataset_button = gr.Button("Load Selected Dataset")
#                dataset_status = gr.Textbox(label="Dataset Status")

                # Add a state to store the loaded dataset
#                loaded_dataset = gr.State(None)
                
                

            with gr.Tab("Dataset Generation"):
                with gr.Row():
                    num_workers = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Workers")
                    num_generations = gr.Number(value=10, label="Number of Generations", precision=0)
                
                with gr.Row():
                    output_file_path = gr.Textbox(label="Output File Path", value=OUTPUT_FILE_PATH)
                
                start_generation_btn = gr.Button("Start Generation")
                generation_status = gr.Textbox(label="Generation Status")
                generation_output = gr.TextArea(label="Generation Output", lines=10)

            with gr.Tab("LLM Configuration"):
                with gr.Row():
                    provider = gr.Dropdown(choices=["local-model", "anything-llm"], label="LLM Provider")
                with gr.Row():
                    base_url = gr.Textbox(label="Base URL (for local model)", visible=False)
                    model = gr.Textbox(label="Model (for local model)", visible=False)
                with gr.Row():
                    workspace = gr.Textbox(label="Workspace (for AnythingLLM)", visible=False)
                    api_key = gr.Textbox(label="API Key (for AnythingLLM)", visible=False)
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        max_tokens = gr.Slider(minimum=100, maximum=4096, value=2048, step=1, label="Max Tokens")
                        temperature = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, label="Temperature")
                    with gr.Row():
                        top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.01, label="Top P")
                        frequency_penalty = gr.Slider(minimum=0, maximum=2, value=0.0, step=0.01, label="Frequency Penalty")
                        presence_penalty = gr.Slider(minimum=0, maximum=2, value=0.0, step=0.01, label="Presence Penalty")
                
                save_llm_config_btn = gr.Button("Save LLM Configuration")
                llm_config_status = gr.Textbox(label="LLM Config Status")

                with gr.Row():
                    save_dataset_config_btn = gr.Button("Save Dataset Configuration", variant="primary")
                    dataset_config_status = gr.Textbox(label="Status")

    add_topic.click(
        lambda x: x + [["New Topic"]],
        inputs=[topics],
        outputs=[topics]
    )

    remove_topic.click(
        lambda x: x[:-1] if len(x) > 0 else x,
        inputs=[topics],
        outputs=[topics]
    )

    edit_all_topics.click(
        edit_all_topics_func,
        inputs=[topics],
        outputs=[topics, all_topics_edit, format_info]
    )

    all_topics_edit.submit(
        update_topics_from_text,
        inputs=[all_topics_edit],
        outputs=[topics, all_topics_edit, format_info]
    )

    load_button.click(
        load_dataset,
        inputs=[file_upload],
        outputs=[row_editor, row_index, total_rows, current_row_display, quality_label, *tag_components, other_description]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    prev_button.click(
        navigate_rows,
        inputs=[file_upload, row_index, gr.State("prev"), config],
        outputs=[row_editor, row_index, total_rows, current_row_display, quality_label, *tag_components, other_description]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    next_button.click(
        navigate_rows,
        inputs=[file_upload, row_index, gr.State("next"), config],
        outputs=[row_editor, row_index, total_rows, current_row_display, quality_label, *tag_components, other_description]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    save_row_button.click(
        save_row_with_metadata,
        inputs=[file_upload, row_index, row_editor, config, quality_label, 
                tag_components[0], tag_components[1], tag_components[2], other_description],
        outputs=[editor_status]
    ).then(
        lambda: "",
        outputs=[preview_output]
    )

    to_markdown_button.click(
        json_to_markdown,
        inputs=[row_editor],
        outputs=[row_editor]
    )

    to_json_button.click(
        markdown_to_json,
        inputs=[row_editor],
        outputs=[row_editor]
    )

    demo.load(
        load_config_to_ui,
        inputs=[config],
        outputs=[quality_scale_name, quality_scale_description, quality_scale, tag_categories, free_text_fields]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    save_config_btn.click(
        save_config_from_ui,
        inputs=[quality_scale_name, quality_scale_description, quality_scale, tag_categories, free_text_fields, topics, all_topics_edit],
        outputs=[config_status, config]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    preview_button.click(
        generate_preview,
        inputs=[row_editor, quality_label, *tag_components, other_description],
        outputs=[preview_output]
    )

    demo.load(
        load_dataset_config,
        outputs=[vodalus_system_message, prompt_1, topics, max_tokens, temperature, top_p, frequency_penalty, presence_penalty]
    )

    save_dataset_config_btn.click(
        save_dataset_config,
        inputs=[vodalus_system_message, prompt_1, topics, max_tokens, temperature, top_p, frequency_penalty, presence_penalty],
        outputs=[dataset_config_status]
    )

    start_generation_btn.click(
        run_generate_dataset,
        inputs=[num_workers, num_generations, output_file_path, llm_provider_state],
        outputs=[generation_status, generation_output]
    )

    demo.load(
        load_llm_config,
        outputs=[provider, base_url, model, workspace, api_key, max_tokens, temperature, top_p, frequency_penalty, presence_penalty]
    )

    save_llm_config_btn.click(
        save_llm_config,
        inputs=[provider, base_url, model, workspace, api_key, max_tokens, temperature, top_p, frequency_penalty, presence_penalty],
        outputs=[llm_config_status]
    )

    msg.submit(chat_with_llm, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)


    for button in [load_button, prev_button, next_button]:
        button.click(
            update_chat_context,
            inputs=[row_editor, row_index, total_rows, quality_label, *tag_components, other_description],
            outputs=[chatbot]
        )

    provider.change(
        lambda x: x,
        inputs=[provider],
        outputs=[llm_provider_state]
    )

    # search_button.click(
    #     search_huggingface_datasets,
    #     inputs=[dataset_search],
    #     outputs=[dataset_results, dataset_input]
    # )

    # dataset_results.change(
    #     lambda choice: choice,
    #     inputs=[dataset_results],
    #     outputs=[dataset_input]
    # )

    # load_dataset_button.click(
    #     load_dataset_wrapper,
    #     inputs=[dataset_input, dataset_split],
    #     outputs=[loaded_dataset, dataset_status]
    # )

    # Modify the start_generation_btn.click to include the loaded dataset
    start_generation_btn.click(
        run_generate_dataset,
        inputs=[num_workers, num_generations, output_file_path, llm_provider_state],
        outputs=[generation_status, generation_output]
    )

    demo.load(
        load_dataset_config_for_ui,
        outputs=[
            vodalus_system_message,
            prompt_1,
            topics,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True)
