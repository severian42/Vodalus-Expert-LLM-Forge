import gradio as gr
import json
import re
from datetime import datetime
from typing import Literal
import os
import importlib
from llm_handler import send_to_llm
from main import generate_data, PROMPT_1
from topics import TOPICS
from system_messages import SYSTEM_MESSAGES_VODALUS
import random

ANNOTATION_CONFIG_FILE = "annotation_config.json"
OUTPUT_FILE_PATH = "dataset.jsonl"

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

def save_annotation_config(config):
    with open(ANNOTATION_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def load_jsonl_dataset(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_row(file_path, index, row_data):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lines[index] = row_data + '\n'
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    return f"Row {index} saved successfully"

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

def navigate_rows(file_path: str, current_index: int, direction: Literal[-1, 1], metadata_config):
    new_index = max(0, current_index + direction)
    return load_and_show_row(file_path, new_index, metadata_config)

def load_and_show_row(file_path, index, metadata_config):
    row_data, total = get_row(file_path, index)
    if not row_data:
        return ("", index, total, "3", [], [], [], "")
    
    try:
        data = json.loads(row_data)
    except json.JSONDecodeError:
        return (row_data, index, total, "3", [], [], [], "Error: Invalid JSON")
    
    metadata = data.get("metadata", {}).get("annotation", {})
    
    quality = metadata.get("quality", "3")
    high_quality_tags = metadata.get("tags", {}).get("high_quality", [])
    low_quality_tags = metadata.get("tags", {}).get("low_quality", [])
    toxic_tags = metadata.get("tags", {}).get("toxic", [])
    other = metadata.get("free_text", {}).get("Additional Notes", "")
    
    return (row_data, index, total, quality, 
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

def save_config_from_ui(name, description, scale, categories, fields):
    new_config = {
        "quality_scale": {
            "name": name,
            "description": description,
            "scale": [{"value": row[0], "label": row[1]} for row in scale]
        },
        "tag_categories": [{"name": row[0], "type": row[1], "tags": row[2].split(", ")} for row in categories],
        "free_text_fields": [{"name": row[0], "description": row[1]} for row in fields]
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
    # Load VODALUS_SYSTEM_MESSAGE from system_messages.py
    with open("system_messages.py", "r") as f:
        system_messages_content = f.read()
        vodalus_system_message = re.search(r'SYSTEM_MESSAGES_VODALUS = \[(.*?)\]', system_messages_content, re.DOTALL).group(1).strip()[3:-3]  # Extract the content between triple quotes

    # Load PROMPT_1 from main.py
    with open("main.py", "r") as f:
        main_content = f.read()
        prompt_1 = re.search(r'PROMPT_1 = """(.*?)"""', main_content, re.DOTALL).group(1).strip()

    # Load TOPICS from topics.py
    topics_module = importlib.import_module("topics")
    topics_list = topics_module.TOPICS

    return vodalus_system_message, prompt_1, [[topic] for topic in topics_list]

def save_dataset_config(system_messages, prompt_1, topics):
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
    
    return "Dataset configuration saved successfully"

# Modify the chat_with_llm function to use Gradio's built-in async capabilities
def chat_with_llm(message, history, selected_llm):
    try:
        msg_list = [{"role": "system", "content": "You are an AI assistant helping with dataset annotation and quality checking."}]
        for h in history:
            msg_list.append({"role": "user", "content": h[0]})
            msg_list.append({"role": "assistant", "content": h[1]})
        msg_list.append({"role": "user", "content": message})

        response, _ = send_to_llm(selected_llm, msg_list)
        
        return history + [[message, response]]
    except Exception as e:
        print(f"Error in chat_with_llm: {str(e)}")
        return history + [[message, f"Error: {str(e)}"]]

def update_chat_context(row_data, index, total, quality, high_quality_tags, low_quality_tags, toxic_tags, other):
    context = f"""Current app state:
    Row: {index + 1}/{total}
    Data: {row_data}
    Quality: {quality}
    High Quality Tags: {', '.join(high_quality_tags)}
    Low Quality Tags: {', '.join(low_quality_tags)}
    Toxic Tags: {', '.join(toxic_tags)}
    Additional Notes: {other}
    """
    return [[None, context]]  # Return as a list of message pairs

# Add this function to handle dataset generation
async def run_generate_dataset(num_workers, num_generations, output_file_path, selected_llm):
    generated_data = []
    for _ in range(num_generations):
        topic_selected = random.choice(TOPICS)
        system_message_selected = random.choice(SYSTEM_MESSAGES_VODALUS)
        data = await generate_data(topic_selected, PROMPT_1, system_message_selected, output_file_path, selected_llm)
        if data:
            generated_data.append(json.dumps(data))
    
    # Write the generated data to the output file
    with open(output_file_path, 'a') as f:
        for entry in generated_data:
            f.write(entry + '\n')
    
    return f"Generated {num_generations} entries and saved to {output_file_path}", "\n".join(generated_data[:5]) + "\n..."

demo = gr.Blocks()

with demo:
    gr.Markdown("# Vodalus Dataset Editor and Annotation Tool")

    config = gr.State(load_annotation_config())

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Dataset Editor"):
                with gr.Row():
                    file_path = gr.Textbox(label="JSONL File Path", value=OUTPUT_FILE_PATH)
                    load_button = gr.Button("Load Dataset")
                
                with gr.Row():
                    prev_button = gr.Button("← Previous")
                    row_index = gr.Number(value=0, label="Current Row", precision=0)
                    total_rows = gr.Number(value=0, label="Total Rows", precision=0)
                    next_button = gr.Button("Next →")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        row_editor = gr.TextArea(label="Edit Row", lines=20)
                    
                    with gr.Column(scale=2):
                        quality_label = gr.Radio(label="Relevance for Training", choices=[])
                        tag_components = [gr.CheckboxGroup(label=f"Tag Group {i+1}", choices=[]) for i in range(3)]
                        other_description = gr.Textbox(label="Additional annotations", lines=3)
                
                with gr.Row():
                    to_markdown_button = gr.Button("Convert to Markdown")
                    to_json_button = gr.Button("Convert to JSON")
                    preview_button = gr.Button("Preview with Metadata")
                    save_row_button = gr.Button("Save Current Row", variant="primary")
                
                preview_output = gr.TextArea(label="Preview", lines=20, interactive=False)
                editor_status = gr.Textbox(label="Editor Status")

            with gr.Tab("Annotation Configuration"):
                with gr.Row():
                    with gr.Column():
                        quality_scale_name = gr.Textbox(label="Quality Scale Name")
                        quality_scale_description = gr.Textbox(label="Quality Scale Description")
                        quality_scale = gr.Dataframe(
                            headers=["Value", "Label"],
                            datatype=["str", "str"],
                            label="Quality Scale",
                            interactive=True
                        )
                
                with gr.Row():
                    tag_categories = gr.Dataframe(
                        headers=["Name", "Type", "Tags"],
                        datatype=["str", "str", "str"],
                        label="Tag Categories",
                        interactive=True
                    )
                
                with gr.Row():
                    free_text_fields = gr.Dataframe(
                        headers=["Name", "Description"],
                        datatype=["str", "str"],
                        label="Free Text Fields",
                        interactive=True
                    )
                
                save_config_btn = gr.Button("Save Configuration")
                config_status = gr.Textbox(label="Status")

            with gr.Tab("Dataset Configuration"):
                with gr.Row():
                    vodalus_system_message = gr.TextArea(label="VODALUS_SYSTEM_MESSAGE", lines=10)
                    prompt_1 = gr.TextArea(label="PROMPT_1", lines=10)
                
                with gr.Row():
                    topics = gr.Dataframe(
                        headers=["Topic"],
                        datatype=["str"],
                        label="TOPICS",
                        interactive=True
                    )
                
                save_dataset_config_btn = gr.Button("Save Dataset Configuration")
                dataset_config_status = gr.Textbox(label="Status")

            with gr.Tab("Dataset Generation"):
                with gr.Row():
                    num_workers = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Workers")
                    num_generations = gr.Number(value=10, label="Number of Generations", precision=0)
                
                with gr.Row():
                    output_file_path = gr.Textbox(label="Output File Path", value=OUTPUT_FILE_PATH)
                
                start_generation_btn = gr.Button("Start Generation")
                generation_status = gr.Textbox(label="Generation Status")
                generation_output = gr.TextArea(label="Generation Output", lines=10)

        with gr.Column(scale=1):
            gr.Markdown("## AI Assistant")
            selected_llm = gr.Radio(["local-model", "anything-llm"], label="Select LLM", value="local-model")
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Chat with AI Assistant")
            clear = gr.Button("Clear")

    load_button.click(
        load_and_show_row,
        inputs=[file_path, gr.Number(value=0), config],
        outputs=[row_editor, row_index, total_rows, quality_label, *tag_components, other_description]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    prev_button.click(
        navigate_rows,
        inputs=[file_path, row_index, gr.Number(value=-1), config],
        outputs=[row_editor, row_index, total_rows, quality_label, *tag_components, other_description]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    next_button.click(
        navigate_rows,
        inputs=[file_path, row_index, gr.Number(value=1), config],
        outputs=[row_editor, row_index, total_rows, quality_label, *tag_components, other_description]
    ).then(
        update_annotation_ui,
        inputs=[config],
        outputs=[quality_label, *tag_components, other_description]
    )

    save_row_button.click(
        save_row_with_metadata,
        inputs=[file_path, row_index, row_editor, config, quality_label, 
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
        inputs=[quality_scale_name, quality_scale_description, quality_scale, tag_categories, free_text_fields],
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
        outputs=[vodalus_system_message, prompt_1, topics]
    )

    save_dataset_config_btn.click(
        save_dataset_config,
        inputs=[vodalus_system_message, prompt_1, topics],
        outputs=[dataset_config_status]
    )

    start_generation_btn.click(
        run_generate_dataset,
        inputs=[num_workers, num_generations, output_file_path, selected_llm],
        outputs=[generation_status, generation_output]
    )

    msg.submit(chat_with_llm, [msg, chatbot, selected_llm], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    # Update chat context when navigating rows or loading dataset
    for button in [load_button, prev_button, next_button]:
        button.click(
            update_chat_context,
            inputs=[row_editor, row_index, total_rows, quality_label, *tag_components, other_description],
            outputs=[chatbot]
        )

if __name__ == "__main__":
    demo.launch(share=True)
