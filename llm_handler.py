import requests
import json
from openai import OpenAI
from params import OPENAI_MODEL, OPENAI_API_KEY

# Create an instance of the OpenAI class for the local model
client = OpenAI(api_key="local-model", base_url="http://localhost:11434/v1")

def send_to_chatgpt(msg_list):
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.6,
            messages=msg_list
        )
        chatgpt_response = completion.choices[0].message.content
        chatgpt_usage = completion.usage
        return chatgpt_response, chatgpt_usage
    except Exception as e:
        print(f"Error in send_to_chatgpt: {str(e)}")
        return f"Error: {str(e)}", None

def send_to_anything_llm(msg_list):
    url = 'http://localhost:3001/api/v1/workspace/mycoworks/chat'
    headers = {
        'accept': 'application/json',
        'Authorization': 'Bearer 0MACR41-7804XQB-MGC1GS0-FGSKB44',
        'Content-Type': 'application/json'
    }
    message_content = " ".join(msg["content"] for msg in msg_list if "content" in msg)
    data = {
        "message": message_content,
        "mode": "chat"
    }
    data_json = json.dumps(data)
    try:
        response = requests.post(url, headers=headers, data=data_json)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_data = response.json()
        chatgpt_response = response_data.get("textResponse")
        chatgpt_usage = response_data.get("usage", {})
        return chatgpt_response, chatgpt_usage
    except requests.RequestException as e:
        print(f"Error in send_to_anything_llm: {str(e)}")
        return f"Error: {str(e)}", None

def send_to_llm(provider, msg_list):
    if provider == "local-model":
        return send_to_chatgpt(msg_list)
    elif provider == "anything-llm":
        return send_to_anything_llm(msg_list)
    else:
        raise ValueError(f"Unknown provider: {provider}")
