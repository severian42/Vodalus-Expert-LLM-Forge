import requests
import json

# Import the OpenAI class from the openai library to interact with OpenAI's API.
from openai import OpenAI

# Import specific constants from the params.py file which includes OPENAI_MODEL and OPENAI_API_KEY.
from params import OPENAI_MODEL, OPENAI_API_KEY

# OPEN AI METHOD #################################################

# Create an instance of the OpenAI class with specific parameters for API key and base URL.
# This instance is configured to interact with a local model running on localhost at port 1234.
client = OpenAI(api_key="local-model", base_url="http://localhost:1234/v1")

# Define a function named send_to_chatgpt which takes a list of messages as input.
def send_to_chatgpt(msg_list):
    # This line sends the list of messages to the chat model and requests a completion.
    # It sets the model to use (from params.py), a temperature parameter that influences the randomness of the response.
    completion = client.chat.completions.create(model=OPENAI_MODEL, temperature=0.6, messages=msg_list)
    
    # Extract the content of the message from the first choice of the completion response.
    chatgpt_response = completion.choices[0].message.content
    
    # Extract the usage information from the completion response, which includes tokens used.
    chatgpt_usage = completion.usage
    
    # Return both the response content and the usage information.
    return chatgpt_response, chatgpt_usage


# Define a function named send_to_llm which takes a provider name and a list of messages as input.
def send_to_llm(provider, msg_list):
    # Check if the provider specified is "local-model".
    if provider == "local-model":
        # If it is, call the send_to_chatgpt function with the list of messages.
        response, usage = send_to_chatgpt(msg_list)
        
        # Return the response and usage information received from send_to_chatgpt.
        return response, usage

##########################################################


# ANYTHING-LLM METHOD #################################################

"""
def send_to_llm(provider, msg_list):
    if provider == "local-model":
        url = 'http://localhost:3001/api/v1/workspace/<workspace-name>/chat'
        headers = {
            'accept': 'application/json',
            'Authorization': 'Bearer 0W76065-XXXXXXX-XXXXXXX-XXXXXXX',
            'Content-Type': 'application/json'
        }
        # Extract the 'content' from each message dictionary and join them into a single string
        message_content = " ".join(msg["content"] for msg in msg_list if "content" in msg)
        data = {
            "message": message_content,
            "mode": "chat"
        }
        data_json = json.dumps(data)
        try:
            response = requests.post(url, headers=headers, data=data_json)
            response_data = response.json()
            chatgpt_response = response_data.get("textResponse")
            chatgpt_usage = response_data.get("usage", {})
            return chatgpt_response, chatgpt_usage
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None, None
"""

##################################################
    