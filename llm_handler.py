import requests
import json
from openai import OpenAI
from params import load_params
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_client():
    params = load_params()
    if params['PROVIDER'] == 'local-model':
        return OpenAI(api_key="local-model", base_url=params['BASE_URL'])
    return None

def send_to_chatgpt(msg_list):
    try:
        client = get_client()
        if client is None:
            raise ValueError("Failed to initialize OpenAI client")
        
        params = load_params()
        logger.info(f"Sending request to: {params['BASE_URL']}")
        logger.info(f"Using model: {params['MODEL']}")
        logger.debug(f"Input messages: {json.dumps(msg_list, indent=2)}")
        
        completion = client.chat.completions.create(
            model=params['MODEL'],
            temperature=params['temperature'],
            messages=msg_list
        )
        chatgpt_response = completion.choices[0].message.content
        chatgpt_usage = completion.usage
        logger.debug(f"LLM response: {chatgpt_response}")
        logger.debug(f"Usage: {chatgpt_usage}")
        return chatgpt_response, chatgpt_usage
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in send_to_chatgpt: {str(e)}")
        return f"Error: Connection failed - {str(e)}", None
    except Exception as e:
        logger.error(f"Error in send_to_chatgpt: {str(e)}")
        return f"Error: {str(e)}", None

def send_to_anything_llm(msg_list):
    params = load_params()
    url = f"{params['BASE_URL']}/api/v1/workspace/{params['WORKSPACE']}/chat"
    headers = {
        'accept': 'application/json',
        'Authorization': f"Bearer {params['API_KEY']}",
        'Content-Type': 'application/json'
    }
    message_content = " ".join(msg["content"] for msg in msg_list if "content" in msg)
    data = {
        "message": message_content,
        "mode": "chat"
    }
    data_json = json.dumps(data)
    logger.debug(f"Sending to AnythingLLM: {data_json}")
    try:
        response = requests.post(url, headers=headers, data=data_json)
        response.raise_for_status()
        response_data = response.json()
        chatgpt_response = response_data.get("textResponse")
        chatgpt_usage = response_data.get("usage", {})
        logger.debug(f"AnythingLLM response: {chatgpt_response}")
        logger.debug(f"AnythingLLM usage: {chatgpt_usage}")
        return chatgpt_response, chatgpt_usage
    except requests.RequestException as e:
        logger.error(f"Error in send_to_anything_llm: {str(e)}")
        return f"Error: {str(e)}", None

def send_to_llm(msg_list):
    params = load_params()
    logger.info(f"Using provider: {params['PROVIDER']}")
    if params['PROVIDER'] == "local-model":
        return send_to_chatgpt(msg_list)
    elif params['PROVIDER'] == "anything-llm":
        return send_to_anything_llm(msg_list)
    else:
        raise ValueError(f"Unknown provider: {params['PROVIDER']}")

def send_to_llm_wrapper(msg_list):
    logger.info("Sending message to LLM")
    return send_to_llm(msg_list)
