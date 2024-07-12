import json

DEFAULT_PARAMS = {
    'PROVIDER': 'local-model',
    'BASE_URL': 'http://localhost:11434/v1',
    'WORKSPACE': 'mycoworks',
    'API_KEY': '0MACR41-7804XQB-MGC1GS0-FGSKB44',
    'OUTPUT_FILE_PATH': './dataset.jsonl',
    'NUM_WORKERS': 1
}

def load_params():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_PARAMS

def save_params(params):
    with open('config.json', 'w') as f:
        json.dump(params, f, indent=2)

# For backwards compatibility
OPENAI_MODEL = "phi3:latest"
OPENAI_API_KEY = ""
OUTPUT_FILE_PATH = DEFAULT_PARAMS['OUTPUT_FILE_PATH']
NUM_WORKERS = DEFAULT_PARAMS['NUM_WORKERS']
PROVIDER = DEFAULT_PARAMS['PROVIDER']
