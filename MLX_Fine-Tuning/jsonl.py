import json

# The path to your current JSONL file
input_file_path = 'data/train.jsonl'
# The path where the reformatted JSONL file will be saved
output_file_path = 'data/reformatted_train.jsonl'

try:
    # Open the input file and create the output file
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            try:
                # Load the JSON object from the current line
                json_object = json.loads(line)
                # Write the JSON object as a single line in the output file
                output_file.write(json.dumps(json_object) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line}")
                print(f"Error message: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print(f'Reformatted JSONL file saved to {output_file_path}')