import json

def copy_json_lines(source_file, destination_file, start_line):
    """
    Copy content from source_file to destination_file starting from start_line (1-indexed).
    Assumes each line is a valid JSON object or array.
    Saves each object as a new line in JSONL format.
    """
    with open(source_file, 'r', encoding='utf-8') as src:
        lines = src.readlines()

    if start_line < 1 or start_line > len(lines):
        raise ValueError("start_line is out of range")

    selected_lines = lines[start_line - 1:]

    with open(destination_file, 'w', encoding='utf-8') as dest:
        for line in selected_lines:
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)  # Validate JSON
                    dest.write(json.dumps(json_obj) + '\n')  # Write as one line
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                    continue

# Example usage
source_path = r'C:\Users\hp\lawRAG2\data\cases_2.jsonl'
destination_path = r'C:\Users\hp\lawRAG2\data\smol_cases_2.jsonl'
start_from_line = 199

copy_json_lines(source_path, destination_path, start_from_line)
