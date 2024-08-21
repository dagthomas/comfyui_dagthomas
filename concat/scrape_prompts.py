import os
import json
import re
from multiprocessing import Pool, cpu_count
from itertools import chain

def extract_segments(text):
    return [segment.strip() for segment in re.split(r'[,.]', text) if segment.strip()]

def is_json(content):
    # Check if the content starts with ```json
    if content.strip().startswith('```json'):
        return True
    
    # Check if the content is a JSON array or object
    try:
        json_content = content.strip()
        if (json_content.startswith('{') and json_content.endswith('}')) or \
           (json_content.startswith('[') and json_content.endswith(']')):
            json.loads(json_content)
            return True
    except json.JSONDecodeError:
        pass
    
    return False

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='iso-8859-1') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return []
    
    if is_json(content):
        print(f"Skipping JSON content in file: {file_path}")
        return []
    
    return extract_segments(content)

def process_batch(file_batch):
    return list(chain.from_iterable(process_file(file_path) for file_path in file_batch))

def process_folder(folder_path, batch_size=100):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    # Create batches of file paths
    batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    # Use multiprocessing to process batches in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_batch, batches)
    # Flatten the results
    return list(chain.from_iterable(results))

def save_to_json(segments, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

# Main execution
if __name__ == '__main__':
    folder_path = '../prompts'  # Replace with your folder path
    output_file = 'output2.json'
    extracted_segments = process_folder(folder_path)
    save_to_json(extracted_segments, output_file)
    print(f"Processed all .txt files. Results saved to {output_file}")