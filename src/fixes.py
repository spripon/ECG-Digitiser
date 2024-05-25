import os
import json

def find_empty_json_files(directory):
    empty_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if not content or content == "{}" or content == "[]":
                            empty_files.append(file_path)
                except json.JSONDecodeError:
                    # Handle the case where the file is not a valid JSON
                    pass
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return empty_files


def delete_files_with_prefix(root_folder, prefix):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith(prefix):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    # directory = '/data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset100_Signals'
    # empty_json_files = find_empty_json_files(directory)
    # print("Empty JSON files:", empty_json_files)
    
    root_folder = '/data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset300_FullImages'
    prefix = '16'
    delete_files_with_prefix(root_folder, prefix)
    
    