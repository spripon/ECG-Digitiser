import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def delete_key_from_dict(d, key):
    if key in d:
        del d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            delete_key_from_dict(v, key)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    delete_key_from_dict(item, key)


def process_json_file(file_path, key_to_delete):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    delete_key_from_dict(data, key_to_delete)
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
            

def remove_keys(dir, key_name):
    error_list = []
    for root, _, files in os.walk(dir):
        for file in tqdm(files):
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    process_json_file(file_path, key_name)
                except Exception as e:
                    error_list.append((e, file_path))
    print("Errors:")
    print(error_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample plotted pixels in a directory.")
    parser.add_argument(
        '--key_to_remove',
        type=str,
        default="dense_plotted_pixels",
        help='Keys to remove'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default="ptb-xl/records500_prepared_w_images",
        help='Directory containing plotted pixels to be resampled.'
    )
    args = parser.parse_args()

    # Increase pixel density
    print(f"Running {args.dir}...")
    remove_keys(args.dir, args.key_to_remove)
    print("All files corrected.")
