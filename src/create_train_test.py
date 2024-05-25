#!/usr/bin/env python

# This file contains functions for creating training_data, vali_data, test_data:
#
#   python create_train_test.py \
#   -i input_data \
#   -d database_file \
#   -o output_folder
#
# where 'database_file' is the path to 'ptbxl_database.csv',
# 'input_data' is a folder containing the your data,
# 'outputs' is a folder for saving your outputs, and
#  -m is an optional argument to move files instead of copying them.

import json
import argparse
import glob
import os
import os
import shutil
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from helper_code import *


# Parse arguments.
def get_parser():
    description = "Run the data split."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", "--database_file", type=str, required=True)
    parser.add_argument("-i", "--input_data", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument(
        "-m", "--move", action="store_true", help="move files instead of copying"
    )
    parser.add_argument(
        "--rgba_to_rgb",
        action="store_true",
        default=False,
        help="Convert all rgba images to rgb images",
    )
    parser.add_argument(
        "--gray_to_rgb",
        action="store_true",
        default=False,
        help="Convert all gray scale images to rgb images",
    )
    return parser


# Function to either copy or move files
def transfer_files(source_paths, target_dir, move=False):
    for file_path in tqdm(source_paths):
        if move:
            shutil.move(file_path, target_dir)
        else:
            shutil.copy(file_path, target_dir)


# Function to convert rgba to rgb
def convert_rgba_to_rgb(file_path):
    img = Image.open(file_path)
    if img.mode == "RGBA":
        rgb_img = img.convert("RGB")
        rgb_img.save(file_path)


# Function to convert gray scale to rgb
def convert_gray_to_rgb(file_path):
    img = Image.open(file_path)
    if img.mode == "L":
        rgb_img = img.convert("RGB")
        rgb_img.save(file_path)


# Run conversion in parallel
def convert_gray_to_rgb_parallel(file_paths, num_workers=-1):
    if num_workers == -1:
        workers = os.cpu_count() - 2
    else:
        workers = num_workers
    print(f"Using {workers}/{os.cpu_count()} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        _ = list(
            tqdm(executor.map(convert_gray_to_rgb, file_paths), total=len(file_paths))
        )


# Run conversion in parallel
def convert_rgba_to_rgb_parallel(file_paths, num_workers=-1):
    if num_workers == -1:
        workers = os.cpu_count() - 2
    else:
        workers = num_workers
    print(f"Using {workers}/{os.cpu_count()} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        _ = list(
            tqdm(executor.map(convert_rgba_to_rgb, file_paths), total=len(file_paths))
        )


# Run the code.
def run(args):
    # Data groups from the PTB-XL database
    strat_fold_train = [1, 2, 3, 4, 5, 6, 7, 8]
    strat_fold_vali = [9]
    strat_fold_test = [10]

    # Load the data
    dg = pd.read_csv(args.database_file, index_col="ecg_id")
    dg["file_start"] = dg.index.map(lambda x: str(x).zfill(5))

    # Get file paths
    data_groups = {"imagesTr": [], "imagesTs": [], "imagesTv": []}
    all_file_paths = glob.glob(f"{args.input_data}/**/*", recursive=True)
    count_files = 0
    print("Starting to determine data groups...")
    for _, row in tqdm(dg.iterrows(), total=dg.shape[0]):
        file_start = row["file_start"]
        strat_fold = row["strat_fold"]
        matching_paths = [path for path in all_file_paths if f"{file_start}_lr" in path]
        count_files += len(matching_paths)
        if strat_fold in strat_fold_train:
            data_groups["imagesTr"].extend(matching_paths)
        if strat_fold in strat_fold_vali:
            data_groups["imagesTv"].extend(matching_paths)
        if strat_fold in strat_fold_test:
            data_groups["imagesTs"].extend(matching_paths)
    print(
        f"In total found {len(data_groups['imagesTr']) + len(data_groups['imagesTv']) + len(data_groups['imagesTs'])} files, compared to {count_files} files in the input folder."
    )

    # Create target directories and transfer files
    if args.move:
        print("Moving files...")
    else:
        print("Copying files...")
    for group_name, file_paths in tqdm(data_groups.items()):
        target_dir = os.path.join(args.output_folder, group_name)
        os.makedirs(target_dir, exist_ok=True)
        transfer_files(file_paths, target_dir, args.move)

    # Move all mask to separate folder
    print("Moving masks to separate folder...")
    endings_to_move = ["_mask.png", "_mask_augmented.png"]
    endings_to_rename = {
        "_0000_mask.png": ".png",
        "_0000_mask_augmented.png": "_augmented.png",
    }
    for folder in ["imagesTr", "imagesTv", "imagesTs"]:
        old_folder_path = os.path.join(args.output_folder, folder)
        new_folder_path = old_folder_path.replace("imagesT", "labelsT")
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        for ending in endings_to_move:
            files_to_move = [
                file for file in os.listdir(old_folder_path) if file.endswith(ending)
            ]
            for file in files_to_move:
                new_file = file
                for old, new in endings_to_rename.items():
                    new_file = new_file.replace(old, new)
                shutil.move(
                    os.path.join(old_folder_path, file),
                    os.path.join(new_folder_path, new_file),
                )

    # Optional: Convert all rgba to rgb
    if args.rgba_to_rgb:
        print("Converting all rgba images to rgb images...")
        for folder in ["imagesTr", "imagesTv", "imagesTs"]:
            print(f"Converting images in {folder}...")
            folder_path = os.path.join(args.output_folder, folder)
            file_paths_to_convert = [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.endswith(".png")
            ]
            convert_rgba_to_rgb_parallel(file_paths_to_convert)

    # Optional: Convert all labels to rgb
    if args.gray_to_rgb:
        print("Converting all masks to rgb images...")
        for folder in ["labelsTr", "labelsTv", "labelsTs"]:
            print(f"Converting labels in {folder}...")
            folder_path = os.path.join(args.output_folder, folder)
            file_paths_to_convert = [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.endswith(".png")
            ]
            convert_gray_to_rgb_parallel(file_paths_to_convert)

    # Print number of files in each group
    num_training_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTr"))
            if file.endswith(".png")
        ]
    )
    num_validation_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTv"))
            if file.endswith(".png")
        ]
    )
    num_test_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTs"))
            if file.endswith(".png")
        ]
    )
    print(f"Training data: {num_training_data} images")
    print(f"Validation data: {num_validation_data} images")
    print(f"Test data: {num_test_data} images")

    # Create summary dataset.json file
    dataset_json_dict = {
        "channel_names": {"0": "Signals"},
        "labels": {"background": 0, "signal": 1},
        "numTraining": num_training_data,
        "file_ending": ".png",
    }
    with open(os.path.join(args.output_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json_dict, f)


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
    print("Files have been transferred successfully.")
