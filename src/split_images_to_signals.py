#!/usr/bin/env python

# This file contains functions for splitting the data up into the individual signals.
#
#   python split_images_to_signals.py \
#   -i input_data \
#   -o output_folder
#
# 'input_data' is a folder containing the your data,
# 'outputs' is a folder for saving your outputs.

import json
import argparse
import os
import sys
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from helper_code import *
from team_code import *


# Parse arguments.
def get_parser():
    description = "Run the data split."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input_data", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("--run_parallel", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=4)
    return parser


def save_single_images(data_dict, output_folder_image, output_folder_label):

    # Get names and paths.
    lead_name = data_dict["info_dict"]["lead_name"]
    image_name = os.path.basename(data_dict["info_dict"]["image_path"])
    label_name = os.path.basename(data_dict["info_dict"]["mask_path"])
    image_name_new = image_name.replace("_0000.png", f"-{lead_name}_0000.png")
    label_name_new = label_name.replace(".png", f"-{lead_name}.png")
    
    # Get output names
    output_file_image = os.path.join(output_folder_image, image_name_new)
    output_file_label = os.path.join(output_folder_label, label_name_new)
    
    if (not os.path.exists(output_file_image)) or (not os.path.exists(output_file_label)):
        
        # Get image and mask.
        image = data_dict["image"]
        label = data_dict["mask"]

        # Convert mask to binary.
        label = torch.where(label >= 1, 1, 0).type(torch.uint8)

        # Save
        write_png(image, output_file_image)
        write_png(label, output_file_label)


# Run saving in parallel
def save_single_images_parallel(
    folder_data, output_folder_image, output_folder_label, num_workers=-1
):
    save_single_images_partial = partial(
        save_single_images,
        output_folder_image=output_folder_image,
        output_folder_label=output_folder_label,
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        _ = list(
            tqdm(
                executor.map(save_single_images_partial, folder_data),
                total=len(folder_data),
            )
        )


# Run the code.
def run(args):
    # Check which paths exist in output folder.
    data_paths = []
    for folder in ["imagesTr", "imagesTv", "imagesTs"]:
        folder_path = os.path.join(args.input_data, folder)
        if os.path.exists(folder_path):
            data_paths.append(folder_path)

    # Get all image files split.
    records, data, loader = dataloader_wrapper(
        list_of_paths=data_paths,
        test_settings=[False, False, False],
        shuffle_settings=[False, False, False],
        transform=None,
        single_signals=True,
    )

    # Save to output folder.
    for record, folder_data, folder_path in zip(records, data, data_paths):
        output_folder_image = os.path.join(
            args.output_folder, os.path.basename(folder_path)
        )
        output_folder_label = output_folder_image.replace("imagesT", "labelsT")
        if not os.path.exists(output_folder_image):
            os.makedirs(output_folder_image)
        if not os.path.exists(output_folder_label):
            os.makedirs(output_folder_label)
        if args.run_parallel:
            print(f"Processing {folder_path} data in parallel...")
            if args.max_workers == -1:
                num_workers = os.cpu_count() - 2
            else:
                num_workers = min(args.max_workers, os.cpu_count() - 2)
            print(f"Using {num_workers}/{os.cpu_count()} workers")
            save_single_images_parallel(
                folder_data,
                output_folder_image,
                output_folder_label,
                num_workers=num_workers,
            )
        else:
            print(f"Processing {folder_path} data in loop...")
            for data_dict in tqdm(folder_data):
                save_single_images(data_dict, output_folder_image, output_folder_label)

    # Crate json dict:
    num_training_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTr"))
            if file.endswith(".png")
        ]
    )
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
