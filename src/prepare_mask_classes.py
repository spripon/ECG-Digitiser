#!/usr/bin/env python

# This file contains functions for creating different mask classes.
#
#   python prepare_mask_classes.py \
#   -i input_folder \
#   -o output_folder
#
# 'input_folder' is a folder containing the your data,
# 'output_folder' is a folder for saving your outputs.

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
    description = "Prepare signal specific masks."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("--run_parallel", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=4)
    return parser


def copy_images(data_dict, output_folder_image):
    img_name = os.path.basename(data_dict["info_dict"]["image_path"])
    output_file_name = os.path.join(output_folder_image, img_name)
    if not os.path.exists(output_file_name):
        img = data_dict["image"]
        write_png(img, output_file_name)


def convert_masks(data_dict, output_folder_label):
    # Get mask and other information
    box_type = "lead_bounding_box"
    mask = data_dict["mask"]
    mask_name = os.path.basename(data_dict["info_dict"]["mask_path"])
    output_file_name = os.path.join(output_folder_label, mask_name)
    
    # Get more infos and save mask
    if not os.path.exists(output_file_name):
        img_height = mask.shape[1]
        mask_values = BOX_TYPE_LABEL_MAPPING[box_type]
        full_mode_lead = data_dict["info_dict"]["full_mode_lead"]
        boxes_of_type = data_dict["info_dict"][box_type]
        boxes_of_type_filtered = filter_for_full_lead(
            boxes_of_type, full_mode_lead, box_type
        )
        mask_new = torch.zeros_like(mask)
        for bbox in boxes_of_type_filtered:
            lead_name = bbox["lead_name"]
            lead_value = mask_values[lead_name]
            bbox_converted = convert_json_to_bbox(bbox, img_height)
            x1 = bbox_converted["x1"]
            y2 = bbox_converted["y2"]
            x2 = bbox_converted["x2"]
            y1 = bbox_converted["y1"]
            mask_new[:, y1:y2, x1:x2][mask[:, y1:y2, x1:x2] != 0] = lead_value
        write_png(mask_new, output_file_name)


# Run saving in parallel
def convert_masks_parallel(folder_data, output_folder_label, num_workers=-1):
    if num_workers == -1:
        num_workers = os.cpu_count() - 2
    print(f"Using {num_workers}/{os.cpu_count()} workers")
    convert_masks_partial = partial(
        convert_masks,
        output_folder_label=output_folder_label,
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        _ = list(
            tqdm(
                executor.map(convert_masks_partial, folder_data),
                total=len(folder_data),
            )
        )
        

# Run copying in parallel
def copy_images_parallel(folder_data, output_folder_image, num_workers=-1):
    if num_workers == -1:
        num_workers = os.cpu_count() - 2
    print(f"Using {num_workers}/{os.cpu_count()} workers")
    copy_images_partial = partial(
        copy_images,
        output_folder_image=output_folder_image,
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        _ = list(
            tqdm(
                executor.map(copy_images_partial, folder_data),
                total=len(folder_data),
            )
        )


# Run the code.
def run(args):
    
    # Check that we are not overwriting
    assert args.input_folder != args.output_folder, "Input and output folder are the same."
    
    # Get number of workers
    if args.max_workers == -1:
        num_workers = os.cpu_count() - 2
    else:
        num_workers = min(args.max_workers, os.cpu_count() - 2)
    
    # Check which paths exist in output folder.
    data_paths = []
    for folder in ["imagesTr", "imagesTv", "imagesTs"]:
        folder_path = os.path.join(args.input_folder, folder)
        if os.path.exists(folder_path):
            data_paths.append(folder_path)

    # Get all image files.
    records, data, loader = dataloader_wrapper(
        list_of_paths=data_paths,
        test_settings=[False, False, False],
        shuffle_settings=[False, False, False],
        transform=None,
    )

    # Convert masks and save.
    for folder_data, folder_path in zip(data, data_paths):
        
        # Prepare output folders
        print(f"Processing {folder_path} data...")
        output_folder_image = os.path.join(
            args.output_folder, os.path.basename(folder_path)
        )
        output_folder_label = output_folder_image.replace("imagesT", "labelsT")
        if not os.path.exists(output_folder_image):
            os.makedirs(output_folder_image)
        if not os.path.exists(output_folder_label):
            os.makedirs(output_folder_label)
            
        # Copy images
        if args.run_parallel:
            print(f"Copying images from {folder_path} to {output_folder_image} in parallel...")
            copy_images_parallel(
                folder_data,
                output_folder_image,
                num_workers=num_workers,
            )
        else:
            print(f"Copying images from {folder_path} to {output_folder_image} sequentially...")
            for data_dict in tqdm(folder_data):
                copy_images(data_dict, output_folder_image)

        # Convert masks and save
        if args.run_parallel:
            print(f"Processing {folder_path} data in parallel...")
            convert_masks_parallel(
                folder_data,
                output_folder_label,
                num_workers=num_workers,
            )
        else:
            print(f"Processing {folder_path} data in loop...")
            for data_dict in tqdm(folder_data):
                convert_masks(data_dict, output_folder_label)

    # Crate json dict:
    num_training_data = len(
        [
            file
            for file in os.listdir(os.path.join(args.output_folder, "imagesTr"))
            if file.endswith(".png")
        ]
    )
    labels_dict = {"background": 0}
    labels_dict.update(BOX_TYPE_LABEL_MAPPING["lead_bounding_box"])
    dataset_json_dict = {
        "channel_names": {"0": "Signals"},
        "labels": labels_dict,
        "numTraining": num_training_data,
        "file_ending": ".png",
    }
    with open(os.path.join(args.output_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json_dict, f)


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
    print("Files have been transferred successfully.")
