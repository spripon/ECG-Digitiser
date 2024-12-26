# Run the digitization of ECG images.

import argparse
import cv2
import numpy as np
import pandas as pd
import os
import shutil
import subprocess
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.io.image import read_image, write_png
from torchvision.transforms.functional import rotate
import wfdb

from config import (
    DATASET_NAME,
    IMAGE_TYPE,
    FREQUENCY,
    LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
    SIGNAL_UNITS,
    LEAD_LABEL_MAPPING,
)


# Parse arguments.
def get_parser():
    description = "Run the trained Challenge models."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        required=True,
        help="Folder containing the images to digitize.",
    )
    parser.add_argument(
        "-m",
        "--model_folder",
        type=str,
        required=True,
        help="Folder containing the nnUNet folder nnUNet_results.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the digitized images.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--allow_failures", action="store_true")
    return parser


def get_rotation_angle(np_image):
    """Get the rotation angle of the image."""
    lines = get_lines(np_image, threshold_HoughLines=1200)
    filtered_lines = filter_lines(
        lines, degree_window=30, parallelism_count=3, parallelism_window=2
    )
    if filtered_lines is None:
        rot_angle = np.nan
    else:
        rot_angle = get_median_degrees(filtered_lines)
    return rot_angle


def get_median_degrees(lines):
    """Get the median angle of the lines."""
    lines = lines[:, 0, :]
    line_angles = [-(90 - line[1] * 180 / np.pi) for line in lines]
    return round(np.median(line_angles), 4)


def is_within_x_degrees_of_horizontal(theta, degree_window):
    """Check if the line is within x degrees of horizontal (90 degrees)."""
    theta_degrees = theta * 180 / np.pi
    deviation_from_horizontal = abs(90 - theta_degrees)
    return deviation_from_horizontal < degree_window


def get_lines(np_image, threshold_HoughLines=1380, rho_resolution=1):
    """Get the lines in the image."""
    # Convert the image to a grayscale NumPy array
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detector to find edges in the image
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Use HoughLines to find lines in the edge-detected image
    lines = cv2.HoughLines(
        edges, rho_resolution, np.pi / 180, threshold_HoughLines, None, 0, 0
    )

    return lines


def filter_lines(lines, degree_window=20, parallelism_count=0, parallelism_window=2):
    """Filter the lines to get the rotation angle."""
    parallelism_radian = np.deg2rad(parallelism_window)
    filtered_lines = []
    line_angles = []

    # Filter lines to be within the degree window of horizontal
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if is_within_x_degrees_of_horizontal(theta, degree_window):
                    filtered_lines.append((rho, theta))
                    line_angles.append(theta)

    # Further filter lines based on parallelism
    parallel_lines = []
    if len(filtered_lines) > 0:
        for rho, theta in filtered_lines:
            count = 0
            for comp_rho, comp_theta in filtered_lines:
                if (
                    abs(theta - comp_theta) < parallelism_radian
                    or abs((theta - comp_theta) - np.pi) < parallelism_radian
                ):
                    count += 1
            if count >= parallelism_count:
                parallel_lines.append((rho, theta))

    if len(parallel_lines) == 0:
        parallel_lines = None
    else:
        parallel_lines = np.array(parallel_lines)[:, np.newaxis, :]

    return parallel_lines


def predict_mask_nnunet(image, dataset_name, model_folder):
    """Predict the mask using nnUNet."""

    # Define temporary folders and paths
    temp_folder_input = "data/temp_nnUNet_input"
    temp_folder_output = "data/temp_nnUNet_output"
    temp_folder_output_pp = "data/temp_nnUNet_output_pp"
    image_path_temp = os.path.join(temp_folder_input, "00000_temp_0000.png")
    mask_path_temp = os.path.join(temp_folder_output, "00000_temp.png")

    # Define run commands
    command_run = f"nnUNetv2_predict -d {dataset_name} -i {temp_folder_input} -o {temp_folder_output} -f all -tr nnUNetTrainer -c 2d -p nnUNetPlans"

    # Set env variabels (nnUNet needs them to be set)
    os.environ["nnUNet_results"] = os.path.join(model_folder, "nnUNet_results")

    # Create temp folders:
    shutil.rmtree(temp_folder_input, ignore_errors=True)
    shutil.rmtree(temp_folder_output, ignore_errors=True)
    shutil.rmtree(temp_folder_output_pp, ignore_errors=True)
    os.makedirs(temp_folder_input, exist_ok=True)
    os.makedirs(temp_folder_output, exist_ok=True)
    os.makedirs(temp_folder_output_pp, exist_ok=True)

    # Save image
    write_png(image, image_path_temp)

    # Run inference
    subprocess.run(command_run, shell=True)

    # Load mask
    mask = read_image(mask_path_temp)

    # Delete all temporary folders and files
    shutil.rmtree(temp_folder_input, ignore_errors=True)
    shutil.rmtree(temp_folder_output, ignore_errors=True)
    shutil.rmtree(temp_folder_output_pp, ignore_errors=True)

    return mask


def cut_to_mask(img, mask, return_y1=False):
    """Cut the image to the mask."""
    coords = torch.where(mask[0] >= 1)
    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()
    img = img[:, y_min : y_max + 1, x_min : x_max + 1]
    if return_y1:
        return img, y_min
    else:
        return img


def cut_binary(mask_to_use, image_rotated):
    """Cut the binary mask into single binary masks."""
    signal_masks = {}
    signal_images = {}
    signal_positions = {}
    mask_values = list(pd.Series(mask_to_use.numpy().flatten()).value_counts().index)
    possible_lead_names = LEAD_LABEL_MAPPING
    lead_names_in_mask = {
        k: v for k, v in possible_lead_names.items() if v in mask_values
    }
    for lead_name, lead_value in lead_names_in_mask.items():
        binary_mask = torch.where(mask_to_use == lead_value, 1, 0)
        signal_img, y1 = cut_to_mask(image_rotated, binary_mask, True)
        signal_mask = cut_to_mask(binary_mask, binary_mask)
        signal_images[lead_name] = signal_img
        signal_masks[lead_name] = signal_mask
        signal_positions[lead_name] = y1

    return signal_masks, signal_positions, signal_images


def vectorise(
    image_rotated, mask, signal_cropped, sec_per_pixel, mV_per_pixel, y_shift_ratio, lead
):
    """Vectorise the image."""

    # Get scaling info
    total_seconds_from_mask = round(torch.tensor(sec_per_pixel).item() * mask.shape[2], 1)
    if total_seconds_from_mask > (LONG_SIGNAL_LENGTH_SEC / 2):
        total_seconds = LONG_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio["full"]
    else:
        total_seconds = SHORT_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio[lead]
    values_needed = int(total_seconds * FREQUENCY)

    # Scale y
    # The code aligns and scales a signal based on a mask's non-zero regions and a vertical shift ratio. It computes the mean vertical position of non-zero elements in the mask, adjusts the signal's vertical position using y_shift_ratio_, and scales the result into physical units (e.g., millivolts) for further analysis.
    non_zero_mean = torch.tensor(
        [
            torch.mean(torch.nonzero(mask[0, :, i]).type(torch.float32))
            for i in range(mask.shape[2])
        ]
    )
    signal_cropped_shifted = (1 - y_shift_ratio_) * image_rotated.shape[
        1
    ] - signal_cropped
    predicted_signal = (signal_cropped_shifted - non_zero_mean) * mV_per_pixel

    # Scale x
    # The code reshapes the predicted signal into a 3D tensor for interpolation and resamples it to a specified size using linear interpolation. It then flattens the resampled data back into a 1D tensor for further use.
    n = predicted_signal.shape[0]
    data_reshaped = predicted_signal.view(1, 1, n)
    resampled_data = F.interpolate(
        data_reshaped, size=values_needed, mode="linear", align_corners=False
    )
    predicted_signal_sampled = resampled_data.view(-1)

    return predicted_signal_sampled


# Run the code.
def run(args):
    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the team's models on the Challenge data.
    if args.verbose:
        print("Running digitization model...")

    # Iterate over the records.
    image_files = [
        f for f in os.listdir(args.data_folder) if f.endswith(f".{IMAGE_TYPE}")
    ]
    for _, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        # Get record and header files.
        image_file_path = os.path.join(args.data_folder, image_file)
        record = image_file.replace(f".{IMAGE_TYPE}", "")
        os.makedirs(args.output_folder, exist_ok=True)
        image = read_image(image_file_path)
        image = image[:3]

        # Rotate
        rot_angle = get_rotation_angle(image.permute(1, 2, 0).numpy().astype(np.uint8))
        image_rotated = rotate(image, rot_angle)

        # Segment
        mask_to_use = predict_mask_nnunet(image_rotated, DATASET_NAME, args.model_folder)

        # Use mask to cut into single, binary masks
        signal_masks_cropped, signal_positions_cropped, _ = cut_binary(
            mask_to_use, image_rotated
        )

        # Vecotrise
        x_pixel_list = [v.shape[2] for v in signal_masks_cropped.values()]
        x_pixel_list_median = np.median(x_pixel_list)
        x_pixel_list_below_2x_median_mean = np.mean(
            [v for v in x_pixel_list if v < 2 * x_pixel_list_median]
        )
        sec_per_pixel = 2.5 / x_pixel_list_below_2x_median_mean
        mm_per_pixel = 25 * sec_per_pixel
        sec_per_pixel = mm_per_pixel / 25
        mV_per_pixel = mm_per_pixel / 10
        signals_predicted = {}
        for lead, mask in signal_masks_cropped.items():
            signals_predicted[lead] = vectorise(
                image_rotated,
                mask,
                signal_positions_cropped[lead],
                sec_per_pixel,
                mV_per_pixel,
                Y_SHIFT_RATIO,
                lead,
            )

        # Save Challenge outputs.
        signals = np.array(
            [
                signals_predicted[signal_name].numpy()
                for signal_name in LEAD_LABEL_MAPPING.keys()
            ]
        ).T
        wfdb.wrsamp(
            record,
            fs=FREQUENCY,
            units=SIGNAL_UNITS,
            sig_name=LEAD_LABEL_MAPPING.keys(),
            p_signal=signals,
            write_dir=args.output_folder,
        )

    if args.verbose:
        print("Done.")


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
