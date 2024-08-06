#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

# General libraries
import librosa
import multiprocessing
import joblib
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import random
import re
import subprocess
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from collections import OrderedDict
from tempfile import TemporaryDirectory
from tqdm import tqdm
from typing import List, Callable, Optional, Tuple

# ML libraries
import cv2
from PIL import Image
from imgaug import augmenters as iaa
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# PyTorch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from torchvision.datasets import VisionDataset
from torchvision.io.image import read_image, write_png
from torchvision.models import get_model, get_weight
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import (
    to_pil_image,
    rotate,
    rgb_to_grayscale,
    resize,
)
from torchvision.transforms import v2
from torchvision.ops import box_iou, generalized_box_iou_loss
from torch.nn.functional import cross_entropy
import torch.nn.functional as F

# Own methods
from helper_code import *


################################################################################
#
# Settings
#
################################################################################
# Root folder
ROOT = "/data/wolf6245"

# Device settings
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# DEVICE = torch.device("cpu")
NUM_WORKERS = 8
multiprocessing.set_start_method('spawn', force=True)
if DEVICE.type == "cuda":
    WORLD_SIZE = torch.cuda.device_count()
else:
    WORLD_SIZE = 1

# General settings
BATCH_SIZE = 32
SEED = 42

# Classification settings
DO_CLASSIFICATION = True
VALI_SIZE = 0.2
EPOCHS_CLASSIFICATION = 200
USE_BEST_MODEL = False
CLASSIFICATION_THRESHOLD=0.5
IMAGE_BASED_CLASSIFICATION = False
USE_SPECTROGRAMS = IMAGE_BASED_CLASSIFICATION
MODEL_NAME_CLASSIFICATION = "resnet50"
NC_CLASSIFICATION = 3

# General digitization settings
TRAIN_NNUNET = True
X_FREQUENCY = 500
NC = 3
IMG_SIZE = (50, 500)
IMAGES_PARTS_FOR_GRID_PREDICTION = (0.7, 0.05, 1.0, 0.25)  # relative: (x1,y1,x2,y2)

# Get bbox settings
PIXELS_TO_SHIFT = (0, -4)  # (x shift, y shift) # TODO: Check why this is necessary
NUM_EPOCHS_BBOX = 100
LR_BBOX = 0.005
MOMENTUM_BBOX = 0.9
WEIGHT_DECAY_BBOX = 0.0005
STEP_SIZE_BBOX = 3
GAMMA_BBOX = 0.1
BOX_TYPES_TO_USE = ["lead_bounding_box"]
BOX_TYPE_LABEL_MAPPING = {
    "lead_bounding_box": {
        "I": 1,
        "II": 2,
        "III": 3,
        "aVR": 4,
        "aVL": 5,
        "aVF": 6,
        "V1": 7,
        "V2": 8,
        "V3": 9,
        "V4": 10,
        "V5": 11,
        "V6": 12,
    },
    "grid_text_bounding_box_x": {"25mm/s": 13},
    "grid_text_bounding_box_y": {"10mm/mV": 14},
    "text_bounding_box": {
        "I": 15,
        "II": 16,
        "III": 17,
        "aVR": 18,
        "aVL": 19,
        "aVF": 20,
        "V1": 21,
        "V2": 22,
        "V3": 23,
        "V4": 24,
        "V5": 25,
        "V6": 26,
    },
}

# Mask color coding
MASK_COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 128, 0),
    7: (128, 0, 0),
    8: (0, 0, 128),
    9: (128, 128, 0),
    10: (128, 0, 128),
    11: (0, 128, 128),
    12: (128, 128, 128),
}

# y 0 alignment
Y_VALUES_PER_LEAD = {
    "I": 12.6 / 21.59,
    "II": 9 / 21.59,
    "III": 5.4 / 21.59,
    "aVR": 12.6 / 21.59,
    "aVL": 9 / 21.59,
    "aVF": 5.4 / 21.59,
    "V1": 12.59 / 21.59,
    "V2": 9 / 21.59,
    "V3": 5.4 / 21.59,
    "V4": 12.59 / 21.59,
    "V5": 9 / 21.59,
    "V6": 5.4 / 21.59,
    "full": 2.1 / 21.59,
}

SIGNAL_START = {
    "I": 0.0,
    "II": 0.0,
    "III": 0.0,
    "aVR": 2.5,
    "aVL": 2.5,
    "aVF": 2.5,
    "V1": 5.0,
    "V2": 5.0,
    "V3": 5.0,
    "V4": 7.5,
    "V5": 7.5,
    "V6": 7.5,
}

# nnUNet settings
NNUNET_RAW = f"{os.getcwd()}/data/ptb-xl"
NNUNET_PREPROCESSED = f"{os.getcwd()}/model/nnUNet_preprocessed"
NNUNET_RESULTS = f"{os.getcwd()}/model/nnUNet_results"

# TODO: Train on float rotated images
# TODO: Lead boxes: Do we need separate models for lead and lead name? Should we use one box per line?
# TODO: Grid info: Do we need a model for grid information or do we assume them to be constant? Do we need a model for pixels per grid cell ("scale info")?
# Questions:
# - [ ] Shifted start
# - [ ] 0 alignment
# - [ ] What are the units of vector data? E.g. 1 = 10mV?
# - [ ] Pixel shift


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
def train_models(data_folder, model_folder, verbose):
    
    train_nnunet = TRAIN_NNUNET
    train_classification = DO_CLASSIFICATION
    
    #############################################################################################################################
    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.
    if train_classification:
        
        if verbose:
            print()
            print('Training classification model...')

        # Get datasets and loader
        dataset = ClassificationDataset(data_folder)
        vali_size = int(len(dataset) * VALI_SIZE)
        train_size = len(dataset) - vali_size
        train_dataset, vali_dataset = torch.utils.data.random_split(dataset, [train_size, vali_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        vali_loader = DataLoader(vali_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        # Define torch model
        model_to_use = get_model_own(MODEL_NAME_CLASSIFICATION, len(dataset.classes), image_based=IMAGE_BASED_CLASSIFICATION, input_shape=dataset[0]['image'].shape[-2:])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_to_use.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        classification_model_trainer = Trainer(
            model=model_to_use,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=EPOCHS_CLASSIFICATION,
            device=DEVICE,
            model_dir=model_folder,
            use_best_model=USE_BEST_MODEL,
        )
        
        # Train the model
        os.makedirs(model_folder, exist_ok=True)
        classification_model = classification_model_trainer.fit(train_loader, vali_loader)
        
        # Save
        model_dict = {
            "classes": dataset.classes,
            "image_based": IMAGE_BASED_CLASSIFICATION,
            "input_shape": dataset[0]['image'].shape[-2:],
            "model_state_dict": classification_model.state_dict(),
            "model_name": MODEL_NAME_CLASSIFICATION,
            "image_based": IMAGE_BASED_CLASSIFICATION
        }
        torch.save(model_dict, f"{model_folder}/classification_model.pth")
        
    else:
        if verbose:
            print('No classification model trained.')


    #############################################################################################################################
    # Train the digitization model.
    if train_nnunet:
        
        if verbose:
            print('Training digitization model...')
        
        def convert_dict_to_args(args_dict):
            args_list = []
            for key, value in args_dict.items():
                if isinstance(value, bool):
                    if value:
                        args_list.append(f'--{key}')
                else:
                    args_list.append(f'--{key}')
                    args_list.append(str(value))
            return args_list
        
        # Generate images and json files
        from gen_ecg_images_from_data_batch import get_parser as get_parser_gen, run as gen_ecg_images_from_data_batch
        args_dict = {
            'input_directory': data_folder,
            'output_directory': data_folder,
            'seed': 10,
            'mask_unplotted_samples': True,
            'print_header': True,
            'store_config': 2,
            'lead_name_bbox': True,
            'lead_bbox': True,
            'random_print_header': 0.7,
            'calibration_pulse': 0.8,
            'wrinkles': True,
            'augment': True,
            'rotate': 5,
            'num_images_per_ecg': 1,
            'overwrite': True,
        }
        args = get_parser_gen().parse_args(convert_dict_to_args(args_dict))
        if verbose:
            print('--------- Generating images... ---------')
            print(args)
        gen_ecg_images_from_data_batch(args)
        
        # Prepare images
        from prepare_image_data import get_parser as get_parser_prep, run as prepare_image_data
        args_dict = {
            'input_folder': data_folder,
            'output_folder': data_folder,
        }
        if verbose:
            print('--------- Preparing images... ---------')
        prepare_image_data(get_parser_prep().parse_args(convert_dict_to_args(args_dict)))
        
        # Replot pixels
        from replot_pixels import resample_pixels_in_dir
        if verbose:
            print('--------- Resampling pixels... ---------')
        resample_pixels_in_dir(data_folder, 5)
        
        # Create train test split
        from create_train_test import get_parser as get_parser_create, run as create_train_test
        args_dict = {
            'input_data': data_folder,
            'output_folder': os.path.join(data_folder, 'Dataset500_Signals'),
            'database_file': 'ptbxl_database.csv',
            'rgba_to_rgb': True,
            'gray_to_rgb': True,
            'mask': True,
            'mask_multilabel': True,
            'rotate_image': True,
            'plotted_pixels_key': "dense_plotted_pixels",
            "no_split": True,
            "move": True,
        }
        if verbose:
            print('--------- Creating train test split... ---------')
        create_train_test(get_parser_create().parse_args(convert_dict_to_args(args_dict)))
        print()
        print()
        
        # Train model
        if verbose:
            print('--------- Training nnUNet model... ---------')
        os.environ["nnUNet_raw"] = data_folder
        os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
        os.environ["nnUNet_results"] = NNUNET_RESULTS
        command_run_preprocess = "nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity"
        command_run_train = "nnUNetv2_train 500 2d all -device cuda --c"
        subprocess.run(command_run_preprocess, shell=True)
        subprocess.run(command_run_train, shell=True)
    
    else:
        if verbose:
            print('No digitization model trained.')
        
    if verbose:
        print('Done.')
        print()


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):

    #############################################################################################################################
    # Run the digitization model.
    if verbose:
        print('Running digitization model...')
    
    # Load info and image
    header_file = get_header_file(record)
    header = load_text(header_file)
    num_samples = get_num_samples(header)
    signal_names = get_signal_names(header)
    path = os.path.split(record)[0]
    image_file = get_image_files(record)[0]
    image_file_path = os.path.join(path, image_file)
    image = read_image(image_file_path)
    image = image[:3]

    # Rotate
    rot_angle = get_rotation_angle(image.permute(1, 2, 0).numpy().astype(np.uint8))
    image_rotated = rotate(image, rot_angle)
    
    # Segment
    mask_to_use = predict_mask_nnunet(image_rotated, f"Dataset{X_FREQUENCY}_Signals")
    
    # Use mask to cut into single, binary masks
    signal_masks_cropped, signal_positions_cropped, _ = cut_binary(mask_to_use, image_rotated, signal_names)

    # Vecotrise
    ## mm per pixel:
    x_pixel_list = [v.shape[2] for v in signal_masks_cropped.values()]
    x_pixel_list_median = np.median(x_pixel_list)
    x_pixel_list_below_2x_median_mean = np.mean([v for v in x_pixel_list if v < 2 * x_pixel_list_median])
    sec_per_pixel = 2.5/x_pixel_list_below_2x_median_mean
    mm_per_pixel = 25 * sec_per_pixel
    ## sec and mV per pixel
    sec_per_pixel = get_sec_per_pixel(mm_per_pixel)
    mV_per_pixel = get_mV_per_pixel(mm_per_pixel)
    ## Vectorise
    signals_predicted = {}
    for lead, mask in signal_masks_cropped.items():
        signals_predicted[lead] = vectorise(
            image_rotated, 
            mask, 
            signal_positions_cropped[lead], 
            sec_per_pixel, 
            mV_per_pixel, 
            Y_VALUES_PER_LEAD["full"], 
            Y_VALUES_PER_LEAD[lead], 
            num_samples,
            interpolate_nans=False
        )
    
    # Prep saving # TODO: Check if this is needed if we improve cut_binary
    signal_list = []
    for signal_name in signal_names:
        if signal_name not in signals_predicted:
            signal = np.empty(num_samples)
            signal[:] = np.nan
        else:
            signal = signals_predicted[signal_name].numpy()
        if len(signal) < num_samples:
            nan_signal = np.empty(num_samples)
            nan_signal[:] = np.nan
            signal_start = SIGNAL_START[signal_name] * num_samples/10
            nan_signal[int(signal_start):int(signal_start + len(signal))] = signal
            signal_list.append(nan_signal)
        else:
            signal_list.append(signal)
    signal = np.array(signal_list).T
    
    
    #############################################################################################################################
    # Run the classification model.
    if DO_CLASSIFICATION:
        if verbose:
            print()
            print('Running classification model...')
        labels = run_classification_model(signal, classification_model["model"], classification_model["classes"])
    else:
        labels = None
        if verbose:
            print()
            print('No classification model run.')
            print()

    return signal, labels


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

################################################################################
# Classification model
################################################################################

def get_signals(records: List[str], data_folder: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    classification_signals = list()
    classification_labels = list()
    sampling_frequencies = list()
    for record_name in records:
        record_path = os.path.join(data_folder, record_name)
        signals, fields = load_signals(record_path)
        header = load_header(record_path)
        frequency = get_sampling_frequency(header)
        labels = load_labels(record_path)
        if any(label for label in labels):
            classification_signals.append(signals)
            classification_labels.append(labels)
            sampling_frequencies.append(frequency)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = compute_one_hot_encoding(classification_labels, classes)
    classification_signals = np.array(classification_signals)
    
    assert classification_labels.shape[1] == len(classes), f"Number of classes {len(classes)} does not match the number of labels {labels.shape[1]}"
    assert classification_labels.shape[0] == classification_signals.shape[0], f"Number of signals {signals.shape[0]} does not match the number of labels {labels.shape[0]}"
    
    return classification_signals, classification_labels, classes, sampling_frequencies


def standardise_signal(signal, median_freq, seconds=10, num_channels=1, use_longer = False):
    # Filter for short signals
    num_nans = np.isnan(signal).sum(axis=0) # Number of nans per signal
    num_nans_median = np.median(num_nans) # Median number of nans divided by 2
    if use_longer:
        signal_filtered = signal[:, num_nans <= (num_nans_median/2)]
    else:
        signal_filtered = signal[:, num_nans >= (num_nans_median/2)] # Only use signals that have more nans than num_nans_median_2

    # If more than num_channels signals left, use the first num_channels
    if signal_filtered.shape[1] > num_channels:
        signal_filtered = signal_filtered[:, :num_channels]
        
    # If less than num_channels signals left, repeat
    if signal_filtered.shape[1] <= num_channels:
        num_to_repeat = math.ceil(num_channels / signal_filtered.shape[1])
        signal_filtered = np.tile(signal_filtered, num_to_repeat)[:, :num_channels]

    # Filter all signals to non nans and cut them to required_samples or pad them
    required_samples = int(seconds * median_freq)
    signal_filtered_non_nan = []
    for i in range(signal_filtered.shape[1]):
        signal_i = signal_filtered[:, i]
        signal_i_non_nan = signal_i[~np.isnan(signal_i)]
        if len(signal_i_non_nan) < required_samples:
            signal_i_non_nan = np.pad(signal_i_non_nan, (0, required_samples - len(signal_i_non_nan)))
        else:
            signal_i_non_nan = signal_i_non_nan[:required_samples]
        signal_filtered_non_nan.append(signal_i_non_nan)
    signal_filtered_non_nan = np.array(signal_filtered_non_nan)
    
    return signal_filtered_non_nan


# Convert to spectrograms using librosa
def convert_to_spectrogram(signal, median_freq, n_fft=1024, hop_length=512, n_mels=64):
    """Convert a signal to a spectrogram using librosa.
    
    Args:
        signal (np.array): The signal to convert.
        median_freq (int): The median frequency of the signal.
        n_fft (int): The number of samples to use for each fourier transform. Larger n_fft values result in higher frequency resolution but lower time resolution.
        hop_length (int): The number of samples between each frame. Smaller hop_length values increase the overlap, resulting in more windows and better time resolution. However, this also increases computational cost.
        n_mels (int): The number of Mel bands to use. More Mel bands provide finer frequency resolution in the Mel spectrogram but increase computational cost.
        
    Returns:
        np.array: The spectrogram.
    """
    # Get spectrograms
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=median_freq, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


# Use the classification model to predict the labels
def run_classification_model(signal: np.ndarray, model, classes: List[str], frequency: int = 500) -> List[str]:
    """Run the classification model on the signal.
    
    Args:
        signal: The signal to classify of shape (num_samples, num_leads). Only for the full signal 5000 samples will be available.
        model: The classification model.
        classes (List[str]): The classes to predict.
    """
        
    # Get signals
    if USE_SPECTROGRAMS:
        transform = None
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        signal_standardized = standardise_signal(signal, frequency, 10, 1, True)
        spectrogram = convert_to_spectrogram(signal_standardized, frequency, n_fft=32, hop_length=16, n_mels=32)
        spectrogram = torch.tensor(spectrogram)
        if NC_CLASSIFICATION == 3:
            if spectrogram.shape[0] == 1:
                spectrogram = torch.cat([spectrogram, spectrogram, spectrogram], dim=0)
        if NC_CLASSIFICATION == 1:
            if spectrogram.shape[0] == 3:
                spectrogram = spectrogram[0, :, :]
        if transform:
            signals = transform(spectrogram)
    else:
        signal_standardized = standardise_signal(signal, frequency, 2.5, 11, False)
        signals = torch.tensor(signal_standardized)
        signals = nn.functional.normalize(signals)
    
    signals = signals.to(torch.float32).to(DEVICE)
    
    # Predict labels
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        predicted = model(signals)
        prob_predicted = torch.sigmoid(predicted)
    
    # Convert the labels
    labels = [classes[i] for i in range(len(classes)) if prob_predicted[0, i] > CLASSIFICATION_THRESHOLD]
    
    return labels


def calculate_f1_score(y_pred, y_true, average="macro", threshold=CLASSIFICATION_THRESHOLD):
    y_true = y_true.detach().cpu().numpy().astype(int)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = (y_pred > threshold).astype(int)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=np.nan)
    return f1
    
    
class ClassificationDataset(Dataset):
    def __init__(self, data_folder, records_to_use=None):
        self.data_folder = data_folder
        self.records_to_use = records_to_use
        if self.records_to_use:
            self.records = self.records_to_use
        else:
            self.records = find_records(data_folder)
        self.signals, self.labels, self.classes, sampling_frequencies = get_signals(self.records, self.data_folder)
        self.median_freq = np.median(sampling_frequencies)
        self.n_fft = 32
        self.hop_length = 16
        self.n_mels = 32
        self.spectrograms = USE_SPECTROGRAMS
        self.nc = NC_CLASSIFICATION
        self.target_size = [256, 256]
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        
        # Get labels
        label = self.labels[idx]
        
        # Prepare signals
        signal = self.signals[idx]
        if self.spectrograms:
            signal_standardized = standardise_signal(signal, self.median_freq, 10, 1, True)
            spectrogram = convert_to_spectrogram(signal_standardized, self.median_freq, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            spectrogram = torch.tensor(spectrogram)
            if self.nc == 3:
                if spectrogram.shape[0] == 1:
                    spectrogram = torch.cat([spectrogram, spectrogram, spectrogram], dim=0)
            if self.nc == 1:
                if spectrogram.shape[0] == 3:
                    spectrogram = spectrogram[0, :, :]
            if self.transform:
                signals = self.transform(spectrogram)
        else:
            signal_standardized = standardise_signal(signal, self.median_freq, 2.5, 11, False)
            signals = torch.tensor(signal_standardized)
            signals = nn.functional.normalize(signals)
        
        # Convert
        signals = signals.to(torch.float32)
        labels = torch.tensor(label).to(torch.float32)
        return {"image": signals, "label": labels}


def get_model_own(model_name, num_classes, image_based=True, input_shape=None):
    if image_based:
        model = eval(f"models.{model_name}(pretrained=True)")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = ResNetMultiLabel(model)
    else:
        model = SimpleNN(input_shape, num_classes)
    return model


class ResNetMultiLabel(nn.Module):
    def __init__(self, base_model):
        super(ResNetMultiLabel, self).__init__()
        self.base_model = base_model
        
    def forward(self, x):
        x = self.base_model(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleNN, self).__init__()
        
        # Calculate the number of input features
        self.input_features = input_shape[0] * input_shape[1]
        
        # Define the layers of the network
        self.fc1 = nn.Linear(self.input_features, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer
        self.fc3 = nn.Linear(256, 256)  # Third fully connected layer
        self.fclast = nn.Linear(256, num_classes)  # Output layer
        
    def forward(self, x):
        x = x.view(-1, self.input_features)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fclast(x)  # Output layer
        return x


################################################################################
# Save and load models
################################################################################

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):

    # Load the digitization model
    digitization_model = None
    
    # Get classification model
    if DO_CLASSIFICATION:
        classification_dict = torch.load(os.path.join(model_folder, "classification_model.pth"))
        classes = classification_dict["classes"]
        input_shape = classification_dict["input_shape"]
        model_state_dict = classification_dict["model_state_dict"]
        model_name = classification_dict["model_name"]
        image_based = classification_dict["image_based"]
        model = get_model_own(model_name, len(classes), image_based=image_based, input_shape=input_shape)
        model.load_state_dict(model_state_dict)
        classification_model = {"model": model, "classes": classes}
    else:
        classification_model = None
    
    return digitization_model, classification_model


# Save your trained digitization model.
def save_torch_model(model_folder, model, model_name):
    filename_model = os.path.join(model_folder, f"{model_name}.pth")
    torch.save(model.state_dict(), filename_model)


################################################################################
# Data loaders
################################################################################

# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])


# Interpolate the signal with pandas.
def interpolate_with_pandas(tensor, method="linear"):
    series = pd.Series(tensor.numpy())
    interpolated_series = series.interpolate(method=method)
    interpolated_tensor = torch.from_numpy(interpolated_series.to_numpy())
    return interpolated_tensor


# Cut the image to positive mask area
def cut_to_mask(img, mask, return_y1=False):
    coords = torch.where(mask[0] >= 1)
    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()
    img = img[:, y_min : y_max + 1, x_min : x_max + 1]
    if return_y1:
        return img, y_min
    else:
        return img


# Load the image(s) for a record.
def load_mask(record, extension, label):
    path = os.path.split(record)[0]
    image_files = get_image_files(record)

    images = list()
    for image_file in image_files:
        image_file_path = os.path.join(path, image_file)
        if os.path.isfile(image_file_path):
            image_file_path = image_file_path.replace(extension, f"{label}{extension}")
            if os.path.isfile(image_file_path):
                image = Image.open(image_file_path)
                images.append(image)
            else:
                images.append(None)

    return images


# Load the json file(s) for a record.
def load_json(record):
    path = os.path.split(record)[0]
    image_files = get_image_files(record)
    json_files = [f.replace(".png", ".json") for f in image_files]

    json_dicts = list()
    for json_file in json_files:
        json_file_path = os.path.join(path, json_file)
        if os.path.isfile(json_file_path):
            with open(json_file_path) as f:
                json_dict = json.load(f)
                json_dicts.append(json_dict)

    return json_dicts, json_files


def convert_json_to_bbox(bbox, img_height):
    left_upper = (
        bbox["0"][1] + PIXELS_TO_SHIFT[0],
        bbox["0"][0] + PIXELS_TO_SHIFT[1],
    )
    left_bottom = (
        bbox["3"][1] + PIXELS_TO_SHIFT[0],
        bbox["3"][0] + PIXELS_TO_SHIFT[1],
    )
    right_upper = (
        bbox["1"][1] + PIXELS_TO_SHIFT[0],
        bbox["1"][0] + PIXELS_TO_SHIFT[1],
    )
    right_bottom = (
        bbox["2"][1] + PIXELS_TO_SHIFT[0],
        bbox["2"][0] + PIXELS_TO_SHIFT[1],
    )

    x1 = left_upper[0]
    y1 = left_upper[1]
    x2 = right_bottom[0]
    y2 = right_bottom[1]

    return {
        "left_upper": left_upper,
        "left_bottom": left_bottom,
        "right_upper": right_upper,
        "right_bottom": right_bottom,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def convert_bbox_to_json(bbox, img_height):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_json = {}
    bbox_json["0"] = [y1 - PIXELS_TO_SHIFT[1], x1 - PIXELS_TO_SHIFT[0]]
    bbox_json["1"] = [y1 - PIXELS_TO_SHIFT[1], x2 - PIXELS_TO_SHIFT[0]]
    bbox_json["2"] = [y2 - PIXELS_TO_SHIFT[1], x2 - PIXELS_TO_SHIFT[0]]
    bbox_json["3"] = [y2 - PIXELS_TO_SHIFT[1], x1 - PIXELS_TO_SHIFT[0]]
    return bbox_json


def cut_image_to_bbox(image, bbox, return_y1=False):
    img_height = image.shape[1]
    bbox_converted = convert_json_to_bbox(bbox, img_height)
    x1 = bbox_converted["x1"]
    y2 = bbox_converted["y2"]
    x2 = bbox_converted["x2"]
    y1 = bbox_converted["y1"]
    image_cropped = image[:, y1:y2, x1:x2]

    if return_y1:
        return image_cropped, y1
    else:
        return image_cropped


def select_signal(signals, fields, lead_name):
    input_channels = fields["sig_name"]
    output_channels = [lead_name]
    filtered_signal = reorder_signal(signals, input_channels, output_channels)
    return filtered_signal


def get_bbox_inputs(images, model=None, device=DEVICE):
    """The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    """

    images = list(image.to(device) for image in images)
    return images


def select_highest_scored_box(output):
    # Get the labels needed
    labels_to_check = [
        list(v.values())
        for k, v in BOX_TYPE_LABEL_MAPPING.items()
        if k in BOX_TYPES_TO_USE
    ]
    labels_to_check = [item for sublist in labels_to_check for item in sublist]

    # Reorder and select highest scored one
    return_dict = {}
    for label in labels_to_check:
        if label in output["labels"]:
            bboxes = output["boxes"][output["labels"] == label]
            scores = output["scores"][output["labels"] == label]
            bbox_with_highest_score = bboxes[scores.argmax()]
            return_dict[label] = list(np.array(bbox_with_highest_score))
        else:
            return_dict[label] = None
    return return_dict


def convert_box_to_integer(bbox):
    for k, v in bbox.items():
        if k in range(1, 13):
            bbox[k] = [int(round(x, 0)) for x in v]
    return bbox


def bbox_prediction_to_json(bbox_predicted):
    json_per_image = {}
    img_height = bbox_predicted["image_height"]
    label_keys = [k for k in bbox_predicted.keys() if k != "image_height"]
    for label in label_keys:
        bbox = bbox_predicted[label]
        box_type_of_label = [
            k for k, v in BOX_TYPE_LABEL_MAPPING.items() if label in v.values()
        ][0]
        label_name = [
            k
            for k, v in BOX_TYPE_LABEL_MAPPING[box_type_of_label].items()
            if v == label
        ][0]
        box_dict = {"lead_name": label_name}
        if bbox is not None:
            bbox_converted = convert_bbox_to_json(bbox, img_height)
            box_dict.update(bbox_converted)
        else:
            box_dict.update(
                {
                    "0": [np.nan, np.nan],
                    "1": [np.nan, np.nan],
                    "2": [np.nan, np.nan],
                    "3": [np.nan, np.nan],
                }
            )
        box_type_key = box_type_of_label + "_predicted"
        if box_type_key in json_per_image:
            json_per_image[box_type_key].append(box_dict)
        else:
            json_per_image[box_type_key] = [box_dict]

    return json_per_image


def filter_for_full_lead(
    boxes_of_type, full_mode_lead, box_type=None, image_path=None, i=None, verbose=0
):
    # Prepare
    boxes_of_type_copy = boxes_of_type.copy()
    if i is not None:
        lead_names_in_dict = [box["lead_name"][i] for box in boxes_of_type]
    else:
        lead_names_in_dict = [box["lead_name"] for box in boxes_of_type]
    double_lead_names = [
        lead_name
        for lead_name in lead_names_in_dict
        if lead_names_in_dict.count(lead_name) > 1
    ]

    # Check for correct format
    assert all(
        [d == full_mode_lead for d in list(set(double_lead_names))]
    ), f"There is some error in the {box_type} dict. Only full_mode_lead {full_mode_lead} can appear twice, but {double_lead_names} appear multiple times."
    assert (
        len(double_lead_names) <= 2
    ), "There are more than two full_mode_leads in the dict."

    # Get the longest box
    if len(double_lead_names) < 2:
        if verbose > 0:
            warnings.warn(
                f"There are no two full_mode_leads in the dict {box_type}. Is it possible that it was already filtered? Image {image_path}: lead_names_in_dict: {lead_names_in_dict}, double_lead_names: {double_lead_names}"
            )
    else:
        if i is not None:
            lead_boxes_x = [
                (b["0"][0][i], b["2"][0][i])
                for b in boxes_of_type
                if b["lead_name"][i] == full_mode_lead
            ]
            lead_boxes_length = [b[1] - b[0] for b in lead_boxes_x]
        else:
            lead_boxes_x = [
                (b["0"][0], b["2"][0])
                for b in boxes_of_type
                if b["lead_name"] == full_mode_lead
            ]
            lead_boxes_length = [b[1] - b[0] for b in lead_boxes_x]
        idx_longest_box = lead_boxes_length.index(max(lead_boxes_length))
        assert (
            idx_longest_box == 1
        ), f"The full_mode_lead is not the second box in the list. The list is {lead_boxes_length} and the index of the longest box is {idx_longest_box}."
        for r, box in enumerate(boxes_of_type):
            if i is not None:
                if box["lead_name"][i] == full_mode_lead:
                    break
            else:
                if box["lead_name"] == full_mode_lead:
                    break
        del boxes_of_type_copy[r]

    return boxes_of_type_copy


def get_bbox_type_targets(batch_dict, model=None, device=DEVICE, verbose=True):
    """During training, the model expects both the input tensors and targets (list of dictionary), containing:
    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
      ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
    - labels (Int64Tensor[N]): the class label for each ground-truth box
    """

    # TODO: Rn we are overwriting the short signal with the full signal. Check for solution where both are predicted individually.

    # Checks
    assert all(
        [box in batch_dict["info_dict"] for box in BOX_TYPES_TO_USE]
    ), f"Missing bounding box in info_dict: {BOX_TYPES_TO_USE}"
    if model is None:
        if verbose:
            print(
                "get_bbox_type_targets(): No model provided. Not checking the number of classes."
            )
    else:
        different_possible_labels = []
        for box_type in BOX_TYPES_TO_USE:
            different_possible_labels.append(BOX_TYPE_LABEL_MAPPING[box_type].values())
        num_of_different_labels = len(
            set([item for sublist in different_possible_labels for item in sublist])
        )
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_classes = model.module.roi_heads.box_predictor.cls_score.out_features
        else:
            num_classes = model.roi_heads.box_predictor.cls_score.out_features
        assert (
            num_of_different_labels == num_classes - 1
        ), f"Number of classes in the model ({num_classes}) (minus background) does not match the number of bounding box labels ({num_of_different_labels})"

    # Prepare targets
    targets = []
    assume_list = len(batch_dict["image"].shape) == 4
    num_of_images = len(batch_dict["image"]) if assume_list else 1
    for i in range(num_of_images):
        if assume_list:
            img_height = batch_dict["image"][i].shape[1]
            full_mode_lead = batch_dict["info_dict"]["full_mode_lead"][i]
            image_path = batch_dict["info_dict"]["image_path"][i]
        else:
            img_height = batch_dict["image"].shape[1]
            full_mode_lead = batch_dict["info_dict"]["full_mode_lead"]
            image_path = batch_dict["info_dict"]["image_path"]
        boxes = []
        labels = []
        for box_type in BOX_TYPES_TO_USE:
            boxes_of_type = batch_dict["info_dict"][box_type]
            if assume_list:
                i_to_pass = i
            else:
                i_to_pass = None
            boxes_of_type_filtered = filter_for_full_lead(
                boxes_of_type,
                full_mode_lead,
                box_type,
                image_path,
                i_to_pass,
            )

            # Convert to bbox
            for box in boxes_of_type_filtered:
                if assume_list:
                    bbox = {
                        "0": [box["0"][0][i], box["0"][1][i]],
                        "1": [box["1"][0][i], box["1"][1][i]],
                        "2": [box["2"][0][i], box["2"][1][i]],
                        "3": [box["3"][0][i], box["3"][1][i]],
                    }
                else:
                    bbox = {
                        "0": [box["0"][0], box["0"][1]],
                        "1": [box["1"][0], box["1"][1]],
                        "2": [box["2"][0], box["2"][1]],
                        "3": [box["3"][0], box["3"][1]],
                    }
                bbox_converted = convert_json_to_bbox(bbox, img_height)
                x1 = bbox_converted["x1"]
                y2 = bbox_converted["y2"]
                x2 = bbox_converted["x2"]
                y1 = bbox_converted["y1"]
                if assume_list:
                    label_name = box["lead_name"][i]
                else:
                    label_name = box["lead_name"]
                label_of_box = BOX_TYPE_LABEL_MAPPING[box_type][label_name]
                boxes.append([x1, y1, x2, y2])
                labels.append(label_of_box)
        image_dict = {
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(labels),
        }
        targets.append(image_dict)

    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return targets


def dataloader_wrapper(
    list_of_paths,
    test_settings,
    shuffle_settings,
    transform,
    single_signals=False,
    run_in_parallel=False,
    rank=None,
    world_size=WORLD_SIZE,
    num_workers=NUM_WORKERS,
    load_argumented=True,
):
    records_list = []
    data_list = []
    loader_list = []

    for path, test_setting, shuffle_setting in zip(
        list_of_paths, test_settings, shuffle_settings
    ):
        print(f"Loading data from {path}")
        records = find_records(path)
        data = ECGSignalDataset(
            data_folder=path,
            records=records,
            test=test_setting,
            transform=transform,
            rotate_back=False,
            rotate=True,
            nc=NC,
            single_signals=single_signals,
            load_argumented=load_argumented,
        )

        if run_in_parallel:
            assert rank is not None, "Rank must be provided for parallel processing."
            sampler = DistributedSampler(
                data,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle_setting,
                drop_last=False,
            )
            num_workers = 0
        else:
            sampler = None
            num_workers = num_workers

        dataloader = DataLoader(
            data,
            batch_size=BATCH_SIZE,
            shuffle=shuffle_setting if sampler is None else False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            sampler=sampler,
        )

        records_list.append(records)
        data_list.append(data)
        loader_list.append(dataloader)
        if rank is None or rank == 0:
            print(
                f"Loaded {len(records)} records with {len(data)} images and {len(dataloader)} batches from {path}"
            )

    return records_list, data_list, loader_list


# Load the image(s) for a record.
class ECGSignalDataset(VisionDataset):
    def __init__(
        self,
        data_folder,
        records,
        test=False,
        transform=None,
        rotate_back=False,
        rotate=False,
        rotation_model=None,
        nc=3,
        single_signals=False,
        load_argumented=True,
    ):
        self.data_folder = data_folder
        self.records = records
        self.records_with_path = [
            os.path.join(data_folder, record) for record in records
        ]
        if isinstance(self.records, str):
            self.records = [self.records]
        self.test = test
        self.transform = transform
        self.rotate = rotate
        self.rotate_back = rotate_back
        self.rotation_model = rotation_model
        self.nc = nc
        self.single_signals = single_signals
        self.load_argumented = load_argumented

        # Get the list of image files, json files, mask files, and augmented mask files and
        self.images = [
            [
                os.path.join(os.path.split(record)[0], image_file)
                for image_file in get_image_files(record)
            ]
            for record in self.records_with_path
        ]
        self.signal_paths = [
            [record for image_file in get_image_files(record)]
            for record in self.records_with_path
        ]
        self.signal_paths = [item for sublist in self.signal_paths for item in sublist]
        self.signal_paths = [
            os.path.join(self.data_folder, record) for record in self.signal_paths
        ]
        self.images = [item for sublist in self.images for item in sublist]
        self.json_files = [f.replace(".png", ".json") for f in self.images]
        self.masks = [
            os.path.join(
                os.path.split(f)[0].replace("/imagesT", "/labelsT"), os.path.split(f)[1]
            ).replace("_0000.png", ".png").replace('_original', '')
            for f in self.images
        ]

        # Cut the images to signals
        if self.single_signals:
            self.lead_names_single = []
            self.lead_bbox_single = []
            self.images_single = []
            self.signal_paths_single = []
            self.json_files_single = []
            self.masks_single = []
            for image_path, signal_path, json_file, mask in zip(
                self.images, self.signal_paths, self.json_files, self.masks
            ):
                json_path = image_path.replace(".png", ".json")
                with open(json_path) as f:
                    try: 
                        json_dict = json.load(f)
                    except Exception as e:
                        print(f"Error in {json_path}")
                        raise e
                for bbox in [{**lead["lead_bounding_box"], "lead_name": lead["lead_name"]} for lead in json_dict["leads"]]:
                    self.images_single.append(image_path)
                    self.signal_paths_single.append(signal_path)
                    self.json_files_single.append(json_file)
                    self.masks_single.append(mask)
                    self.lead_names_single.append(bbox["lead_name"])
                    self.lead_bbox_single.append(bbox)
            self.images = self.images_single
            self.signal_paths = self.signal_paths_single
            self.json_files = self.json_files_single
            self.masks = self.masks_single
            self.lead_names = self.lead_names_single
            self.lead_bbox = self.lead_bbox_single

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load info
        info_dict = {}
        if not self.test:
            with open(self.json_files[idx]) as f:
                json_dict = json.load(f)

        # Load image
        info_dict["image_path"] = self.images[idx]
        image = read_image(self.images[idx])
        if self.nc != image.shape[0]:
            if self.nc == 3 and image.shape[0] == 4:
                image = image[:3]
            elif self.nc == 1 and image.shape[0] == 3:
                image = rgb_to_grayscale(image)
            else:
                raise ValueError(
                    f"For nc = {self.nc}, the number of channels {image.shape[0]} is not supported."
                )
                
        if self.rotate_back:
            image = rotate(image, -json_dict["rotate"])

        # Rotate image
        rot_angle = get_rotation_angle(image.permute(1, 2, 0).numpy().astype(np.uint8))
        info_dict["rot_angle_predicted"] = rot_angle
        if self.rotate:
            if not self.test:
                try:
                    if json_dict["augment"]:
                        image_rotated = rotate(image, json_dict["rotate"])
                    else:
                        image_rotated = image
                except Exception as e:
                    print(self.images[idx])
                    print(json_dict)
                    raise e
            else:
                image_rotated = rotate(image, rot_angle)
        else:
            image_rotated = image

        # Predict pixels per grid cell
        pixels_per_grid, sec_per_pixel, mV_per_pixel = get_grid_info(image_rotated)
        info_dict["pixels_per_grid_predicted"] = pixels_per_grid
        info_dict["sec_per_pixel_predicted"] = sec_per_pixel
        info_dict["mV_per_pixel_predicted"] = mV_per_pixel

        # Prep image
        if self.transform:
            image_rotated = self.transform(image_rotated)

        # Load all other information
        if not self.test:
            # Only have full lead
            full_lead_length = max(
                [
                    lead["end_sample"] - lead["start_sample"]
                    for lead in json_dict["leads"]
                    if lead["lead_name"] == json_dict["full_mode_lead"]
                ]
            )
            json_dict["leads"] = [
                lead
                for lead in json_dict["leads"]
                if lead["lead_name"] != json_dict["full_mode_lead"]
                or lead["end_sample"] - lead["start_sample"] == full_lead_length
            ]
            
            # Prepare info dict
            info_dict["signal_path"] = self.signal_paths[idx]
            info_dict["full_mode_lead"] = json_dict["full_mode_lead"]
            info_dict["text_bounding_box"] = [{**lead["text_bounding_box"], "lead_name": lead["lead_name"]} for lead in json_dict["leads"]]
            info_dict["lead_bounding_box"] = [{**lead["lead_bounding_box"], "lead_name": lead["lead_name"]} for lead in json_dict["leads"]]
            info_dict["lead_name"] = "all"
            info_dict["x_resolution"] = json_dict["x_resolution"]
            info_dict["y_resolution"] = json_dict["y_resolution"]
            info_dict["augment"] = json_dict["augment"]
            info_dict["x_grid"] = json_dict["x_grid"]
            info_dict["y_grid"] = json_dict["y_grid"]
            mm_per_pixel_x = get_mm_per_pixel(info_dict["x_grid"])
            mm_per_pixel_y = get_mm_per_pixel(info_dict["y_grid"])
            sec_per_pixel = get_sec_per_pixel(mm_per_pixel_x)
            mV_per_pixel = get_mV_per_pixel(mm_per_pixel_y)
            info_dict["sec_per_pixel"] = sec_per_pixel
            info_dict["mV_per_pixel"] = mV_per_pixel
            info_dict["rotation"] = json_dict["rotate"]

            # Load mask
            mask = read_image(self.masks[idx])
            info_dict["mask_path"] = self.masks[idx]

            # Load augmented mask
            if json_dict["augment"] and self.load_argumented:
                json_dict["leads_augmented"] = [
                    lead
                    for lead in json_dict["leads_augmented"]
                    if lead["lead_name"] != json_dict["full_mode_lead"]
                    or lead["end_sample"] - lead["start_sample"] == full_lead_length
                ]
                mask_augmented = read_image(
                    self.masks[idx].replace(".png", "_augmented.png")
                )
                info_dict["lead_bounding_box_augmented"] = [{**lead["lead_bounding_box"], "lead_name": lead["lead_name"]} for lead in json_dict["leads_augmented"]]
                info_dict["text_bounding_box_augmented"] = [{**lead["text_bounding_box"], "lead_name": lead["lead_name"]} for lead in json_dict["leads_augmented"]]
            else:
                mask_augmented = mask
                info_dict["lead_bounding_box_augmented"] = info_dict["lead_bounding_box"]
                info_dict["text_bounding_box_augmented"] = info_dict["text_bounding_box"]
            signals, fields = load_signals(self.signal_paths[idx])

        else:
            print(
                "Running in test mode. Not looking for records. Returning empty values for info_dict, mask, mask_augmented, and signals."
            )
            signals = np.nan
            mask = np.nan
            mask_augmented = np.nan
            info_dict = {}

        if self.single_signals:
            # Select signal
            info_dict["lead_name"] = self.lead_names[idx]
            bbox = self.lead_bbox[idx]
            image_rotated = cut_image_to_bbox(image_rotated, bbox)
            mask = cut_image_to_bbox(mask, bbox)
            signals = select_signal(signals, fields, self.lead_names[idx])
            # Resize
            original_size_image = image_rotated.shape
            original_size_mask = mask.shape
            try:
                image_rotated = resize(
                    image_rotated, IMG_SIZE
                )  # TODO: Check if this is necessary and what to do with full size signals
            except Exception as e:
                print(self.images[idx])
                print(self.lead_names[idx])
                raise e
            mask = resize(mask, IMG_SIZE)
            info_dict["original_size_image"] = original_size_image
            info_dict["original_size_mask"] = original_size_mask
            info_dict["image_size"] = IMG_SIZE

        return {
            "image": image_rotated,
            "image_original": image,
            "signals": signals.astype(np.float32),
            "info_dict": info_dict,
            "mask": mask,
            "mask_augmented": mask_augmented,
        }


################################################################################
# Models and training
################################################################################

def cut_binary(mask_to_use, image_rotated, signal_names): # TODO: Make sure we have enough signals
    signal_masks = {}
    signal_images = {}
    signal_positions = {}
    mask_values = list(pd.Series(mask_to_use.numpy().flatten()).value_counts().index)
    possible_lead_names = BOX_TYPE_LABEL_MAPPING["lead_bounding_box"]
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
        

# TODO: How can peaks better be handled?
def vectorise(image_rotated, mask, signal_cropped, sec_per_pixel, mV_per_pixel, y_values_per_lead_full, y_values_per_lead_lead, num_samples, interpolate_nans=False):
    
    # Get scaling info
    total_seconds_from_mask = round(
        torch.tensor(sec_per_pixel).item() * mask.shape[2], 1
    )
    if total_seconds_from_mask > 5: # TODO: Check if this is a valid assumption to make
        total_seconds = 10
    else:
        total_seconds = 2.5
    values_needed = int(num_samples/(10/total_seconds))

    # Get mask values
    non_zero_mean = torch.tensor(
        [
            torch.mean(torch.nonzero(mask[0, :, i]).type(torch.float32))
            for i in range(mask.shape[2])
        ]
    )

    # y-scale by shifting
    if total_seconds == 10:
        zero_pixel_total_image = (1 - y_values_per_lead_full) * image_rotated.shape[1]
    else:
        zero_pixel_total_image = (1 - y_values_per_lead_lead) * image_rotated.shape[1]
    zero_pixel = zero_pixel_total_image - signal_cropped
    predicted_signal = (zero_pixel - non_zero_mean) * mV_per_pixel

    # x-Scale by interpolation
    n = predicted_signal.shape[0]
    data_reshaped = predicted_signal.view(1, 1, n)
    resampled_data = F.interpolate(
        data_reshaped, size=values_needed, mode="linear", align_corners=False
    )
    predicted_signal_sampled = resampled_data.view(-1)
    if interpolate_nans:  # TODO: Should this be done or should nans be kept?
        predicted_signal_filled = interpolate_with_pandas(predicted_signal_sampled)
    else:
        predicted_signal_filled = predicted_signal_sampled

    return predicted_signal_filled


def predict_mask_nnunet(image, dataset_name):
    # Define paths
    temp_folder_input = "data/temp_nnUNet_input"
    temp_folder_output = "data/temp_nnUNet_output"
    temp_folder_output_pp = "data/temp_nnUNet_output_pp"
    image_path_temp = os.path.join(temp_folder_input, "00000_temp_0000.png")
    mask_path_temp = os.path.join(temp_folder_output, "00000_temp.png")
    #mask_path_temp = os.path.join(temp_folder_output_pp, "00000_temp.png")

    # Define run commands
    command_run = f"nnUNetv2_predict -d {dataset_name} -i {temp_folder_input} -o {temp_folder_output} -f  all -tr nnUNetTrainer -c 2d -p nnUNetPlans"
    command_post_process = f"nnUNetv2_apply_postprocessing -i {temp_folder_output} -o {temp_folder_output_pp} -pp_pkl_file model/nnUNet_results/{dataset_name}/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/postprocessing.pkl -np 8 -plans_json model/nnUNet_results/{dataset_name}/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/plans.json"

    # Create temp folders:
    os.makedirs(temp_folder_input, exist_ok=True)
    os.makedirs(temp_folder_output, exist_ok=True)
    os.makedirs(temp_folder_output_pp, exist_ok=True)

    # Set env variabels (nnUNet needs them to be set)
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS

    # Save image
    write_png(image, image_path_temp)

    # Run inference
    subprocess.run(command_run, shell=True)

    # Run postprocessing
    #subprocess.run(command_post_process, shell=True)

    # Load mask
    mask = read_image(mask_path_temp)

    # Delete all temporary folders and files
    shutil.rmtree(temp_folder_input)
    shutil.rmtree(temp_folder_output)
    shutil.rmtree(temp_folder_output_pp)

    return mask


def get_bbox_model(weights=None, box_score_thresh=0.05, num_classes=12 + 1):
    # Get model
    model_name = "fasterrcnn_resnet50_fpn_v2"
    model = get_model(model_name, weights=weights, box_score_thresh=box_score_thresh)
    if weights:
        preprocess = get_weight(weights).transforms()
    else:
        preprocess = get_weight(
            "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
        ).transforms()

    # Adjust model
    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )  # Get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )  # Replace the pre-trained head with a new one

    return model, preprocess


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    num_epochs: int
    device: torch.device
    criterion: Optional[torch.nn.Module] = None
    metrics: List[Callable] = field(default_factory=list)
    use_default_metric: bool = True
    verbose: bool = True
    use_best_model: bool = True
    metric_index: int = 0
    epochs_to_save: int = 5
    model_dir: str = "temp_models"
    continue_training: bool = False
    input_transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    run_in_parallel: bool = False
    print_freq: int = 10
    world_size: int = WORLD_SIZE
    rank: int = None

    def __post_init__(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.run_in_parallel:
            assert (
                self.rank is not None
            ), "Rank must be provided when running in parallel."
            if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                print(f"Training the model in parallel on {self.world_size} GPUs.")
            self.model = self.model.to(self.rank)
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
            )
        else:
            if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                print(f"Training the model on one device {self.device}.")
            self.model = self.model.to(self.device)
        if self.criterion is None:
            if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                print(
                    "No criterion provided. The last model epoch will be used since validation is not possible."
                )
            self.use_best_model = False
        else:
            if self.metrics == []:
                if self.use_default_metric:
                    print("Using default metric: F1 score")
                    self.metrics = [calculate_f1_score]
                else:
                    if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                        print(
                            "No metrics provided. The last model epoch will be used since comparison is not possible."
                        )
                    self.use_best_model = False

    def fit(self, training_dataloader: DataLoader, vali_dataloader: DataLoader):
        # Prep training
        start_time = time.time()
        self.dataloarders = {"train": training_dataloader, "val": vali_dataloader}
        self.dataset_sizes = {
            "train": len(training_dataloader.dataset),
            "val": len(vali_dataloader.dataset),
        }
        last_checkpoint_path = os.path.join(self.model_dir, "last_checkpoint.pth")
        best_checkpoint_path = os.path.join(self.model_dir, "best_checkpoin.pth")
        log_path = os.path.join(self.model_dir, "log.txt")
        if self.continue_training:
            checkpoint_dict = torch.load(last_checkpoint_path)
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
            self.best_value = checkpoint_dict["best_value"]
            self.best_epoch = checkpoint_dict["best_epoch"]
            start_epoch = checkpoint_dict["epoch"] + 1
            if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                print(f"Continuing training from epoch {start_epoch}")
            with open(log_path, "r") as f:
                log_str = f.readlines()
        else:
            log_str = []
            self.best_value = -99999.9
            self.best_epoch = 0
            start_epoch = 0
            
        self.model = self.model.to(self.device)

        if self.verbose and (self.rank == 0 or not self.run_in_parallel):
            if self.criterion is None:
                print(
                    f"No criterion provided. The model will be trained using its own loss function. Validation will not be performed."
                )
            if self.run_in_parallel:
                print(
                    f"Training the model for {self.num_epochs} epochs on {self.world_size} GPUs, on device {self.rank}, starting from epoch {start_epoch} with a batch size of {BATCH_SIZE}."
                )

            else:
                print(
                    f"Training the model for {self.num_epochs} epochs on one device {self.device}, starting from epoch {start_epoch} with a batch size of {BATCH_SIZE}."
                )

        # Train
        for epoch in range(start_epoch, self.num_epochs):
            if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                print(f"Epoch {epoch}/{self.num_epochs-1} - Training...")

            if self.run_in_parallel:
                for dataloader in self.dataloarders.values():
                    dataloader.sampler.set_epoch(epoch)

            # Train one epoch
            epoch_start_time = time.time()
            train_loss, train_metrics = Trainer.train_one_epoch(
                self.model,
                self.input_transform,
                self.target_transform,
                self.criterion,
                self.metrics,
                self.optimizer,
                self.dataloarders["train"],
                self.device,
                epoch,
                print_freq=self.print_freq,
                verbose=self.verbose
                if (self.rank == 0 or not self.run_in_parallel)
                else False,
            )
            train_time = (time.time() - epoch_start_time) / 60
            if self.criterion is None or not self.use_best_model:
                vali_loss = 0.0
                vali_metrics = [0.0] * len(self.metrics)
            else:
                vali_loss, vali_metrics = Trainer.validate(
                    self.model,
                    self.input_transform,
                    self.target_transform,
                    self.criterion,
                    self.metrics,
                    self.dataloarders["val"],
                    self.device,
                )
            vali_time = ((time.time() - epoch_start_time) / 60) - train_time
            self.scheduler.step()

            # Evaluate
            epoch_train_loss = train_loss / self.dataset_sizes["train"]
            epoch_vali_loss = vali_loss / self.dataset_sizes["val"]
            epoch_train_metrics = [
                m / self.dataset_sizes["train"] for m in train_metrics
            ]
            epoch_vali_metrics = [m / self.dataset_sizes["val"] for m in vali_metrics]
            if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                train_metrics_str = " ".join(
                    [
                        f"{m.__name__}: {v:.4f}"
                        for m, v in zip(self.metrics, epoch_train_metrics)
                    ]
                )
                vali_metrics_str = " ".join(
                    [
                        f"{m.__name__}: {v:.4f}"
                        for m, v in zip(self.metrics, epoch_vali_metrics)
                    ]
                )
                epoch_str = f"Epoch {epoch}/{self.num_epochs-1} - Trained in {train_time:.1f} min , validated in {vali_time:.1f} min - Train loss: {epoch_train_loss:.4f} - Validation loss: {epoch_vali_loss:.4f} - Train metrics: {train_metrics_str} - Validation metrics: {vali_metrics_str}"
                log_str.append(epoch_str)
                print(epoch_str)

            # Save the last model and the best value and epoch
            save_best = False
            if (self.use_best_model) and (
                epoch_vali_metrics[self.metric_index] > self.best_value
            ):
                self.best_value = epoch_vali_metrics[self.metric_index]
                self.best_epoch = epoch
                save_best = True
            else:
                self.best_value = 0.0
                self.best_epoch = epoch
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_loss": epoch_train_loss,
                "vali_loss": epoch_vali_loss,
                "vali_metrics": epoch_vali_metrics,
                "best_metric_index": self.metric_index,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
            }
            if (epoch % self.epochs_to_save == 0) or (epoch == self.num_epochs - 1):
                if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                    print(f"Saving checkpoint at epoch {epoch}...")
                torch.save(checkpoint_dict, last_checkpoint_path)
                with open(log_path, "w") as f:
                    for line in log_str:
                        f.write(f"{line}\n")
            if save_best:
                if self.verbose and (self.rank == 0 or not self.run_in_parallel):
                    print(f"Saving best checkpoint at epoch {epoch}...")
                torch.save(checkpoint_dict, best_checkpoint_path)
        if self.verbose and (self.rank == 0 or not self.run_in_parallel):
            print(f"Training completed in {(time.time() - start_time) / 60:.1f} min")
        if self.criterion is not None:
            print(
                f"Best validation metric: {self.best_value:.4f} at epoch {self.best_epoch}"
            )

        # Use the best model
        if self.use_best_model:
            best_checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(best_checkpoint["model_state_dict"])

        return self.model

    @staticmethod
    def train_one_epoch(
        model,
        inputs_transform,
        target_transform,
        criterion,
        metrics,
        optimizer,
        dataloader,
        device,
        epoch,
        print_freq=10,
        verbose=True,
    ):
        model.train()
        running_loss = 0.0
        running_metrics = [0.0] * len(metrics)
        input_count = 0

        for i, batch_dict in enumerate(dataloader):
            if inputs_transform is None:
                inputs = batch_dict["image"].to(device)
            else:
                inputs = inputs_transform(batch_dict["image"], model, device)

            if target_transform is None:
                targets = batch_dict["label"].to(device)
            else:
                targets = target_transform(batch_dict, model, device)

            optimizer.zero_grad()

            if criterion:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            else:
                loss_dict = model(inputs, targets)
                loss = sum(loss for loss in loss_dict.values())

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(inputs)
            input_count += len(inputs)
            metrics = [calculate_f1_score] # TODO: Fix this, don't define this here.
            metrics = [m(outputs, targets) for m in metrics]
            running_metrics = [r + m for r, m in zip(running_metrics, metrics)]

            if verbose and i % print_freq == 0:
                print(
                    f"Epoch {epoch} - Batch {i}/{len(dataloader)-1} - Train loss: {(running_loss / input_count):.4f}"
                )

        return running_loss, running_metrics

    @staticmethod
    def validate(
        model,
        inputs_transform,
        target_transform,
        criterion,
        metrics,
        dataloader,
        device,
    ):
        model.eval()
        validation_loss = 0.0
        validation_metrics = [0.0] * len(metrics)

        with torch.no_grad():
            for i, batch_dict in enumerate(dataloader):
                if inputs_transform is None:
                    inputs = batch_dict["image"].to(device)
                else:
                    inputs = inputs_transform(batch_dict["image"], model, device)

                if target_transform is None:
                    targets = batch_dict["label"].to(device)
                else:
                    targets = target_transform(batch_dict, model, device)

                if criterion:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                else:
                    if (
                        "FasterRCNN" in model.__class__.__name__
                    ):  # TODO: Implement loss for FasterRCNN
                        # outputs = model(inputs)
                        # loss_dict = Trainer.FasterRCNN_loss(outputs, targets)
                        # loss = sum(loss for loss in loss_dict.values())
                        loss = torch.tensor(0.0)
                    else:
                        raise ValueError(
                            "No criterion provided and no loss implemented."
                        )
                validation_loss += loss.item() * len(inputs)
                metrics = [calculate_f1_score] # TODO: Fix this, don't define this here.
                metrics = [m(outputs, targets) for m in metrics]
                validation_metric = [r + m for r, m in zip(validation_metrics, metrics)]

        return validation_loss, validation_metric

    @staticmethod
    def FasterRCNN_loss(outputs, targets):
        total_box_loss = 0.0
        total_cls_loss = 0.0

        for pred, target in zip(outputs, targets):
            # Box IoU loss
            box_loss = generalized_box_iou_loss(pred["boxes"], target["boxes"])
            total_box_loss += box_loss

            # Classification loss - Convert scores to probabilities using softmax if necessary
            # pred['labels'] and target['labels'] need to be compatible with cross_entropy
            if "scores" in pred:
                scores = torch.softmax(pred["scores"], dim=-1)
                target_labels = torch.zeros_like(scores).scatter_(
                    1, target["labels"].unsqueeze(1), 1
                )
                cls_loss = cross_entropy(scores, target_labels, reduction="sum")
            else:
                cls_loss = cross_entropy(pred["labels"], target["labels"])
            total_cls_loss += cls_loss

        avg_cls_loss = total_cls_loss / len(outputs)
        avg_box_loss = total_box_loss / len(outputs)

        return {"cls_loss": avg_cls_loss, "box_loss": avg_box_loss}


################################################################################
# Helper functions
################################################################################


def compute_snr_batch(output, target, reduction="mean"):
    snrs = list()
    for i in range(output.shape[0]):
        for j in range(output[i].shape[1]):
            snr, p_signal, p_noise = compute_snr(
                output[i][:, j].detach().cpu().numpy(),
                target[i][:, j].detach().cpu().numpy(),
            )
            snrs.append(snr)

    if reduction == "mean":
        if not np.all(np.isnan(snrs)):
            return np.nanmean(snrs)
        else:
            return float("nan")
    elif reduction == "sum":
        if not np.all(np.isnan(snrs)):
            return np.sum(snrs)
        else:
            return float("nan")
    else:
        return snrs


def get_grid_info(image_with_grid_lines):
    # Prep image
    x_min = int(IMAGES_PARTS_FOR_GRID_PREDICTION[0] * image_with_grid_lines.shape[2])
    y_min = int(IMAGES_PARTS_FOR_GRID_PREDICTION[1] * image_with_grid_lines.shape[1])
    x_max = int(IMAGES_PARTS_FOR_GRID_PREDICTION[2] * image_with_grid_lines.shape[2])
    y_max = int(IMAGES_PARTS_FOR_GRID_PREDICTION[3] * image_with_grid_lines.shape[1])
    image_cropped_np = (
        image_with_grid_lines.permute(1, 2, 0)
        .numpy()
        .astype(np.uint8)[y_min:y_max, x_min:x_max]
    )

    # Get lines
    lines = get_lines(image_cropped_np, threshold_HoughLines=430)
    lines_filtered = filter_lines(lines, degree_window=5, parallelism_count=4)

    # Get units
    if (lines_filtered is not None) and (len(lines_filtered) >= 2):
        pixels_per_grid = get_median_distance_between_consecutive_lines(lines_filtered)
        mm_per_pixel = get_mm_per_pixel(pixels_per_grid)
        sec_per_pixel = get_sec_per_pixel(mm_per_pixel)
        mV_per_pixel = get_mV_per_pixel(mm_per_pixel)
    else:
        pixels_per_grid = np.nan
        sec_per_pixel = np.nan
        mV_per_pixel = np.nan

    return pixels_per_grid, sec_per_pixel, mV_per_pixel


def get_mm_per_pixel(pixel_per_grid, mm_per_grid=5):
    return mm_per_grid / pixel_per_grid


def get_sec_per_pixel(mm_per_pixel, mm_per_sec=25):
    return mm_per_pixel / mm_per_sec


def get_mV_per_pixel(mm_per_pixel, mm_per_mV=10):
    return mm_per_pixel / mm_per_mV


def get_median_distance_between_consecutive_lines(
    lines, distance_threshold_large_small=10
):
    # Extract the rho values from the lines array and sort them
    rho_values = np.sort(lines[:, 0, 0])  # Sorting the rho values in ascending order

    # Compute distances between consecutive rho values
    distances = np.diff(
        rho_values
    )  # np.diff calculates the difference between consecutive elements

    # Filter out distances that are below the threshold
    distances_small = distances[distances <= distance_threshold_large_small]
    distances_large = distances[distances > distance_threshold_large_small]

    # Get the median
    small_median = np.median(distances_small)
    large_median = np.median(distances_large)

    # Calculate the median of these distances
    median_distance = (
        large_median + small_median
    )  # The idea is that a grid line is surrounded by two lines, hence the distance between two grid-line-surrounding-lines has to be added back to the distance between the grid lines

    return round(median_distance, 3)


def get_median_degrees(lines):
    lines = lines[:, 0, :]
    line_angles = [-(90 - line[1] * 180 / np.pi) for line in lines]
    return round(np.median(line_angles), 4)


################################################################################
# Rotation
################################################################################


def is_within_x_degrees_of_horizontal(theta, degree_window):
    theta_degrees = theta * 180 / np.pi
    deviation_from_horizontal = abs(90 - theta_degrees)
    return deviation_from_horizontal < degree_window


def filter_lines(lines, degree_window=20, parallelism_count=0, parallelism_window=2):
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


def get_lines(np_image, threshold_HoughLines=1380, rho_resolution=1):
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


def get_rotation_angle(np_image):
    lines = get_lines(np_image, threshold_HoughLines=1200)
    filtered_lines = filter_lines(
        lines, degree_window=30, parallelism_count=3, parallelism_window=2
    )
    if filtered_lines is None:
        rot_angle = np.nan
    else:
        rot_angle = get_median_degrees(filtered_lines)
    return rot_angle


################################################################################
# Plots
################################################################################


def get_image_with_lines(np_image, lines, width=None):
    if width is None:
        width = np_image.shape[1]

    if isinstance(lines, list):
        lines = np.array(lines)[:, np.newaxis, :]

    opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + width * (-b))
            y1 = int(y0 + width * (a))
            x2 = int(x0 - width * (-b))
            y2 = int(y0 - width * (a))
            cv2.line(
                opencv_image, (x1, y1), (x2, y2), (255, 0, 0), 2
            )  # Red line with thickness 2
    final_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

    return final_image


def plot_image_with_torch(ax, batch_dict, j=None, color_dict=None):
    # Prep image
    if j is None:
        image = batch_dict["image"]
    else:
        image = batch_dict["image"][j]
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
        torch.uint8
    )
    image = image[:3, ...]

    # Prep target
    targets = get_bbox_type_targets(batch_dict, verbose=False)
    if j is None:
        target = targets[0]
    else:
        target = targets[j]
    pred_boxes = target["boxes"].long()
    pred_labels = [f"{l}" for l in target["labels"]]
    if color_dict is None:
        colors = ["red" for l in target["labels"]]
    else:
        colors = [color_dict[int(l.cpu().numpy())] for l in target["labels"]]
    output_image = draw_bounding_boxes(
        image,
        pred_boxes,
        pred_labels,
        colors=colors,
        width=2,
        font_size=40,
        font="arial.ttf",
    )
    ax.imshow(output_image.permute(1, 2, 0))
    return ax


def plot_image_with_mask_bbox(
    ax, image, mask, j, bboxes_dict, boxes_list, c_list, img_height, img_width
):
    plot_image(ax, image, mask)
    for box_name, c in zip(boxes_list, c_list):
        bboxes = bboxes_dict[box_name]
        plot_bbox(ax, j, bboxes, c, img_height, img_width)
    return ax


def plot_image(ax, image, mask, replace_signals=True, mask_colors=MASK_COLORS):
    # Convert to numpy
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    mask = mask[:1, :, :].squeeze().numpy().astype(np.uint8)
    image_masked = image.copy()

    # Mask image
    for m, c in mask_colors.items():
        image_masked[mask == m] = c

    # Plot
    if replace_signals:
        ax.imshow(image_masked)
    else:
        binary_mask = mask > 0
        ax.imshow(image)
        ax.imshow(binary_mask, cmap="jet", alpha=0.5)

    return ax


def plot_bbox(ax, j, bboxes, c, img_height, img_width):
    for box in bboxes:
        # Define the corners of the box based on the provided indices
        bbox = {
            "0": [box["0"][0][j], box["0"][1][j]],
            "1": [box["1"][0][j], box["1"][1][j]],
            "2": [box["2"][0][j], box["2"][1][j]],
            "3": [box["3"][0][j], box["3"][1][j]],
        }
        box_converted = convert_json_to_bbox(bbox, img_height)
        left_upper = box_converted["left_upper"]
        left_bottom = box_converted["left_bottom"]
        right_upper = box_converted["right_upper"]
        right_bottom = box_converted["right_bottom"]

        # Create a list of tuples for the vertices of the polygon
        vertices = [left_upper, right_upper, right_bottom, left_bottom]

        # Create a Polygon patch with the vertices and color
        polygon = Polygon(vertices, closed=True, edgecolor=c, fill=False, linewidth=0.5)

        # Add the polygon to the axes
        ax.add_patch(polygon)

    return ax


def plot_signals(ax, record_path):
    # Load the signals
    label_signal, label_fields = load_signals(
        record_path
    )  # Shape of label_signal is (1000,12) for 12 leads
    sig_names = label_fields["sig_name"]

    # Create subplots within the provided axis using a 4x3 grid
    inner_grid = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=ax)

    # Plot the signals
    for i in range(label_signal.shape[1]):
        sub_ax = plt.subplot(inner_grid[i])
        sub_ax.plot(label_signal[:, i], label=sig_names[i])
        sub_ax.set_title(sig_names[i])
        sub_ax.set_xlim(0, X_FREQUENCY*10)  # Set x-axis limits
        sub_ax.set_ylim(-1.5, 1.5)    # Set y-axis limits
        sub_ax.set_xticks([0, X_FREQUENCY*5, X_FREQUENCY*10])  # Set specific x-axis ticks
        sub_ax.set_yticks([-1, 0, 1])  # Set specific y-axis ticks
        sub_ax.minorticks_off()
        sub_ax.label_outer()  # Clean up labels to only show on the outer edges

    return ax


def inspection_plots(loader_to_use, num_images_to_plot=1, plot_augmented=True):
    i = 0
    num_cols = 4 if plot_augmented else 3
    fig, ax = plt.subplots(
        num_images_to_plot, num_cols, figsize=(40, num_images_to_plot * 10)
    )
    for batch_dicts in loader_to_use:
        for j in range(len(batch_dicts["image"])):
            # Plot with torch
            ax[i][0].set_title(
                f'Torch: {os.path.split(batch_dicts["info_dict"]["image_path"][j])[-1]} - Lead: {batch_dicts["info_dict"]["lead_name"][j]}'
            )
            ax[i][0] = plot_image_with_torch(
                ax[i][0],
                batch_dicts,
                j,
                # {0: "red", 1: "green", 2: "blue", 3: "blue"}
            )

            # Plot the original image
            ax[i][1].set_title(
                f'Rotated plots: {os.path.split(batch_dicts["info_dict"]["image_path"][j])[-1]} - Lead: {batch_dicts["info_dict"]["lead_name"][j]}'
            )
            ax[i][1] = plot_image_with_mask_bbox(
                ax[i][1],
                batch_dicts["image"][j],
                batch_dicts["mask"][j],
                j,
                batch_dicts["info_dict"],
                [
                    "lead_bounding_box",
                    "text_bounding_box",
                ],
                ["r", "g", "b", "b"],
                batch_dicts["image"][j].shape[1],
                batch_dicts["image"][j].shape[2],
            )

            # Plot the rotated image
            if batch_dicts["info_dict"]["augment"][j] and plot_augmented:
                ax[i][2].set_title(
                    f'Original image: {os.path.split(batch_dicts["info_dict"]["image_path"][j])[-1]}'
                )
                ax[i][2] = plot_image_with_mask_bbox(
                    ax[i][2],
                    batch_dicts["image_original"][j],
                    batch_dicts["mask_augmented"][j],
                    j,
                    batch_dicts["info_dict"],
                    [
                        "lead_bounding_box_augmented",
                        "text_bounding_box_augmented",
                    ],
                    ["r", "g", "b", "b"],
                    batch_dicts["image"][j].shape[1],
                    batch_dicts["image"][j].shape[2],
                )

            # Plot the signals
            ax[i][num_cols - 1].set_title(
                f'Signals: {os.path.split(batch_dicts["info_dict"]["image_path"][j].split(".")[0][:-9])[-1]}'
            )
            ax[i][num_cols - 1] = plot_signals(
                ax[i][num_cols - 1],
                batch_dicts["info_dict"]["image_path"][j].split(".")[0][:-9],
            )

            i += 1
            if i >= num_images_to_plot:
                break
        if i >= num_images_to_plot:
            break

    # fig.suptitle("Torch plot, 4 point plot, rotated plot")
    # fig.tight_layout()
    plt.show()
