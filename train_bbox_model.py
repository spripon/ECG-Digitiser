#!/usr/bin/env python

# This file contains functions for training the bbox models. You can run it as follows:
#
#   python train_bbox_model.py -t data_folder_train -v data_folder_vali -m model_folder
#

import argparse
import sys

from helper_code import *
from team_code import *


# Parse arguments.
def get_parser():
    description = "Train the Challenge model(s)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-t", "--data_folder_train", type=str, required=True)
    parser.add_argument("-v", "--data_folder_vali", type=str, required=True)
    parser.add_argument("-m", "--model_folder", type=str, required=True)
    parser.add_argument("--run_in_parallel", action="store_true", default=False)
    return parser


# Run the code.
def run(args, run_in_parallel=False, rank=None):
    # Get args
    data_training_path = args.data_folder_train
    data_vali_path = args.data_folder_vali
    model_folder = args.model_folder
    model_folder_checkpoints = os.path.join(model_folder, "checkpoints")

    # Get the model and weights
    model_pretrained, preprocess = get_bbox_model(
        "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
    )

    # Get the data
    records, data, loader = dataloader_wrapper(
        list_of_paths=[data_training_path, data_vali_path],
        test_settings=[False, False],
        shuffle_settings=[True, False],
        transform=preprocess,
        run_in_parallel=run_in_parallel,
        rank=rank,
    )
    if rank is None or rank == 0:
        print(
            f"Data loaded. Training on {len(data[0])} images. Validating on {len(data[1])} images."
        )

    # Initiate trainer
    params = [p for p in model_pretrained.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR_BBOX, momentum=MOMENTUM_BBOX, weight_decay=WEIGHT_DECAY_BBOX
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE_BBOX, gamma=GAMMA_BBOX
    )
    trainer = Trainer(
        model=model_pretrained,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        num_epochs=NUM_EPOCHS_BBOX,
        device=DEVICE,
        target_transform=get_bbox_type_targets,
        input_transform=get_bbox_inputs,
        model_dir=model_folder_checkpoints,
        criterion=None,  # We will use the loss function from the model
        run_in_parallel=run_in_parallel,
        rank=rank,
    )

    # Train
    trained_model = trainer.fit(
        training_dataloader=loader[0], vali_dataloader=loader[1]
    )

    # Save the model
    save_torch_model(model_folder, trained_model, "lead_bbox_detection")


def main_parallel(rank, args):
    setup(rank, WORLD_SIZE)
    run(args, run_in_parallel=True, rank=rank)
    cleanup()


if __name__ == "__main__":
    args_to_use = get_parser().parse_args(sys.argv[1:])

    # Check
    if args_to_use.run_in_parallel:
        assert (
            args_to_use.run_in_parallel and DEVICE.type == "cuda"
        ), "Cannot run in parallel (RUN_IN_PARALLEL) without CUDA (DEVICE)."
        assert (
            args_to_use.run_in_parallel and WORLD_SIZE > 1
        ), "Cannot run in parallel (RUN_IN_PARALLEL) without multiple GPUs (WORLD_SIZE)."

    if args_to_use.run_in_parallel:
        print(f"Running in parallel on {WORLD_SIZE} processes.")
        mp.spawn(
            main_parallel,
            args=(args_to_use,),
            nprocs=WORLD_SIZE,
        )
    else:
        print("Running in serial.")
        run(args_to_use)
    print("Training of bbox model complete.")
