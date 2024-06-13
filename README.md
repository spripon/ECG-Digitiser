# Python code for the George B. Moody PhysioNet Challenge 2024

## What's in this repository?

This repository contains our team's code for a Python entry to the [George B. Moody PhysioNet Challenge 2024](https://physionetchallenges.org/2024/). It builds on the default repository provided by the Challenge. You can try it by running the steps describe below.

At the moment, it does the following:

- For the example code, we implemented a random forest model with several simple features. (This simple example is **not** designed to perform well, so you should **not** use it as a baseline for your approach's performance.)


## How do I get started?

1. Clone this repository `git clone https://github.com/felixkrones/physionet24.git`
2. Move into repo `cd physionet24`
3. Create a new environment:

    Using pip:

        python3.11 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt

    Using conda:

        conda create --name env-name python=3.11
        conda activate env-name
        pip install --upgrade pip
        pip install -r requirements.txt

    For nnU-Net we need to define the locations for raw data, preprocessed data and trained models, by setting environment variables. Please find instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

    At the moment, the official nnU-Net repository contains a bug and is not working with RGB png images. Please use the following installation instructions for now:

        git clone https://github.com/felixkrones/nnUNet.git
        cd nnUNet
        pip install -e .

    To also use nnSAM, run the following after:

        pip install git+https://github.com/ChaoningZhang/MobileSAM.git
        pip install timm
        pip install git+https://github.com/Kent0n-Li/nnSAM.git

4. Create the data. So far, only the waveforms are provided with all the metadata in separate files. We a) need to combine the metadata with the header files and b) generate the images from the signals. See the description below on `How do I create data for these scripts?`. This is using the 500Hz images at the moment.
5. Create your train and test data. We use the splits suggested in `ptbxl_database.csv`. See below under `How do I create train and test data?`.
6. Prepare the images and masks for the segmentation model. See below under `How do I prepare the images for the segmentation`.
7. Make changes: Before you change anything, create a new branch: `git checkout -b your_branch`. You should only make changes on `team_code.py`. If you want to make changes on the data split, you can change `create_train_test.py`. If you need new packages, add them to `requirements.txt`. You can use `analysis.ipynb` or create your own notebook for some data analysis and experiments.
8. Run the code as decribed below under `## How do I run these scripts?`.


## How do I create the data for these scripts?

You need xx GB of free storage.
Downloading the data will need 3-4 GB of space. Step 3 will increase the data from 3 GB to xx GB.

1. Download (and unzip) the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) and [PTB-XL+ dataset](https://physionet.org/content/ptb-xl-plus/). These instructions use `ptb-xl` as the folder name that contains the data for these commands (the full folder name for the PTB-XL dataset is currently `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`, and the full folder name for the PTB-XL+ dataset is currently `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1`), but you can replace it with the absolute or relative path on your machine. Replace the name of the folder that contains a file named `ptbxl_database.csv` with `ptb-xl/`. Replace the paths below with the actual path to that folder.

2. Add information from various spreadsheets from the PTB-XL dataset to the WFDB header files.

        python prepare_ptbxl_data.py \
            -i ptb-xl/records500 \
            -pd ptb-xl/ptbxl_database.csv \
            -pm ptb-xl/scp_statements.csv \
            -sd ptb-xl-p/labels/12sl_statements.csv \
            -sm ptb-xl-p/labels/mapping/12slv23ToSNOMED.csv \
            -o ptb-xl/records500_prepared

    For example:

        python prepare_ptbxl_data.py \
            -i /data/wolf6245/data/ptb-xl/records500 \
            -pd /data/wolf6245/data/ptb-xl/ptbxl_database.csv \
            -pm /data/wolf6245/data/ptb-xl/scp_statements.csv \
            -sd /data/wolf6245/data/ptb-xl-p/labels/12sl_statements.csv \
            -sm /data/wolf6245/data/ptb-xl-p/labels/mapping/12slv23ToSNOMED.csv \
            -o /data/wolf6245/data/ptb-xl/records500_prepared

3. [Generate synthetic ECG images](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator) on the dataset: (This step is using a separate repository)

    1. Deactivate your current environment: `deactivate`
    2. Move to the location where you want to install the generator repo: `cd ..`
    3. Clone the repo: `git clone https://github.com/felixkrones/ecg-image-kit.git`
    4. Move into generator piece: `cd ecg-image-kit/codes/ecg-image-generator/`
    5. Create a new environment. For this, follow the instructions in the README of the generator repo. You can define the environment name and the Python version in the `environment_droplet.yml` file.
    6. Now you can run the following code. Careful though, this can take very long, around 10 min per subfolder (approx. 1000 files) or 4h in total and will increase the necessary disk space by approx. 15x, adding another 8GB for the 500Hz data. To test, better to run it on one single subfolder (e.g., add /00000):

            python gen_ecg_images_from_data_batch.py \
                -i ptb-xl/records500_prepared \
                -o ptb-xl/records500_prepared_w_images \
                --print_header \
                --store_config 2 \
                --mask_unplotted_samples

        For example:

            python gen_ecg_images_from_data_batch.py \
                -i /data/wolf6245/data/ptb-xl/records500_prepared/09000 \
                -o /data/wolf6245/data/ptb-xl/records500_prepared_w_images/09000 \
                -se 10 \
                --mask_unplotted_samples \
                --print_header \
                --store_config 2 \
                --lead_name_bbox \
                --lead_bbox \
                --random_print_header 0.7 \
                --calibration_pulse 0.6 \
                --fully_random \
                -rot 5 \
                --num_images_per_ecg 4 \
                --random_bw 0.1 \
                --add_qr_code \
                --run_in_parallel \
                --num_workers 8
                
    7. Deactivate the environment again, move back to your original repo and activate the environment there again:

            deactivate
            cd 
            cd your-repo-path
            source .venv/bin/activate

4. Add the file locations and other information for the synthetic ECG images to the WFDB header files. (The expected image filenames for record `12345` are of the form `12345-0.png`, `12345-1.png`, etc., which should be in the same folder.) You can use the `ptb-xl/records500` folder for the `train_model` step:

        python prepare_image_data.py \
            -i ptb-xl/records500_prepared_w_images \
            -o ptb-xl/records500_prepared_w_images

    For example:

        python prepare_image_data.py \
            -i /data/wolf6245/data/ptb-xl/records500_prepared_w_images \
            -o /data/wolf6245/data/ptb-xl/records500_prepared_w_images

5. Remove the waveforms, certain information about the waveforms, and the demographics and classes to create a version of the data for inference. You can use the `ptb-xl/records500_hidden/00000` folder for the `run_model` step, but it would be better to repeat the above steps on a new subset of the data that you will not use to train your model:

        python remove_hidden_data.py \
            -i ptb-xl/records500_prepared_w_images \
            -o ptb-xl/records500_prepared_w_images_hidden \
            --include_images

    For example:

        python remove_hidden_data.py \
            -i /data/wolf6245/data/ptb-xl/records500_prepared_w_images \
            -o /data/wolf6245/data/ptb-xl/records500_prepared_w_images_hidden \
            --include_images

## How do I create train and test data?

We use the suggested splits from `ptbxl_database.csv`. 
Run the code twice, once for full data and once for inference data:

    python create_train_test.py \
        -i ptb-xl/records500_prepared_w_images \
        -d ptb-xl/ptbxl_database.csv \
        -o ptb-xl/splits500

For example:

    python create_train_test.py \
        -i /data/wolf6245/data/ptb-xl/records500_prepared_w_images_prepped \
        -d /data/wolf6245/data/ptb-xl/ptbxl_database.csv \
        -o /data/wolf6245/data/ptb-xl/Dataset500_Signals \
        --rgba_to_rgb \
        --gray_to_rgb \
        --mask \
        --mask_multilabel \
        --rotate_image \
        --num_workers 8


## How do I run these scripts?

### Pre-training

Before we can run the main pipeline, we need to first train the bbox model and then the segmentation model. The bbox model is trained on the images with the bounding boxes. The segmentation model is trained on the single signals.

#### Bbox model

To train the bbox model, run:

    python train_bbox_model.py -t data_folder_train -v data_folder_vali -m model_folder

For example:

    python train_bbox_model.py -t /data/wolf6245/data/ptb-xl/Dataset500_Signals/imagesTr -v /data/wolf6245/data/ptb-xl/Dataset500_Signals/imagesTv -m model --run_in_parallel

#### Segmentation model

To then train the segmentation model, we use [nnU-Net](https://github.com/felixkrones/nnUNet.git); this involves multiple steps:

1. Set the environment variables for nnU-Net.

        # Set environment variables
        export nnUNet_raw="/data/wolf6245/data/ptb-xl"
        export nnUNet_preprocessed="/data/wolf6245/src/phd/physionet24/model/nnUNet_preprocessed"
        export nnUNet_results="/data/wolf6245/src/phd/physionet24/model/nnUNet_results"

        # Check
        echo ${nnUNet_raw}

    If you are planning to use nnSAM, define the model:

        set MODEL_NAME=nnsam

2. Experiment planning and preprocessing

        nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

    For example for the single signals:

        nnUNetv2_plan_and_preprocess -d 200 --clean -c 2d --verify_dataset_integrity

    Or for the full images:

        nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity

3. Model training

        nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD

    For example for a single fold:

        nnUNetv2_train 200 2d 0 -device cuda --npz

    Or for the full images:

        nnUNetv2_train 500 2d 0 -device cuda --c

    Or select the device (one per fold):

        CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 500 2d 1 --c

4. Determine the best configuration

        nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS

    For example for the single signals:

        nnUNetv2_find_best_configuration 200 -c 2d -f 0

    Or for the full images:

        nnUNetv2_find_best_configuration 500 -c 2d --disable_ensembling

5. Inference
    
        See instructions in `inference_instructions.txt`

    For example for the single signals:

        nnUNetv2_predict -d Dataset200_SingleSignals -i /data/wolf6245/data/ptb-xl/Dataset200_SingleSignals/imagesTv -o /data/wolf6245/src/phd/physionet24/data/nnUNet_output -f  0 -tr nnUNetTrainer -c 2d -p nnUNetPlans

    Or for the full images:

        nnUNetv2_predict -d Dataset500_Signals -i /data/wolf6245/data/ptb-xl/Dataset500_Signals/imagesTv -o /data/wolf6245/src/phd/physionet24/data/nnUNet_output -f  0 -tr nnUNetTrainer -c 2d -p nnUNetPlans

6. Postprocessing

        See instructions in `inference_instructions.txt`

    For example for the single signals:

        nnUNetv2_apply_postprocessing -i /data/wolf6245/src/phd/physionet24/data/nnUNet_output -o /data/wolf6245/src/phd/physionet24/data/nnUNet_output_pp -pp_pkl_file /data/wolf6245/src/phd/physionet24/model/nnUNet_results/Dataset200_SingleSignals/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/postprocessing.pkl -np 8 -plans_json /data/wolf6245/src/phd/physionet24/model/nnUNet_results/Dataset200_SingleSignals/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/plans.json

    Or for the full images:

        nnUNetv2_apply_postprocessing -i /data/wolf6245/src/phd/physionet24/data/nnUNet_output -o /data/wolf6245/src/phd/physionet24/data/nnUNet_output_pp -pp_pkl_file /data/wolf6245/src/phd/physionet24/model/nnUNet_results/Dataset500_FullImages/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/postprocessing.pkl -np 8 -plans_json /data/wolf6245/src/phd/physionet24/model/nnUNet_results/Dataset500_FullImages/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0/plans.json


### Challenge pipeline

You can train your model(s) by running

    python train_model.py -d training_data -m model -v

For example:

    python train_model.py -d /data/wolf6245/data/ptb-xl/Dataset500_Signals/imagesTr -m model -v

where

- `training_data` (input; required) is a folder with the training data files, including the images and classes (you can use the `ptb-xl/splits_500/imagesTr` folder from the steps before); and
- `model` (output; required) is a folder for saving your model(s).

We are asking teams to include working training code and a pre-trained model. Please include your pre-trained model in the `model` folder so that we can load it with the below command.

You can run your trained model(s) by running

    python run_model.py -d test_data -m model -o data/test_outputs -v

e.g.

    python run_model.py -d /Users/Felix_Krones/code/data/ptb-xl/test_images -m model -o data/test_outputs -v

where

- `test_data` (input; required) is a folder with the validation or test data files, excluding the images and diagnoses (you can use the `ptb-xl/Dataset500_Signals_hidden/imagesTs` folder from the steps before);
- `model` (input; required) is a folder for loading your model(s); and
- `test_outputs` is a folder for saving your model outputs.

We are asking teams to include working training code and a pre-trained model. Please include your pre-trained model in the `model` folder so that we can load it with the above command.

The [Challenge website](https://physionetchallenges.org/2024/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by running

    python evaluate_model.py -d labels -o data/test_outputs -s data/evaluation/scores.csv

e.g.

    python evaluate_model.py -d /data/wolf6245/data/ptb-xl/test -o data/test_outputs -s data/evaluation/scores.csv

where

- `labels` is a folder with labels for the data. This is basically the folder which we used in the step before for test_data (you can use the `ptb-xl/Dataset500_Signals/imagesTs` folder from the steps before);
- `test_outputs` is a folder containing files with your model's outputs for the data; and
- `scores.csv` (optional) is file with a collection of scores for your model.


## Which scripts I can edit?

Please edit the following script to add your code:

* `team_code.py` is a script with functions for training and running your trained model(s).

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model(s).
* `run_model.py` is a script for running your trained model(s).
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my models?

You can choose to create digitization and/or classification models.

To train and save your model(s), please edit the `train_models` function in the `team_code.py` script. Please do not edit the input or output arguments of this function.

To load and run your trained model(s), please edit the `load_models` and `run_models` functions in the `team_code.py` script. Please do not edit the input or output arguments of these functions.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 1000 records.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2024/#data). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2024.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-example-2024  test_data  test_outputs  training_data

        user@computer:~/example$ cd python-example-2024/

        user@computer:~/example/python-example-2024$ sudo docker build -t image .

        ------ IF THIS FAILS, TRY: Delete the line `"credsStore": "desktop",` from `~/.docker/config.json` and try again. ------

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2024$ docker run -it -v ~/example/model:/challenge/model -v ~/example/test_data:/challenge/test_data -v ~/example/test_outputs:/challenge/test_outputs -v ~/example/training_data:/challenge/training_data image bash

        docker run -it -v /data/wolf6245/src/phd/physionet24/model:/challenge/model -v /data/wolf6245/data/ptb-xl/Dataset500_Signals/imagesTs:/challenge/test_data -v /data/wolf6245/src/phd/physionet24/data/test_outputs:/challenge/test_outputs -v /data/wolf6245/data/ptb-xl/Dataset500_Signals/imagesTr:/challenge/training_data image bash

        root@[...]:/challenge# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py      [...]

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d test_data -m model -o test_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d test_data -o test_outputs
        [...]

        root@[...]:/challenge# exit
        Exit


---------------------------------------------------------------------------------------
## What else do I need?

This repository does not include data or the code for generating ECG images. Please see the above instructions for how to download and prepare the data.

## How do I learn more? How do I share more?

Please see the [Challenge website](https://physionetchallenges.org/2024/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges). Please do not make pull requests, which may share information about your approach.

## Useful links

* [Challenge website](https://physionetchallenges.org/2024/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2024)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2024)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2024/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)


## Credits

- We used nnU-Net:

        Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

- The base for this repository was [PhysioNet 2024](https://github.com/physionetchallenges/python-example-2024) by the PhysioNet team.

