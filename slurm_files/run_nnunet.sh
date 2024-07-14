#! /bin/bash 
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --ntasks-per-node=28
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=nn_train_all

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wolf6245@ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/ph24
conda info --env

# 1. Set the environment variables for nnU-Net.
export nnUNet_raw="/data/inet-multimodal-ai/wolf6245/data/ptb-xl"
export nnUNet_preprocessed="/data/inet-multimodal-ai/wolf6245/src/phd/physionet24/model/nnUNet_preprocessed"
export nnUNet_results="/data/inet-multimodal-ai/wolf6245/src/phd/physionet24/model/nnUNet_results"

# Optional: Use nnsam
# export MODEL_NAME="nnunet"

# 2. Experiment planning and preprocessing # 20 h
# nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity

# 3. Model training # 14 h
# nnUNetv2_train 500 2d 0 -device cuda --c
nnUNetv2_train 500 2d all -device cuda --c

# (Optional) 4. Determine the best configuration
# nnUNetv2_find_best_configuration 500 -c 2d -f 0 --disable_ensembling

# 5. Save
# nnUNetv2_export_model_to_zip

# 6. Unpack
# nnUNetv2_install_pretrained_model_from_zip