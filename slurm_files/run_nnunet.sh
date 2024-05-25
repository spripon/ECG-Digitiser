#! /bin/bash 
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --ntasks-per-node=28
#SBATCH --time=10:10:00
#SBATCH --partition=short
#SBATCH --job-name=nn_pre

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wolf6245@ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet2024
conda info --env

# 1. Set the environment variables for nnU-Net.
export nnUNet_raw="/data/inet-multimodal-ai/wolf6245/data/ptb-xl"
export nnUNet_preprocessed="/data/inet-multimodal-ai/wolf6245/src/phd/physionet2024/data/nnUNet_preprocessed"
export nnUNet_results="/data/inet-multimodal-ai/wolf6245/src/phd/physionet2024/data/nnUNet_results"

# 2. Experiment planning and preprocessing
nnUNetv2_plan_and_preprocess -d 300 --clean -c 2d --verify_dataset_integrity

# 3. Model training # 14 h
# nnUNetv2_train 300 2d 0 -device cuda
# nnUNetv2_train 300 2d 1 -device cuda
# nnUNetv2_train 300 2d 2 -device cuda
# nnUNetv2_train 300 2d 3 -device cuda
# nnUNetv2_train 300 2d 4 -device cuda

# 4. Determine the best configuration
# nnUNetv2_find_best_configuration 300 -c 2d -f 0 #--disable_ensembling