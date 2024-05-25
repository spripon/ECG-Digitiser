#! /bin/bash 
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --ntasks-per-node=20
#SBATCH --time=36:10:00
#SBATCH --partition=medium
#SBATCH --job-name=split

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wolf6245@ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet2024
conda info --env

# python fix_json.py

# python prepare_ptbxl_data.py \
#     -i /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records100 \
#     -d /data/inet-multimodal-ai/wolf6245/data/ptb-xl/ptbxl_database.csv \
#     -s /data/inet-multimodal-ai/wolf6245/data/ptb-xl/scp_statements.csv \
#     -o /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records100_prepared

# python add_image_filenames.py \
#     -i /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records100_prepared_w_images \
#     -o /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records100_prepared_w_images

# python create_train_test.py \
#         -i /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records100_prepared_w_images \
#         -d /data/inet-multimodal-ai/wolf6245/data/ptb-xl/ptbxl_database.csv \
#         -o /data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset100_Signals \
#         --rgba_to_rgb \
#         --gray_to_rgb

python split_images_to_signals.py \
        -i /data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset100_Signals \
        -o /data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset200_SingleSignals

# python prepare_mask_classes.py \
#         -i /data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset100_Signals \
#         -o /data/inet-multimodal-ai/wolf6245/data/ptb-xl/Dataset300_FullImages