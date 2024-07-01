#! /bin/bash 
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=20
#SBATCH --time=11:10:00
#SBATCH --partition=short
#SBATCH --job-name=prep

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wolf6245@ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/ph24
conda info --env


python remove_keys_from_json.py \
        --dir /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records500_prepared_w_images/15000


python remove_keys_from_json.py \
        --dir /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records500_prepared_w_images/16000


python remove_keys_from_json.py \
        --dir /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records500_prepared_w_images/17000


python remove_keys_from_json.py \
        --dir /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records500_prepared_w_images/18000
