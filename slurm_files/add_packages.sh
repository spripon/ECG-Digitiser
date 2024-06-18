#! /bin/bash 
#SBATCH --nodes=1
#SBATCH --mem=25G
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=packages


module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/ph24

pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install timm
pip install git+https://github.com/Kent0n-Li/nnSAM.git
