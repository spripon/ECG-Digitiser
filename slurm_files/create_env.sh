#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --time=01:30:00
#SBATCH --partition=short
#SBATCH --job-name=ph24_env

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/envs/ph24
conda create --prefix $CONPREFIX python=3.11

# Activate your environment
source activate $CONPREFIX

# Install packages...
pip install -r requirements.txt

# Install nnunet
cd nnUNet
pip install -e .

# Install nnsam
# pip install git+https://github.com/ChaoningZhang/MobileSAM.git
# pip install timm
# pip install git+https://github.com/Kent0n-Li/nnSAM.git
