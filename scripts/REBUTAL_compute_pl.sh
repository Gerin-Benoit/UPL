#!/bin/bash

#SBATCH --job-name=pl_8_imagenet
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --array=0-0%1

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg

datasets=("ssimagenet")
archs=("anay_ViT_B_16")


module purge
module load nvidia/cuda/11.7.0-515.43.04
module load devel/python/3.9.13
source ~/.venv/dassl/bin/activate


bash get_info.sh ssimagenet anay_ViT_B_16 end 16 -1 False &> ~/8_pl_logs/anay_ViT_B_16_ssimagenet.txt

