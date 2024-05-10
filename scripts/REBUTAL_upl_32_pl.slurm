#!/bin/bash

#SBATCH --job-name=zs_32pl
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg

module purge
module load devel/python/3.9.13
module load nvidia/cuda/11.7.0-515.43.04
source ~/.venv/dassl/bin/activate

bash upl_train.sh ssimagenet anay_ViT_B_16 end 16 32 False True anay_ViT_B_32_random_init_32pl

