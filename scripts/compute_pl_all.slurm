#!/bin/bash

#SBATCH --job-name=pl_all_imagenets
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --array=0-2%3

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg

datasets=("ssimagenet")
archs=("anay_rn101" "anay_ViT_B_32" "anay_ViT_L_14")


index=$SLURM_ARRAY_TASK_ID
arch_idx=$(( index ))
arch=${archs[arch_idx]}

module purge
module load nvidia/cuda/11.7.0-515.43.04
module load devel/python/3.9.13
source ~/.venv/dassl/bin/activate


bash get_info.sh ssimagenet "$arch" end 16 -1 False &> ~/pl_logs/"$arch"_ssimagenet.txt

