#!/bin/bash

#SBATCH --job-name=zs_CT
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --array=0-14%15

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg
datasets=("ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet") # "ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")
archs=("anay_rn101" "anay_ViT_B_32" "anay_ViT_L_14")

index=$SLURM_ARRAY_TASK_ID
dataset_idx=$(( index / 3 ))
temp=$(( index % 3 ))
arch_idx=$(( temp ))

dataset=${datasets[$dataset_idx]}
arch=${archs[arch_idx]}


module purge
module load devel/python/3.9.13
module load nvidia/cuda/11.7.0-515.43.04
source ~/.venv/dassl/bin/activate

bash upl_train_TEMPLATES.sh "$dataset" "$arch" end 16 16 False True "$arch"_random_init > ~/zs_TEMPLATES/"$arch"_"$dataset".txt

