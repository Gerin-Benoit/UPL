#!/bin/bash

#SBATCH --job-name=fs_imagenet
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=1-23:00:00
#SBATCH --array=0-2%

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg
datasets=("ssimagenet") # "ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")
archs=("anay_ViT_B_16")
suffixes=("vit16")
shots=(16)
seeds=(1 2 3)

index=$SLURM_ARRAY_TASK_ID
dataset_idx=$(( index / 3 ))
temp=$(( index % 3 ))
arch_idx=$(( temp / 3 ))
temp=$(( temp % 3 ))
shot_idx=$(( temp / 3 ))
temp=$(( temp % 3 ))
seed_idx=$(( temp ))

dataset=${datasets[$dataset_idx]}
arch=${archs[arch_idx]}
suffix=${suffixes[arch_idx]}
shot=${shots[shot_idx]}
seed=${seeds[seed_idx]}

module purge
module load devel/python/3.9.13
module load nvidia/cuda/11.7.0-515.43.04
source ~/.venv/dassl/bin/activate

bash coopupl_train_and_val_"${suffix}".sh "$dataset" "$arch" end 16 "$shot" False True "$seed" > ~/trainval_logs/"$arch"_"$dataset"_"$shot"shots_seed"$seed".txt

