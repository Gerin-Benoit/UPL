#!/bin/bash

#SBATCH --job-name=pl_all_finegrained
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --array=0-49%50

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssoxford_flowers" "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101")
archs=("anay_rn50" "anay_rn101" "anay_ViT_B_16" "anay_ViT_B_32" "anay_ViT_L_14")


index=$SLURM_ARRAY_TASK_ID
dataset_idx=$(( index / 5 ))
temp=$(( index % 5 ))
arch_idx=$(( temp ))

dataset=${datasets[$dataset_idx]}
arch=${archs[arch_idx]}

module purge
module load CUDA/11.7.0
module load Python/3.9.16-el8
source ~/.venv/dassl/bin/activate


bash get_info.sh "$dataset" "$arch" end 16 -1 False &> ~/pl_logs/"$arch"_"$dataset".txt



