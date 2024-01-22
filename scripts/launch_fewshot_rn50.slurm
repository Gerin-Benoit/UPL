#!/bin/bash

#SBATCH --job-name=fs_rn50_finegrained
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --array=0-149%30

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=ariacpg

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssoxford_flowers" "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101") # "ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")
shots=(1 2 4 8 16)
seeds=(1 2 3)


module purge
module load CUDA/11.7.0
module load Python/3.9.16-el8
source ~/.venv/dassl/bin/activate


for dataset in "${datasets[@]}"; do
  for shot in "${shots[@]}"; do
    for seed in "${seeds[@]}"; do
      bash coopupl_train_and_val_rn50.sh "$dataset" rn50_ep50 end 16 "$shot" False True "$seed" > ~/trainval_logs/rn50_"$dataset"_"$shot"shots_seed"$seed".txt
    done
  done
done

