#!/bin/bash

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssoxford_flowers" "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101" "ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")
shots=(1 2 4 8 16)
seeds=(1 2 3)


for dataset in "${datasets[@]}"; do
  for shot in "${shots[@]}"; do
    for seed in "${seeds[@]}"; do
      bash coopupl_train_and_val_vit16.sh "$dataset" anay_ViT_B_16 end 16 "$shot" False True "$seed" > ~/trainval_logs/vitb16_"$dataset"_"$shot"shots_seed"$seed".txt
    done
  done
done

