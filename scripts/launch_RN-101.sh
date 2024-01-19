#!/bin/bash

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssoxford_flowers" "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101" "ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")


for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=1 bash get_info.sh "$dataset" anay_rn101 end 16 -1 False
done

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=1 bash upl_train.sh "$dataset" anay_rn101 end 16 16 False True rn101_random_init > ~/train_logs_rn101_$dataset.txt
done




