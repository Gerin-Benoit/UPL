#!/bin/bash

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssoxford_flowers" "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101" "ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")


#for dataset in "${datasets[@]}"; do
#  CUDA_VISIBLE_DEVICES=1 bash get_info.sh "$dataset" anay_ViT_B_16 end 16 -1 False
#done

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=1 bash upl_train.sh "$dataset" anay_ViT_B_16 end 16 16 False True ViT_B_16_random_init > ~/train_logs_ViTB16_$dataset.txt
done

for dataset in "${datasets[@]}"; do
  bash upl_test_existing_logits.sh "$dataset" anay_ViT_B_16 end 16 16 False True > ~/eval_logs_ViTB16_$dataset.txt
done
