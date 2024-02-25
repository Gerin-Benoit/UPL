#!/bin/bash

datasets=("ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")


for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=3 bash get_info_TEMPLATES.sh "$dataset" anay_rn101 end 16 -1 False
done

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=3 bash get_info_TEMPLATES.sh "$dataset" anay_ViT_B_32 end 16 -1 False
done

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=3 bash get_info_TEMPLATES.sh "$dataset" anay_ViT_L_14 end 16 -1 False
done


