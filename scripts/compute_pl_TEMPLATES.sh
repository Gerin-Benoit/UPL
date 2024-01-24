#!/bin/bash

datasets=("ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2")


for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=3 bash get_info_TEMPLATES.sh "$dataset" rn50_ep50 end 16 -1 False
done

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=3 bash get_info_TEMPLATES.sh "$dataset" anay_ViT_B_16 end 16 -1 False
done


