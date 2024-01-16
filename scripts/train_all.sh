#!/bin/bash

#datasets=("ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2")
datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "sssun397" "ssucf101")
for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=3 bash upl_train.sh "$dataset" rn50_ep50 end 16 16 False True rn50_random_init > ~/train_logs_$dataset.txt
done


for dataset in "${datasets[@]}"; do
  bash upl_test_existing_logits.sh "$dataset" rn50_ep50 end 16 16 False True > ~/eval_logs_$dataset.txt
done
