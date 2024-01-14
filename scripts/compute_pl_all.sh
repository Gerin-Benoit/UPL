#!/bin/bash

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "sssun397" "ssucf101")


for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=0 bash get_info.sh "$dataset" anay_rn50 end 16 -1 False > ~/getpl_logs_$dataset.txt
done

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=0 bash upl_train.sh "$dataset" rn50_ep50 end 16 16 False True rn50_random_init > ~/train_logs_$dataset.txt
done

for dataset in "${datasets[@]}"; do
  bash upl_test_existing_logits.sh "$dataset" rn50_ep50 end 16 16 False True > ~/eval_logs_$dataset.txt
done

