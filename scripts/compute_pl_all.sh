#!/bin/bash

#datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssoxford_flowers" "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101")
datasets=("ssimagenet-a" "ssimagenet-sketch" "ssimagenet-v2")

for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=0 bash get_info.sh "$dataset" anay_rn50 end 16 -1 False # > ~/getpl_logs.txt
done

