#!/bin/bash

datasets=("ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")

for dataset in "${datasets[@]}"; do
  bash upl_test_existing_logits.sh "$dataset" rn50_ep50 end 16 16 False True > ~/eval_logs_$dataset.txt
done