#!/bin/bash

datasets=("ssdtd" "sseurosat")
shots=(1 2 4 8 16)
lss=(0.1 0.3 0.5 0.7 0.9)


#datasets=("ssimagenet")

for dataset in "${datasets[@]}"; do
  for shot in "${shots[@]}"; do
    for ls in "${lss[@]}"; do
      CUDA_VISIBLE_DEVICES=2 bash coopupl_train.sh "$dataset" rn50_ep50 end 16 "$shot" False True "$ls" "$(1-$ls)"
      done
  done
done
