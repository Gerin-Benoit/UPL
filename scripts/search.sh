#!/bin/bash

datasets=("ssdtd" "sseurosat")
shots=(1 2 4 8 16)
lss=(3 4 5)


#datasets=("ssimagenet")

for dataset in "${datasets[@]}"; do
  for shot in "${shots[@]}"; do
    for ls in "${lss[@]}"; do
      CUDA_VISIBLE_DEVICES=2 bash coopupl_train.sh "$dataset" rn50_ep50 end 16 "$shot" False True "$ls" 1
      done
  done
done
