#!/bin/bash

datasets=("ssimagenet-a" "ssimagenet-r" "ssimagenet-sketch" "ssimagenet-v2" "ssimagenet")
archs=("anay_rn101" "anay_ViT_B_32" "anay_ViT_L_14") # "anay_ViT_B_16")

for arch in "${archs[@]}"; do
  for dataset in "${datasets[@]}"; do

    bash upl_train_TEMPLATES.sh "$dataset" "$arch" end 16 16 False True "$arch"_random_init

  done
done
