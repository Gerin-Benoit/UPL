#!/bin/bash

datasets=("sscaltech101" "ssdtd" "sseurosat" "ssfgvc_aircraft" "ssfood101" "ssimagenet" "ssoxford_flowers") # "ssoxford_pets" "ssstanford_cars" "sssun397" "ssucf101")

for dataset in "${datasets[@]}"; do
  bash upl_test_existing_logits.sh "$dataset" rn50_ep50 end 16 16 False True
done