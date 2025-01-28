#!/bin/bash

dataset="$1"

for method in parce draem fastflow padim patchcore reverse rkde stfpm ganomaly
do
    python src/comparison/regional/evaluate.py $method --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/inpaint/ --data_file results/$dataset/competency/regional/data/$method.csv --estimator_file models/$dataset/anomaly/$method.p
done
