#!/bin/bash

dataset="$1"

for method in parce softmax dropout ensemble temperature kl entropy openmax energy odin mahalanobis knn dice
do
    python src/comparison/overall/evaluate.py $method --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --save_file results/$dataset/unmodified/data/$method.csv --estimator_file models/$dataset/overall/$method.p
done
