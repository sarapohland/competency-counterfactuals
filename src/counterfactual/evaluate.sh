#!/bin/bash

dataset="$1"
decoder="reconstruct"

for method in orig igd fgd reco lgd lnn
do
    python src/counterfactual/evaluate.py --method $method --test_data $dataset --decoder_dir models/$dataset/$decoder/ --output_dir results/$dataset/explainability/ --save_images
done

# for method in fgd lgd lnn
# do
#     for metric in l2 l1 cos
#     do
#         python src/counterfactual/evaluate.py --method $method --metric $metric --test_data $dataset --decoder_dir models/$dataset/$decoder/ --output_dir results/$dataset/explainability/
#     done
# done