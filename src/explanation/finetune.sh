#!/bin/bash

dataset="$1"
decoder="reconstruct"

for method in none reco lgd lnn #orig igd fgd 
do
    for response in free #mc
    do
        python src/explanation/finetune.py $dataset --counterfactual $method --response_type $response  --output_dir results/$dataset/explainability/  --decoder_dir models/$dataset/$decoder/
    done
done