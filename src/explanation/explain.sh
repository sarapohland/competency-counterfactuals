#!/bin/bash

dataset="$1"
decoder="reconstruct"

for method in none reco lgd lnn #orig igd fgd 
do
    for response in free #mc
    do
        python src/explanation/explain.py $dataset --output_dir results/$dataset/explainability/ --counterfactual $method --response_type $response
    done
done