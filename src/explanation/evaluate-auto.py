import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.counterfactual.utils import *

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    args = parser.parse_args()

    # Create dictionary to store results
    countertfactuals = ['none', 'orig', 'igd', 'fgd', 'reco', 'lgd', 'lnn']
    accuracies = {method: [] for method in countertfactuals}

    # Create list of key words for each image modification
    if args.response_type == 'free':
        keywords = {
            'spatial': ['spatial', 'structure', 'object', 'squiggle', 'grid', 'person', \
                            'uncommon', 'unfamiliar', 'unseen', 'unexpected', \
                            'out-of-distribution', 'not part of', 'not one of'],
            'brightness': ['brightness', 'lighting', 'low light', 'low-light', \
                            'bright', 'underexposed', 'overexposed', 'exposure', \
                            'rainbow', 'reflection', 'glare', 'lens flare', \
                            'light source', 'amount of light'],
            'contrast': ['low contrast', 'low-contrast', 'lower contrast', 'lower-contrast', \
                            'high contrast', 'high-contrast', 'higher contrast', 'higher-contrast' \
                            'enhanced contrast', 'increased contrast', 'strong contrast', \
                            'lack of contrast', 'reduced contrast', 'washed', 'muted', \
                            'level of contrast', 'detail and contrast', 'difference in contrast', \
                            'textured', 'level of texture', 'varied texture', 'uniform texture', \
                            'insufficient texture', 'lack of texture', 'defined edges', \
                            'highlights', 'shadows'], 
            'saturation': ['saturation', 'saturate', 'color intensity', 'color balance', \
                            'coloration', 'color distortion', 'color variation', \
                            'distorted color', 'unusual color', 'different color', \
                            'non-standard color', 'difference in color', 'altered the color', \
                            'vibrant', 'vivid', 'intense', 'muted', 'faded', 'subdued', \
                            'colorful', 'multiple colors', 'rainbow', ], 
            'noise': ['noise', 'noisy', 'noisiness', 'noisier', \
                        'grainy', 'graininess', 'grainier', \
                        'fuzzy', 'fuzziness', 'fuzzier', 'speckles', ],
            'pixelate': ['pixelate', 'pixelation', 'resolution', 'compressed', 'compression', \
                            'downscaled', 'downscaling', 'downsampled', 'downscaling', \
                            'blocky', 'blockiness'],
        }
    elif args.response_type == 'mc':
        keywords = {
            'spatial': ['1'], 
            'brightness': ['2'],
            'contrast': ['3'],
            'saturation': ['4'],
            'noise': ['5'],
            'pixelate': ['6'],
        }
    else:
        raise NotImplementedError('The response type should either be `free` or `mc`.')
    

    expl_dir = os.path.join(args.output_dir, 'explanations')
    for counterfactual in countertfactuals:
        for mod in mods[1:]:
            # Read saved responses
            expl_subdir = os.path.join(expl_dir, mod)
            expl_subdir = os.path.join(expl_subdir, args.response_type)
            expl_file = os.path.join(expl_subdir, f'{counterfactual}.json')
            with open(expl_file, 'r') as file:
                responses = json.load(file)
            
            # Check if responses contain keywords of corresponding modification
            num_correct = 0
            pattern = re.compile(r"(" + "|".join(re.escape(k) for k in keywords[mod]) + r")", re.IGNORECASE)
            for response in responses:
                if pattern.search(response):
                    num_correct += 1

            accuracy = num_correct / len(responses)
            accuracies[counterfactual].append(accuracy)

            # print(f'Accuracy of {counterfactual} method with {args.response_type} response for {mod} modification: {accuracy}')

    # Display accuracy for counterfactual methods
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("coolwarm_r")
    
    data = np.array([accuracies[method] for method in countertfactuals])

    heatmap = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Accuracy")

    ax.set_xticks(np.arange(len(mods[1:])))
    ax.set_yticks(np.arange(len(countertfactuals)))
    ax.set_xticklabels(mods[1:])
    ax.set_yticklabels(countertfactuals)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(countertfactuals)):
        for j in range(len(mods[1:])):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_xlabel("Source of Low Competency")
    ax.set_ylabel("Counterfactual Method")
    ax.set_title("Accuracy of Low Competency Explanations")

    results_dir = os.path.join(args.output_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{args.response_type}.png'))

if __name__ == "__main__":
    main()