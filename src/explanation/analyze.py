import os
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from tabulate import tabulate

from src.counterfactual.utils import *

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    args = parser.parse_args()

    # Create dictionary to store accuracies
    counterfactuals = ['none', 'reco', 'lgd', 'lnn']
    accuracies = {method: {} for method in counterfactuals}

    # Compute accuracies across image modifications
    expl_dir = os.path.join(args.output_dir, 'explanations')
    for mod in mods[1:]:
        expl_subdir = os.path.join(expl_dir, mod)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        for method in counterfactuals:
            # Load list of saved labels
            label_file = os.path.join(expl_subdir, f'{method}.npz')
            labels = np.load(open(label_file, 'rb'))['labels']

            # Compute accuracy of explanations
            accuracy = np.sum(labels) / len(labels)
            accuracies[method][mod] = accuracy

    # Compute average accuracy across all modifications
    for method in counterfactuals:
        avg_acc = np.mean([accuracies[method][mod] for mod in mods[1:]])
        accuracies[method]['average'] = avg_acc

    # Save table of accuracies
    col_titles = mods[1:] + ['average']
    df = pd.DataFrame.from_dict(accuracies, orient='index', columns=col_titles)
    results_folder = os.path.join(args.output_dir, 'results')
    df.to_csv(os.path.join(results_folder, 'explanations.csv'))
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f"))
    # print(tabulate(df, headers='keys', tablefmt='latex', floatfmt=".2f"))
    
if __name__ == "__main__":
    main()