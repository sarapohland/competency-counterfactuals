import os
import pickle
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from tabulate import tabulate

from src.counterfactual.utils import *

def save_table(all_data, all_methods, labels, metric, file):
    table = {'Method': all_methods}
    for mod in mods:
        table[f"{mod.capitalize()}"] = [np.mean(data[metric][labels == mod]) for data in all_data]

    df = pd.DataFrame(table)
    df.to_csv(file, index=False)
    print(metric.replace('_', ' ').title())
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    args = parser.parse_args()

    # Read saved data
    data_dir = os.path.join(args.output_dir, 'data')
    all_data, all_methods = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.p'):
            file_path = os.path.join(data_dir, filename)
            all_data.append(pickle.load(open(file_path, 'rb')))
            all_methods.append(Path(filename).stem)
    labels = np.load(open(os.path.join(data_dir, 'labels.npz'), 'rb'))['labels']
    labels = int_to_mod(labels)

    # Generate plots of competency estimates
    plot_dir = os.path.join(args.output_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    scores_by_mod = []
    for method, data in zip(all_methods, all_data):
        scores = data['scores']
        scores_by_mod = [scores[labels == mod] for mod in mods]
        save_file = os.path.join(plot_dir, '{}.png'.format(method))
        plot_competency(scores_by_mod, mods, save_file)
    
    # Create tables comparing generated images
    results_dir = os.path.join(args.output_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # # Create table comparing latent representation similarity
    # save_file = os.path.join(results_dir, 'latent.csv')
    # save_table(all_data, all_methods, labels[~(labels=='none')], 'latent_similarity', save_file)
        
    # # Create table comparing feature vector similarity
    # save_file = os.path.join(results_dir, 'feature.csv')
    # save_table(all_data, all_methods, labels[~(labels=='none')], 'feature_similarity', save_file)

    # # Create table comparing perceptual image similarity
    # save_file = os.path.join(results_dir, 'perceptual.csv')
    # save_table(all_data, all_methods, labels[~(labels=='none')], 'perceptual_similarity', save_file)

    # Create a summary table across image modifications
    summary_table = {'Method': all_methods}
    summary_table['Time'] = [f"{data['time']:.4f}" for data in all_data]
    # summary_table['Success'] = [f"{(np.sum(data['scores'] > COMP_THRESH) / len(data['scores'])):.4f}" for data in all_data]
    summary_table['Success'] = [f"{(np.sum(data['scores'][~(labels=='none')] > COMP_THRESH) / len(data['scores'][~(labels=='none')])):.4f}" 
                                for data in all_data]

    for metric in ['latent_similarity', 'feature_similarity', 'perceptual_similarity']:
        summary_table[f"{metric.replace('_', ' ').title()}"] = [
            f"{np.mean(data[metric]):.4f} ± {np.std(data[metric]):.4f}"
            for data in all_data
        ]

    df = pd.DataFrame(summary_table)
    save_file = os.path.join(results_dir, 'summary.csv')
    df.to_csv(save_file, index=False)
    print('Summary')
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == "__main__":
    main()
