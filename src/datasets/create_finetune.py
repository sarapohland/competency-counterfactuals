import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.setup_dataloader import setup_loader
from src.analysis.utils import factors
from src.analysis.modify import modify_image
from src.counterfactual.utils import *

NUM_EXAMPLES = 500

def main():

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--start_idx', type=int, default=100)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Load trained competency estimator
    file = os.path.join(args.decoder_dir, 'parce.p')
    estimator = pickle.load(open(file, 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    estimator.set_device(device)

    # Create dictionaries to store data
    images = {mod: [] for mod in mods[1:]}
    values = {mod: [] for mod in mods[1:]}

    # Add modified images to dataset
    dataloader = setup_loader(args.dataset, batch_size=1, test=True)
    for idx, (data, label) in enumerate(dataloader):
        if idx < args.start_idx:
            continue
        
        # End once we have enough example images
        if np.all(np.array([len(images[mod]) for mod in mods[2:]]) >= NUM_EXAMPLES):
            break

        # Get prediction of classification model
        output = estimator.model(data)
        pred = torch.argmax(output, dim=1).cpu()
        if not pred == label:
            continue

        # Compute competency score of images
        score = estimator.comp_scores(data, output.detach().numpy())

        # Try to generate low-competency image from high-competency one
        if score > COMP_THRESH:
            for mod in mods[2:]:
                if len(images[mod]) < NUM_EXAMPLES:
                    # Generate modified versions of this image
                    for factor in factors[args.dataset][mod]:
                        new_img = modify_image(data, mod, factor).float()
                        output = estimator.model(new_img)
                        score = estimator.comp_scores(new_img, output.detach().numpy())

                        # Save low-competency image
                        if score < COMP_THRESH:
                            images[mod].append(new_img)
                            values[mod].append(factor)
                            break

    # Add OOD images to dataset
    multi = int(NUM_EXAMPLES / 100)
    dataloader = setup_loader(args.dataset, batch_size=1, ood=True)
    for idx, (data, label) in enumerate(dataloader):
        # End once we have enough example images
        if len(images['spatial']) >= NUM_EXAMPLES:
            break

        if args.dataset == 'speed':
            for _ in range(multi):
                images['spatial'].append(data)
                values['spatial'].append(0)

        elif idx >= args.start_idx:
            # Compute competency score of images
            score = estimator.comp_scores(data, output.detach().numpy())

            if score < COMP_THRESH:
                images['spatial'].append(data)
                values['spatial'].append(0)

    # Convert data to numpy arrays
    for mod in mods[1:]:
        images[mod] = torch.vstack(images[mod]).numpy()
        values[mod] = np.array(values[mod])[:,None]
        print(mod, images[mod].shape, values[mod].shape)

    all_imgs = np.vstack([images[mod] for mod in mods[1:]])
    all_vals = np.vstack([values[mod] for mod in mods[1:]])
    all_mods = np.vstack([np.array([mod] * len(images[mod]))[:,None] for mod in mods[1:]])
    print('all', np.shape(all_imgs), np.shape(all_vals), np.shape(all_mods))
    print(all_mods.flatten())
    print(all_vals.flatten())

    # Save all data to a npz file
    data_path = os.path.join('./data', args.dataset)
    file = os.path.join(data_path, 'finetune.npz')
    np.savez(file, data=all_imgs, props=all_mods, factors=all_vals)


if __name__=="__main__":
    main()   