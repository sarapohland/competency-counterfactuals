import os
import time
import json
import torch
import pickle
import argparse
import numpy as np

from lpips import LPIPS
from torch.nn.functional import cosine_similarity

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader
from src.counterfactual.utils import *
from src.counterfactual.methods import *
from src.utils.visualize import visualize_img

torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--method', type=str, default='reco')
    parser.add_argument('--metric', type=str, default='none')
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    assert(args.method in methods)
    if args.method in ['fgd', 'lgd', 'lnn'] and args.metric != 'none':
        name = '{}-{}'.format(args.method, args.metric)
    else:
        name = args.method

    # Create folder to save results
    data_dir = os.path.join(args.output_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file = os.path.join(data_dir, '{}.p'.format(name))
    label_file = os.path.join(data_dir, 'labels.npz')

    if args.save_images:
        img_dir = os.path.join(args.output_dir, 'images')
        for mod in mods:
            img_subdir = os.path.join(img_dir, mod)
            if not os.path.exists(img_subdir):
                os.makedirs(img_subdir)

    # Create dictionary to store data
    data = {'time': 0, 'losses': [], 'scores': [],
            'latent_similarity': [], 'feature_similarity': [], 
            'perceptual_similarity': [], 
            }
    labels = []
    
    # Set perceptual loss function
    lpips_loss = LPIPS(net='alex')
    
    # Load trained competency estimator
    file = os.path.join(args.decoder_dir, 'parce.p')
    estimator = pickle.load(open(file, 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    estimator.set_device(device)

    if args.method == 'igd':
        # Create object to perform image gradient descent
        IGD = ImageGradientDescent(estimator, thresh=COMP_THRESH)

    elif args.method == 'fgd':
        # Create object to perform latent gradient descent
        if args.metric != 'none':
            FGD = FeatureGradientDescent(estimator, thresh=COMP_THRESH, metric=args.metric)
        else:
            FGD = FeatureGradientDescent(estimator, thresh=COMP_THRESH)

    elif args.method == 'lgd':
        # Create object to perform latent gradient descent
        if args.metric != 'none':
            LGD = LatentGradientDescent(estimator, thresh=COMP_THRESH, metric=args.metric)
        else:
            LGD = LatentGradientDescent(estimator, thresh=COMP_THRESH)

    elif args.method == 'lnn':
        # Create object to perform latent nearest neighbors
        if args.metric != 'none':
            LNN = LatentNearestNeighbors(estimator, thresh=COMP_THRESH, metric=args.metric)
        else:
            LNN = LatentNearestNeighbors(estimator, thresh=COMP_THRESH)

    # Create data loader for example set
    dataloader = setup_loader(args.test_data, batch_size=50, example=True)
    
    # Collect data from test set
    total = 0
    count = {mod: 0 for mod in mods}
    for X, y in dataloader:
        X = X.to(device)
        labels.append(y)

        # Compute losses and competency scores for original images
        output = estimator.model(X).cpu()
        orig_scores, orig_losses = estimator.comp_scores_and_losses(X, output.detach().numpy())

        # Generate counterfactual images with desired method
        start = time.time()
        X_new = X.clone()
        X_low = X[orig_scores < COMP_THRESH]
        
        if args.method == 'reco':
            # Get reconstructed image for low-competency images
            X_new[orig_scores < COMP_THRESH] = estimator.decoder(X_low)

        elif args.method == 'lgd':
            # Perform latent gradient descent (LGD) for low-competency images
            X_new[orig_scores < COMP_THRESH] = LGD.get_counterfactual(X_low)

        elif args.method == 'lnn':
            # Perform latent nearest neighbors (LNN) for low-competency images
            X_new[orig_scores < COMP_THRESH] = LNN.get_counterfactual(X_low)

        elif args.method == 'igd':
            # Perform image gradient descent (IGD) for low-competency images
            X_new[orig_scores < COMP_THRESH] = IGD.get_counterfactual(X_low)

        elif args.method == 'fgd':
            # Perform feature gradient descent (FGD) for low-competency images
            X_new[orig_scores < COMP_THRESH] = FGD.get_counterfactual(X_low)

        data['time'] += (time.time() - start)

        # Compute losses and competency scores for counterfactual images
        output = estimator.model(X_new).cpu()
        new_scores, new_losses = estimator.comp_scores_and_losses(X_new, output.detach().numpy())
        data['losses'].append(new_losses)
        data['scores'].append(new_scores)

        if len(X_low) <= 0:
            continue

        # Get feature and latent vectors of original and counterfactual images
        mask = (orig_scores < COMP_THRESH) & (new_scores > COMP_THRESH)
        ori_imgs, new_imgs = X[mask], X_new[mask]
        ori_latent = estimator.decoder.encode_image(ori_imgs).detach().cpu()
        new_latent = estimator.decoder.encode_image(new_imgs).detach().cpu()
        ori_feature = estimator.model.get_feature_vector(ori_imgs).detach().cpu()
        new_feature = estimator.model.get_feature_vector(new_imgs).detach().cpu()

        # Compute similarity between generated images and originals
        data['latent_similarity'].append(cosine_similarity(new_latent, ori_latent, dim=1).numpy())
        data['feature_similarity'].append(cosine_similarity(new_feature, ori_feature, dim=1).numpy())
        data['perceptual_similarity'].append(lpips_loss(2*ori_imgs.cpu()-1, 2*new_imgs.cpu()-1).detach().numpy().flatten())

        # Visualize generated images
        if args.save_images:
            for idx in range(len(X)):
                if orig_scores[idx] < COMP_THRESH:
                    mod = int_to_mod(y[idx].numpy())[0]
                    img_subdir = os.path.join(img_dir, mod)

                    visualize_img(X[idx])
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(round(orig_scores[idx], 3), fontsize=20)
                    plt.savefig(os.path.join(img_subdir, '{}_orig.png'.format(count[mod])), bbox_inches='tight')

                    visualize_img(X_new[idx])
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(round(new_scores[idx], 3), fontsize=20)
                    plt.savefig(os.path.join(img_subdir, '{}_{}.png'.format(count[mod], name)), bbox_inches='tight')

                    count[mod] += 1
           
        total += len(X)
    labels = np.vstack(labels)

    # Compute average computation time
    data['time'] /= total

    # Stack similarity metrics
    data['latent_similarity'] = np.hstack(data['latent_similarity'])
    data['feature_similarity'] = np.hstack(data['feature_similarity'])
    data['perceptual_similarity'] = np.hstack(data['perceptual_similarity'])

    # Stack reconstruction losses and comptency estimates
    data['losses'] = np.hstack(data['losses'])
    data['scores'] = np.hstack(data['scores'])

    # Save data
    print(data_file)
    pickle.dump(data, open(data_file, 'wb'))
    np.savez(label_file, labels=labels)

if __name__ == "__main__":
    main()
