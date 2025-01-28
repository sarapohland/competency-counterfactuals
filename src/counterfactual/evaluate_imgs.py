import os
import pickle
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from tabulate import tabulate
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from src.counterfactual.utils import *
from src.utils.visualize import pil_to_tensor

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    args = parser.parse_args()

    # Define distance functions to measure realism of generated images
    fid = FrechetInceptionDistance(reset_real_features=False, normalize=True, feature=64)
    kid = KernelInceptionDistance( reset_real_features=False, normalize=True, feature=64, subset_size=50)

    # Create dictionaries to store data
    FID = {'Method': methods}
    KID = {'Method': methods}

    # Define real images from calibration set
    real_imgs = []
    file = os.path.join(args.decoder_dir, 'parce.p')
    estimator = pickle.load(open(file, 'rb'))
    for X, _ in estimator.dataloader:
        real_imgs.append(X)
    real_imgs = torch.vstack(real_imgs)
    real_imgs = (real_imgs * 255).type(torch.uint8)
    fid.update(real_imgs, real=True)
    kid.update(real_imgs, real=True)

    # Compute realism scores for each counterfactual method
    img_dir = os.path.join(args.output_dir, 'images')
    for mod in mods[1:]:
        mod_dir = os.path.join(img_dir, mod)
        all_fid, all_kid = [], []
        for method in methods:
            # Gather all generated counterfactual images
            gen_imgs = []
            for filename in [f for f in os.listdir(mod_dir) if method in f]:
                img_path = os.path.join(mod_dir, filename)
                pil_img = Image.open(img_path).convert("RGB")
                gen_imgs.append(pil_to_tensor(pil_img).float())
            gen_imgs = torch.vstack(gen_imgs)
            gen_imgs = (gen_imgs * 255).type(torch.uint8)

            # Compute FID score
            fid.reset()
            fid.update(gen_imgs, real=False)
            fid_score = fid.compute()
            all_fid.append(fid_score.item())

            # Compute KID score
            kid.reset()
            kid.update(gen_imgs, real=False)
            kid_score, _ = kid.compute()
            all_kid.append(kid_score.item())

            print(mod, method, fid_score.item(), kid_score.item())
        FID[f"{mod.capitalize()}"] = all_fid
        KID[f"{mod.capitalize()}"] = all_kid

    # Create tables comparing realism of generated images
    results_dir = os.path.join(args.output_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create Frechet Inception Distance (FID) table
    df = pd.DataFrame(FID)
    save_file = os.path.join(results_dir, 'fid.csv')
    df.to_csv(save_file, index=False)
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

    # Create Kernel Inception Distance (KID) table
    df = pd.DataFrame(KID)
    save_file = os.path.join(results_dir, 'kid.csv')
    df.to_csv(save_file, index=False)
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == "__main__":
    main()
