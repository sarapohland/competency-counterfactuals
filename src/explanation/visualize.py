import os
import json
import argparse
import numpy as np

from src.counterfactual.utils import *
from src.explanation.utils import display_images

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('counterfactual', type=str)
    parser.add_argument('--modification', type=str, default='none')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    # Check image modification type
    assert args.modification in mods[1:]
    mod = args.modification

    # Get locations of images and explanations
    img_dir = os.path.join(args.output_dir, 'images')
    expl_dir = os.path.join(args.output_dir, 'explanations')
    img_subdir = os.path.join(img_dir, mod)
    expl_subdir = os.path.join(expl_dir, mod)
    expl_subdir = os.path.join(expl_subdir, args.response_type)

    # Read language model response
    expl_file = os.path.join(expl_subdir, f'{args.counterfactual}.json')
    with open(expl_file, 'r') as file:
        responses = json.load(file)
    response = responses[args.index]

    # Open original image
    orig_path = os.path.join(img_subdir, f'{args.index}_orig.png')
    orig_img = Image.open(orig_path).convert('RGB')

    # Open generated image
    if args.counterfactual == 'none':
        gen_img = None
    else:
        gen_path = os.path.join(img_subdir, f'{args.index}_{args.counterfactual}.png')
        gen_img = Image.open(gen_path).convert('RGB')

    # Save figure of image(s) with explanation
    save_dir = os.path.join(args.output_dir, 'examples')
    save_subdir = os.path.join(save_dir, args.modification)
    save_subdir = os.path.join(save_subdir, args.response_type)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    save_file = os.path.join(save_subdir, f'{args.index}_{args.counterfactual}.png')    
    display_images(response, orig_img, gen_img, file=save_file)

    
if __name__ == "__main__":
    main()