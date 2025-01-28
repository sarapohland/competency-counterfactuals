import os
import re
import json
import argparse
import numpy as np

from src.counterfactual.utils import *
from src.explanation.utils import display_images

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    elif val in ('s', 'skip', '?', 'unknown'):
        return -1
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
def collect_labels(expl, orig, gen=None, buffer=250):
    # Ask user for explanation correctness
    display_images(expl, orig, gen, buffer)
    choice = input('Is this an accurate explanation for this image pair? ')
    label = strtobool(choice)
    plt.close()
    return label

def main():
    # Get list of image modifications
    if args.modification == 'none':
        eval_mods = mods[1:]
    else:
        eval_mods = [args.modification]

    # Create list of key words for each image modification
    if args.response_type == 'free':
        keywords = {
            'spatial': ['spatial', 'structure', 'object', 'person', 'yellow', '20', \
                            'common', 'familiar', 'seen', 'expected', 'typical', 'unique', \
                            'novel', 'not seen', 'not part of', 'not one of', 'outlier', \
                            'out-of-distribution', 'anomaly'],
            'brightness': ['light', 'bright', 'expos', 'rainbow', 'reflection', 'glare', 'flare'],
            'contrast': ['contrast', 'texture', 'smooth', 'sharp', 'defined', 'rainbow', \
                         'washed', 'highlight', 'shadow'] , 
            'saturation': ['saturat', 'color', 'tone', 'rainbow', 'vibrant', \
                           ' red ', 'orange', 'yellow', 'green', 'blue', 'purple'], 
            'noise': ['nois', 'grain', 'fuzz', 'speckle'],
            'pixelate': ['pixelat', 'resolution', 'compress', 'downscal', 'downsampl', 'block'],
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
    
    img_dir = os.path.join(args.output_dir, 'images')
    expl_dir = os.path.join(args.output_dir, 'explanations')
    for mod in eval_mods:
        # Create/load list to store labels
        img_subdir = os.path.join(img_dir, mod)
        expl_subdir = os.path.join(expl_dir, mod)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        label_file = os.path.join(expl_subdir, f'{args.counterfactual}.npz')
        try:
            labels = np.load(open(label_file, 'rb'))['labels']
            print(f'Loaded {len(labels)} labels for {args.counterfactual} explanation of {mod} modification.')
        except:
            labels = np.array([])
            print(f'Creating new list for {args.counterfactual} explanation of {mod} modification.')

        # Read language model responses
        expl_file = os.path.join(expl_subdir, f'{args.counterfactual}.json')
        with open(expl_file, 'r') as file:
            responses = json.load(file)
    
        pattern = re.compile(r"(" + "|".join(re.escape(k) for k in keywords[mod]) + r")", re.IGNORECASE)
        for idx, response in enumerate(responses):
            if idx < len(labels):
                continue
            else:
                print(f'Explanation {idx} of {args.counterfactual} counterfactual for {mod} modification.')

            # Check if responses contain keywords of corresponding modification
            if pattern.search(response):
                # Open original image
                orig_path = os.path.join(img_subdir, f'{idx}_orig.png')
                orig_img = Image.open(orig_path).convert('RGB')

                # Open generated image
                if args.counterfactual == 'none':
                    gen_img = None
                else:
                    gen_path = os.path.join(img_subdir, f'{idx}_{args.counterfactual}.png')
                    gen_img = Image.open(gen_path).convert('RGB')

                # Query user about explanation
                label = collect_labels(response, orig_img, gen_img)
                labels = np.append(labels, [label])

            else:
                labels = np.append(labels, [0])

            # Dump labels to file
            np.savez(label_file, labels=labels)

def test():
    # Get list of image modifications
    if args.modification == 'none':
        eval_mods = mods
    else:
        eval_mods = [args.modification]

    img_dir = os.path.join(args.output_dir, 'images')
    expl_dir = os.path.join(args.output_dir, 'explanations')
    for mod in eval_mods:
        # Load list of saved labels
        img_subdir = os.path.join(img_dir, mod)
        expl_subdir = os.path.join(expl_dir, mod)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        label_file = os.path.join(expl_subdir, f'{args.counterfactual}.npz')
        labels = np.load(open(label_file, 'rb'))['labels']
        print(f'Loaded {len(labels)} labels for {args.counterfactual} explanation of {mod} modification.')
        print('Current accuracy: ', np.sum(labels) / len(labels))

        # Read language model responses
        expl_file = os.path.join(expl_subdir, f'{args.counterfactual}.json')
        with open(expl_file, 'r') as file:
            responses = json.load(file)
    
        for idx, response in enumerate(responses):
            if idx < args.start_idx:
                continue
            else:
                print(f'Explanation {idx} of {args.counterfactual} counterfactual for {mod} modification: {labels[idx]}.')

            # Open original image
            orig_path = os.path.join(img_subdir, f'{idx}_orig.png')
            orig_img = Image.open(orig_path).convert('RGB')

            # Open generated image
            if args.counterfactual == 'none':
                gen_img = None
            else:
                gen_path = os.path.join(img_subdir, f'{idx}_{args.counterfactual}.png')
                gen_img = Image.open(gen_path).convert('RGB')

            display_images(response, orig_img, gen_img)

            # Ask user whether to correct label
            choice = input('Does this image need to be corrected? ')
            correct = strtobool(choice)
            if correct:
                labels[idx] = 1 - labels[idx]
            plt.close()

        # Dump labels to file
        np.savez(label_file, labels=labels)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('counterfactual', type=str)
    parser.add_argument('--modification', type=str, default='none')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    test() if args.test else main()