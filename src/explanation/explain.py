import os
import json
import argparse

from tqdm import tqdm
from PIL import Image
from natsort import natsorted

from src.utils.visualize import numpy_to_pil, tensor_to_pil
from src.datasets.custom_dataset import ExampleDataset
from src.datasets.setup_dataloader import setup_loader
from src.counterfactual.utils import *
from src.counterfactual.methods import *
from src.explanation.llama import Llama
from src.explanation.utils import *

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--counterfactual', type=str, default='none')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    args = parser.parse_args()

    # Load Llama model for language explanations
    max_new_tokens = get_max_new_tokens(args.response_type)
    try:
        model_name = f"llama_{args.dataset}_{args.counterfactual}"
        llama = Llama(model_name, max_new_tokens)
        expl_dir = os.path.join(args.output_dir, 'explanations_finetune')
    except:
        llama = Llama("unsloth/Llama-3.2-11B-Vision-Instruct", max_new_tokens)
        expl_dir = os.path.join(args.output_dir, 'explanations')

    # Create folder to save results
    for mod in mods[1:]:
        expl_subdir = os.path.join(expl_dir, mod)
        if not os.path.exists(expl_subdir):
            os.makedirs(expl_subdir)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        if not os.path.exists(expl_subdir):
            os.makedirs(expl_subdir)

    # Generate counterfactual images (if applicable)
    if not args.counterfactual == 'none':

        # Load trained competency estimator
        file = os.path.join(args.decoder_dir, 'parce.p')
        estimator = pickle.load(open(file, 'rb'))
        # estimator.set_device(device)

        # Create object for generating counterfactuals (if applicable)
        if args.counterfactual == 'igd':
            gen = ImageGradientDescent(estimator, thresh=COMP_THRESH)
        elif args.counterfactual == 'fgd':
            gen = FeatureGradientDescent(estimator, thresh=COMP_THRESH)
        elif args.counterfactual == 'lgd':
            gen = LatentGradientDescent(estimator, thresh=COMP_THRESH)
        elif args.counterfactual == 'lnn':
            gen = LatentNearestNeighbors(estimator, thresh=COMP_THRESH)

        # Set up data loader for example dataset
        dataloader = setup_loader(args.dataset, batch_size=50, example=True)

        # Generate counterfactuals for each batch
        counterfactuals = []
        for X, y in dataloader:
            # X = X.to(device)

            # Compute losses and competency scores for original images
            output = estimator.model(X).cpu()
            scores = estimator.comp_scores(X, output.detach().numpy())

            # Generate counterfactual images with desired method
            X_new = X.clone()
            X_low = X[scores < COMP_THRESH]
            
            if args.counterfactual == 'reco':
                X_new[scores < COMP_THRESH] = estimator.decoder(X_low)
            else:
                X_new[scores < COMP_THRESH] = gen.get_counterfactual(X_low)
            counterfactuals.append(X_new)
        counterfactuals = torch.vstack(counterfactuals).detach().cpu()

    # Generate example dataset
    dataset = ExampleDataset(os.path.join('./data/', args.dataset))
    images, labels = [], []
    for idx, (data, label) in enumerate(zip(dataset.data, dataset.mods)):
        if args.counterfactual == 'none':
            images.append(numpy_to_pil(data))
            labels.append(label)

        else:
            orig_img = numpy_to_pil(data)
            gen_img = tensor_to_pil(counterfactuals[idx])
            image = combine_images(orig_img, gen_img)
            images.append(image)
            labels.append(label)
    # images = np.stack(images)
    labels = np.array(labels).flatten()
    
    # Get instructions for Llama model
    instructions = get_llama_instructions(args.dataset, args.response_type, args.counterfactual)

    # Generate explanations for each image modification
    for mod in mods[1:]:

        responses = []
        expl_subdir = os.path.join(expl_dir, mod)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        expl_file = os.path.join(expl_subdir, f'{args.counterfactual}.json')

        # Generate response from Llama
        for image in [images[idx] for idx in np.where(labels == mod)[0]]:
            answer = llama.query_model(image, instructions)
            responses.append(answer)

        # Save Llama response
        with open(expl_file, 'w') as file:
            print(f'Saving response to {expl_file}')
            json.dump(responses, file, indent=4)

if __name__ == "__main__":
    main()