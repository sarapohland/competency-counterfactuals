import os
import json
import argparse

from tqdm import tqdm
from PIL import Image
from natsort import natsorted

from src.counterfactual.utils import *
from src.explanation.llama import Llama

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--counterfactual', type=str, default='none')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    # parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Generate descriptions of the scenario
    if args.dataset == 'lunar':
        description = (
            "I trained a CNN for image classification from a set of images obtained from a "
            "simulated lunar environment. The classifier learns to distinguish between "
            "different regions in this environment, such as regions with smooth terrain, "
            "regions with bumpy terrain, regions at the edge of a crater, regions inside "
            "a crater, and regions near a hill."
        )
    elif args.dataset == 'speed':
        description = (
            "I trained a CNN for image classification from a dataset containing speed "
            "limit signs. The classifier learns to distinguish between seven (7) "
            "different speed limits: 30, 50, 60, 70, 80, 100, and 120 km/hr."
        )
    else:
        description = "I trained a CNN for image classification."
    
    description += (
        " In addition to the classification model, I trained a reconstruction-based "
        "competency estimator that estimates the probability that the classifier's "
        "prediction is accurate for a given image."
    )

    # Generate instructions for the language model
    if args.response_type == 'free':
        max_new_tokens = 100
        if args.counterfactual == 'none':
            instructions = (
                " I am going to show you an image for which the classifier is not "
                "confident. In a single sentence, explain what properties of the "
                "image itself might lead to the observed reduction in model confidence."
            )
            query = (
                "Here is the image for which my classifier is not confident. Could you "
                "please try to explain why the model is not confident?"
            )
        else:
            instructions = (
                " I am going to show two images side-by-side. The first (on the left) "
                "is the original image, for which my classifier is not confident. The "
                "second image (on the right) is a similar image, for which my model is "
                "more confident. In a single sentence, explain what properties of the "
                "original image might have led to the observed reduction in model confidence."
            )
            query = (
                "On the left is the image for which my classifier is not confident, and "
                "on the right is a similar image resulting in a higher level of confidence. "
                "Could you please try to explain why the model is less confident for the "
                "first image?"
            )
    
    elif args.response_type == 'mc':
        max_new_tokens = 40
        if args.counterfactual == 'none':
            instructions = (
                " I am going to show you an image for which the classifier is not "
                "confident. There are six potential reasons the model may not be "
                "confident for this particular image: (1) the image contains spatial "
                "anaomlies in particular regions of the image, (2) the image has an "
                "unusual brightness level, (3) the image has an unusual contrast level,"
                " (4) the image has an unusual saturation level, (5) the image is noisy,"
                " or (6) the image has been pixelated. Please be sure to respond with "
                "the number corresponding to your selection and then explain your "
                "response."
            )
            query = (
                "Here is the image for which my classifier is not confident. Please "
                "select one of the following reasons for low model confidence: (1) "
                "spatial, (2) brightness, (3) contrast, (4) saturation, (5) noise, "
                "or (6) pixelation."
            )
        else:
            instructions = (
                " I am going to show two images side-by-side. The first (on the left) "
                "is the original image, for which my classifier is not confident. The "
                "second image (on the right) is a similar image, for which my model is "
                "more confident. There are six potential reasons the model may not be "
                "confident for the original image: (1) the image contains spatial "
                "anaomlies in particular regions of the image, (2) the image has an "
                "unusual brightness level, (3) the image has an unusual contrast level,"
                " (4) the image has an unusual saturation level, (5) the image is noisy,"
                " or (6) the image has been pixelated. Please be sure to respond with "
                "the number corresponding to your selection and then explain your "
                "response."
            )
            query = (
                "On the left is the image for which my classifier is not confident, and "
                "on the right is a similar image resulting in a higher level of confidence. "
                "Please select one of the following reasons for low model confidence: (1) "
                "spatial, (2) brightness, (3) contrast, (4) saturation, (5) noise, "
                "or (6) pixelation."
            )
    
    else:
        raise NotImplementedError('The response type should either be `free` or `mc`.')
    
    instructions = description + instructions

    # Create folder to save results
    expl_dir = os.path.join(args.output_dir, 'explanations')
    for mod in mods[1:]:
        expl_subdir = os.path.join(expl_dir, mod)
        if not os.path.exists(expl_subdir):
            os.makedirs(expl_subdir)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        if not os.path.exists(expl_subdir):
            os.makedirs(expl_subdir)

    # Load Llama model for language explanations
    llama = Llama(max_new_tokens)

    # Generate explanations for each image modification
    img_dir = os.path.join(args.output_dir, 'images')
    for mod in mods[1:]:
        responses = []
        img_subdir = os.path.join(img_dir, mod)
        expl_subdir = os.path.join(expl_dir, mod)
        expl_subdir = os.path.join(expl_subdir, args.response_type)
        expl_file = os.path.join(expl_subdir, f'{args.counterfactual}.json')

        if args.counterfactual == 'none':
            all_images = natsorted([f for f in os.listdir(img_subdir) if 'orig' in f])
            for filename in tqdm(all_images):
                # Open original image
                image_path = os.path.join(img_subdir, filename)
                index, rest = filename.split("_")
                image = Image.open(image_path).convert('RGB')

                # Generate response from Llama and save
                answer = llama.query_model(image, query, instructions)
                responses.append(answer)
 
        else:
            all_images = natsorted([f for f in os.listdir(img_subdir) if args.counterfactual in f])
            for filename in tqdm(all_images):
                # Open original and counterfactual image
                image_path = os.path.join(img_subdir, filename)
                index, rest = filename.split("_")
                method = rest.split(".")[0]
                orig_path = os.path.join(img_subdir, f'{index}_orig.png')
                orig_img = Image.open(orig_path).convert('RGB')
                gen_img = Image.open(image_path).convert('RGB')

                # Save images side-by-side
                width = orig_img.width + gen_img.width
                height = max(orig_img.height, gen_img.height)
                image = Image.new("RGB", (width, height))
                image.paste(orig_img, (0, 0))
                image.paste(gen_img, (orig_img.width, 0))

                # Generate response from Llama and save
                answer = llama.query_model(image, query, instructions)
                responses.append(answer)

        with open(expl_file, 'w') as file:
            print(f'Saving response to {expl_file}')
            json.dump(responses, file, indent=4)

if __name__ == "__main__":
    main()