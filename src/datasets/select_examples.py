import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from src.datasets.setup_dataloader import setup_loader
from src.analysis.modify import modify_image
from src.counterfactual.utils import *

NUM_EXAMPLES = 100


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def display_image(img, score=None):
    plt.figure()
    plt.imshow(np.swapaxes(np.swapaxes(img, 1, 2), 2, 3)[0,:,:,:])
    if score is not None:
        plt.title('Competency score: {}'.format(np.round(score[0], 2)))
    plt.axis('off')
    plt.show(block=False)

def save_image(img, file):
    # Reformat input image
    np_img = img.numpy()
    np_img = np.squeeze(np_img * 255).astype(np.uint8)
    np_img = np.swapaxes(np.swapaxes(np_img, 0, 1), 1, 2)
    pil_img = Image.fromarray(np_img)

    # Save input image
    pil_img.save(file)

def query_user(data, mod, score=None, estimator=None):
    # Generate modified image (if applicable)
    if mod in ['brightness', 'contrast', 'saturation', 'pixelate', 'noise']:
        assert estimator is not None

        # Display image
        display_image(data)

        # Ask user for property factor
        score = 1.0
        while score > COMP_THRESH:
            try:
                factor = float(input('Select {} factor. '.format(mod)))
                new_img = modify_image(data, mod, factor).float()
            except:
                plt.close()
                print('Not generating an image with modified {}.'.format(mod))
                return False, data
            output = estimator.model(new_img)
            score = estimator.comp_scores(new_img, output.detach().numpy())
            if score > COMP_THRESH:
                print('This image does not result in low levels of competency.')
        plt.close()

        # Ask user whether to save image
        display_image(new_img, score)
        save = strtobool(input('Do you want to save this image? '))
        plt.close()
        if not save:
            retry = strtobool(input('Do you want to try a different factor? '))
            if retry:
                query_user(data, mod, estimator=estimator)
        return save, new_img

    else:
        # Ask user whether to save image
        display_image(data, score)
        save = strtobool(input('Do you want to save this image? '))
        plt.close()
        return save


def main():

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--decoder_dir', type=str, default='None')
    parser.add_argument('--data_type', type=str, default='test')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Create folders to save images
    data_dir = './data/{}/'.format(args.dataset)
    for mod in mods:
        subfolder = os.path.join(data_dir, mod)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # Create data loader
    if args.data_type == 'test':
        dataloader = setup_loader(args.dataset, batch_size=1, test=True)
    elif args.data_type == 'ood':
        dataloader = setup_loader(args.dataset, batch_size=1, ood=True)
    else:
        raise NotImplementedError('Unkown data type `{}`'.format(args.data_type))

    # Load trained competency estimator
    file = os.path.join(args.decoder_dir, 'parce.p')
    estimator = pickle.load(open(file, 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    estimator.set_device(device)

    # Select example images
    for idx, (data, label) in enumerate(dataloader):
        if idx < args.start_idx:
            continue
        print('{} {} set: {}'.format(args.dataset, args.data_type, str(idx)))

        # Get prediction of classification model
        output = estimator.model(data)
        pred = torch.argmax(output, dim=1).cpu()

        # Query user about saving image
        if args.data_type == 'test' and pred == label:
            # Compute competency score of images
            score = estimator.comp_scores(data, output.detach().numpy())

            if score > COMP_THRESH:
                subfolder = os.path.join(data_dir, 'none')
                # num_saved = len(os.listdir(subfolder)) - 1
                # if num_saved < NUM_EXAMPLES:
                # print('{}: {}/{}'.format('none', num_saved, NUM_EXAMPLES))

                # Ask user whether to save unmodified ID image
                if query_user(data, 'none', score=score):
                    # Save image to appropriate folder
                    file = os.path.join(subfolder, '{}.png'.format(str(idx)))
                    save_image(data, file)

                    for mod in ['brightness', 'contrast', 'saturation', 'noise', 'pixelate']:
                        subfolder = os.path.join(data_dir, mod)
                        num_saved = len([f for f in os.listdir(subfolder) if f.endswith('.png')])
                        if num_saved < NUM_EXAMPLES:
                            print('{}: {}/{}'.format(mod, num_saved, NUM_EXAMPLES))

                            # Generate modified versions of this image
                            save, new_img = query_user(data, mod, estimator=estimator)
                            
                            # Save image to appropriate folder
                            if save:
                                file = os.path.join(subfolder, '{}.png'.format(str(idx)))
                                save_image(new_img, file)

        elif args.data_type == 'ood':
            # Compute competency score of images
            score = estimator.comp_scores(data, output.detach().numpy())

            if score < COMP_THRESH:
                subfolder = os.path.join(data_dir, 'spatial')
                num_saved = len([f for f in os.listdir(subfolder) if f.endswith('.png')])
                if num_saved < NUM_EXAMPLES:
                    # Ask user whether to save OOD image
                    if query_user(data, 'spatial', score=score):
                        # Save image to appropriate folder
                        file = os.path.join(subfolder, '{}.png'.format(str(idx)))
                        save_image(data, file)
                else:
                    break
        
        else:
            continue


if __name__=="__main__":
    main()   