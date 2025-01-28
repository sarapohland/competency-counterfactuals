import os
import argparse
import numpy as np

from PIL import Image
from pathlib import Path

# from natsort import natsorted

from src.counterfactual.utils import *

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('data_path', type=str)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    args = parser.parse_args()

    # Get labels of unmodified data
    file = os.path.join(args.data_path, 'dataset.npz')
    dataset = np.load(open(file, 'rb'))
    test_labels = dataset['test_labels']
    ood_labels = dataset['ood_labels']

    # Create dictionary to store data and labels
    data = {mod: [] for mod in mods}
    mod_labels = {mod: None for mod in mods}
    class_labels = {mod: [] for mod in mods}

    for mod in mods:
        # Read images from folder
        idxs = []
        folder = os.path.join(args.data_path, mod)
        for file in os.listdir(folder):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = Image.open(os.path.join(folder, file))
                height = args.height if args.height is not None else img.size[1]
                width  = args.width  if args.width  is not None else img.size[0]
                img = img.resize((width, height))
                img = np.array(img)
                img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
                img = np.expand_dims(img, axis=0)
                data[mod].append(img)
                idxs.append(int(Path(file).stem))

        # Convert data to numpy array
        data[mod] = np.vstack(data[mod])
        data[mod] = np.float16(data[mod] / 255)
        mod_labels[mod] = np.array([mod] * len(data[mod]))[:,None]
        if mod == 'spatial':
            class_labels[mod] = ood_labels[idxs]
        else:
            class_labels[mod] = test_labels[idxs]
        print(mod, np.shape(data[mod]), np.shape(mod_labels[mod]), np.shape(class_labels[mod]))

    # Save all data to a npz file
    all_data = np.vstack([data[mod] for mod in mods])
    all_mods = np.vstack([mod_labels[mod] for mod in mods])
    all_lbls = np.vstack([class_labels[mod] for mod in mods])
    print('all', np.shape(all_data), np.shape(all_mods), np.shape(all_lbls))
    file = os.path.join(args.data_path, 'examples.npz')
    np.savez(file, data=all_data, mods=all_mods, labels=all_lbls)


if __name__ == "__main__":
    main()