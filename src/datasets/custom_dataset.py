import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from src.counterfactual.utils import *

from src.utils.segment import mask_images
from src.analysis.modify import modify_images


class CustomDataset(Dataset):
    def __init__(self, data_dir, key):
        # Load saved data
        file = os.path.join(data_dir, 'dataset.npz')
        dataset = np.load(open(file, 'rb'))
        self.data = dataset[key + '_data']
        self.labels = dataset[key + '_labels']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:])
        return data, labels

class SegmentedDataset(Dataset):
    def __init__(self, data_dir, key, params):
        # Load saved data
        file = os.path.join(data_dir, 'dataset.npz')
        dataset = np.load(open(file, 'rb'))
        orig_data = dataset[key + '_data']

        # Segment and mask images
        self.data, self.labels = mask_images(orig_data, params)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:,:,:]).float()
        return data, labels

class ModifiedDataset(Dataset):
    def __init__(self, data_dir, key, params):
        # Load saved data
        file = os.path.join(data_dir, 'dataset.npz')
        dataset = np.load(open(file, 'rb'))
        orig_data = dataset[key + '_data']
        self.labels = dataset[key + '_labels']

        # Modify image properties
        self.data = modify_images(orig_data, params)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:])
        return data, labels

class ExampleDataset(Dataset):
    def __init__(self, data_dir):
        # Load saved data
        file = os.path.join(data_dir, 'examples.npz')
        dataset = np.load(open(file, 'rb'))
        self.data = dataset['data']
        self.labels = mod_to_int(dataset['mods'])[:,None]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:,:,:]).float()
        labels = torch.from_numpy(self.labels[idx,:])
        return data, labels