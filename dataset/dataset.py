'''
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import numpy as np
import os
import torchvision
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return root+basename+extension#os.path.join(root, basename,extension)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class train(Dataset):

    def __init__(self, input_transform=None, target_transform=None):
	
        self.images_root = './data/train/Images/'
        self.labels_root = './data/train/Labels/'
               
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')

        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class test(Dataset):

    def __init__(self, input_transform=None, target_transform=None):
	
        self.images_root = './data/test/image/'
        self.labels_root = './data/test/label/'

        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)
