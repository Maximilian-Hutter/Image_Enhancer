from numpy.random.mtrand import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import os
import numpy as np
from data_augmentation import *
import torchvision

class ImageDataset(Dataset):
    def __init__(self, root, augmentation=0):

        self.augmentation = augmentation

        #self.imgs = sorted(glob.glob(root + "/*.*"))
        self.label = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):   # get images to dataloader
        #img = Image.open(self.imgs[index % len(self.recovered)])
        label = Image.open(self.label[index % len(self.label)])

        img = create_input(label)
        lightmap = lightmap_gen(img)

        imgs = {"img": img,"lightmap":lightmap, "label": label}   # create imgs dictionary

        return imgs

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.label)