from numpy.random.mtrand import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import os
import numpy as np
from data_augmentation import *
import torchvision
import torchvision.transforms as transforms
from data_utils import read_imgdata, expo_correct

class ImageDataset(Dataset):
    def __init__(self, root, augmentation=0):

        self.augmentation = augmentation

        #self.imgs = sorted(glob.glob(root + "/*.*"))
        self.label = sorted(glob.glob(root + "/label/*.*"))
        self.image = sorted(glob.glob(root + "/input/*.*"))
        self.aligns = sorted(glob.glob(root + "/align/*.*"))
        self.expo = sorted(glob.glob(root+ "/expo/*.*"))

    def __getitem__(self, index):   # get images to dataloader
        #img = Image.open(self.imgs[index % len(self.recovered)])
        label = Image.open(self.label[index % len(self.label)])
        image = Image.open(self.image[index % len(self.image)])
        alignratio = np.load(self.aligns[index % len(self.aligns)]).astype(np.float32)
        label = read_imgdata(self.label[index % len(self.label)], ratio=alignratio)
        expo = np.load(self.expo[index % len(self.label)]).astype(np.float32)

        label = expo_correct(label, expo, 1)

        SIZE = 896
        SMALL_SIZE = SIZE / 2
        SMALL_SIZE = int(SMALL_SIZE)

        label = Image.fromarray((label * 4096).astype(np.uint8))
        b,g,r = label.split()
        label = Image.merge("RGB", (r, g, b))
        img = image.resize((SMALL_SIZE, SMALL_SIZE))
        label = label.resize((SIZE,SIZE))
        
        transform = transforms.Compose([
        transforms.PILToTensor()
        ])

        lightmap = lightmap_gen(img)

        img = img.resize((SMALL_SIZE,SMALL_SIZE))
        lightmap = lightmap.resize((SMALL_SIZE,SMALL_SIZE))
        img = transform(img)
        lightmap = transform(lightmap)
        label = transform(label)
        
        imgs = {"img": img,"lightmap":lightmap, "label": label}   # create imgs dictionary

        return imgs

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.label)