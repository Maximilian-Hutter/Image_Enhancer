import os
from time import sleep 
from PIL import Image
import numpy as np
import torch.utils.data as data
import cv2

path ="D:/Data/NTIRE/"

files = sorted(os.listdir(path))

alignratio = np.load(path + "align/" + "0000_alignratio.npy").astype(np.float32)
img = util.read_imgdata(path + "label/" + "0000_gt.png.png", ratio=alignratio)

img.show()