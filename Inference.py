import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from models import *
from torchvision import transforms, utils
from PIL import Image
import argparse
import torchvision


parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
parser.add_argument('--modelpath', type=str, default="weights/1499HDR.pth", help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='C:/Users/Redmi/Downloads/', help=("Path to image folder"))
parser.add_argument('--imagename', type=str, default='test.png', help=("filename of the image"))
parser.add_argument('--gpu_mode', type=bool, default=True, help=('enable cuda'))
parser.add_argument('--channels',type=int, default=3, help='number of channels R,G,B for img / number of input dimensions 3 times 2dConv for img')
parser.add_argument('--filters',type=int, default=64, help='number of filters')

      

opt = parser.parse_args()

PATH = opt.modelpath
imagepath = (opt.inferencepath + opt.imagename)
image = Image.open(imagepath)

image = image.resize((448,448))
lightmap = lightmap_gen(image)


lightmap = pil_to_tensor(lightmap)
image = pil_to_tensor(image)

image = image.unsqueeze(0)
lightmap = lightmap.unsqueeze(0)

image= image.to(torch.float32)
lightmap = lightmap.to(torch.float32)

model=NeuralNet(3, "PReLU", opt.filters)

if opt.gpu_mode == False:
    device = torch.device('cpu')

if opt.gpu_mode:
    	device = torch.device('cuda:0')

model.load_state_dict(torch.load(PATH,map_location=device))

model.eval()

out = model(image, lightmap)
utils.save_image(out,'results/HDR.png')