from numpy import byte, dtype
import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from models import *
from torchvision import transforms, utils
from PIL import Image
import argparse
import torchvision
from torchvision.transforms.functional import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
parser.add_argument('--modelpath', type=str, default='weights/PReLUHDR.pth', help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='C:/Data/DIV2K_train_HR/DIV2K_train_HR/', help=("Path to image folder"))
parser.add_argument('--imagename', type=str, default='0010.png', help=("filename of the image"))
parser.add_argument('--hr_height', type=int, default= 320, help='high res. image height')
parser.add_argument('--hr_width', type=int, default= 480, help='high res. image width')
parser.add_argument('--gpu_mode', type=bool, default=False, help=('enable cuda'))
parser.add_argument('--channels',type=int, default=3, help='number of channels R,G,B for img / number of input dimensions 3 times 2dConv for img')
parser.add_argument('--n_resblock', type=int, default=7, help='number of Res Blocks')
parser.add_argument('--upsample', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--imgchannels', type=int, default=3, help=("set the channels of the Image (default = RGBD -> 4)"))
parser.add_argument('--filters', type=int, default=64, help="set number of filters")
parser.add_argument('--activation', type=str, default="PReLU", help=("set activation function"))
    
def create_input(img):

    colorjitter = torchvision.transforms.ColorJitter(0.3,0.3,0.4,0.01)
    blur = torchvision.transforms.GaussianBlur(5, 1)
    img = colorjitter(img)
    img = blur(img)
    img = adjust_saturation(img,0.8)
    img =autocontrast(img)
    return img

transform = transforms.Compose([
        transforms.PILToTensor()
        ])

opt = parser.parse_args()

PATH = opt.modelpath
imagepath = (opt.inferencepath + opt.imagename)
image = Image.open(imagepath)

if opt.gpu_mode == False:
    image = create_input(image)

if opt.gpu_mode:
    image = create_input(image).cuda()
image = image.resize((448,448))
image = transform(image)

lightmap = lightmap_gen(image)
image = image.float()
lightmap = lightmap.float()
image = image.unsqueeze(0)
lightmap = lightmap.unsqueeze(0)

#print(image.shape)
#print(lightmap.shape)

#utils.save_image(image,'results/SDR.png')
model = NeuralNet(in_features=opt.imgchannels,activation_function=opt.activation, filters=opt.filters)

if opt.gpu_mode == False:
    device = torch.device('cpu')

if opt.gpu_mode:
    	device = torch.device('cuda:0')

model.load_state_dict(torch.load(PATH,map_location=device))

model.eval()

out = model(image, lightmap)
utils.save_image(out,'results/HDR.png')