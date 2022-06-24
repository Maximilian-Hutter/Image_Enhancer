from PIL import Image
import torch
from torchvision.transforms.functional import *
import torchvision.transforms

img = Image.open("C:/Users/Redmi/Downloads/test.jpg")
img.show()

def color_extract(img):

    img = equalize(img)
    img = posterize(img,3)
    img = autocontrast(img)
    #img = adjust_brightness(img, 0.5)

    return img

def create_input(img):

    colorjitter = torchvision.transforms.ColorJitter(0.3,0.3,0.4,0.01)
    blur = torchvision.transforms.GaussianBlur(5, 1)
    img = colorjitter(img)
    img = blur(img)
    img = adjust_saturation(img,0.8)
    return img

#lightmap = color_extract(img)
#lightmap.show()

img = create_input(img)
img.show()
lightmap = color_extract(img)
lightmap.show()