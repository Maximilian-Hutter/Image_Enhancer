from torchvision.transforms.functional import *

def lightmap_gen(img):

    img = equalize(img)
    img = posterize(img,3)
    lightmap = autocontrast(img)

    return lightmap

def create_input(img):

    colorjitter = torchvision.transforms.ColorJitter(0.4,0.4,0.4,0.01)
    blur = torchvision.transforms.GaussianBlur(5, 1)
    img = colorjitter(img)
    img = blur(img)
    img = adjust_saturation(img,0.9)
    return img
