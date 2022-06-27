from torchvision.transforms.functional import *
import torchvision

def lightmap_gen(img):

    lightmap = to_grayscale(img)

    return lightmap