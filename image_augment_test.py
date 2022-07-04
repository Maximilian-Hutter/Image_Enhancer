from PIL import Image, ImageChops
import torch
from torchvision.transforms.functional import *
import torchvision.transforms
from data_utils import *
import glob 
root = "C:/Data/NTIRE_test"
label = sorted(glob.glob(root + "/label/*.*"))
image = sorted(glob.glob(root + "/input/*.*"))
aligns = sorted(glob.glob(root + "/align/*.*"))
expo = sorted(glob.glob(root+ "/expo/*.*"))

#img = Image.open(self.imgs[index % len(self.recovered)])
#label = Image.open(label[0])
image = Image.open(image[0])
alignratio = np.load(aligns[0]).astype(np.float32)
label = read_imgdata(label[0], ratio=alignratio)
expo = np.load(expo[0]).astype(np.float32)
label = np.array(label)
label = expo_correct(label, expo, 1)
SIZE = 896
SMALL_SIZE = SIZE / 2
SMALL_SIZE = int(SMALL_SIZE)

label = Image.fromarray((label * 4096).astype(np.uint8))
b,g,r = label.split()
label = Image.merge("RGB", (r, g, b))
#label = label.resize((SIZE,SIZE))

label.show()