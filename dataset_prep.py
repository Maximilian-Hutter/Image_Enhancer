import os 
from PIL import Image
import numpy as np
path ="C:/Users/Redmi/Downloads/Train0000_0210/Train0000_0210/"

files = sorted(os.listdir(path))
substring = ["short", "long", "exposures"]
for count, i in enumerate(files):
    for string in substring:
        if not(string in i):
 
            if "gt" in i:
                img = Image.open(path + i)
                img.save("D:/Data/NTIRE/label/" + i + ".png")
            elif "medium" in i:
                img = Image.open(path + i)
                img.save("D:/Data/NTIRE/input/" + i + ".png")
            elif "align" in i:
                np_in = np.load(path + i)
                np.save("D:/Data/NTIRE/align/" + i ,np_in)

    print("File {} of {}".format(count, len(files)))
