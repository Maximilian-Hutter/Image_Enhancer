import os
import random
from PIL import Image
import numpy as np

path ="C:/Data/NTIRE21_HDR_valid_input/"


files = sorted(os.listdir(path))
substring = ["short", "long"]
if __name__ == '__main__':
    for count, i in enumerate(files):
        for string in substring:
            if not(string in i):
                
                if "medium" in i:
                    img = Image.open(path + i, "r")
                    img.save("C:/Data/NTIRE_valid/input/" + i)
                elif "expo" in i:
                    np_in = np.load(path + i, "r")
                    np.save("C:/Data/NTIRE_valid/expo/" + i,np_in)

        print("File {} of {}".format(count, len(files)))