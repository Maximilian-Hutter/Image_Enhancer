import os
import random
from PIL import Image
import numpy as np

path ="C:/Data/NTIRE_unprocessed/"


files = sorted(os.listdir(path))
substring = ["short", "long"]
if __name__ == '__main__':
    for count, i in enumerate(files):
        for string in substring:
            if not(string in i):
                
                if "gt" in i:
                    img = Image.open(path + i, "r")
                    img.save("C:/Data/NTIRE_test/label/" + i)
                elif "medium" in i:
                    img = Image.open(path + i, "r")
                    img.save("C:/Data/NTIRE_test/input/" + i)
                elif "align" in i:
                    np_in = np.load(path + i, "r")
                    np.save("C:/Data/NTIRE_test/align/" + i ,np_in)
                elif "expo" in i:
                    np_in = np.load(path + i, "r")
                    np.save("C:/Data/NTIRE_test/expo/" + i,np_in)

        print("File {} of {}".format(count, len(files)))