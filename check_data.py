import os

import cv2
import numpy as np

path = "./data/s1_processed/"

listdir = os.listdir(path)

for i in range(len(listdir)):
    file_path = os.path.join(path, listdir[i])
    if i == 5:
        break
    if file_path == "./data/s1_processed/align":
        align_path = file_path
    else:
        print(file_path)
print(align_path)
