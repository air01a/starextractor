import numpy as np
import sep

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import cv2
from utils import *
from filters import levels
from scipy.interpolate import make_interp_spline
import os

res = []
for file in os.listdir('.'):
    # check only text files
    if file.endswith('.fits'):
        res.append(file)


for file in res:
    img = open_process_fits(file)
    img2 = img.clone()
    img3 = img.clone()


    stretch(img,0.18,0)
    stretch(img2,0.18,1)
    stretch(img3,0.5,2)
    normalize(img)
    normalize(img2)
    normalize(img3)
    #levels(img,3000,1,65536,1,1,1,1)
    save_jpeg(img,file+'_1.jpg')
    save_jpeg(img2,file+'_2.jpg')
    save_jpeg(img3,file+'_3.jpg')