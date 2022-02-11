import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis
import os

# Reading image
def readImg_infolder(img, dir):
    #read image
    for list in dir:
        imgLoc.append('Clubbing/' + list)

    #read all image in the list
    for j in imgLoc:
        img2 = cv2.imread(str(j))
        reimage = cv2.resize(img2, (50, 50))
        img.append(reimage)
    return img

# Color feature extraction
#def color_extraction(img):

# -------------------------------------

imgLoc = []
imgList = []
hsv_split = []
h_values = []
color_features = []
values = []

cols = ['h_mean', 's_mean', 'v_mean', 'h_std', 's_std', 'v_std', 'h_skew', 's_skew', 'v_skew', 'h_kurtosis',
            's_kurtosis', 'v_kurtosis']
color_feature = pd.DataFrame(columns=cols)



# image path
path = "Clubbing"

# array of image list
dir_list = os.listdir(path)

# read image
imgList = readImg_infolder(imgList, dir_list)

# convert RGB to HSV and split H S V
for i in imgList:
    hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #hsv_split.extend([h, s, v])

    data = []
    df = pd.DataFrame()

    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    data.extend([h_mean, s_mean, v_mean])


    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    data.extend([h_std, s_std, v_std])  # Second order moment placed in a feature array

    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_skew = h_skewness ** (1. / 3)
    s_skew = s_skewness ** (1. / 3)
    v_skew = v_skewness ** (1. / 3)
    data.extend([h_skew, s_skew, v_skew])

    h_kurtosis = kurtosis(h, axis=None)
    s_kurtosis = kurtosis(s, axis=None)
    v_kurtosis = kurtosis(v, axis=None)
    data.extend([h_kurtosis, s_kurtosis, v_kurtosis])

    data = np.array(data).flatten()
    values.append(data)
    #print(values)
#--------------------------------------------------------

color_feature = pd.DataFrame(values, columns=cols)
print(color_feature)
#color_feature.to_csv('D:\Color_Features.csv')

#------------------ hanggang dito nalang -----------------























