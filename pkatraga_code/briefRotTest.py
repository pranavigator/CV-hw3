import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy.ndimage as sc_img
import matplotlib.pyplot as plt
from helper import plotMatches

#Q2.1.6

def rotTest(opts):

    #Read the image and convert to grayscale, if necessary
    img1 = cv2.imread('../data/cv_cover.jpg')
    # img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    hist_all = np.zeros(37)
    for i in range(37):
        
        #Rotate Image
        print("Iteration:", i)
        img_test = sc_img.rotate(img1, angle=10*i)
        #Compute features, descriptors and Match features
        matches, _, _ = matchPics(img1, img_test, opts)
        #Update histogram
        count = len(matches[:, 0])
        # print(hist)
        hist_all[i] = count

    #Display histogram
    print(hist_all)
    plt.bar(np.arange(0,370,10), hist_all, width=5)
    plt.xlabel('Angle of Rotation')
    plt.ylabel('Match Count')
    plt.show()  

    #Calculating individual angles for display
    # i = 34
    # img_test = sc_img.rotate(img1, angle=10*i)
    # #Compute features, descriptors and Match features
    # matches, locs1, locs2 = matchPics(img1, img_test, opts)
    # plotMatches(img1, img_test, matches, locs1, locs2)
if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
