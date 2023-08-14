import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
import planarH
from matchPics import matchPics as mP

# Q2.2.4

def warpImage(opts):
    cover_img = cv2.imread('../data/cv_cover.jpg')
    desk_img = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    #Resizing HP cover to match CV cover
    hp_cover_resize = cv2.resize(hp_cover, dsize=(cover_img.shape[1], cover_img.shape[0]))
    matches, locs1, locs2 = mP(cover_img, desk_img, opts)
    H2to1, inliers = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
    # print(inliers)

    hp_composite = planarH.compositeH(H2to1, hp_cover_resize, desk_img)

    #Saving image without resize
    # cv2.imwrite('../data/hp_warp_noresize.png', hp_composite)

    #Saving image after resize
    cv2.imwrite('../data/hp_warp_resized_1_10tol_test.png', hp_composite)

    pass

if __name__ == "__main__":  

    opts = get_opts()
    warpImage(opts)


