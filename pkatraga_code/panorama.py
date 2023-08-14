import numpy as np
import cv2

# Import necessary functions
import planarH
from cpselect.cpselect import cpselect
from matchPics import matchPics
from opts import get_opts
from matplotlib import pyplot as plt

# Q4
opts = get_opts()
# left = cv2.imread('../data/pano_left.jpg')
# right = cv2.imread('../data/pano_right.jpg')

left = cv2.imread('../data/left_img_resize.jpg')
right = cv2.imread('../data/right_img_resize.jpg')

#Creating an initial panorama with the left image and an array of zeros the size of the right image
panorama_init = np.hstack((left, np.zeros(right.shape))).astype(np.uint8)

matches, locs1, locs2 = matchPics(right, panorama_init, opts)
H2to1, _ = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)

panorama = planarH.compositeH(H2to1, right, panorama_init)

plt.imshow(panorama)
plt.axis('off')
plt.show()

cv2.imwrite('../data/panorama_new2.png', panorama)

#Ignore. Opposite of above logic. Left image warped onto right

# panorama_init = np.hstack((np.zeros(left.shape), right))

# panorama_init = np.hstack((np.zeros(left.shape), right)).astype(np.uint8)

# matches, locs1, locs2 = matchPics(left, panorama_init, opts)
# H2to1, _ = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)

# # plt.imshow(panorama_init)
# # plt.axis('off')
# # plt.show()

# panorama = planarH.compositeH(H2to1, left, panorama_init)

# plt.imshow(panorama)
# plt.axis('off')
# plt.show()

# cv2.imwrite('../data/panorama_new2.png', panorama)
