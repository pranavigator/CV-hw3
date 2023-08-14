import numpy as np
import cv2
import multiprocessing
from matplotlib import pyplot as plt

#Import necessary functions
from matchPics import matchPics
import planarH
from opts import get_opts

from helper import loadVid

#Write script for Q3.1
def vidHomography(args):
    i, ar_frame, cv_cover, book_frame, opts = args

    print("Frame num:", i)

    matches, locs1, locs2 = matchPics(cv_cover, book_frame, opts)

    H2to1, _ = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)

    composite_frame = planarH.compositeH(H2to1, ar_frame, book_frame)

    return composite_frame

ar_source = loadVid('../data/ar_source.mov')
book_vid = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

height_cover = cv_cover.shape[0]
width_cover = cv_cover.shape[1]

height_ar = ar_source.shape[1]
width_ar = ar_source.shape[2]

height_crop = np.abs(height_ar - height_cover)
width_crop = np.abs(width_ar - width_cover)

width_crop_start = np.abs((width_ar - width_cover)//2)
# print(cv_cover.shape)
# print(ar_source.shape)

#ar_source.shape[2]/5
new_frame = np.zeros((ar_source.shape[0], height_cover, width_cover, ar_source.shape[3])).astype(np.uint8)

# print(new_frame.shape)

for i in range(ar_source.shape[0]):
    frame = ar_source[i, :, :, :]
    frame_h_resize = cv2.resize(frame, dsize = (frame.shape[1], height_cover))
    # print(frame_h_resize.shape)
    new_frame[i, :, :, :] = frame_h_resize[:, width_crop_start:width_crop_start+width_cover, :]

# out = cv2.VideoWriter('cropped_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (new_frame.shape[0], new_frame.shape[1]))


# for i in range(new_frame.shape[0]):
#     # plt.imshow(ar_source[:, :, i,:])
#     cv2.imshow('frame:', new_frame[i, :, :,:])
#     cv2.waitKey(1)
    # plt.show()
# out.release()

opts = get_opts()
# args = []

# for i in range(2):
#     args.append([i, new_frame[i, :, :, :], cv_cover, book_vid[i, :, :, :], opts])

# n_workers = multiprocessing.cpu_count()
# p = multiprocessing.Pool(processes = n_workers)
# composite_vid = p.map(vidHomography, args)
# p.close()
# p.join()

composite_vid = np.zeros((new_frame.shape[0], book_vid.shape[1], book_vid.shape[2], book_vid.shape[3]))

vid = cv2.VideoWriter('../data/finalvid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                       (composite_vid.shape[2], composite_vid.shape[1]))
for i in range(new_frame.shape[0]):
    args = [i, new_frame[i, :, :, :], cv_cover, book_vid[i, :, :, :], opts]
    composite_frame = vidHomography(args).astype(np.uint8)
    # composite_vid.append(composite_frame)
    vid.write(composite_frame)
    # cv2.imshow('frame:', composite_frame)
    # cv2.waitKey(10)
# args = [435, new_frame[435, :, :, :], cv_cover, book_vid[435, :, :, :], opts]
# composite_frame = vidHomography(args).astype(np.uint8)

# composite_vid = np.array(composite_vid)

vid.release()
# print(composite_vid.shape[0])
# for i in range(composite_vid.shape[0]):

    





