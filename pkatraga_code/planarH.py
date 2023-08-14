import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    A = np.zeros((x1.shape[0]*2, 9))
    for i in range(x1.shape[0]):
        A[2*i, :] = np.array([x2[i, 0], x2[i, 1], 1, 0, 0, 0, -x2[i, 0]*x1[i, 0], -x2[i, 1]*x1[i, 0], -x1[i, 0]])
        A[2*i+1, :] = np.array([0, 0, 0, x2[i, 0], x2[i, 1], 1, -x2[i, 0]*x1[i, 1], -x2[i, 1]*x1[i, 1], -x1[i, 1]])

    # print(A)
    U, S, VT = np.linalg.svd(A.T @ A)

    h = VT[-1,:] #Taking last column of V

    H2to1 = np.reshape(h, (3,3))
    # print(H2to1)

    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    x1_xcentroid = np.sum(x1[0,:])/x1.shape[0]
    x1_ycentroid = np.sum(x1[1,:])/x1.shape[0]
    x1_centroid = np.hstack((x1_xcentroid, x1_ycentroid))

    x2_xcentroid = np.sum(x2[0,:])/x2.shape[0]
    x2_ycentroid = np.sum(x2[1,:])/x2.shape[0]
    x2_centroid = np.hstack((x2_xcentroid, x2_ycentroid))

    #Shift the origin of the points to the centroid
    x1_new = x1 - x1_centroid
    x2_new = x2 - x2_centroid

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    s_x1 = np.sqrt(2) / np.max(np.sqrt(x1_new[:,0]**2 + x1_new[:,1]**2))
    s_x2 = np.sqrt(2) / np.max(np.sqrt(x2_new[:,0]**2 + x2_new[:,1]**2))

    t_x1 = -s_x1 * x1_xcentroid
    t_y1 = -s_x1 * x1_ycentroid

    t_x2 = -s_x2 * x2_xcentroid
    t_y2 = -s_x2 * x2_ycentroid

    #Similarity transform 1
    T1 = np.array([[s_x1, 0, t_x1], [0, s_x1, t_y1], [0, 0, 1]])
    x1_normalized = s_x1 * x1_new
    # print(x1_normalized)

    #Similarity transform 2
    T2 = np.array([[s_x2, 0, t_x2], [0, s_x2, t_y2], [0, 0, 1]])
    x2_normalized = s_x2 * x2_new
    # print(x2_normalized.shape)
    # assert np.linalg.norm(x2_normalized, axis=1).all() <= np.sqrt(2)

    #Compute homography
    H2to1_hat = computeH(x1_normalized, x2_normalized)

    #Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1_hat @ T2

    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    point_len = locs1.shape[0]
    inliers = np.zeros((locs1.shape[0], 1))
    inliers_prevcount = 0
    bestH2to1 = np.zeros((3,3))
    
    #Flipping columns as we receive the locations as (y, x)
    locs1 = np.fliplr(locs1)
    locs2 = np.fliplr(locs2)
    locs2_homo = np.hstack((locs2, np.ones((point_len, 1))))

    for i in range(max_iters):
        #Ran into issues where the number of samples was less than four so set replace=True for those cases
        try:
            point_pairs = np.random.choice(point_len, 4, replace = False)
        
        except ValueError:
            point_pairs = np.random.choice(point_len, 4, replace = True)

        locs1_sample = locs1[point_pairs]
        locs2_sample = locs2[point_pairs]

        H2to1 = computeH_norm(locs1_sample, locs2_sample)
        locs1_pred = (H2to1 @ locs2_homo.T).T
        
        # print("initial preds:", locs1_pred)
        for j in range(locs1_pred.shape[0]):
            if locs1_pred[j,2] != 0:
                locs1_pred[j, :] = locs1_pred[j, :] / locs1_pred[j,2]

        err = np.linalg.norm(locs1 - locs1_pred[:,:2], axis=1)
        # print(err)
        inliers_curr = err <= inlier_tol


        inliers_count = np.sum(inliers_curr)
        # print(inliers_count)
        if inliers_count > inliers_prevcount:
                bestH2to1 = H2to1
                inliers = inliers_curr
                inliers_prevcount = inliers_count
        
        if inliers_count == point_len:
             break
    
    # print("Final:", inliers_prevcount)
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template
    mask = np.ones((template.shape))
    
    #Warp mask by appropriate homography
    mask = cv2.warpPerspective(mask, np.linalg.inv(H2to1), dsize=(img.shape[1], img.shape[0]))
    
    #Warp template by appropriate homography
    warp_template = cv2.warpPerspective(template, np.linalg.inv(H2to1), dsize=(img.shape[1], img.shape[0]))

    #Use mask to combine the warped template and the image
    idx = mask == 0
    idx = idx.astype(int)
    composite_img = idx*img + warp_template

    return composite_img


