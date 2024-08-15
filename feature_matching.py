# Import Necessary Libraries
import cv2
import numpy as np
from itertools import compress


# Define a Function to Perform Feature Matching
def FeatureMatching(keypoints1, descriptors1, keypoints2, descrptors2):
    
    """
    Match feature points from two images
    
    @param {tuple} keypoints1 - a series of keypoint(type: cv2.KeyPoint) from the first image
    @param {numpy.ndarray} descriptors1 - a 2d array of descriptors from the first image
    @param {tuple} keypoints2 - a series of keypoint(type: cv2.KeyPoint) from the second image
    @param {numpy.ndarray} descrptors2 - a 2d array of descriptors from the second image

    @return {cv2.DMatch} a list of DMatch objects
    """
    
    # Define Parameters for Flann Matching
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    search_params = dict(checks = 50)

    # Create a Flann Based Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching from 1 to 2
    if(descriptors1 is not None and len(descriptors1)>2 and descrptors2 is not None and len(descrptors2)>2):

        matches_1_2 = flann.knnMatch(descriptors1, descrptors2, k = 2)
    else:
        good_matches = []
        return good_matches
    
    # Matching from 1 to 2
    if(descriptors1 is not None and len(descriptors1)>2 and descrptors2 is not None and len(descrptors2)>2):

        matches_2_1 = flann.knnMatch(descrptors2, descriptors1, k = 2)

    else:
        good_matches = []
        return good_matches

    # ------ Some of matches desc1->desc2 or desc2->desc1 are tuple (length 0 or length 1) which means not pair... --
    # ------ Code block below detect such cases and save in the list for deleting -----------------------------------
    matches_1_2 = list(matches_1_2)
    matches_1_2 = [value for value in matches_1_2 if len(value) > 1]

    matches_2_1 = list(matches_2_1)
    matches_2_1 = [value for value in matches_2_1 if len(value) > 1]
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------

    # Ratio Test 1 -> 2
    keep_matches_1 = []
    for (m, n) in matches_1_2:
        if (m.distance / n.distance) < 0.7:
            keep_matches_1.append(m)

    # Ratio Test 2 -> 1
    keep_matches_2 = []
    for (m, n) in matches_2_1:
        if (m.distance / n.distance) < 0.7:
            keep_matches_2.append(m)

    # Bi-directional Check
    bi_direction_matches = []
    for m in keep_matches_1:
        for n in keep_matches_2:
            if (m.queryIdx == n.trainIdx) and (m.trainIdx == n.queryIdx):
                bi_direction_matches.append(m)

    # Do RANSAC to filter out outliers
    if len(bi_direction_matches)>10:
        src_pts = np.float32([[keypoints1[m.queryIdx][0], keypoints1[m.queryIdx][1]] for m in bi_direction_matches])
        dst_pts = np.float32([[keypoints2[m.trainIdx][0], keypoints2[m.trainIdx][1]] for m in bi_direction_matches])
        _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1)
        matchesMask = mask.ravel().tolist()
        good_matches = list(compress(bi_direction_matches, matchesMask))
    else:
        good_matches = bi_direction_matches

    # Return the Matches
    return good_matches