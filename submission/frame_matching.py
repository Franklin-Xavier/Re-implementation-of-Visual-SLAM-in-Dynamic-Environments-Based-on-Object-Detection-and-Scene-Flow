# Import Necessary Libraries
from feature_points import FeaturePoints
from feature_matching import FeatureMatching
import numpy as np


# Define a Function to Perform Feature Matching between Previous and Current Features
def FrameMatching(previous_keypoints, current_keypoints):
    
    """
    Match current feature points with previous featrue points

    @param {FeaturePoints class} previous_keypoints - FeaturePoints object from the previous frame
    @param {FeaturePoints class} current_keypoints - FeaturePoints object from the current frame

    @return {numpy.array} a numpy array of shape (n, 5), consists of paired featurepoints in the format [u_current, v_current, x_previous, y_previous, z_previous]
    """
    
    # Match the Left and Right Keypoints
    left_matches = FeatureMatching(previous_keypoints.left_pts, previous_keypoints.left_descriptors, current_keypoints.left_pts, current_keypoints.left_descriptors)
    right_matches = FeatureMatching(previous_keypoints.right_pts, previous_keypoints.right_descriptors, current_keypoints.right_pts, current_keypoints.right_descriptors)

    # Bi-image Check
    good_matches = []
    for i in range(len(left_matches)):
        for j in range(len(right_matches)):
            if (left_matches[i].queryIdx == right_matches[j].queryIdx) and (left_matches[i].trainIdx == right_matches[j].trainIdx):
                good_matches.append(left_matches[i])


    # Return Paired Current 2D Point and Previous 3D Point in a Numpy Array
    paired_feature_points_current_2d_previous_3d = np.empty([0, 5])
    paired_feature_points_current_2d_previous_2d = np.empty([0, 4])
    
    for m in left_matches:
        current_2d = current_keypoints.left_pts[m.trainIdx]
        previous_2d = previous_keypoints.left_pts[m.queryIdx]
        previous_3d = previous_keypoints.pt3ds[m.queryIdx]
        
        # [u_current, v_current, X_previous, Y_previous, Z_previous]
        paired_feature_points_current_2d_previous_3d = np.vstack([paired_feature_points_current_2d_previous_3d, [current_2d[0], current_2d[1], previous_3d[0], previous_3d[1], previous_3d[2]]])
        
        paired_feature_points_current_2d_previous_2d = np.vstack([paired_feature_points_current_2d_previous_2d, 
                                                [current_2d[0], current_2d[1], 
                                                 previous_2d[0], previous_2d[1]]])
        
    # Return the Paired Feature Points
    return paired_feature_points_current_2d_previous_3d, paired_feature_points_current_2d_previous_2d