# Import Necessary Modules
from feature_points import FeaturePoints
from feature_matching import FeatureMatching

# Import Necessary Libraries
import numpy as np
import cv2


# Define a Function to Extract Feature Points
def FeatureExtraction(left_img, right_img, camera_param):
    
    """
    Extract the features in the time frame and reconstruct the feature points

    @param {numpy.ndarray} left_img - a image read by cv2.imread
    @param {numpy.ndarray} right_img - a image read by cv2.imread
    @param {dict} camera_param - camera parameter, cosisting of focal length and baseline 

    @return {Frame class} a Frame consists of kps, descs, and 3d coordinate
    """

    # Create an ORB Object to Extract Keypoints
    orb = cv2.ORB_create(3000)

    # Find the Keypoints of Left and Right Images
    left_keypoints, left_desc = orb.detectAndCompute(left_img, None) 
    right_keypoints, right_desc = orb.detectAndCompute(right_img, None)

    # Convert keypoints to ndarray
    left_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in left_keypoints])
    right_keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in right_keypoints])

    # Match the Left and Right Keypoints
    matches = FeatureMatching(left_keypoints, left_desc, right_keypoints, right_desc)

    # Create Feature Points Object
    feature_points = FeaturePoints()
    feature_points.left_pts = np.empty([0, 2])
    feature_points.left_descriptors = np.empty([0, left_desc.shape[1]], dtype = np.uint8)
    feature_points.right_pts = np.empty([0, 2])
    feature_points.right_descriptors = np.empty([0, right_desc.shape[1]], dtype = np.uint8)
    feature_points.disparity = np.empty([0, 1])
    feature_points.depth = np.empty([0, 1])
    feature_points.pt3ds = np.empty([0, 3])

    # For all Matches
    for m in matches:

        # Get the Indices of Matches
        left_index = m.queryIdx
        right_index = m.trainIdx

        # Compute Feature Points and Disparity
        if (abs(left_keypoints[left_index][1] - right_keypoints[right_index][1]) > 1e-9):
            continue
        disparity = left_keypoints[left_index][0] - right_keypoints[right_index][0]
        if (disparity == 0):
            continue
    
        f_result = np.array([[left_keypoints[left_index][0]], [left_keypoints[left_index][1]], [1]]).T @ camera_param['fundamental'] @ np.array([[right_keypoints[right_index][0]], [right_keypoints[right_index][1]], [1]])
        
        # Save the Coordinates of Feature point of Left Image
        feature_points.left_pts = np.vstack([feature_points.left_pts, [left_keypoints[left_index][0], left_keypoints[left_index][1]]])
        feature_points.left_descriptors = np.vstack([feature_points.left_descriptors, left_desc[left_index]])
            
        # Save the Coordinates of Feature point of Right Image
        feature_points.right_pts = np.vstack([feature_points.right_pts, [right_keypoints[right_index][0], right_keypoints[right_index][1]]])
        feature_points.right_descriptors = np.vstack([feature_points.right_descriptors, right_desc[right_index]])

        # Read the Baseline and Focal Length
        baseline = camera_param['baseline']
        focal_length = camera_param['focal_length']

        # Calculate and Store Disparity and Depth
        depth = baseline * focal_length / disparity
        feature_points.disparity = np.vstack([feature_points.disparity, disparity])
        feature_points.depth = np.vstack([feature_points.depth, depth])

        # Compute 3D coordinates in Camera Frame            
        pt2d_z = np.array([depth * left_keypoints[left_index][0], depth * left_keypoints[left_index][1], depth])   # [z * u, z * v, z]
        pt3d = np.linalg.pinv(camera_param['left_projection']) @ pt2d_z
        # pt3d_cv = cv2.triangulatePoints(camera_param['left_projection'], camera_param['right_projection'], left_keypoints[left_index], right_keypoints[right_index])
        # pt3d_cv = pt3d_cv[0:3] / pt3d_cv[3]
        feature_points.pt3ds = np.vstack([feature_points.pt3ds, pt3d[0: 3]])

    # Determine Number of Feature Points
    feature_points.num_fp = int(feature_points.left_pts.shape[0])
    # print('Number of matches', feature_points.num_fp)

    # Return the Feature Points
    return feature_points
