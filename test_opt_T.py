# Import Necessary Modules
from feature_extraction import FeatureExtraction
from read_camera_param import ReadCameraParam
from frame_matching import FrameMatching
from scipy.stats import entropy
from perform_yolo import PerformYolo

# Import Necessary Libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def project_points(T, points):
    """ Project 3D points using the transformation matrix T. """
    projected = T @ points.T  # Apply transformation
    projected /= projected[2]  # Perspective divide (simplified)
    return projected[:2].T  # Return only x, y

def compute_error(T, world_points, image_points):
    """ Compute reprojection error. """
    predicted_points = project_points(T, world_points)
    return np.linalg.norm(predicted_points - image_points[:, :2], axis=1).sum()  # Sum of Euclidean distances


def compute_gradient(T, world_points, image_points, epsilon=1e-6):
    """ Compute numerical gradient of the error function with respect to transformation matrix T. """
    grad = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T_plus = T.copy()
            T_plus[i, j] += epsilon
            error_plus = compute_error(T_plus, world_points, image_points)
            
            T_minus = T.copy()
            T_minus[i, j] -= epsilon
            error_minus = compute_error(T_minus, world_points, image_points)
            
            grad[i, j] = (error_plus - error_minus) / (2 * epsilon)
    return grad


def gradient_descent(T, world_points, image_points, learning_rate=10, iterations=500000):
    """ Perform gradient descent to minimize the error function. """
    for _ in range(iterations):
        grad = compute_gradient(T, world_points, image_points)
        T -= learning_rate * grad  # Update rule
        print(f"Current error: {compute_error(T, world_points, image_points)}")
    return T




# Define Main Function
if __name__ == "__main__":

        # Get the Folders for Left & Right Stereo Images
    left_images_folder = 'Dataset_3/Left_Images/'
    right_images_folder = 'Dataset_3/Right_Images/'

    # Get the Images Path list
    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)

    # Sort Imgae Paths by Name
    left_images = sorted(left_images, key=lambda x:int(x.split('.')[0]))
    right_images = sorted(right_images, key=lambda x:int(x.split('.')[0]))

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]
    
    
    file_path = './Dataset_3/true_T.txt'
    true_T = np.loadtxt(file_path, dtype=np.float64)


    # For the Left and Right Images Dataset
    for ind in range(len(left_images)):

        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])


        next_left_image = cv2.imread(left_images[ind+1])
        next_right_image = cv2.imread(right_images[ind+1])

        left_boxes, right_boxes = PerformYolo(next_left_image, next_right_image)

        camera_param = ReadCameraParam('./Dataset_3/calib.txt')
        feature_points = FeatureExtraction(left_image, right_image, camera_param)
        next_feature_points = FeatureExtraction(next_left_image, next_right_image, camera_param)


        paired_features = FrameMatching(feature_points, next_feature_points)


        predicted_list = []

        for ind, pt3d in enumerate(paired_features[:, 2:]):
            
            temp = np.vstack((true_T[ind].reshape(3, 4), [0, 0, 0, 1])) @ np.hstack((pt3d, [1]))
            temp_2 = np.linalg.inv(np.vstack((true_T[ind+1].reshape(3, 4), [0, 0, 0, 1])))@temp
        
            M = camera_param['left_projection']

            projected_homogeneous = M@temp_2

            x_prime = projected_homogeneous[0] / projected_homogeneous[2]
            y_prime = projected_homogeneous[1] / projected_homogeneous[2]

            if np.isnan(x_prime)==False or np.isnan(y_prime)==False:
                predicted_list.append([x_prime, y_prime])

        predicted_list = np.array(predicted_list)


        current_2ds = []
        previous_2ds = []

        for ind, one_pair in enumerate(paired_features):
            
            previous_2d = M@np.hstack((one_pair[2:], [1]))

            x_prime = previous_2d[0] / previous_2d[2]
            y_prime = previous_2d[1] / previous_2d[2]

            if np.isnan(x_prime)==False or np.isnan(y_prime)==False:
                previous_2ds.append([x_prime, y_prime])
                current_2ds.append(one_pair[:2])


        # Mock data
        # 3D world points (homogeneous coordinates)
        # world_points = np.array([
        #     [1, 2, 3, 1],
        #     [4, 5, 6, 1],
        #     [7, 8, 9, 1]
        # ])

        world_points = np.hstack((paired_features[:, 2:], np.ones((len(paired_features), 1))))

        # Corresponding 2D points in the image (homogeneous coordinates, for simplicity)
        # image_points = np.array([
        #     [0.1, 0.2, 1],
        #     [0.4, 0.5, 1],
        #     [0.7, 0.8, 1]
        # ])
        image_points = np.hstack((current_2ds, np.ones((len(paired_features), 1))))
        



        # Initial guess for the transformation matrix T (4x4)
        T = np.eye(4)

        T_optimized = gradient_descent(T, world_points, image_points)
        print("Optimized Transformation Matrix:")
        print(T_optimized)
        print()