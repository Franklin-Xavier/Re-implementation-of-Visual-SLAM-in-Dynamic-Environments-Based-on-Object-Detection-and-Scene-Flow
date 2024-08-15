import cv2
import os
import matplotlib.pyplot as plt
from feature_extraction import FeatureExtraction
from read_camera_param import ReadCameraParam
import numpy as np

if __name__ == "__main__":

    # Read Calib File
    camera_param = ReadCameraParam('Dataset_1/calib.txt')


    left_images_folder = 'Dataset_1/Left_Images/'
    right_images_folder = 'Dataset_1/Right_Images/'
    

    left_images = os.listdir(left_images_folder)
    right_images = os.listdir(right_images_folder)

    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]

    left_image = cv2.imread(left_images[0])
    right_image = cv2.imread(right_images[0])

    feature_points = FeatureExtraction(left_image, right_image, camera_param)
    # It seems like for dataset2, the left image is actually the right image
    # feature_points = FeatureExtraction(right_image, left_image)

    # print(feature_points)
    radius = 5

    sorted_indices = np.argsort(feature_points.pt3ds[:, 2])
    feature_points.pt3ds = feature_points.pt3ds[sorted_indices]
    
    feature_points.left_pts = feature_points.left_pts[sorted_indices]
    feature_points.left_descriptors = feature_points.left_descriptors[sorted_indices]
    
    feature_points.right_pts = feature_points.right_pts[sorted_indices]
    feature_points.right_descriptor = feature_points.right_descriptor[sorted_indices]
    # fp_sorted = sorted(feature_points, key=lambda x: x.pt3ds[2])
    
    for i in range(feature_points.num_fp):
        lu = int(feature_points.left_pts[i][0])
        lv = int(feature_points.left_pts[i][1])
        cv2.circle(left_image, (lu, lv), radius=radius, color=(255, 0, 0), thickness=-1)

        ru = int(feature_points.right_pts[i][0])
        rv = int(feature_points.right_pts[i][1])
        cv2.circle(right_image, (ru, rv), radius=radius, color=(255, 0, 0), thickness=-1)

        fig, axes = plt.subplots(1, 2, figsize=(50, 10))
        axes[0].imshow(left_image)
        axes[0].set_title('Left 1')
        axes[1].imshow(right_image)
        axes[1].set_title('Right 2')
        plt.tight_layout()
        plt.text(0, 0, f'({feature_points.pt3ds[i][2]})')
        plt.show()

        cv2.circle(left_image, (lu, lv), radius=radius, color=(0, 0, 255), thickness=-1)
        cv2.circle(right_image, (ru, rv), radius=radius, color=(0, 0, 255), thickness=-1)
