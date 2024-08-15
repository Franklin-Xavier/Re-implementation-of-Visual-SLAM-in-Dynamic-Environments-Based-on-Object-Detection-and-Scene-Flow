from read_camera_param import ReadCameraParam
from feature_extraction import FeatureExtraction
from perform_yolo import PerformYolo
from filter_feature_points import FilterFeaturePoints
from frame_matching import FrameMatching
from pose_estimator import PoseEstimator
from draw_trajectory import DrawTrajectory

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def draw_frame_matching(img, paired_kps, pre_T_true, cur_T_true, projection):
    current_kps = paired_kps[:, 0:2].T
    current_kps = np.vstack([current_kps, np.ones(current_kps.shape[1])])

    pre_3dkps = paired_kps[:, 2:5].T
    pre_3dkps = np.vstack([pre_3dkps, np.ones(pre_3dkps.shape[1])])

    pre_T_true = np.vstack([pre_T_true.reshape(3, 4), [0, 0, 0, 1]])
    cur_T_true = np.vstack([cur_T_true.reshape(3, 4), [0, 0, 0, 1]])
    
    temp = pre_T_true @ pre_3dkps
    temp_2 = np.linalg.inv(cur_T_true) @ temp

    projected_homogeneous = projection @ temp_2
    predict_kps = projected_homogeneous[0:2] / projected_homogeneous[2]

    for i in range(predict_kps.shape[1]):
        u_pred = int(predict_kps[0][i])
        v_pred = int(predict_kps[1][i])
        cv2.circle(img, (u_pred, v_pred), radius=3, color=(255, 0, 0), thickness=-1)

        u_ob = int(current_kps[0][i])
        v_ob = int(current_kps[1][i])
        cv2.circle(img, (u_ob, v_ob), radius=3, color=(0, 255, 0), thickness=-1)

    plt.imshow(img)
    plt.show()

DATASET = './Dataset_4'

if __name__ == "__main__":
    
    # Read Calib File
    camera_param = ReadCameraParam(DATASET + '/calib.txt')
    T_true_path = DATASET + '/true_T.txt'
    T_true = np.loadtxt(T_true_path, dtype=np.float64)

    # Get the Folders for Left & Right Stereo Images
    left_images_folder = DATASET + '/Left_Images/'
    right_images_folder = DATASET + '/Right_Images/'

    # Get the Images Path list
    left_images = sorted(os.listdir(left_images_folder))
    right_images = sorted(os.listdir(right_images_folder))

    # Get the Path of Images
    left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images]
    right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images]

    # Read the First Frame of Left and Right Images
    left_image = cv2.imread(left_images[0])
    right_image = cv2.imread(right_images[0])
    previous_feature_points = FeatureExtraction(left_image, right_image, camera_param)

    pre_img = left_image

    # Set up Camera Pose List
    Transformation_list = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])

    for ind in range(1, len(left_images)-1):

        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        feature_points = FeatureExtraction(left_image, right_image, camera_param)

        paired_static_features = FrameMatching(previous_feature_points, feature_points)
        print("Number of pairs: ", paired_static_features.shape[0])
        previous_feature_points = feature_points
        pre_img = left_image


        # draw_frame_matching(left_image, paired_static_features, T_true[ind-1], T_true[ind], camera_param['left_projection'])



        # ############################## Compute Reprojection Error #########################
        # pe = PoseEstimator(paired_static_features, camera_param['left_projection'], T_true[ind - 1])
        pe = PoseEstimator(paired_static_features, camera_param['left_projection'], Transformation_list[ind - 1])
        # pe.ComputeReprojError(T_true[ind])
        dof = pe.minimize_error()
        T_current = pe.convert2T(dof)
        print("Cal: ", sum(pe.ComputeReprojError(dof)))
        dof_true = pe.convert2dof(np.vstack([T_true[ind].reshape(3, 4), [0, 0, 0, 1]]))
        print("True: ", sum(pe.ComputeReprojError(dof_true)))
        # a = T_current.reshape(4, 4)
        # print(a)
        # print("------------")
        # b = np.vstack([T_true[ind].reshape(3, 4), [0, 0, 0, 1]])
        # print(b)
        # print(np.linalg.inv(b))
        # T_current = np.vstack([T_current.reshape(3, 4), [0, 0, 0, 1]])
        # T_pre = np.vstack([T_true[ind - 1].reshape(3, 4), [0, 0, 0, 1]])
        # T_pre = np.vstack([Transformation_list[ind - 1].reshape(3, 4), [0, 0, 0, 1]])
        # T_current = T_pre @ T_current.reshape(4, 4)
        Transformation_list = np.vstack([Transformation_list, T_current[0:3, :].flatten()])

    # # Plot the error between ground truth and calculated
    # error = np.linalg.norm((T_true[0:11] - Transformation_list), axis=1)
    # print(error)
    # print(Transformation_list[:,11])
    # DrawTrajectory(Transformation_list)
    # DrawTrajectory(T_true[:11])
    plt.plot(Transformation_list[:len(left_images), 11])
    plt.plot(T_true[:len(left_images), 11])
    plt.show()