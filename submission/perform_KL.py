# Import Necessary Modules
from feature_extraction import FeatureExtraction
from read_camera_param import ReadCameraParam
from frame_matching import FrameMatching
from scipy.stats import entropy
from perform_yolo import PerformYolo
import visulizations
from filter_feature_points import check_point_in_bbox

# Import Necessary Libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

def normlize_descriptors(desc):
    norm_desc = np.abs(desc)
    total = np.sum(norm_desc)
    if total>0:
        norm_desc /= total

    return norm_desc

def kl_divergence_scipy(P, Q, epsilon=1e-10):

    P = np.array(P) + epsilon
    Q = np.array(Q) + epsilon
    return entropy(P, Q)

    """ Compute reprojection error. """
    predicted_points = project_points(T, world_points)
    return np.linalg.norm(predicted_points - image_points[:, :2], axis=1).sum()  # Sum of Euclidean distances

def compute_error(pred_points, observed_points):

    return np.linalg.norm(pred_points - observed_points, axis=1).sum()


# Define Main Function
def PerformKL(pre_left_image, pre_right_image,
              cur_left_image, cur_right_image, 
              left_boxes, right_boxes,
              camera_param,
              T_previous,
              T_current, 
              paired_poten_dy_fpts_cur_2d_pre_2d,
              paired_poten_dy_fpts_cur_2d_pre_3d):

    # # Get the Folders for Left & Right Stereo Images
    # left_images_folder = 'Dataset_3/Left_Images/'
    # right_images_folder = 'Dataset_3/Right_Images/'

    # # Get the Images Path list
    # left_images = os.listdir(left_images_folder)
    # right_images = os.listdir(right_images_folder)

    # # Sort Imgae Paths by Name
    # left_images = sorted(left_images, key=lambda x:int(x.split('.')[0]))
    # right_images = sorted(right_images, key=lambda x:int(x.split('.')[0]))

    # # Get the Path of Images
    # left_images = [os.path.abspath(left_images_folder + '/' + left_image) for left_image in left_images][-20:]
    # right_images = [os.path.abspath(right_images_folder + '/' + right_image) for right_image in right_images][-20:]
    
    
    file_path = './Dataset_3/true_T.txt'
    true_T = np.loadtxt(file_path, dtype=np.float64)


    # # Read the Images
    pre_left_image = cv2.imread(pre_left_image)
    pre_right_image = cv2.imread(pre_right_image)


    cur_left_image = cv2.imread(cur_left_image)
    cur_right_image = cv2.imread(cur_right_image)


    predicted_list = []
    new_potential_dynamic_features = []

    # for pt2d, pt3d in zip(potential_dynamic_feature_points.left_pts[:, :], potential_dynamic_feature_points.pt3ds[:, :3]):
    for (x_cur_2d, y_cur_2d, x_pre_3d, y_pre_3d, z_pre_3d) in paired_poten_dy_fpts_cur_2d_pre_3d:

        temp = np.vstack((T_previous.reshape(3, 4), [0, 0, 0, 1])) @ np.hstack((np.array([x_pre_3d, y_pre_3d, z_pre_3d]).T, np.ones([1])))
        temp_2 = np.linalg.inv(np.vstack((T_current.reshape(3, 4), [0, 0, 0, 1])))@temp
    
        M = camera_param

        projected_homogeneous = M@temp_2

        x_prime = projected_homogeneous[0] / projected_homogeneous[2]
        y_prime = projected_homogeneous[1] / projected_homogeneous[2]

        if np.isnan(x_prime)==False or np.isnan(y_prime)==False:
            predicted_list.append([x_prime, y_prime])
            new_potential_dynamic_features.append((x_cur_2d, y_cur_2d, x_pre_3d, y_pre_3d, z_pre_3d))

    # pre -> cur feature point prediction
    predicted_list = np.array(predicted_list)
    new_potential_dynamic_features = np.array(new_potential_dynamic_features)

    # save potential dynamic feature points in current image seperately
    # current_2ds = []
    # previous_2ds = []

    # for one_2d, one_3d in zip(potential_dynamic_feature_points.left_pts, potential_dynamic_feature_points.pt3ds):
        
    #     previous_2d = M@np.hstack((one_3d, [1]))

    #     x_prime = previous_2d[0] / previous_2d[2]
    #     y_prime = previous_2d[1] / previous_2d[2]

    #     if np.isnan(x_prime)==False or np.isnan(y_prime)==False:
    #         previous_2ds.append([x_prime, y_prime])
    #         current_2ds.append(one_2d)


    # Visualizations
    # visulizations.draw_features_on_image_vertical(pre_left_image, cur_left_image, previous_2ds, predicted_list, 
    #                                                 pic_title='Observed Fpts in T0 vs Predicted Fpts in T1')

    # visulizations.draw_features_on_image_vertical(pre_left_image, cur_left_image, previous_2ds, current_2ds, 
    #                                                 pic_title='Observed Fpts Matching between T0 and T1')

    # visulizations.draw_features_on_image_vertical(cur_left_image, cur_left_image, predicted_list, current_2ds, 
    #                                                 pic_title='Predicted Fpts in T1 vs Observed Fpts in T1')
    # visulizations.draw_features_on_image_together(cur_left_image, cur_left_image, predicted_list, current_2ds, 
    #                                                 pic_title='Predicted Fpts in T1 vs Observed Fpts in T1')


    # find descriptors of predicted and observe and comapre
    orb = cv2.ORB_create()
    
    keypoints = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=50) for pt in predicted_list]
    __, preds_descriptors = orb.compute(cur_left_image, keypoints)
    preds_descriptors = np.array(preds_descriptors, dtype=np.float64)

    keypoints = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=50) for pt in new_potential_dynamic_features[:, :2]]
    __, observed_descriptors = orb.compute(cur_left_image, keypoints)     
    observed_descriptors = np.array(observed_descriptors, dtype=np.float64)

    kl_values = []

    # calculate KL value
    for one_pred_desc, one_observed_desc in zip(preds_descriptors, observed_descriptors):

        norm_one_pred_desc = normlize_descriptors(one_pred_desc)
        norm_one_observed_desc = normlize_descriptors(one_observed_desc)

        one_entropy = kl_divergence_scipy(norm_one_pred_desc, norm_one_observed_desc)

        kl_values.append(round(one_entropy, 3))
            

    # num_diff_boxes = len(right_boxes) - len(left_boxes)
    # if num_diff_boxes>0:
    #     all_boxes = left_boxes+right_boxes[-(num_diff_boxes):]
    # else:
    #     all_boxes = left_boxes
    all_boxes = []

    for right_box in right_boxes:
        all_boxes.append(right_box)
            
        
    for left_box in left_boxes:
        all_boxes.append(left_box)


    # visulizations.visualize_KL(cur_left_image, cur_left_image, predicted_list, new_potential_dynamic_features[:, :2], kl_values, all_boxes, all_boxes, 
    #                             font_size=5, pic_title='Compare Predicted Fpts in T1 & Observed Fpts in T1')
    
    # visulizations.visualize_KL_result(cur_left_image, cur_left_image, predicted_list, new_potential_dynamic_features[:, :2], kl_values, all_boxes, all_boxes,
    #                                   pic_title='Compare Predicted Fpts in T1 & Observed Fpts in T1')
    

    
    more_static_points = []


    for (x1, y1, x2, y2) in all_boxes:
        increase_x_scale = round((x2-x1)*0.2)
        increase_y_scale = round((y2-y1)*0.2)          

        list_points = []
        kl_values_in_one_bb = []

        erased_points_indices = []
        count = 0

        for (pred_x, pred_y), (x_cur_2d, y_cur_2d, x_pred_3d, y_pred_3d, z_pred_3d), one_kl_value in zip(predicted_list, new_potential_dynamic_features, kl_values):

                # if check_point_in_bbox((x1-increase_x_scale, y1-increase_y_scale, x2+increase_x_scale, y2+increase_y_scale), (pred_x, pred_y)):
                if check_point_in_bbox((x1, y1, x2, y2), (pred_x, pred_y)):

                    list_points.append([x_cur_2d, y_cur_2d, x_pred_3d, y_pred_3d, z_pred_3d])
                    kl_values_in_one_bb.append(one_kl_value)
                    erased_points_indices.append(count)

                count+=1


        if np.mean(kl_values_in_one_bb)<0.3:
            for i in list_points:
                more_static_points.append(i)

        # print(new_potential_dynamic_features[erased_points_indices])
        new_potential_dynamic_features = np.delete(new_potential_dynamic_features, erased_points_indices, axis=0)
        

    more_static_points = np.array(more_static_points)
    return more_static_points