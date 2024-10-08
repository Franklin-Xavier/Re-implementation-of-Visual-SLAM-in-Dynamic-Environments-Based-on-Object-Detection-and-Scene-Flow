# Import Necessary Modules
from read_camera_param import ReadCameraParam
from perform_yolo import PerformYolo
from feature_extraction import FeatureExtraction
from filter_feature_points import FilterFeaturePoints_for_kl
from frame_matching import FrameMatching
from pose_estimator import PoseEstimator
# from display_images import DisplayImages
from evaluation import compute_ate_t, compute_ate_R, compute_rpe_t, compute_rpe_R
import visulizations
import perform_KL

# Import Necessary Libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Define Dataset Folder
DATASET = './Dataset_00'

# Define Main Function
if __name__ == "__main__":

    # perform_kl == 0 : SLAM without KL
    # perorm_kl == 1: SLAM with KL
    perform_kl = 1

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

    left_boxes, right_boxes = PerformYolo(left_image, right_image)

    previous_feature_points = FeatureExtraction(left_image, right_image, camera_param)
    previous_static_feature_points, previous_dynamic_feature_points = FilterFeaturePoints_for_kl(left_boxes, right_boxes, previous_feature_points, 
                                                                        use_kmeans = False, num_clusters = 2)


    Transformation_list = np.array([T_true[0]])
    Transformation_list_with_dynamic = np.array([T_true[0]])

    Yaw_list = np.empty([0, 1])
    true_Yaw = np.empty([0, 1])


    # For the Left and Right Images Dataset
    for ind in range(1, len(left_images) - 1):
        
        print(f"Image {ind-1} to Image {ind}")

        ####################### Preprocess the Images #######################
        # Read the Images
        left_image = cv2.imread(left_images[ind])
        right_image = cv2.imread(right_images[ind])

        ########################### Perform YOLO on Both Images ########################
        left_boxes, right_boxes = PerformYolo(left_image, right_image)
        
        ############## Extract FeaturePoints from Both Images ##############
        # camera_param below is for Kitti dataset (Dataset_1, Dataset_3 not for Dataset_2)
        feature_points = FeatureExtraction(left_image, right_image, camera_param)

        ######################## Remove Static Features using Depth map ################

        static_feature_points, dynamic_feature_points = FilterFeaturePoints_for_kl(left_boxes, right_boxes, 
                                                                            feature_points, use_kmeans = False, 
                                                                            num_clusters = 2)


        ############################## Display both the Images #########################
        # DisplayImages(left_image, right_image, left_boxes, right_boxes, static_feature_points, dynamic_feature_points)

        ############################## Match Feature Between Frames #########################


        paired_poten_dy_fpts_cur_2d_pre_3d, paired_poten_dy_fpts_cur_2d_pre_2d \
            = FrameMatching(previous_dynamic_feature_points, dynamic_feature_points)


        paired_static_features_current_2d_previous_3d, paired_static_features_current_2d_previous_2d = \
            FrameMatching(previous_static_feature_points, static_feature_points)
        

        previous_static_feature_points = static_feature_points

        ############# Compute Transformation matrix of Camera onto Next Frame (Before KL & aftrer KL)###############


        if perform_kl == 1:

            dynamic_included = 0

            while dynamic_included < 2:

                min_error = float('inf')
                list_errors = []
                list_T = []

                if dynamic_included == 1:
                    if more_static_points.shape[0]!=0:
                        paired_static_features_current_2d_previous_3d = np.vstack((paired_static_features_current_2d_previous_3d, more_static_points))

                
                for _ in range(1000):
                    
                    sample_idx = np.random.choice(range(paired_static_features_current_2d_previous_3d.shape[0]), 50)
                
                    pe = PoseEstimator(paired_static_features_current_2d_previous_3d[sample_idx], camera_param['left_projection'], Transformation_list[ind - 1])        
                    dof = pe.minimize_error()

                    T_current = pe.convert2T(dof)

                    error_now = np.sum(np.linalg.norm(pe.ComputeReprojError(dof)))

                    list_errors.append(error_now)
                    list_T.append(T_current)


                if dynamic_included == 0:

                    print('num candidates fpts before KL for RANSAC is', paired_static_features_current_2d_previous_3d.shape[0])

                    min_error = np.argmin(list_errors)
                    T_current = list_T[min_error]
                    dof_true = pe.convert2dof(np.vstack([T_true[ind].reshape(3, 4), [0, 0, 0, 1]]))
                    Transformation_list = np.vstack([Transformation_list, T_current[0:3, :].flatten()])
                    

                if dynamic_included == 1:

                    print('num candidates fpts after KL for RANSAC is', paired_static_features_current_2d_previous_3d.shape[0])

                    min_error = np.argmin(list_errors)
                    T_current = list_T[min_error]
                    dof_true = pe.convert2dof(np.vstack([T_true[ind].reshape(3, 4), [0, 0, 0, 1]]))
                    Transformation_list_with_dynamic = np.vstack([Transformation_list_with_dynamic, T_current[0:3, :].flatten()])

                
                if dynamic_included == 0:
                    if len(paired_poten_dy_fpts_cur_2d_pre_2d)!=0:
                        more_static_points = perform_KL.PerformKL(left_images[ind-1], right_images[ind-1], 
                                            left_images[ind], right_images[ind],
                                            left_boxes, right_boxes,
                                            camera_param['left_projection'], 
                                            Transformation_list[ind-1],
                                            Transformation_list[ind],
                                            paired_poten_dy_fpts_cur_2d_pre_2d,
                                            paired_poten_dy_fpts_cur_2d_pre_3d)
                        
                    else:
                        more_static_points = np.array([])
                    
                dynamic_included += 1

            previous_dynamic_feature_points = dynamic_feature_points




        else:

            min_error = float('inf')
            list_errors = []
            list_T = []

            print('num candidates fpts before KL for RANSAC is', paired_static_features_current_2d_previous_3d.shape[0])

            for _ in range(1000):
                
                sample_idx = np.random.choice(range(paired_static_features_current_2d_previous_3d.shape[0]), 50)
            
                pe = PoseEstimator(paired_static_features_current_2d_previous_3d[sample_idx], camera_param['left_projection'], Transformation_list[ind - 1])        
                dof = pe.minimize_error()

                T_current = pe.convert2T(dof)

                error_now = np.sum(np.linalg.norm(pe.ComputeReprojError(dof)))

                list_errors.append(error_now)
                list_T.append(T_current)


            min_error = np.argmin(list_errors)
            T_current = list_T[min_error]
            dof_true = pe.convert2dof(np.vstack([T_true[ind].reshape(3, 4), [0, 0, 0, 1]]))
            Transformation_list = np.vstack([Transformation_list, T_current[0:3, :].flatten()])


        # Draw the Final Trajectory Yaw
        angles, _ = cv2.Rodrigues(T_current[0:3, 0:3])
        Yaw_list = np.vstack([Yaw_list, angles[1]])

        true_angles, _ = cv2.Rodrigues(T_true[ind].reshape(3, 4)[0:3, 0:3])
        true_Yaw = np.vstack([true_Yaw, true_angles[1]])

        plt.plot(Yaw_list[:len(left_images)])
        plt.plot(true_Yaw[:ind+1])
        plt.title('Yaw angle of the Camera')
        plt.xlabel('Frame')
        plt.ylabel('Yaw Angle(rad)')
        plt.legend(['Our Estimation', 'Ground Truth'])
        
        if ind!=len(left_images)-2:
            plt.pause(2)
            plt.close()
        else:
            plt.show()
        
        if perform_kl == 1:
            matrices_ours = Transformation_list_with_dynamic.reshape(-1, 3, 4)
        else:
            matrices_ours = Transformation_list.reshape(-1, 3, 4)

        # Extract the translation components (x, y) from the last column of the [3, 4] matrices_ours
        x_coords_ours = [mat[0, 3] for mat in matrices_ours]
        y_coords_ours = [mat[2, 3] for mat in matrices_ours]

        matrices_true = T_true[:+ind+1].reshape(-1, 3, 4)

        # Extract the translation components (x, y) from the last column of the [3, 4] matrices_true
        x_coords_true = [mat[0, 3] for mat in matrices_true]
        y_coords_true = [mat[2, 3] for mat in matrices_true]

        # Plotting the trajectory in 2D
        plt.figure(figsize=(8, 6))
        plt.plot(x_coords_ours, y_coords_ours, marker='o', linestyle='-', color='b')
        plt.plot(x_coords_true, y_coords_true, marker='o', linestyle='-', color='r')

        plt.title('2D Trajectory of the Camera (Ignoring Height)')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.grid(True)
        plt.axis('equal')  # Keeps the scale of x and y the same
        plt.legend(['Our Estimation', 'Ground Truth'])
        
        if ind!=len(left_images)-2:
            plt.pause(2)
            plt.close()
        else:
            plt.show()
            
    # Evaluation
    ate_error_t = compute_ate_t(T_true[:len(left_images)-1], Transformation_list)
    ate_error_R = compute_ate_R(T_true[:len(left_images)-1], Transformation_list)
    rpe_error_t = compute_rpe_t(T_true[:len(left_images)-1], Transformation_list)
    rpe_error_R = compute_rpe_R(T_true[:len(left_images)-1], Transformation_list)

    print("The ATE_t error: ", ate_error_t)
    print("The ATE_R error: ", ate_error_R)
    print("The RPE_t error: ", rpe_error_t)
    print("The RPE_R error: ", rpe_error_R)

