# Computer Vision Final Project
Our report, slides, and all of our code is included within this zip file.


# Documents:
- Computer_Vision_Final_Presentation_Reimplementation_of_Visual_SLAM_in_Dynamic_Environments_Based_on_Object_Detection_and_Scene_Flow.pdf: Presentation
- Computer_Vision_Final_Report_Reimplementation_of_Visual_SLAM_in_Dynamic_Environments_Based_on_Object_Detection_and_Scene_Flow.pdf: Final report


# Precomputed Results:
- Result: Saved graphs of Yaw, Trajectory for Every Feature Points, Static Feature Points, With Dynamic Feature Points for all Datasets


# Program Files
- evaluation.py: Script to Evaluate our Calculated Trajectory with the Ground Truth Trajectory
- feature_extraction.py: Script to Extract Data for Feature Points
- feature_matching.py: Script to Perform Feature Matching across Stereo Images at Current Timestamp
- feature_points.py: Script to Implement Class for Feature Points 
- filter_feature_points.py: Script to Filter Feature Points based on whether it Falls inside a Bounding Box
- frame_matching.py: Script to Perform Feature Matching across Previous and Current Left/Right Frames
- main.py: Main Script to Run the Project
- perform_KL.py: Script to Perform KL Divergence to Discriminate Dynamic and Potential Dynamic Feature Points
- perform_yolo.py: Script to Perform YOLO on Stereo Images and Return Bounding box Coordinates
- pose_estimator.py: Script to Implement a Pose Estimator Class to compute Transformation Matrix
- read_camera_param.py: Script to Read the Camera Calibration Parameters from Text File
- visualizations.py: Script to Visualize KL Divergence


# Parameter Files:
- yolov8n.pt: Pretrained YOLOv8n Model

# Data Files
These files describe the modified data set we used for our project (a subset of KITTI)
- Dataset_00, Dataset_02, Dataset_05, Dataset_08, Dataset_10: 200 Stereo images, calib.txt having Camera Calibration Parameters, times.txt having Timestamp of every Image frame, true_T.txt having True Transformation Matrices.