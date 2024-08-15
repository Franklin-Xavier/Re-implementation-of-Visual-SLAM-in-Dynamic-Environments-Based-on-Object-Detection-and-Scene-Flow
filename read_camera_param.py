# Import Necessary Libraries
import numpy as np
import cv2


# Define a Function to Read Calibration File
def ReadCalibFile(filepath):

    # Initialise Dictionary to Store Data
    data = {}

    # Store the Data into Dictionary
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    
    # Return the Data
    return data


# Define a Function to Read Camera Parameters
def ReadCameraParam(filepath):
    
    # Create Camera Parameter Dictionary
    camera_param = {}
     
    # Read Calib File
    calib_data = ReadCalibFile(filepath)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(calib_data['P0'], (3, 4))
    P_rect_10 = np.reshape(calib_data['P1'], (3, 4))

    # Extract Projection Matrices, Focal Length, and Baselie
    camera_param['left_projection'] = P_rect_00
    camera_param['right_projection'] = P_rect_10
    camera_param['focal_length'] = P_rect_00[0][0]
    camera_param['baseline'] = -P_rect_10[0, 3] / P_rect_10[0, 0]
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P_rect_10)
    t = t[:3] / t[3]
    t_skew = np.array([[0, -t[2][0], t[1][0]], [t[2][0], 0, -t[0][0]], [-t[1][0], t[0][0], 0]])
    camera_param['fundamental']  = np.linalg.inv(k.T) @ t_skew @ r @ np.linalg.inv(k)
    # camera_param['principle_point'] = (P_rect_00[0, 2], P_rect_00[1, 2])

    # Return the Camera Parameters
    return camera_param