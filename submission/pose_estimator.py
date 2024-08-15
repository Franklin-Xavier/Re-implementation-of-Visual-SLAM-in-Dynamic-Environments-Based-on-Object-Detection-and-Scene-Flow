# Import Necessary Libraries
import numpy as np
from scipy.optimize import least_squares
import cv2


# Define a Class for Pose Estimator
class PoseEstimator:
    
    # Initialise the required Variables
    def __init__(self, paired_static_features, camera_projection, T_previous) -> None:
        
        # Stack all the 3D points into one matrix
        self.pt3Ds = paired_static_features[:, 2:5].T
        self.pt3Ds = np.vstack([self.pt3Ds, np.ones([1, self.pt3Ds.shape[1]])])
        
        # Stack all 2d points into one matrix
        self.u_current = paired_static_features[:, 0].flatten()
        self.v_current = paired_static_features[:, 1].flatten()

        # Get the projection matrix
        self.projection = camera_projection

        # Get the previous transfomation matrix
        self.T_previous = T_previous.reshape(3, 4)

        # Set the current transformatnio matrix to identity matrix
        # self.T_current = np.eye(4, 4).flatten()
        self.dof = np.zeros(6)


    # Define a Function to Compute Reprojection Error
    def ComputeReprojError(self, dof):
        T_current = self.convert2T(dof)

        # Compute the Projected Homogenous Coordinates
        temp = np.vstack((self.T_previous.reshape(3, 4), [0, 0, 0, 1])) @ self.pt3Ds
        temp_2 = np.linalg.inv(T_current) @ temp
        
        projected_homogeneous = self.projection @ temp_2

        # Compute the Predicted Coordinates
        u_predict = projected_homogeneous[0] / projected_homogeneous[2]
        v_predict = projected_homogeneous[1] / projected_homogeneous[2]

        # Compute the Reprojection Error and Return
        # reprojection_error = np.sqrt((self.u_current - u_predict) ** 2 + (self.v_current - v_predict) ** 2)
        reprojection_error = np.vstack([(self.u_current - u_predict), (self.v_current - v_predict)]).flatten()

        return reprojection_error
    
    def convert2T(self, dof):
        r = dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = dof[3:]
        T_current = np.eye(4, dtype=np.float64)
        T_current[:3, :3] = R
        T_current[:3, 3] = t

        return T_current
    
    def convert2dof(self, transform):
        t = transform[:3, 3]
        R = transform[:3, :3]
        r, _ = cv2.Rodrigues(R)
        dof = np.empty(6)
        dof[:3] = r.flatten()
        dof[3:] = t.flatten()

        return dof
    
    # Define a Function to Minimize Error using Least Squares
    def minimize_error(self):
        self.dof = least_squares(self.ComputeReprojError, self.dof, method='lm', max_nfev=200).x

        return self.dof

