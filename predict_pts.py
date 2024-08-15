# Import Necessary Libraries
import numpy as np
from scipy.optimize import least_squares


def PredictPts(paired_static_features, camara_projection, T_previous, T_current):

    pt3Ds = paired_static_features[:, 2:5].T
    pt3Ds = np.vstack([pt3Ds, np.ones([1, pt3Ds.shape[1]])])
    

    previous_2d = camara_projection @ pt3Ds
    u_previous = previous_2d[0] / previous_2d[2]
    v_previous = previous_2d[1] / previous_2d[2]

    previous_pts = []
    for u, v in zip(u_previous, v_previous):
        previous_pts.append((u, v))
    previous_pts = np.array(previous_pts)



    temp = np.vstack((T_previous.reshape(3, 4), [0, 0, 0, 1])) @ pt3Ds
    temp_2 = np.linalg.inv(np.vstack((T_current.reshape(3, 4), [0, 0, 0, 1]))) @ temp

    projected_homogeneous = camara_projection @ temp_2

    u_predict = projected_homogeneous[0] / projected_homogeneous[2]
    v_predict = projected_homogeneous[1] / projected_homogeneous[2]

    predicted_pts = []
    for u, v in zip(u_predict, v_predict):
        predicted_pts.append((u, v))
    predicted_pts = np.array(predicted_pts)

    return previous_pts, predicted_pts
    

