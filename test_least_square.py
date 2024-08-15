from scipy.optimize import least_squares
import numpy as np

a = np.array([1, 2, 3, 4])

static_feature_points = np.array([[        403,         145,     -8.6439,     -1.6617,      29.703],
       [        373,         176,     -12.795,    -0.49504,      38.614],
       [        372,         173,     -12.849,    -0.60247,      38.614],
       [        333,         210,      -6.684,     0.62555,      16.789],
       [      500.4,       121.2,     -15.935,     -9.5519,      107.26],
       [      254.4,       254.4,     -6.7037,      1.3462,      12.871],
       [      385.2,       153.6,     -9.2292,     -1.2866,      29.253],
       [      424.8,      164.16,     -13.715,     -1.5709,      53.631],
       [     437.76,      159.84,     -12.748,     -1.8932,      53.631],
       [     293.76,       122.4,     -8.1528,      -1.598,      17.877],
       [     437.18,       160.7,     -10.677,     -1.5239,      44.693],
       [      518.4,      179.71,     -13.532,    -0.85544,      111.73],
       [     495.94,      158.98,     -11.528,     -2.7189,      74.488],
       [     323.48,      213.58,     -6.9744,     0.71683,      16.929],
       [     499.74,      178.33,     -13.918,    -0.89192,       93.11],
       [     437.53,      159.67,     -11.122,     -1.6546,      46.555],
       [     292.38,      234.32,     -6.6039,      1.0611,      14.325],
       [     355.83,      236.39,     -5.6949,      1.2122,      15.518],
       [     440.43,      174.18,     -12.179,    -0.79393,      51.728],
       [     323.48,         214,     -7.1035,        0.75,      17.243],
       [     333.43,      226.44,     -6.8648,      1.0484,      17.243],
       [     444.91,      152.29,     -7.4327,      -1.481,       32.33],
       [     351.15,      182.74,     -6.7554,     -0.0618,      17.961],
       [     472.98,      186.33,      -20.12,     0.16637,      107.77],
       [     498.06,      186.33,     -8.1801,    0.083184,      53.883],
       [     537.48,      186.33,     -10.451,     0.16637,      107.77],
       [      512.4,      189.91,     -7.1058,     0.35177,      53.883]])

camera = np.array([[     718.86,           0,      607.19,           0],
       [          0,      718.86,      185.22,           0],
       [          0,           0,           1,           0]])

def objective(T_current):
    T_current = T_current.reshape(3, 4)
    T_previous = np.eye(3, 4)

    pt3Ds = static_feature_points[:, 2:5].T
    pt3Ds = np.vstack([pt3Ds, np.ones([1, pt3Ds.shape[1]])])
    temp = np.vstack((T_previous.reshape(3, 4), [0, 0, 0, 1])) @ pt3Ds
    temp_2 = np.linalg.inv(np.vstack((T_current.reshape(3, 4), [0, 0, 0, 1]))) @ temp

    M = camera

    projected_homogeneous = M @ temp_2

    u_predict = projected_homogeneous[0] / projected_homogeneous[2]
    v_predict = projected_homogeneous[1] / projected_homogeneous[2]

    u = static_feature_points[:, 0].flatten()
    v = static_feature_points[:, 1].flatten()

    reprojecgtion_error = sum(np.sqrt((u_predict - u) ** 2 + (v_predict - v) ** 2))
    
    return reprojecgtion_error

if __name__ == "__main__":
    file_path = './Dataset_1/true_T.txt'
    true_T = np.loadtxt(file_path, dtype=np.float64)
    T = np.eye(3, 4)
    print(objective(T.flatten()))
    result = least_squares(objective, T.flatten())

    T_cal = result.x.reshape(3, 4)
    print(T_cal)
    print(objective(result.x))
    T_true = true_T[1].reshape(3, 4)
    print(T_true)

    error = np.linalg.norm(T_true - T_cal)
    print(error)

