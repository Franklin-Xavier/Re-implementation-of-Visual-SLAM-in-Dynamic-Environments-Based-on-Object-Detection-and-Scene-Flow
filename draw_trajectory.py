import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def DrawTrajectory(T):

    x = T[:, 3] # left right
    y = T[:, 7] # height
    z = T[:, 11] # forward backword

    fig = plt.figure()

    plt.subplot(3, 1, 1)
    plt.title('left right')
    plt.tight_layout()
    plt.plot(x)

    plt.subplot(3, 1, 2)
    plt.title('height')
    plt.tight_layout()
    plt.plot(y)


    plt.subplot(3, 1, 3)
    plt.title('forward backward')
    plt.tight_layout()
    plt.plot(z)


    plt.show()

    # Draw Euler Angles ------------------------------------------------------------

    # euler_angle_x = []
    # euler_angle_y = []
    # euler_angle_z = []
    
    # for one_true in true_T:
    #     # Convert the rotation matrix to a rotation object
    #     rotation = R.from_matrix(one_true.reshape(3, 4)[:, :3])

    #     # Convert to Euler angles
    #     euler_angles_rad = rotation.as_euler('xyz', degrees=False)  # radians
    #     euler_angles_deg = rotation.as_euler('xyz', degrees=True)  # degrees

    #     euler_angle_x.append(euler_angles_deg[0])
    #     euler_angle_y.append(euler_angles_deg[1])
    #     euler_angle_z.append(euler_angles_deg[2])

    # fig = plt.figure()

    # plt.subplot(3, 1, 1)
    # plt.title('Euler angle x')
    # plt.tight_layout()
    # plt.plot(euler_angle_x)

    # plt.subplot(3, 1, 2)
    # plt.title('Euler angle y')
    # plt.tight_layout()
    # plt.plot(euler_angle_y)


    # plt.subplot(3, 1, 3)
    # plt.title('Euler angle z')
    # plt.tight_layout()
    # plt.plot(euler_angle_z)

    # plt.show()

    # ------------------------------------------------------------------------------


# Test codes
if __name__ == "__main__":

    file_path = './Dataset_4/true_T.txt'
    true_T = np.loadtxt(file_path, dtype=np.float64)
    
    DrawTrajectory(true_T)


