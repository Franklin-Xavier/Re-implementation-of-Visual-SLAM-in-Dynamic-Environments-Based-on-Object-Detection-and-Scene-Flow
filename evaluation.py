import numpy as np
import cv2

def compute_ate_t(gt_poses, est_poses):
    """
    Compute Absolute Trajectory Error (ATE) between ground truth poses and estimated poses for translation.
    """

    absolute_errors = np.linalg.norm(gt_poses[:, 9:12] - est_poses[:, 9:12], axis=1)
    ate = np.mean(absolute_errors)
    return ate

def compute_ate_R(gt_poses, est_poses):
    """
    Compute Relative Pose Error (RPE) between ground truth poses and estimated poses for rotation.
    """

    r_gt = np.empty([0, 1])   # Caculate only yaw
    r_est = np.empty([0, 1])

    for i in range(gt_poses.shape[0]):
        Ri_gt = gt_poses[i].reshape(3, 4)
        Ri_gt = Ri_gt[:3, :3]
        ri_gt, _ = cv2.Rodrigues(Ri_gt)
        r_gt = np.vstack([r_gt, ri_gt[1]])

        Ri_est = est_poses[i].reshape(3, 4)
        Ri_est = Ri_est[:3, :3]
        ri_est, _ = cv2.Rodrigues(Ri_est)
        r_est = np.vstack([r_est, ri_est[1]])

    absolute_errors = np.linalg.norm(r_gt[:] - r_est[:], axis=1)
    ate = np.mean(absolute_errors)
    return ate

def compute_rpe_t(gt_poses, est_poses):
    """
    Compute Relative Pose Error (RPE) between ground truth poses and estimated poses for translation.
    """

    relative_errors = np.linalg.norm(gt_poses[1:, 9:12] - gt_poses[:-1, 9:12] - (est_poses[1:, 9:12] - est_poses[:-1, 9:12]), axis=1)
    rpe = np.mean(relative_errors)
    return rpe

def compute_rpe_R(gt_poses, est_poses):
    """
    Compute Relative Pose Error (RPE) between ground truth poses and estimated poses for rotation.
    """

    r_gt = np.empty([0, 1])   # Caculate only yaw
    r_est = np.empty([0, 1])

    for i in range(gt_poses.shape[0]):
        Ri_gt = gt_poses[i].reshape(3, 4)
        Ri_gt = Ri_gt[:3, :3]
        ri_gt, _ = cv2.Rodrigues(Ri_gt)
        r_gt = np.vstack([r_gt, ri_gt[1]])

        Ri_est = est_poses[i].reshape(3, 4)
        Ri_est = Ri_est[:3, :3]
        ri_est, _ = cv2.Rodrigues(Ri_est)
        r_est = np.vstack([r_est, ri_est[1]])

    relative_errors = np.linalg.norm(r_gt[1:] - r_gt[:-1] - (r_est[1:] - r_est[:-1]), axis=1)
    rpe = np.mean(relative_errors)
    return rpe