import cv2 
from matplotlib import pyplot as plt
import numpy as np

def orb(left_img, right_img):

    orb = cv2.ORB_create() 

    leftKeypoints, leftDescriptors = orb.detectAndCompute(left_img,None) 
    rightKeypoints, rightDescriptors = orb.detectAndCompute(right_img,None)

    # img1 = cv2.drawKeypoints(left_img, leftKeypoints, None, color=(0,255,0), flags=0)
    # img2 = cv2.drawKeypoints(right_img, rightKeypoints, None, color=(0,255,0), flags=0)
    # f, axarr = plt.subplots(2)
    # axarr[0].imshow(img1)
    # axarr[1].imshow(img2)
    # plt.show() 
    
    return leftKeypoints, leftDescriptors, rightKeypoints, rightDescriptors

if __name__ == "__main__":

    image_folder_L = 'dataset/image_0/'
    image_folder_R = 'dataset/image_1/'

    left_img = cv2.imread(image_folder_L + '000000.png', 0) 
    right_img = cv2.imread(image_folder_R + '000000.png', 0)

    left_kp, left_desc, right_kp, right_desc = orb(left_img, right_img)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(left_img, right_img)
    # plt.imshow(disparity,'gray')
    # plt.show()

    matcher = cv2.BFMatcher()
    matches = matcher.match(left_desc, right_desc)

    BFM_match_image = cv2.drawMatches(left_img, left_kp, right_img, right_kp, matches[:50],None)
    plt.imshow(BFM_match_image)
    plt.show()

    disparity_values = []
    diss = [] 
    for m in matches:
        index_L = m.queryIdx
        index_R = m.trainIdx

        disparity_value = disparity[int(left_kp[index_L].pt[1])][int(left_kp[index_L].pt[0])]
        dis = right_kp[index_R].pt[0] - left_kp[index_L].pt[0]
        disparity_values.append(disparity_value)
        diss.append(dis)
        print(disparity_value, dis)

    # norm = np.linalg.norm(diss)
    # diss /= norm
    n = 20
    plt.plot(range(n), disparity_values[:n], 'b')
    plt.plot(range(n), diss[:n], 'r')
    plt.show()
    # for i in range(len(left_kp)):
    #     disparity_value = abs(disparity[int(left_kp[i].pt[1])][int(left_kp[i].pt[0])])
    #     dis = abs(left_kp[i].pt[0] - right_kp[i].pt[0])
    #     if disparity_value != dis:
    #         print(left_kp[i].pt, right_kp[i].pt, disparity_value)