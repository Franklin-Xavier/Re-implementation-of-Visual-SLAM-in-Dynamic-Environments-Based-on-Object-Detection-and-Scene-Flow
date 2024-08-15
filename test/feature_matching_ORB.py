import numpy as np 
import cv2 
import matplotlib.pyplot as plt
      

def featureExtraction(left_img, right_img):

    # Extract keypoints
    orb = cv2.ORB_create() 

    kp_L, desc_L = orb.detectAndCompute(left_img,None) 
    kp_R, desc_R = orb.detectAndCompute(right_img,None)

    # Matching keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(desc_L, desc_R)

    featurePoints = []   
    for m in matches:
        index_L = m.queryIdx
        index_R = m.trainIdx

        # featurePoint = [left_kp_x, left_kp_y, right_kp_x, right_kp_y, disparity]
        featurePoints.append([kp_L[index_L].pt[0], kp_L[index_L].pt[1], kp_R[index_R].pt[0], kp_R[index_R].pt[1], (kp_L[index_L].pt[0] - kp_R[index_R].pt[0])])
    
    featurePoints = np.array(featurePoints)

    return featurePoints


def BFM(left_img, right_img, left_kp, left_desc, right_kp, right_desc):

    matcher = cv2.BFMatcher() 
    
    matches = matcher.match(left_desc, right_desc) 
    
    match_img = cv2.drawMatches(left_img, left_kp, 
                                right_img, right_kp, matches[:50],None) 
    
    match_img = cv2.resize(match_img, (1200,400)) 
    
    return match_img



def FLANN(left_img, right_img, left_kp, left_desc, right_kp, right_desc):
    
    FLANN_INDEX_LSH = 6

    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,  
                        key_size = 12,   
                        multi_probe_level = 1)  
    
    search_params = dict(checks=50)  
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.match(left_desc, right_desc)

    matches = sorted(matches, key = lambda x:x.distance)


    N_MATCHES = 50 

    match_img = cv2.drawMatches(left_img, left_kp, 
                                right_img, right_kp, 
                                matches[:N_MATCHES], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    match_img = cv2.resize(match_img, (1200,400))  
     
    return match_img

if __name__ == "__main__":

    image_folder_L = 'Ben_0/image_0/'
    image_folder_R = 'Ben_0/image_1/'

    left_img = cv2.imread(image_folder_L + '000000.png', 0) 
    right_img = cv2.imread(image_folder_R + '000000.png', 0)

    featurePoints = featureExtraction(left_img, right_img)

    BFM_match_image = BFM(left_img, right_img, left_kp, left_desc, right_kp, right_desc)

    # Show the final image 
    # cv2.imshow("Matches", BFM_match_image)
    plt.imshow(BFM_match_image)
    plt.show()

    FLANN_match_image = FLANN(left_img, right_img, left_kp, left_desc, right_kp, right_desc)

    # Display the matched keypoints
    plt.imshow(FLANN_match_image)
    plt.show()

