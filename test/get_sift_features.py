# Import Necessary Libraries
import cv2


# Initialize ORB detector
orb = cv2.ORB_create()

# Define a Function to Detect and Draw Keypoints
def detect_and_draw_keypoints(frame):
    
    # Convert Frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Keypoints
    keypoints, _ = orb.detectAndCompute(gray, None)
    
    # Draw Keypoints on the Frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Return the Keypoints
    return frame_with_keypoints


# Define a Function to Get SIFT Features on Both Images
def get_sift_features_on_images(left_image, right_image):

    # Detect and Draw Keypoints on Both Frames
    left_frame_with_keypoints = detect_and_draw_keypoints(left_image)
    right_frame_with_keypoints = detect_and_draw_keypoints(right_image)
        
    # Display the Frame with Keypoints
    cv2.imshow('Left_Frame with Keypoints', left_frame_with_keypoints)
    cv2.waitKey(1)
    cv2.imshow('Right_Frame with Keypoints', right_frame_with_keypoints)
    cv2.waitKey(1)