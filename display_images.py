# Import Necessary Libraries
import cv2


# Define a List of Colors of Order (B, G, R)
colors = [
    (255, 255, 0),  # Cyan
    (0, 255, 0),    # Lime
    (255, 0, 0),    # Blue
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 0, 255),  # Pink
    (0, 128, 128),  # Olive
    (0, 0, 128)     # Maroon
]


# Define a Function to Display Everything on Both Images
def DisplayImages(left_image, right_image, left_boxes, right_boxes, static_feature_points, dynamic_feature_points):
    
    # Display Bounding Boxes on Left & Right Image
    color_index = 0
    for left_bbox in left_boxes:
        left_x1, left_y1, left_x2, left_y2 = left_bbox[0], left_bbox[1], left_bbox[2], left_bbox[3]
        cv2.rectangle(left_image, (left_x1, left_y1), (left_x2, left_y2), colors[color_index], 2)

    for right_bbox in right_boxes:
        right_x1, right_y1, right_x2, right_y2 = right_bbox[0], right_bbox[1], right_bbox[2], right_bbox[3]
        cv2.rectangle(right_image, (right_x1, right_y1), (right_x2, right_y2), colors[color_index], 2)

    # Display Static Feature Points on Both Images using Red Color
    for ind in range(static_feature_points.num_fp):
        if ind < len(static_feature_points.left_pts) and ind < len(static_feature_points.right_pts):
            left_x, left_y = int(static_feature_points.left_pts[ind][0]), int(static_feature_points.left_pts[ind][1])
            right_x, right_y = int(static_feature_points.right_pts[ind][0]), int(static_feature_points.right_pts[ind][1])
            left_image = cv2.circle(left_image, (left_x, left_y), radius = 2, color = (0, 0, 255), thickness = -1)
            right_image = cv2.circle(right_image, (right_x, right_y), radius = 2, color = (0, 0, 255), thickness = -1)

    # Display Dynamic Feature Points on Both Images using Green Color
    for ind in range(dynamic_feature_points.num_fp):
        if ind < len(dynamic_feature_points.left_pts) and ind < len(dynamic_feature_points.right_pts):
            left_x, left_y = int(dynamic_feature_points.left_pts[ind][0]), int(dynamic_feature_points.left_pts[ind][1])
            right_x, right_y = int(dynamic_feature_points.right_pts[ind][0]), int(dynamic_feature_points.right_pts[ind][1])
            left_image = cv2.circle(left_image, (left_x, left_y), radius = 2, color = (0, 255, 0), thickness = -1)
            right_image = cv2.circle(right_image, (right_x, right_y), radius = 2, color = (0, 255, 0), thickness = -1)

    # Display the Images
    cv2.imshow('Left_Image', left_image)
    cv2.waitKey(1)
    cv2.imshow('Right_Image', right_image)
    cv2.waitKey(1)