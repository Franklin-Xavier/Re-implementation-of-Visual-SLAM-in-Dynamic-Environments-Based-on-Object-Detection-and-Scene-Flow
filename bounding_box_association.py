# Import Necessary Libraries
import collections
import numpy as np


# Define a Function to Check if a Feature Point falls inside Bounding Box
def check_point_in_bbox(bbox, point):

    # Get the Coordinates and Points
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = point[0], point[1]

    # Check if Point falls inside Bounding Box
    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False
    

# Define a Function to Check if Features are Matching on Both Images
def is_Matching_Features(feature_points, left_feature, right_feature):

    # Create Feature from Left and Right Features
    feature = np.array([left_feature[0], left_feature[1], right_feature[0], right_feature[1]])

    # Check if Feature is present in List of Feature Points
    point = np.empty([0, 4])
    for ind in range(feature_points.num_fp):
        point = np.vstack([point, [feature_points.left_pts[ind][0], feature_points.left_pts[ind][1], feature_points.right_pts[ind][0], feature_points.right_pts[ind][1]]])
    
    if feature in point:
        return True
    else:
        return False


# Define a Function to Count Matching Features
def Count_Matching_Features(feature_points, left_features, right_features):
    
    # Initialize match count
    match_count = 0

    # Count Matching Features
    for left_feature in left_features:
        for right_feature in right_features:
            if is_Matching_Features(feature_points, left_feature, right_feature):
                match_count += 1
                break

    # Return the match count
    return match_count


# Define a Function to Associate Bounding Boxes between Left and Right images
def BoundingBoxAssociation(left_boxes, right_boxes, feature_points):

    # Initialise Empty Dictionary to store Objects in Left and Right Images
    objects_on_left_image = dict()
    objects_on_right_image = dict()

    # Store the Bounding Boxes for both Frames
    objects_on_left_image['Bounding_Boxes'] = dict()
    objects_on_left_image['Bounding_Boxes']['Coordinates'] = left_boxes
    objects_on_right_image['Bounding_Boxes'] = dict()
    objects_on_right_image['Bounding_Boxes']['Coordinates'] = right_boxes

    # Store the Feature Points for both Frames
    objects_on_left_image['Feature_Points'] = feature_points.left_pts
    objects_on_right_image['Feature_Points'] = feature_points.right_pts

    # Initialise a List to store Feature Points for Corresponding Bounding Boxes
    objects_on_left_image['Bounding_Boxes']['Feature_Points'] = []
    objects_on_right_image['Bounding_Boxes']['Feature_Points'] = []

    # Initialise a List to store Number of Matching Features for both Frames
    objects_on_left_image['Number_of_Matching_Features'] = []
    objects_on_right_image['Number_of_Matching_Features'] = []

    # Check every Bounding Box Coordinates in Left Image
    for bbox in objects_on_left_image['Bounding_Boxes']['Coordinates']:

        # Initialise List of Points
        points = []

        # Check every Feature Point in Left Image
        for point in objects_on_left_image['Feature_Points']:
            
            # Check if that Points falls inside Bounding box
            if check_point_in_bbox(bbox, point):
                points.append(point)
        
        # Store the Feature Points for that Bounding Box
        objects_on_left_image['Bounding_Boxes']['Feature_Points'].append(points)
    

    # Check every Bounding Box Coordinates in Right Image
    for bbox in objects_on_right_image['Bounding_Boxes']['Coordinates']:

        # Initialise List of Points
        points = []

        # Check every Feature Point in Right Image
        for point in objects_on_right_image['Feature_Points']:
            
            # Check if that Points falls inside Bounding box
            if check_point_in_bbox(bbox, point):
                points.append(point)
        
        # Store the Feature Points for that Bounding Box
        objects_on_right_image['Bounding_Boxes']['Feature_Points'].append(points)
    
    # Delete Feature Points from Objects
    del objects_on_left_image['Feature_Points']
    del objects_on_right_image['Feature_Points']


    # For every Feature Points in Left Image
    for left_features in objects_on_left_image['Bounding_Boxes']['Feature_Points']:

        # Initialise Matches Counts
        matches = []

        # For every Feature Points in Right Image
        for right_features in objects_on_right_image['Bounding_Boxes']['Feature_Points']:

            # Append the Matches count
            matches.append(Count_Matching_Features(feature_points, left_features, right_features))
        
        # Store the Number of Matching Features for Left Image
        objects_on_left_image['Number_of_Matching_Features'].append(matches)
    

    # For every Feature Points in Right Image
    for right_features in objects_on_right_image['Bounding_Boxes']['Feature_Points']:

        # Initialise Matches Counts
        matches = []

        # For every Feature Points in Left Image
        for left_features in objects_on_left_image['Bounding_Boxes']['Feature_Points']:

            # Append the Matches count
            matches.append(Count_Matching_Features(feature_points, right_features, left_features))
        
        # Store the Number of Matching Features for Right Image
        objects_on_right_image['Number_of_Matching_Features'].append(matches)
    

    # Initialise List to store Associated Bounding Boxes
    associated_bounding_boxes = []

    # For every Matching Feature in Left Objects
    for left_match_index in range(len(objects_on_left_image['Number_of_Matching_Features'])):

        # Get the Max Match count and its Index
        max_match = max(objects_on_left_image['Number_of_Matching_Features'][left_match_index])

        # Get its Index
        max_match_index = objects_on_left_image['Number_of_Matching_Features'][left_match_index].index(max_match)

        # Check if the Right Object also has the same Pair
        if left_match_index == objects_on_right_image['Number_of_Matching_Features'][max_match_index].index(max(objects_on_right_image['Number_of_Matching_Features'][max_match_index])):
            associated_bounding_boxes.append([objects_on_left_image['Bounding_Boxes']['Coordinates'][left_match_index], objects_on_right_image['Bounding_Boxes']['Coordinates'][max_match_index]])
    
    # Return Associated Bounding Boxes
    return associated_bounding_boxes