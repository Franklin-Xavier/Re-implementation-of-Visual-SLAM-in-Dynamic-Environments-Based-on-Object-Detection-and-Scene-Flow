# Define a Class to Store Data for Feature Points
class FeaturePoints:
    
    # Initialise the Class members
    def __init__(self) -> None:
        
        # Initialise Number of Feature Points
        self.num_fp = 0
        
        # Initialise Point coordinates and Descriptors for Left image
        self.left_pts = 0
        self.left_descriptors = 0
        
        # Initialise Point coordinates and Descriptors for Right image
        self.right_pts = 0
        self.right_descriptors = 0

        # Initialise Disparity and Depth
        self.disparity = 0
        self.depth = 0

        # Initialise 3D Point Coordinates in Camera Coordinates
        self.pt3ds = 0

        # Initialise Bounding Box ID
        self.bbox_id = 0