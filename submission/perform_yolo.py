# Import Necessary Libraries
from ultralytics import YOLO
import numpy as np


# Define a Function to Perform YOLO on Images
def PerformYolo(left_image, right_image):

    # Load the YOLOv8 Pretrained Model
    model = YOLO('yolov8n.pt')

    # Predict the Objects in Left and Right Images using Model
    detections_left_image = model(left_image, verbose = False)
    detections_right_image = model(right_image, verbose = False)

    # Create Result bboxes
    left_bboxes = []
    right_bboxes = []

    # Visualise Bounding Boxes for all Detected Objects in Left Image
    for obj in detections_left_image:
                    
        # Get the Bounding Box Coordinates and Class of Prediction for that Object
        boxes = obj.boxes
        for box in boxes:

            # Convert Bounding Box Coordinates into Array                    
            bbox = np.array(box.xyxy[0].cpu())

            # Get the Coordinates of Bounding Box of Object and Plot on Frame
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            left_bboxes.append([x1, y1, x2, y2])

    # Visualise Bounding Boxes for all Detected Objects in Right Image
    for obj in detections_right_image:
                    
        # Get the Bounding Box Coordinates and Class of Prediction for that Object
        boxes = obj.boxes
        for box in boxes:

            # Convert Bounding Box Coordinates into Array                    
            bbox = np.array(box.xyxy[0].cpu())
                    
            # Get the Coordinates of Bounding Box of Object and Plot on Frame
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            right_bboxes.append([x1, y1, x2, y2])

    # Return the Bounding Boxes
    return left_bboxes, right_bboxes