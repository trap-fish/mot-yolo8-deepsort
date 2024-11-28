import os
import cv2
from ultralytics import RTDETR
import random
from tracker import Tracker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# define the detection model
model = RTDETR("rtdetr-l.pt")

# initialise Tracker object
tracker = Tracker()

# define colours for bounding boxes
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# set confidence threshold for boundary box assignment
threshold = 0.6

output_dir = "./testing/"
img_path = os.path.join('/media/citi-ai/matthew/MOT17/train/MOT17-02-DPM/img1/')

# Initialize video reader
images = ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg'] 

for img_name in images:
    frame_path = os.path.join(img_path, img_name)
    frame = cv2.imread(frame_path)

    # Get detections from the model
    results = model(frame)
    detections = []
    for result in results:
        for res in result.boxes.data.tolist():
            x1, y1, x2, y2, score, classid = res
            if score >= threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), score])
                
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (colors[int(classid) % len(colors)]), 3)
                cv2.putText(frame, f"ID: {int(classid)}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Save the image
    cv2.imwrite(os.path.join(output_dir, f"out_{img_name}"), frame)
    
    # Debug: Check detections
    print(f"Detections for {img_name}: {detections}")

    # Update tracker with detections
    tracker.update(frame, detections)

    # Debug: Check tracks
    print(f"Tracks after update for {img_name}: {[{'id': track.track_id, 'bbox': track.bbox} for track in tracker.tracks]}")

    # Write tracker outputs to image
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id

        # # Draw the bounding box
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
        #               (colors[track_id % len(colors)]), 3)
        # cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # # Save the image
        # cv2.imwrite(os.path.join(output_dir, f"out_{img_name}"), frame)

        # Debug: Bounding box information
        print(f"Track ID: {track_id}, BBox: {x1, y1, x2, y2}")



