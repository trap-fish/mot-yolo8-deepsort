import os
import cv2
from ultralytics import YOLO, RTDETR
import random
from tracker import Tracker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



video_path = os.path.join('.', 'data/okutama', 'okutama-sample.mov')
video_out_path = os.path.join('.', 'data/okutama', 'okutama-tracked-rt-detr.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, 
                          cv2.VideoWriter_fourcc(*'MP4V'), 
                          cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0])
                          )

# define the detection model
#model = YOLO("yolov8n.pt")
model = RTDETR("rtdetr-l.pt")

# initialise Tracker object
tracker = Tracker()

# define colours for bounding boxes
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# set confidence threshold for boundary box assignment
threshold = 0.6

output_dir = "./testing/"
for seq in os.listdir('/media/citi-ai/matthew/MOT17/train/'):
    seq_path = os.path.join('/media/citi-ai/matthew/MOT17/train/', seq, 'img1/')
    seq_output = os.path.join(output_dir, f'{seq}.txt')

    # Initialize video reader
    images = sorted(os.listdir(seq_path))  # Sorted frame filenames
    frame_id = 0  # Initialize frame counter

    with open(seq_output, 'w') as f:
        for img_name in images:
            frame_id += 1
            frame_path = os.path.join(seq_path, img_name)
            frame = cv2.imread(frame_path)

            # Get detections from the model
            results = model(frame)
            detections = []
            for result in results:
                for res in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, classid = res
                    if score >= threshold:
                        detections.append([int(x1), int(y1), int(x2), int(y2), score])

            # update tracker with new detections on current frame
            tracker.update(frame, detections)

            # Write tracker outputs to file in MOTChallenge format
            for track in tracker.tracks:
                x1, y1, x2, y2 = track.bbox
                w, h = x2 - x1, y2 - y1
                track_id = track.track_id
                confidence = max([detection[4] for detection in detections], default=1.0)  # Use the max detection confidence
                f.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},{confidence},-1,-1,-1\n")



cap.release() # release memory
cap_out.release()
cv2.destroyAllWindows()



