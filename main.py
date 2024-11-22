import os
import cv2
from ultralytics import YOLO, RTDETR
import random
from tracker import Tracker

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

while ret:

    results = model(frame)

    for result in results:
        detections = []
        for res in result.boxes.data.tolist():
            x1, y1, x2, y2, score, classid = res
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            classid = int(classid)

            if score >= threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)


    # cv2.imshow('frame', frame)
    # cv2.waitKey(25)
    cap_out.write(frame)
    ret, frame = cap.read()

cap.release() # release memory
cap_out.release()
cv2.destroyAllWindows()



