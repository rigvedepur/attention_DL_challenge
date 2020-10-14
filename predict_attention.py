from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
import argparse
from utils import *
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="myriad", help="select cpu or myriad")

args = vars(ap.parse_args())

# Some constants
confidence = 0.1
w = 300
h = 300

protoPath = "caffe/COCO/pose_deploy.prototxt"
weightsPath = "caffe/COCO/pose_iter_440000.caffemodel"
num_points = 18


net = cv2.dnn.readNetFromCaffe(protoPath, weightsPath)



if args['device'] == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args['device'] == "myriad":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_MYRIAD)
    print("Using Movidius NCS")


# Start video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)
attentive = 0
distracted = 0
absent = 0
t_start = time.time()
while True:


    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    frameCopy = np.copy(frame)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inputBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (w, h), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inputBlob)
    detection = net.forward()

    H = detection.shape[2]
    W = detection.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(num_points):
        # confidence map of corresponding body's part.
        probMap = detection[0, i, :, :]

        # Find global maxima of the probMap.
        _, prob, _, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > confidence:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    t_elapsed = int(time.time() - t_start)

    if points[16] and points[17]:
        print('Looking straight :)')
        attentive += 1
        stats = (attentive, distracted, absent)
        plot_skeletal_map(frame, [0, 1, 14, 15, 16, 17], points, "Current Status:  Attentive", (0,150,0), stats, t_elapsed)

    if points[16] is None and points[17] is not None:
        print('Person looking right :)')
        distracted += 1
        stats = (attentive, distracted, absent)
        plot_skeletal_map(frame, [0, 1, 14, 15, 16, 17], points, "Current Status:  Distracted", (0,0,150), stats, t_elapsed)

    elif points[17] is None and points[16] is not None:
        print('Person looking left :)')
        distracted += 1
        stats = (attentive, distracted, absent)
        plot_skeletal_map(frame, [0, 1, 14, 15, 16, 17], points, "Current Status:  Distracted", (0,0,150), stats, t_elapsed)

    if points[16] is None and points[17] is None:
        print('Current Status:  Absent')
        absent += 1
        stats = (attentive, distracted, absent)
        plot_skeletal_map(frame, [], points, "Current Status:  Absent", (180, 0, 0), stats, t_elapsed)




    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
