'''
    CS 1390 Introduction to Machine Learning
    Monsoon 2020

    Project: Vehicle Detection on Roads
             Using openCV2 and Yolov3
    
    Akhil Kumar
    akhil.kumar_ug21@ashoka.edu.in
'''

import os
import sys
import cv2
import numpy
import time

class COLORS:
	clear = '\033[0m'
	blue  = '\033[94m'
	green = '\033[92m'
	cyan  = '\033[96m'
	red   = '\033[91m'
	yell  = '\033[93m'
	mag   = '\033[35m'

# if user provides a video, we want to use that
if len(sys.argv) > 1:
    try:
        # check if its a valid path we can open
        video = cv2.VideoCapture (sys.argv[1])
    except:
        # if not, throw an error and quit
        print (COLORS.red + "Unable to load video at " + sys.argv[1] + "\n Usage: python tracker.py <path/to/video>"  + COLORS.clear)
        os._exit (1)
else:
    # default test video
    video = cv2.VideoCapture ('vehicles/Urban/march9.avi')

'''
    YOLO setup
'''
# our pre-trained weights and configuration files
cfg     = 'cfg/tiny.cfg'
weights = 'weights/tiny.weights'

# load yolo nerual network
net = cv2.dnn.readNet (weights, cfg)

# get layer names
layers = net.getLayerNames ()

# get output layers
output_layer = []
for i in net.getUnconnectedOutLayers ():
    output_layer.append (layers[i[0] - 1])

# get a list of all classes 
classes = []

with open ('coco.names', 'r') as fp:
    for line in fp:
        classes.append (line.rstrip ())

'''
    Process the video
'''
#frame = cv2.imread ('traffic.jpg')
starting_time = time.time()
frame_count   = 0

while (video.isOpened()):
    ret, frame = video.read ()
    frame_count += 1

    # dimensions
    height, width, channels = frame.shape

    # convert frame to blob
    # with channel = RGB instead of BGR
    # https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
    blob = cv2.dnn.blobFromImage (frame, 0.00392, (416, 416), (0, 0, 0), True)

    # set our neural network to work on the blob
    net.setInput (blob)
    output = net.forward (output_layer)

    # go through every object detected
    detected_classes = []
    confidences      = []
    rectangles       = []

    objects_detected = 0

    for out in output:
        for detection in out:
            score = detection[5:]
            detected_class = numpy.argmax (score)
            confidence = score [detected_class]

            # if we have at least an 80% confidence
            if confidence > 0.5:
                # we have successfully detected an object!
                objects_detected += 1

                # center of the object
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)

                # width and height 
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectangle coordinates
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                detected_classes.append (detected_class)
                confidences.append (float(confidence))
                rectangles.append ([x, y, w, h])

    # start drawing boxes

    # non-maximum suppression for highest confidence rectangle
    # so that we don't have any overlap
    indexes = cv2.dnn.NMSBoxes (rectangles, confidences, 0.5, 0.4)

    for index in indexes:
        i = index[0]
        x, y, w, h = rectangles[i]
        label = str (classes[detected_classes[i]])

        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText  (frame, label, (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        elapsed_time = time.time() - starting_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Vehicle", frame)

    cv2.waitKey (1)




