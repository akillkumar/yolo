import cv2
import numpy

cap = cv2.VideoCapture ('video.avi')

while True:
    ret, frame = cap.read ()
    cv2.imshow ("Frame", frame)

cv2.waitKey ()
