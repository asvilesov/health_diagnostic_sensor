import cv2 as cv

import numpy as np

cap = cv.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1024, height=600,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1024, height=(int)600, format=(string)BGRx ! videoconvert !  appsink",cv.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open RGB camera.")
    exit()
else:
    cv.namedWindow("RGB",cv.WINDOW_AUTOSIZE)
    print("Running, press ESC or Ctrl-c to exit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive RGB frame. Exiting...")
            break
        
        cv.imshow('RGB',cv.resize(frame,(640,480)))
        if cv.waitKey(5) == 27:
            print("Key pressed. Exiting...")
            break


print("Releasing RGB...")
cap.release()
print("Destroy all...")
cv.destroyAllWindows()
