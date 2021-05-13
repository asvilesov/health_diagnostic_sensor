import cv2 as cv

import numpy as np

cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open IR camera.")
    exit()
else:
    cv.namedWindow("IR",cv.WINDOW_AUTOSIZE)
    print("Running, press ESC or Ctrl-c to exit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive IR frame. Exiting...")
            break
        
        cv.imshow('IR',cv.resize(frame,(640,480)))
        if cv.waitKey(5) == 27:
            print("Key pressed. Exiting...")
            break


print("Releasing IR...")
cap.release()
print("Destroy all...")
cv.destroyAllWindows()
