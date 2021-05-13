import cv2 as cv
from threading import Thread
import time
import numpy as np


class VideoStreamWidget(object):
    def __init__(self, type):
        if type == 'IR':
            src = 1
            self.capture = cv.VideoCapture(src)
        elif type == 'RGB':
            src = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1024, height=600,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1024, height=(int)600, format=(string)BGRx ! videoconvert !  appsink"
            self.capture = cv.VideoCapture(src,cv.CAP_GSTREAMER)
        else:
            raise Exception("Invalid type.")
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.isStopped = False
        self.isStarted = False
        self.thread.start()

    def update(self):
        while not self.isStopped:
            if self.capture.isOpened():
                if self.isStarted == False:
                    self.isStarted = True
                (self.status, self.frame) = self.capture.read()
            else:
                raise Exception("Could not open capture.")
            time.sleep(.01)

if __name__ == '__main__':
    RGB_stream = VideoStreamWidget('RGB')
    IR_stream = VideoStreamWidget('IR')

    cv.namedWindow("demo",cv.WINDOW_AUTOSIZE)
    cv.startWindowThread()
    print("Running, press ESC or Ctrl-c to exit...")
    while True:
        if RGB_stream.isStarted and IR_stream.isStarted:
            dbl_frame = cv.hconcat([cv.resize(RGB_stream.frame,(320,240)),cv.resize(IR_stream.frame,(320,240))])
            cv.imshow('demo',dbl_frame)
        if cv.waitKey(10) == 27:
            print("Key pressed. Exiting...")
            RGB_stream.isStopped = True
            IR_stream.isStopped = True
            RGB_stream.thread.join()
            IR_stream.thread.join()
            break


print("Releasing RGB...")
RGB_stream.capture.release()
print("Releasing IR...")
IR_stream.capture.release()
print("Destroy all...")
cv.destroyAllWindows()
