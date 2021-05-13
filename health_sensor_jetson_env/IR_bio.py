# This will be the script with the bounding box on IR feed - RGB feed will not be shown.
# This script DOES NOT WORK on the Jetson - see line 301 - 303.
# See README.txt for details on this script
import mmcv, cv2
import torch
import time
import numpy as np
import skin_seg as ss
import platform

from PIL import Image, ImageDraw
from IPython import display
from threading import Thread
from facenet_pytorch import MTCNN
from utils import *
from uvctypes import *
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from bioFace import biofaces as bf
from bioFace.illum_models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)



skin_model = ss.ConvAutoEncoderArrayDecoder(num_decoders=1)
skin_model.to(device)
skin_model.load_state_dict(torch.load("models/skin_seg"))
skin_model.eval()

#Load Pretrained Model
bioFinder = bf.BioFaces(illuminants=(illA,illD,illF), 
                        camera_spectral_sensitivities=spec_senstivity,
                        skin_reflectance=skin_reflect_arr,
                        t_mat=t_mat,
                        batch_size = 64)

bioFinder.load_state_dict(torch.load("models/skin_seg_spherical2"))
bioFinder.eval()
bioFinder.to(device)
mean_pixel = np.reshape(np.array([129.1863,104.7624,93.5940]), newshape=(1,1,3))
t_mean_pixel = torch.tensor([129.1863,104.7624,93.5940], dtype = torch.float32, requires_grad=False)
t_mean_pixel = torch.reshape(t_mean_pixel, shape= (1,3,1,1))




BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr): # Taken from uvc-radiometry.py

    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)
    ).reshape(
        frame.contents.height, frame.contents.width
    ) # no copy

    # data = np.fromiter(
    #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
    # ).reshape(
    #   frame.contents.height, frame.contents.width, 2
    # ) # copy

    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return

    if not q.full():
        q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback) # Taken from uvc-radiometry.py

def ktof(val): # Taken from uvc-radiometry.py
    return (1.8 * ktoc(val) + 32.0)

def ktoc(val): # Taken from uvc-radiometry.py
    return (val - 27315) / 100.0

def raw_to_8bit(data): # Taken from uvc-radiometry.py
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color): # Taken from uvc-radiometry.py
    val = ktof(val_k)
    cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

class VideoStreamWidget(object): # OpenCV streaming class
    def __init__(self, type):
        if type == 'IR': # Not used anymore - should delete
            src = 1
            self.capture = cv2.VideoCapture(src)
        elif type == 'RGB':
            src = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=960, height=540,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)960, height=(int)540, format=(string)BGRx ! videoconvert !  appsink"
            self.capture = cv2.VideoCapture(src,cv2.CAP_GSTREAMER)
            self.face_not_visible = True

        else:
            raise Exception("Invalid type.")
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.isStopped = False
        self.isStarted = False
        self.thread.start()
        print('Thread Started')

        
    def update(self):
        while not self.isStopped:
            if self.capture.isOpened():
                if self.isStarted == False:
                    self.isStarted = True
                    print('Starting capture...')
                (self.status, self.frame) = self.capture.read()
            else:
                raise Exception("Could not open capture.")
            time.sleep(.01)
    
    def __del__(self):
        try:
            self.capture.release()
        except:
            print('Capture already released!')




if __name__ == '__main__':
    calibration = [250, 0]
    Fever_thresh = [94, 96]
    Fever = 3
    fever_frame = np.zeros((480,200,3),np.uint8)
    fever_frame[:] = (0,0,255)
    
    norm_frame = np.zeros((480,200,3),np.uint8)
    norm_frame[:] = (0,255,0)
    
    err_frame = np.zeros((480,200,3),np.uint8)
    err_frame[:] = (0,255,255)

    noface_frame = np.zeros((480,200,3),np.uint8)
    noface_frame[:] = (255,255,255)

    # Initialize IR camera via libuvc
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(1)
    print("IR camera initialized (radiometry)")
    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)
        else:
           print("libuvc: device found")
        
        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                print("uvc_open error")
                exit(1)

            print("device opened!")

            print_device_info(devh)
            print_device_formats(devh)

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
            )

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)
        
            # Start capture threads
            RGB_stream = VideoStreamWidget('RGB')
            # IR_stream = VideoStreamWidget('IR')

            cv2.namedWindow("demo",cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow("IR Radiometry",cv2.WINDOW_AUTOSIZE)
            # cv2.startWindowThread()
            print("Running, press ESC or Ctrl-c to exit...")

            try:
                while True:
                    data_ir = q.get(True,500)
                    if data_ir is None:
                        break
                    
                    
                    if RGB_stream.isStarted:
                        # Detect faces
                        frame = Image.fromarray(RGB_stream.frame)
                        RGB_stream.boxes, RGB_stream.probs, RGB_stream.landmarks = mtcnn.detect(frame, landmarks=True)
                        # Draw faces
                        # frame_draw = frame.copy()
                        

                        # Check if net found faces
                        # if(RGB_stream.boxes is not None):
                        if 0:
                            RGB_stream.face_not_visible = False
                            data_ir = cv2.resize(data_ir[:,:],(640,480))
                            for box, landmark in zip(RGB_stream.boxes, RGB_stream.landmarks): # need to fix so that only first box,landmark is chosen
                                
                                box[0] = box[0]-calibration[0]
                                box[1] = box[1]-calibration[1]
                                box[2] = box[2]-calibration[0]
                                box[3] = box[3]-calibration[1]

                                for i in range(0, len(landmark)):
                                    landmark[i][0] = landmark[i][0] - calibration[0]
                                    landmark[i][1] = landmark[i][1] - calibration[1]
                                landmark = landmark[0:3]
                                # idx 0 - left eye
                                # idx 1 - right eye
                                # idx 2 - botom of nose
                               
                              
                            a = [round(landmark[0][0]), round(landmark[0][1])]
                            b = [round(landmark[1][0]), round(landmark[2][1])]
                            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_ir[a[1]:b[1],a[0]:b[0]])
                            minLoc = list(minLoc)
                            maxLoc = list(maxLoc)
                            for i in range(0,2):
                                minLoc[i] = minLoc[i] + a[i]
                                maxLoc[i] = maxLoc[i] + a[i]
                            minLoc = tuple(minLoc)
                            maxLoc = tuple(maxLoc)
                            print(ktof(maxVal))
                            img_ir = raw_to_8bit(data_ir)
                            display_temperature(img_ir,minVal,minLoc, (255,0,0))
                            display_temperature(img_ir,maxVal,maxLoc, (0,0,255))
                            frame_draw = Image.fromarray(img_ir).copy()
                            draw = ImageDraw.Draw(frame_draw)
                            for box, landmark in zip(RGB_stream.boxes, RGB_stream.landmarks): # same here.
                                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                                draw.polygon(landmark[0:3], outline=(255,0,0))
                                # print(landmark.shape)
                                # print(landmark)
                            if ktof(maxVal) > Fever_thresh[1]:
                                Fever = 1
                                print("Fever detected!")
                            elif ktof(maxVal) < Fever_thresh[0]:
                                Fever = 2
                                print("ERROR: Temperature is too low. Please move face closer to camera and/or remove glasses.")
                            else:
                                Fever = 0
                                print("Normal temperature.")
                        else:
                            RGB_stream.face_not_visible = True
                            print("Face not visible!")
                            data_ir = cv2.resize(data_ir[:,:],(640,480))
                            img_ir = raw_to_8bit(data_ir)
                            frame_draw = Image.fromarray(img_ir).copy()


                        idx = 0
                        if(RGB_stream.boxes is not None): 
                            np_frame = np.asarray(frame)
                            area, idx = calc_area(RGB_stream.boxes) # return area and idx of largest box (small filter if there are multiple people)
                            box = RGB_stream.boxes[idx]
                            # calculate center of largest box and get bounding box
                            squares = calc_square_box(box, img_dim = np_frame.shape)
                            # extract 
                            square_seg = np_frame[squares[3]:squares[1], squares[2]:squares[0]]
                            square_frame_original = cv2.resize(square_seg, (64,64), interpolation = cv2.INTER_LINEAR)
                            #Scaling for 64x64 model input
                            square_frame = np.transpose(square_frame_original, (2,0,1)) / 256
                            square_frame = np.reshape(square_frame, newshape=(1,) + square_frame.shape)
                            square_img = torch.tensor(square_frame, dtype=torch.float32)
                            square_img_64 = (256*square_img - t_mean_pixel).repeat(64, 1, 1, 1)
                            square_img = square_img.to(device)
                            square_img_64 = square_img_64.to(device)
                            #Get Mask
                            masks, _ = skin_model(square_img)
                            print('This next line does not work on the Jetson and crashes.')
                            print('This is due to 6GB size of the bioFinder model.')
                            print('We attempted to quantize the model to reduce its size, but suffered compatibility issues.')
                      
                            output = bioFinder(square_img_64)
                            recons = output[0].cpu().detach().numpy()
                            diff   = output[2].cpu().detach().numpy()
                            spec   = output[3].cpu().detach().numpy()
                            fmel   = output[4].cpu().detach().numpy()
                            fhem   = output[5].cpu().detach().numpy()
                            
                            masks = masks[0].cpu().detach().numpy()
                            mask = np.transpose(masks[0], (1,2,0))
                            mask64 = mask > 0.6
                            mask_reshape = cv2.resize(mask, (square_seg.shape[1], square_seg.shape[0]))
                            mask = mask_reshape > 0.6 #mask
                            mask = np.reshape(mask, mask.shape + (1,))
                            img_masked = square_seg * mask #mask overlayed on image
                            
                            print('7')
                            # cv2.imshow('img_crop', square_seg)
                            # cv2.imshow('mask', mask_reshape)
                            cv2.imshow('mask on image', cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR))
                            print('8')
                            recons = cv2.cvtColor(np.clip(np.transpose(recons[0], (1,2,0)) + mean_pixel, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR) * mask64
                            
                            diffs = np.transpose(diff[0], (1,2,0)) * mask64
                            print(diffs.shape, np.mean(diffs, axis =(0,1)), np.std(diffs), print(diffs.max()))
                            # diffs = diffs / (np.max(diffs))
                            diffs = np.clip(np.concatenate((diffs,diffs,diffs), axis = 2) * 255, 0, 255)
                            print(diffs.shape, np.mean(diffs, axis =(0,1)), np.std(diffs))
                            print('9')
                            specs = np.transpose(spec[0], (1,2,0)) * mask64
                            print(specs.shape, np.mean(specs, axis =(0,1)), np.std(specs), print(np.max(specs)))
                            # specs = specs / np.max(specs)
                            specs = np.clip(np.concatenate((specs,specs,specs), axis = 2) * 255, 0, 255)
                            print(specs.shape, np.mean(specs, axis =(0,1)), np.std(specs))
                            print()
                            print('10')
                            fmel = 255 * np.transpose(fmel[0], (1,2,0)) * mask64
                            fmel = cv2.applyColorMap(fmel.astype(np.uint8), cv2.COLORMAP_JET)
                            fhem = 255 * np.transpose(fhem[0], (1,2,0)) * mask64
                            fhem = cv2.applyColorMap(fhem.astype(np.uint8), cv2.COLORMAP_JET)
                            # if(time.perf_counter() - start > 4):
                            #     frames.append(np.mean(fhem))
                            #     frame_num += 1
                            hori = np.concatenate((recons, specs.astype(np.uint8), diffs.astype(np.uint8), fmel, fhem), axis = 1)
                            cv2.imshow('mel', hori)
                            print('11')
                            draw.rectangle(squares.tolist(), outline=(0, 255, 0), width=6)
                            print('12')
                        # Display the resulting frame
                        if Fever == 0:
                            dbl_frame = cv2.hconcat([cv2.resize(np.asarray(frame_draw),(640,480)),norm_frame])
                        elif Fever == 1:
                            dbl_frame = cv2.hconcat([cv2.resize(np.asarray(frame_draw),(640,480)),fever_frame])
                        elif Fever == 2:
                            dbl_frame = cv2.hconcat([cv2.resize(np.asarray(frame_draw),(640,480)),err_frame])
                        else:
                            dbl_frame = cv2.hconcat([cv2.resize(np.asarray(frame_draw),(640,480)),noface_frame])
                        cv2.imshow('demo', dbl_frame)

                        # Hiding raw input (uncomment for debug)
                        # print('Display dual raw input')
                        # dbl_frame = cv2.hconcat([cv2.resize(RGB_stream.frame,(320,240)),cv2.resize(IR_stream.frame,(320,240))])

                    if cv2.waitKey(10) == 27:
                        print("Key pressed. Exiting...")
                        RGB_stream.isStopped = True
                        RGB_stream.thread.join()
                        # IR_stream.isStopped = True
                        # IR_stream.thread.join()
                        break
                
            finally:
                libuvc.uvc_stop_streaming(devh)
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        print("Releasing IR & RGB...")
        libuvc.uvc_exit(ctx)
        RGB_stream.capture.release()
# print("Releasing IR...")
# IR_stream.capture.release()
cv2.destroyAllWindows()
