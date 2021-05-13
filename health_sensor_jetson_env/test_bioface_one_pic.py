#Run Face Detection and Segmentation

from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

from utils import *
import skin_seg as ss

from bioFace import biofaces as bf
from bioFace.illum_models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
bioFinder = torch.quantization.quantize_dynamic(bioFinder, dtype=torch.float16)
bioFinder.load_state_dict(torch.load("models/quantized_biofaces"))
bioFinder.eval()
# bioFinder.to(device)
mean_pixel = np.reshape(np.array([129.1863,104.7624,93.5940]), newshape=(1,1,3))
t_mean_pixel = torch.tensor([129.1863,104.7624,93.5940], dtype = torch.float32, requires_grad=False)
t_mean_pixel = torch.reshape(t_mean_pixel, shape= (1,3,1,1))


face_not_visible = True

square_seg = cv2.imread('./data/frame11.jpg', 1)
square_frame_original = cv2.resize(square_seg, (64,64), interpolation = cv2.INTER_LINEAR)
print(square_frame_original.shape)

#Scaling for 64x64 model input
square_frame = np.transpose(square_frame_original, (2,0,1)) / 256
square_frame = np.reshape(square_frame, newshape=(1,) + square_frame.shape)
square_img = torch.tensor(square_frame, dtype=torch.float32)
square_img_64 = (256*square_img - t_mean_pixel).repeat(64, 1, 1, 1)
square_img = square_img.to(device)
# square_img_64 = square_img_64.to(device)

#Get Mask
masks, _ = skin_model(square_img)
print('1')
output = bioFinder(square_img_64)
print('2')
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

# cv2.imshow('img_crop', square_seg)
# cv2.imshow('mask', mask_reshape)
cv2.imshow('mask on image', cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR))

recons = cv2.cvtColor(np.clip(np.transpose(recons[0], (1,2,0)) + mean_pixel, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR) * mask64

diffs = np.transpose(diff[0], (1,2,0)) * mask64
print(diffs.shape, np.mean(diffs, axis =(0,1)), np.std(diffs), print(diffs.max()))
# diffs = diffs / (np.max(diffs))
diffs = np.clip(np.concatenate((diffs,diffs,diffs), axis = 2) * 255, 0, 255)
print(diffs.shape, np.mean(diffs, axis =(0,1)), np.std(diffs))

specs = np.transpose(spec[0], (1,2,0)) * mask64
print(specs.shape, np.mean(specs, axis =(0,1)), np.std(specs), print(np.max(specs)))
# specs = specs / np.max(specs)
specs = np.clip(np.concatenate((specs,specs,specs), axis = 2) * 255, 0, 255)
print(specs.shape, np.mean(specs, axis =(0,1)), np.std(specs))
print()

fmel = 255 * np.transpose(fmel[0], (1,2,0)) * mask64
fmel = cv2.applyColorMap(fmel.astype(np.uint8), cv2.COLORMAP_JET)
fhem = 255 * np.transpose(fhem[0], (1,2,0)) * mask64
fhem = cv2.applyColorMap(fhem.astype(np.uint8), cv2.COLORMAP_JET)
# if(time.perf_counter() - start > 4):
#     frames.append(np.mean(fhem))
#     frame_num += 1
hori = np.concatenate((recons, specs.astype(np.uint8), diffs.astype(np.uint8), fmel, fhem), axis = 1)
cv2.imshow('mel', hori)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.destroyAllWindows()


