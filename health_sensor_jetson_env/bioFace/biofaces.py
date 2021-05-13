
import torch
import numpy as np
from bioFace.illum_models import *

class ConvAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
       
        #Encoder
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)  
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)  
        self.conv4 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        self.batchNorm1 = torch.nn.BatchNorm2d(32)
        self.batchNorm2 = torch.nn.BatchNorm2d(64)
        self.batchNorm3 = torch.nn.BatchNorm2d(128)
        self.batchNorm4 = torch.nn.BatchNorm2d(256)
        self.batchNorm5 = torch.nn.BatchNorm2d(512)

    def forward(self, x):
        #Encoder
        x1 = torch.nn.functional.relu(self.batchNorm1(self.conv1(x)))
        x2 = self.pool(x1)
        x2 = torch.nn.functional.relu(self.batchNorm2(self.conv2(x2)))
        x3 = self.pool(x2)
        x3 = torch.nn.functional.relu(self.batchNorm3(self.conv3(x3)))
        x4 = self.pool(x3)
        x4 = torch.nn.functional.relu(self.batchNorm4(self.conv4(x4)))
        x5 = self.pool(x4)
        y = torch.nn.functional.relu(self.batchNorm5(self.conv5(x5)))
        
              
        return (y, x4, x3, x2, x1)

class ConvAutoDecoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoDecoder, self).__init__()

        #Decoder
        self.t_conv1 = torch.nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.t_conv2 = torch.nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.t_conv3 = torch.nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.t_conv4 = torch.nn.Conv2d(64  + 32, 32, 3, padding=1)
        self.t_conv5 = torch.nn.Conv2d(32, 1, 3, padding=1) #remember to one channel
        
        self.upsample = torch.nn.Upsample(scale_factor=2, mode = 'nearest')

        self.t_batchNorm1 = torch.nn.BatchNorm2d(256)
        self.t_batchNorm2 = torch.nn.BatchNorm2d(128)
        self.t_batchNorm3 = torch.nn.BatchNorm2d(64)
        self.t_batchNorm4 = torch.nn.BatchNorm2d(32)
    
    def forward(self, y, x4, x3, x2, x1):
        #Decoder
        x_back = self.upsample(y)
        x_back = torch.cat((x_back, x4), 1)
        x = torch.nn.functional.relu(self.t_batchNorm1(self.t_conv1(x_back)))
        x = self.upsample(x)
        x = torch.cat((x,x3), 1)
        x = torch.nn.functional.relu(self.t_batchNorm2(self.t_conv2(x)))
        x = self.upsample(x)
        x = torch.cat((x,x2), 1)
        x = torch.nn.functional.relu(self.t_batchNorm3(self.t_conv3(x)))
        x = self.upsample(x)
        x = torch.cat((x,x1), 1)
        x = torch.nn.functional.relu(self.t_batchNorm4(self.t_conv4(x)))
        # x = torch.nn.functional.sigmoid(self.t_conv5(x))
        x = self.t_conv5(x)

        return x

class ConvEncoderArrayDecoder(torch.nn.Module):
    def __init__(self):
        super(ConvEncoderArrayDecoder, self).__init__()

        self.encoder = ConvAutoEncoder()
        self.decoder = ConvAutoDecoder()


    def forward(self, x):

        y = self.encoder(x)
        x_recon = self.decoder(*y)

        return x_recon

class ConvAutoEncoderArrayDecoder(torch.nn.Module):
    def __init__(self, num_decoders):
        super(ConvAutoEncoderArrayDecoder, self).__init__()

        self.num_decoders = num_decoders
        self.encoder = ConvAutoEncoder()
        self.decoders = torch.nn.ModuleList([ConvAutoDecoder() for i in range(num_decoders)])


    def forward(self, x):

        y = self.encoder(x)
        x_recon = [decoder(*y) for decoder in self.decoders]

        return x_recon, y[0]

class Scale_Bio_Camera(torch.nn.Module):
    def __init__(self):
        super(Scale_Bio_Camera, self).__init__()

        self.num_decoder = 4
        self.illum_weights_size = 15
        self.camera_weights_size = 2
        
        #Encoder (enc-dec)
        self.imgEncoder = ConvAutoEncoderArrayDecoder(self.num_decoder)

        #Extract Camera and Illum Model weights
        self.cnn1 = torch.nn.Conv2d(512, 512, 4, padding=0) #decompose to 1D
        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, self.camera_weights_size + self.illum_weights_size)

        self.bN1 = torch.nn.BatchNorm2d(512)
        self.bN2 = torch.nn.BatchNorm1d(512)

        #Scaling Vals
        self.softmax  = torch.nn.functional.softmax
        self.sigmoid    = torch.nn.functional.sigmoid

    def forward(self, x):

        enc_output, y = self.imgEncoder(x)
        
        y = torch.nn.functional.relu(self.bN1(self.cnn1(y)))
        y = torch.flatten(y, start_dim=1)
        y = torch.nn.functional.relu(self.bN2(self.fc1(y)))
        y = self.fc2(y)

        fmel, fblood, diffuse, specular = enc_output
        #scaling
        fmel    = 2*self.sigmoid(fmel) - 1
        fblood  = 2*self.sigmoid(fblood) - 1
        shading = torch.exp(diffuse)
        specular= torch.exp(specular)
        
        illum_weights   = y[:,0:self.illum_weights_size-1]
        illum_temp      = y[:,self.illum_weights_size]
        camera_weights  = y[:,-self.camera_weights_size:]
        #scaling
        illum_weights   = self.softmax(illum_weights)
        illum_temp      = 21 / (1 + torch.exp(illum_temp))
        camera_weights  = 6*self.sigmoid(camera_weights) - 3
        #TODO bgrid?

        #outputs
        return (fmel, fblood), shading, specular, (illum_weights, illum_temp), camera_weights

''' --------------------------------------------------- Physical Models and Reconstruction --------------------------------------------------- '''

class Illumination(torch.nn.Module):
    def __init__(self, illumA, illumD, illumF, temp_range):
        super(Illumination, self).__init__()
        # self.illumA = torch.nn.Parameter(illumA, requires_grad=False)
        # self.illumD = torch.nn.Parameter(illumD, requires_grad=False)
        # self.illumF = torch.nn.Parameter(illumF, requires_grad=False)
        # self.temp_range = torch.nn.Parameter(temp_range, requires_grad=False)
        # self.illumA = torch.tensor(illumA, dtype = torch.float32, requires_grad=False)
        # self.illumD = torch.tensor(illumD, dtype = torch.float32, requires_grad=False)
        # self.illumF = torch.tensor(illumF, dtype = torch.float32, requires_grad=False)
        # self.temp_range = torch.tensor(temp_range, dtype = torch.float32, requires_grad=False)

        self.register_buffer('illumA', torch.tensor(illumA, dtype = torch.float32, requires_grad=False))
        self.register_buffer('illumD', torch.tensor(illumD, dtype = torch.float32, requires_grad=False))
        self.register_buffer('illumF', torch.tensor(illumF, dtype = torch.float32, requires_grad=False))
        self.register_buffer('temp_range', torch.tensor(temp_range, dtype = torch.float32, requires_grad=False))

    def forward(self, illum_weights):

        illum_weights, illum_temp = illum_weights
        illA_weight = illum_weights[:,0]
        illD_weight = illum_weights[:,1]
        illF_weight = illum_weights[:,2:15]

        #linear Interpolation
        illum_temp_ceil = torch.ceil(illum_temp)
        illum_temp_floor = torch.floor(illum_temp)
        illum_alpha = illum_temp_ceil - illum_temp_ceil
        illum_alpha = torch.reshape(illum_alpha, illum_alpha.shape + (1,))
        # print(illum_alpha.shape)
        illD_weight_ceil = self.illumD[illum_temp_ceil.long()]
        illD_weight_floor = self.illumD[illum_temp_floor.long()]
        # print(illD_weight_ceil.shape)
        illD = illD_weight_floor * illum_alpha + illD_weight_ceil * (1 - illum_alpha)
        # print(illD.shape)
        # print(torch.sum(illD, 1))


        #Pseudo Gaussian Interpolation #TODO Numerical Errors?
        # illum_temp = torch.multiply(torch.reshape(illum_temp, illum_temp.shape + (1,)), torch.ones(illum_temp.shape + self.temp_range.shape) ) 
        # # print((self.temp_range - illum_temp).shape)
        # # print((self.temp_range - illum_temp))
        # temp_weights = torch.exp(1 - torch.abs((self.temp_range - illum_temp)))
        # # print(torch.sum(temp_weights, 0))
        # temp_weights = temp_weights / torch.sum(temp_weights, 0)
        # # print(temp_weights)
        # su = torch.sum(temp_weights, 1)
        # # print(su)
        # # print(su.shape)

        E =     torch.reshape(illA_weight, illA_weight.shape + (1,)) * self.illumA 
        E +=    torch.reshape(illD_weight, illD_weight.shape + (1,)) * illD#torch.matmul(temp_weights, self.illumD) 
        E +=    torch.matmul(illF_weight, self.illumF)

        # print(torch.matmul(temp_weights, self.illumD).shape)
        # print(torch.sum(torch.matmul(temp_weights, self.illumD),1) )
        # print()
        # print(torch.sum(temp_weights, 1))
        # print(torch.sum(self.illumD, 1))
        # print(torch.sum(illum_weights, 1))

        return E

class Camera(torch.nn.Module):
    def __init__(self, mean, pc):
        super(Camera, self).__init__()

        self.register_buffer('mean', torch.tensor(mean, dtype = torch.float32, requires_grad=False))
        self.register_buffer('pc', torch.tensor(pc, dtype = torch.float32, requires_grad=False))
        self.register_buffer('wavelength', torch.tensor(self.mean.shape[0] / 3, dtype = torch.long, requires_grad=False))

    def forward(self, b):

        ss = torch.nn.functional.relu(torch.matmul(b, self.pc) + self.mean)

        ss_r = ss[:,0:self.wavelength]
        ss_g = ss[:,self.wavelength:2*self.wavelength]
        ss_b = ss[:,2*self.wavelength:3*self.wavelength]

        return (ss_r, ss_g, ss_b)

#### Sloppy TODO

class LightColour(torch.nn.Module):
    def __init__(self):
        super(LightColour, self).__init__()

    def forward(self, e, cam_ss):

        ss_r, ss_g, ss_b = cam_ss

        l_r = torch.sum(torch.multiply(e, ss_r), dim=1)
        l_g = torch.sum(torch.multiply(e, ss_g), dim=1)
        l_b = torch.sum(torch.multiply(e, ss_b), dim=1)

        # print(l_r.shape)
        # print(l_g.shape)
        # print(l_b.shape)

        l_t = torch.stack((l_r, l_g, l_b), dim = 1)

        return l_t

class computeSpecularities(torch.nn.Module):
    def __init__(self):
        super(computeSpecularities, self).__init__()

    def forward(self, spec, l_t):

        spec_spd_r = torch.multiply(spec, l_t[:,0])
        spec_spd_g = torch.multiply(spec, l_t[:,1])
        spec_spd_b = torch.multiply(spec, l_t[:,2])

        spec_spd = torch.cat((spec_spd_r, spec_spd_g, spec_spd_b), dim = 1)
        # spec_spd = torch.reshape(spec_spd, shape=spec_spd.shape[0:2] + spec_spd.shape[3:5])

        return spec_spd

class computeSkinReflectance(torch.nn.Module):
    def __init__(self, skin_reflectance_map, batch_size = 64):
        """

        Args:
            skin_reflectance_map ([d1 x d2 x discretewavlengthvec]): [assume d1 axis is fblood and d2 axis is fmel]
        """
        super(computeSkinReflectance, self).__init__()
        skin_reflec = torch.tensor(skin_reflectance_map, dtype = torch.float32, requires_grad=False)
        skin_reflec = torch.reshape(skin_reflec, shape = (1,) + skin_reflec.shape)
        self.register_buffer('skin_reflec', skin_reflec.repeat(batch_size, 1, 1, 1))
        self.register_buffer('skin_dim', torch.tensor(skin_reflectance_map.shape[0], dtype = torch.long, requires_grad=False))

    def forward(self, bio_maps):
        fmel, fblood = bio_maps
        bio_map = torch.stack((fblood, fmel))
        bio_map = torch.reshape(bio_map, shape = bio_map.shape[0:2] +bio_map.shape[3:5])
        bio_map = bio_map.permute(1,2,3,0)
        #Thank god.
        r_biomap = torch.nn.functional.grid_sample(self.skin_reflec, bio_map)

        return r_biomap

class imageFormation(torch.nn.Module):
    def __init__(self):
        super(imageFormation, self).__init__()

    def forward(self, bio_reflectance, spec_sensitivity, illum, specularities, shading):

        # print(spec_sensitivity[0].shape)
        spec_shape = spec_sensitivity[0].shape

        illum = torch.reshape(illum, shape= illum.shape + (1,1,))
        # print(illum.shape)
        spectra_ref = torch.multiply(bio_reflectance, illum)
        # print(spectra_ref.shape)

        r = torch.sum(spectra_ref*torch.reshape(spec_sensitivity[0], spec_shape + (1,1,)), dim = 1)
        g = torch.sum(spectra_ref*torch.reshape(spec_sensitivity[1], spec_shape + (1,1,)), dim = 1)
        b = torch.sum(spectra_ref*torch.reshape(spec_sensitivity[2], spec_shape + (1,1,)), dim = 1)

        albedo = torch.stack((r,g,b), dim = 1)
        # print(shading.shape)
        shadedDiffuse = albedo * shading
        
        # print(shadedDiffuse.shape)

        return shadedDiffuse + specularities

class whiteBalance(torch.nn.Module):
    def __init__(self):
        super(whiteBalance, self).__init__()

    def forward(self, rawAppearence, LightColour):

        whiteBalancedImages = rawAppearence / torch.reshape(LightColour, LightColour.shape + (1,1))

        return whiteBalancedImages

class fromRawTosRGB(torch.nn.Module):
    def __init__(self, T_RAW2XYZ, batch_size=64):
        super(fromRawTosRGB, self).__init__()

        t_mat = torch.tensor(T_RAW2XYZ, dtype = torch.float32, requires_grad=False)
        t_mat = torch.reshape(t_mat, shape = (1,) + t_mat.shape)
        self.register_buffer('T_RAW_XYZ', t_mat.repeat(batch_size, 1, 1, 1))

        self.register_buffer('T_XYZ_RAW', torch.tensor([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.057]], dtype = torch.float32, requires_grad=False))

        self.register_buffer('mean_pixel', torch.tensor([129.1863,104.7624,93.5940], dtype = torch.float32, requires_grad=False))
        self.mean_pixel = torch.reshape(self.mean_pixel, shape= (1,3,1,1))

    def forward(self, whiteBalancedImg, b):

        b = torch.reshape(b, shape = (b.shape[0], 1, 1, b.shape[1]))/3
        t_mat = torch.nn.functional.grid_sample(self.T_RAW_XYZ, b)
        # print(t_mat.shape)
        # print(t_mat)
        # print(torch.sum(t_mat[:,0:3], dim = 1))
        # print(torch.sum(t_mat[:,3:6], dim = 1))
        # print(torch.sum(t_mat[:,6:9], dim = 1))
        t_row1 = torch.reshape(torch.sum(t_mat[:,0:3], dim = 1), shape=(64,1,1,1))
        t_row2 = torch.reshape(torch.sum(t_mat[:,3:6], dim = 1), shape=(64,1,1,1))
        t_row3 = torch.reshape(torch.sum(t_mat[:,6:9], dim = 1), shape=(64,1,1,1))

        Ix = torch.sum(whiteBalancedImg * (t_mat[:,0:3]/t_row1), dim = 1)
        Iy = torch.sum(whiteBalancedImg * (t_mat[:,3:6]/t_row2), dim = 1)
        Iz = torch.sum(whiteBalancedImg * (t_mat[:,6:9]/t_row3), dim = 1)
        Ixyz = torch.stack((Ix,Iy,Iz), axis = 1)

        # print(Ixyz)

        #TODO possible errors
        R = torch.sum(torch.reshape(self.T_XYZ_RAW[0], (1,3,1,1)) * Ixyz, dim = 1)
        G = torch.sum(torch.reshape(self.T_XYZ_RAW[1], (1,3,1,1)) * Ixyz, dim = 1)
        B = torch.sum(torch.reshape(self.T_XYZ_RAW[2], (1,3,1,1)) * Ixyz, dim = 1)

        return 255*torch.nn.functional.relu(torch.stack((R,G,B), dim = 1)) - self.mean_pixel
        


class BioFaces(torch.nn.Module):
    def __init__(self, illuminants, camera_spectral_sensitivities, skin_reflectance, t_mat, batch_size):
        super(BioFaces, self).__init__()

        #Data Prelims
        self.illumA, self.illumD, self.illumF = illuminants
        self.cam_spec_sens = camera_spectral_sensitivities
        self.mean_camera, self.pc_scaled = pca(camera_spectral_sensitivities, 2)
        self.skin_reflectance_map = skin_reflectance
        self.t_mat = t_mat
        self.batch_size = batch_size

        #Torch Layers
        # self.convEncoder    = ConvAutoEncoderArrayDecoder(num_decoders=4)
        self.encoder        = Scale_Bio_Camera()
        self.illum_model    = Illumination(illumA=self.illumA, illumD=self.illumD, illumF=self.illumF, temp_range=np.arange(22))
        self.cam_model      = Camera(self.mean_camera, self.pc_scaled)
        self.light_model    = LightColour()
        self.comp_Spec      = computeSpecularities()
        self.skin_reflec_model = computeSkinReflectance(self.skin_reflectance_map, batch_size=self.batch_size)
        self.image_former   = imageFormation()
        self.w_balancer     = whiteBalance() 
        self.find_sRGB      = fromRawTosRGB(self.t_mat)

    def forward(self, images):

        bio, diffuse, specular, illums, camera_specs = self.encoder(images)

        e = self.illum_model(illums)

        ss = self.cam_model(camera_specs)

        light_spectra = self.light_model(e, ss)

        spec_spd = self.comp_Spec(specular, light_spectra)

        reflect_bio = self.skin_reflec_model(bio)

        form_imgs = self.image_former(reflect_bio, ss, e, spec_spd, diffuse)

        bal_imgs = self.w_balancer(form_imgs, light_spectra)

        rgb_imgs = self.find_sRGB(bal_imgs, camera_specs)

        return rgb_imgs, camera_specs, diffuse, specular, bio[0], bio[1]

class bioFacesLoss(object):
    def __init__(self, appearence_loss=1e-3, camera_loss=1e-4, specular_loss=1e-4, diffuse_loss=1e-6):

        self.loss_weights = [appearence_loss, camera_loss, specular_loss, diffuse_loss]
        self.mse_loss = torch.nn.MSELoss()

    
    def __call__(self, images, masks, true_shading, reconstructed_imgs, camera_specs, specular, diffuse):
        # print(reconstructed_imgs.dtype)
        # print(images.dtype)
        # print(masks.dtype)
        loss =  self.loss_weights[0]*torch.sum(torch.square(images*masks - reconstructed_imgs*masks))/64/64#self.mse_loss(reconstructed_imgs*masks, images*masks)
        loss += self.loss_weights[1]*torch.sum(torch.square(camera_specs)) 
        loss += self.loss_weights[2]*torch.sum(specular)

        scale = torch.sum(input=true_shading*diffuse*masks, dim=(2,3)) / torch.sum(input=torch.square(diffuse)*masks, dim=(2,3))
        diffuse = scale*diffuse

        loss += self.loss_weights[3]*torch.sum(torch.square(( (true_shading-diffuse)* masks)))

        return loss











    




