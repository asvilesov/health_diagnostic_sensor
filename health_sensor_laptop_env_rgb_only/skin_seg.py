#Skin Segmentation NN

import torch
import numpy as np

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
