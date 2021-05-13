
import h5py
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import expanduser

#OS Helper function
home = expanduser("~")
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

class image_dataset(object):

    def __init__(self, transform = True):

        self.image_shape = (64,64,3)
        self.num_images = None
        # self.max_buffer_length = 2000
        self.idxs = None

        self.images = None

        self.load_images()

        self.epoch = 0
        self.batch_num = 0


    def load_images(self, filepath='./data/celebA_img_norm_mask_light.h5', filepath_shade='./data/celebA_shade.h5', shuffle = True):

        #imgs
        f = h5py.File(filepath, 'r')
        dset = f['dataset']
        dset = np.array(dset)
        # dset = np.transpose(dset, (0,2,3,1)) #on GPU NCHW is faster?

        self.mean_pixel = np.reshape(np.array([129.1863,104.7624,93.5940], dtype=np.float32), newshape=(1,3,1,1))
 
        #Extract images and mask
        self.images = dset[:,0:3,:,:].astype(np.float32)
        self.mask   = dset[:,6,:,:].astype(np.float32)

        #spherical harmonics
        f2 = h5py.File(filepath_shade, 'r')
        dset2 = f2['dataset2']
        
        self.shading = np.array(dset2)
        self.shading = np.sum(self.shading, axis = 1) #assume reduction of this axis 3->1
        
        '''Spherical Harmonics Approx'''
        ##compute shading #TODO spherical harmonics
        #normal = dset[:,3:6,:,:]
        #normal = dset[:,3:6,:,:]

        # c = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])
        # self.shading = np.zeros_like(self.mask)
        # print(self.shading.shape)

        # for img in range(dset2.shape[0]):
        #     L = dset2[img]
        #     K = np.array([[L[:,8]*c[0], c[0]*L[:,4], c[0]*L[:,7], c[1]*L[:,3]], 
        #             [c[0]*L[:,4], -c[0]*L[:,8], c[0]*L[:,5], c[1]*L[:,1]],
        #             [c[0]*L[:,7], c[0]*L[:,5], c[2]*L[:,6], c[1]*L[:,2]],
        #             [c[1]*L[:,3], c[1]*L[:,1], c[1]*L[:,2], c[3]*L[:,0] - c[4]*L[:,6]]
        #             ])
        #     n_img = normal[img]
        #     n_img = np.concatenate((n_img, np.ones((1,) + n_img.shape[1:3])), axis= 0)
        #     for i in range(n_img.shape[1]):
        #         for j in range(n_img.shape[2]):
        #             for k in range(3):
        #                 self.shading[img][i][j] = np.sum(n_img[:,i,j]*(K[:,:,k] @ n_img[:,i,j]))
        '''Quick Shading'''
        # light = dset[:,7:10,:,:]
        # self.shading = np.clip(np.sum(np.multiply(normal, light), axis = 1), a_min = 0, a_max = 10).astype(np.float32)

        '''Scaling'''
        self.images = 255 * self.images - self.mean_pixel#(np.power(self.images, 2.2)) - self.mean_pixel
        self.images = self.images.astype(np.float32)
        

        self.num_images = self.max_buffer_length = self.images.shape[0]
        self.image_shape = self.images.shape[1:3]
        
        
        self.idxs = np.arange(self.num_images)
        np.random.shuffle(self.idxs)


    def get_batch(self, batch_size=64):
        indices = self.idxs[self.batch_num*batch_size : min(self.max_buffer_length, (self.batch_num+1)*batch_size)]
        batch = (self.images[indices], self.mask[indices], self.shading[indices])
        
        if((self.batch_num+1)*batch_size > self.max_buffer_length):
            self.batch_num = 0
            self.epoch += 1
        else:
            self.batch_num += 1
        
        return batch

    def _get_batch(self, batch_size=64):
        indices = self.idxs[self.batch_num*batch_size : min(self.max_buffer_length, (self.batch_num+1)*batch_size)]
        batch = (self.images[indices], self.mask[indices], self.shading[indices])
        
        if((self.batch_num+1)*batch_size > self.max_buffer_length):
            self.batch_num = 0
            self.epoch += 1
            np.random.shuffle(self.idxs)
            raise StopIteration
        else:
            self.batch_num += 1
        
        return batch
    
    def __iter__(self):
        return self

    def __next__(self):
        return self._get_batch()
    
    def __len__(self):
        return self.num_images
        

        
    