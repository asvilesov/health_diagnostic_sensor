import h5py
import numpy as np

filepath="./data/zx_7_d10_inmc_celebA_01.hdf5"
filepath2="./data/zx_7_d3_lrgb_celebA_01.hdf5"

f = h5py.File(filepath, 'r')
dset = f['zx_7']
dset = np.array(dset)

#spherical harmonics
f2 = h5py.File(filepath2, 'r')
dset_spher = f2['zx_7']
dset_spher = np.array(dset_spher)

filepath="./data/zx_7_d10_inmc_celebA_0"
filepath2="./data/zx_7_d3_lrgb_celebA_0"

for i in range(1,5):

    f = h5py.File(filepath+str(i+1) + ".hdf5", 'r')
    temp_dset = f['zx_7']
    dset = np.concatenate((dset, temp_dset), axis=0)

    f2 = h5py.File(filepath2+str(i+1) + ".hdf5", 'r')
    temp_dset_spher = f2['zx_7']
    dset_spher = np.concatenate((dset_spher, temp_dset_spher), axis = 0)

#Save original dataset as whole
hf = h5py.File('celebA_img_norm_mask_light.h5', 'w')
hf.create_dataset('dataset', data=dset)
hf.close()

#Extract images and mask
images = dset[:,0:3,:,:].astype(np.float32)
mask   = dset[:,6,:,:].astype(np.float32)
#compute shading #TODO spherical harmonics
normal = dset[:,3:6,:,:]

c = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])
shading = np.zeros_like(normal)

print(shading.shape)

#Shading
for img in range(shading.shape[0]):
    if(img % 500 == 0):
        print("batch: ", img)

    L = dset_spher[img]
    K = np.array([[L[:,8]*c[0], c[0]*L[:,4], c[0]*L[:,7], c[1]*L[:,3]], 
            [c[0]*L[:,4], -c[0]*L[:,8], c[0]*L[:,5], c[1]*L[:,1]],
            [c[0]*L[:,7], c[0]*L[:,5], c[2]*L[:,6], c[1]*L[:,2]],
            [c[1]*L[:,3], c[1]*L[:,1], c[1]*L[:,2], c[3]*L[:,0] - c[4]*L[:,6]]
            ])
    n_img = normal[img]
    n_img = np.concatenate((n_img, np.ones((1,) + n_img.shape[1:3])), axis= 0)
    for i in range(n_img.shape[1]):
        for j in range(n_img.shape[2]):
            for k in range(3):
                shading[img][k][i][j] = np.sum(n_img[:,i,j]*(K[:,:,k] @ n_img[:,i,j]))

hf2 = h5py.File('celebA_shade.h5', 'w')
hf2.create_dataset('dataset2', data=shading)
hf2.close()

print(dset.shape)
print(dset_spher.shape)