import pymaster as nmt

import h5py
import healpy as hp
import sys

# compute coupling matries with Namaster.
import os
import numpy as np
import scipy
from scipy import integrate
from jupyterthemes import jtplot
import math
import time
import sys
import h5py as h5
import healpy as hp
import pickle
import copy

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')


print('loaded')
# Read mask and apodize it on a scale of ~1deg
#mask = nmt.mask_apodization(mask,1., apotype="Smooth")
#hp.mollview(mask, coord=['G', 'C'], title='Apodized mask')
#plt.show()


# load the mask
mask = load_obj("mask_DES_y3_py2")
#save_obj("mask_DES_y3_py2",mask)
print ('f_sky: ', 1./(len(mask)*1./len(mask[mask])))
mask_sm =  hp.sphtfunc.smoothing(mask, (3./60.)*np.pi/180.  )
mask_sm[mask] = 1.
mask = copy.copy(mask_sm)
# computes Cl.
lmax = 1024
nside = 512

print(1)

alms_mask = hp.map2alm(mask, lmax=lmax)  # Spin transform
Cl_mask =  hp.sphtfunc.alm2cl(alms_mask)

print(2)

# Read healpix maps and initialize a spin-0 and spin-2 field
f_0 = nmt.NmtField(mask, [mask])
f_2 = nmt.NmtField(mask, [mask,mask])
bins = nmt.bins.NmtBin.from_lmax_linear(1024, 1, is_Dell=False)#nmt.bins.NmtBin(nside=1024, ells=2048)
print(3)

w = nmt.NmtWorkspace()
print(4)
w.compute_coupling_matrix(f_0, f_0, bins, is_teb=False)
print(5)
Mgg = w.get_coupling_matrix()
print(6)
save_obj('Mgg',Mgg)



w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f_2, f_2, bins, is_teb=False)
M = w.get_coupling_matrix()
ME = (M[::4,:][:,::4])
print(4)


name_file ='mode_coupling_matrix_NaMaster_3arcmin_sm_{0}_{1}.h5'.format(lmax,nside)

# you might need to transpose these. check it ***
h5f = h5py.File(name_file, 'w')
h5f.create_dataset('ME', data=(ME[:1024,:][:,:1024]).reshape(1024*1024))
h5f.create_dataset('MB', data=Mgg*0)
h5f.create_dataset('MgE', data=Mgg*0)
h5f.create_dataset('Mgg', data=(Mgg[:1024,:][:,:1024]).reshape(1024*1024))
h5f.close()
print(5)
