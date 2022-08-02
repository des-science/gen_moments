import os
import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import camb
from camb import model
from jupyterthemes import jtplot
import math
import time
import sys
moments_path = os.path.realpath(os.path.join(os.getcwd(), '../Moments_analysis/'))
sys.path.insert(0, moments_path)
import h5py as h5
import healpy as hp
import pickle
import pickle as pk
from Moments_analysis import gk_inv
from Moments_analysis import moments_map
import copy
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')

jtplot.reset()


fname = '/global/cfs/cdirs/des/shivamp/gen_moments/Moments_analysis_minsu/Covariance/mask_DES_y3_py2'
mask = load_obj(fname)
print ('f_sky: ', 1./(len(mask)*1./len(mask[mask])))
mask_sm =  hp.sphtfunc.smoothing(mask, (13./60.)*np.pi/180.  )
mask_sm[mask] = 1.
mask = copy.copy(mask_sm)
# computes Cl.
lmax = 1024
nside = 512

import pymaster as nmt
print('loaded')
# Read healpix maps and initialize a spin-0 and spin-2 field
f_0 = nmt.NmtField(mask, [mask])
f_2 = nmt.NmtField(mask, [mask,mask])
bins = nmt.bins.NmtBin.from_lmax_linear(lmax, 1, is_Dell=False)#nmt.bins.NmtBin(nside=1024, ells=2048)


w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f_2, f_2, bins, is_teb=False)
M = w.get_coupling_matrix()
ME = (M[::4,:][:,::4])
# import pdb; pdb.set_trace()
saved = {'M':M, 'ME':ME, 'mask':mask, 'lmax':lmax}
pk.dump(saved, open('namaster_stuff.pk','wb'))


