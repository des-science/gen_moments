import sys, os
sys.path.insert(0, '/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/')

from Moments_analysis import moments_map
import pickle 
import healpy as hp
import numpy as np
import os
# from astropy.table import Table
import gc
# import pyfits as pf
# from Moments_analysis import g2k_sphere
import timeit
import os
# from bornraytrace import lensing as brk
import numpy as np
# from bornraytrace import intrinsic_alignments as iaa
# import bornraytrace
# from astropy.table import Table    
import healpy as hp
# import frogress
# import pyfits as pf
from astropy.cosmology import z_at_value
# from astropy.cosmology import FlatLambdaCDM
# from astropy import units as u
# import cosmolopy.distance as cd
# from scipy.interpolate import interp1d
import gc
# import pandas as pd
import pickle
# import multiprocessing
# from functools import partial
import h5py as h5
# import dill


def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out

def IndexToDeclRa(index, nside,nest= False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)

def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA,DEC to hpix coordinates
    """

    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    return pix


def addSourceEllipticity(self,es,es_colnames=("e1","e2"),rs_correction=True,inplace=False):

    """

    :param es: array of intrinsic ellipticities, 

    """

    #Safety check
    assert len(self)==len(es)

    #Compute complex source ellipticity, shear
    es_c = np.array(es[es_colnames[0]]+es[es_colnames[1]]*1j)
    g = np.array(self["shear1"] + self["shear2"]*1j)

    #Shear the intrinsic ellipticity
    e = es_c + g
    if rs_correction:
        e /= (1 + g.conjugate()*es_c)

    #Return
    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
    else:
        return (e.real,e.imag)

        
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
        f.close()

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute



def gk_inv(K,KB,nside,lmax):

    alms = hp.map2alm(K, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsE = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsE[ell == 0] = 0.0

    
    alms = hp.map2alm(KB, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsB = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsB[ell == 0] = 0.0

    _,e1t,e2t = hp.alm2map([kalmsE,kalmsE,kalmsB] , nside=nside, lmax=lmax, pol=True)
    return e1t,e2t# ,r



def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048,nosh=True):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!


    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1] * 1.
        almsB = alms[2] * 1. 
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0

    


    almssm = [alms[0], almsE, almsB]


    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE



def rotate_map_approx(mask, rot_angles, flip=False,nside = 2048):
    alpha, delta = hp.pix2ang(nside, np.arange(len(mask)))

    rot = hp.rotator.Rotator(rot=rot_angles, deg=True)
    rot_alpha, rot_delta = rot(alpha, delta)
    if not flip:
        rot_i = hp.ang2pix(nside, rot_alpha, rot_delta)
    else:
        rot_i = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)
    rot_map = mask*0.
    rot_map[rot_i] =  mask[np.arange(len(mask))]
    return rot_map





def make_maps(seed, rand_rotate_parent_sim=False, jnrealize=0):
    config = dict()
    config['sources_bins'] = [0,1,2,3]
    config['nside'] =  512    
    rot = np.mod(seed,4)
    jsim = seed//4
    # if jsim < 10:
    #     ldir_sims = '/global/cfs/cdirs/des/cosmogrid/DESY3/fiducial/cosmo_fiducial/perm_000' + str(jsim) + '/'
    # elif (jsim > 10) & (jsim < 100):
    #     ldir_sims = '/global/cfs/cdirs/des/cosmogrid/DESY3/fiducial/cosmo_fiducial/perm_00' + str(jsim) + '/'
    # elif (jsim > 100) & (jsim < 1000):
    #     ldir_sims = '/global/cfs/cdirs/des/cosmogrid/DESY3/fiducial/cosmo_fiducial/perm_0' + str(jsim) + '/'
    # else:
    #     pass
    ldir_sims = '/global/cfs/cdirs/lsst/www/shivamp/gen_mom/cosmogrid_kappa/fiducial/ns_512/'
    
    # df_sim = h5.File(fname, 'r')    
    g1_tomo = dict()
    g2_tomo = dict()
    
    # read into memory full sky maps from cosmogrid +++++++++++++
    for tomo_bin in (config['sources_bins']):
        # kappa_dfi = df_sim['kg']['desy3metacal' + str(tomo_bin+1)][:]
        fname = ldir_sims + 'kappa-jr-' + str(jsim) + '-jz-' + str(tomo_bin) + '.fits'
        kappa_dfi = hp.read_map(fname)
        if rand_rotate_parent_sim:
            rand_angle = [360.*np.random.rand(), 360.*np.random.rand(), 360.*np.random.rand()]
            kappa_dfi = rotate_map_approx(kappa_dfi, rand_angle, flip=False,nside = config['nside'])
        lmax = 2*config['nside']
        g1i, g2i = gk_inv(kappa_dfi,0.0*kappa_dfi,config['nside'],lmax)
        g1_tomo[tomo_bin] = g1i
        g2_tomo[tomo_bin] = g2i
        if rot ==0:
            pass
        elif (rot ==1):
            g1_tomo[tomo_bin] = rotate_map_approx(g1_tomo[tomo_bin],[ 180 ,0 , 0], flip=False,nside = config['nside'])
            g2_tomo[tomo_bin] = rotate_map_approx(g2_tomo[tomo_bin],[ 180 ,0 , 0], flip=False,nside = config['nside'])
        elif rot ==2:
            g1_tomo[tomo_bin] = rotate_map_approx(g1_tomo[tomo_bin],[ 90 ,0 , 0], flip=True,nside = config['nside'])
            g2_tomo[tomo_bin] = rotate_map_approx(g2_tomo[tomo_bin],[ 90 ,0 , 0], flip=True,nside = config['nside'])
        elif rot ==3:
            g1_tomo[tomo_bin] = rotate_map_approx(g1_tomo[tomo_bin],[ 270 ,0 , 0], flip=True,nside = config['nside'])
            g2_tomo[tomo_bin] = rotate_map_approx(g2_tomo[tomo_bin],[ 270 ,0 , 0], flip=True,nside = config['nside'])
        

    maps_PKDGRAV = dict()
    sources_maps = dict()
  
    ldir_mcal = '/global/cfs/cdirs/lsst/www/shivamp/gen_mom/process_data/'
    maps_PKDGRAV = dict()

    sources_maps = dict()
    # print ('doing the mpas!')
    for tomo_bin in config['sources_bins']:        
        mcal_here = ldir_mcal + 'mcal_tomo_' + str(tomo_bin) + '.h5'
        df_mcal = h5.File(mcal_here, 'r')
        maps_PKDGRAV[tomo_bin] = dict()
        dec1 = df_mcal['dec'][:]
        ra1 = df_mcal['ra'][:]
        w = df_mcal['w'][:]
        
        pix = convert_to_pix_coord(ra1,dec1, nside=config['nside'])
        
        f = 1.
            
        n_map = np.zeros(hp.nside2npix(config['nside']))
        n_map_sc = np.zeros(hp.nside2npix(config['nside']))

        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

        n_map[unique_pix] += np.bincount(idx_rep, weights=w)
        n_map_sc[unique_pix] += np.bincount(idx_rep, weights=w/f**2)

        g1_ = g1_tomo[tomo_bin][pix]
        g2_ = g2_tomo[tomo_bin][pix]
        
        es1,es2 = apply_random_rotation(df_mcal['e1'][:]/f, df_mcal['e2'][:]/f)
        es1a,es2a = apply_random_rotation(df_mcal['e1'][:]/f, df_mcal['e2'][:]/f)

        del df_mcal
        gc.collect()

        x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))

        e1_map_buzz = np.zeros(hp.nside2npix(config['nside']))
        e2_map_buzz = np.zeros(hp.nside2npix(config['nside']))
        e1r_map_buzz = np.zeros(hp.nside2npix(config['nside']))
        e2r_map_buzz = np.zeros(hp.nside2npix(config['nside']))
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

        e1_map_buzz[unique_pix] += np.bincount(idx_rep, weights= x1_sc*w)
        e2_map_buzz[unique_pix] += np.bincount(idx_rep, weights= x2_sc*w)
        e1r_map_buzz[unique_pix] += np.bincount(idx_rep, weights=es1a*w)
        e2r_map_buzz[unique_pix] += np.bincount(idx_rep, weights=es2a*w)

        mask_sims = n_map_sc != 0.
        e1_map_buzz[mask_sims]  = e1_map_buzz[mask_sims]/(n_map_sc[mask_sims])
        e2_map_buzz[mask_sims] =  e2_map_buzz[mask_sims]/(n_map_sc[mask_sims])
        e1r_map_buzz[mask_sims]  = e1r_map_buzz[mask_sims]/(n_map_sc[mask_sims])
        e2r_map_buzz[mask_sims] =  e2r_map_buzz[mask_sims]/(n_map_sc[mask_sims])

        EE,BB,_   =  g2k_sphere(e1_map_buzz, e2_map_buzz, mask_sims, nside=config['nside'], lmax=config['nside']*2 ,nosh=True)
        EEn,BBn,_ =  g2k_sphere(e1r_map_buzz, e2r_map_buzz, mask_sims, nside=config['nside'], lmax=config['nside']*2 ,nosh=True)

        sources_maps[tomo_bin] = {'EE':EE,'EEn':EEn} 

    mask_sims = hp.read_map('/global/cfs/cdirs/lsst/www/shivamp/gen_mom/process_data/mask_sims.fits',verbose=False)
               
    for tomo_bin in config['sources_bins']: 
        sources_maps[tomo_bin]['mask'] = mask_sims
    
    def compute_phmoments(sources_maps = None,output='',lab='/global/cfs/cdirs/lsst/www/shivamp/gen_mom/temp_data2/'):
        if not os.path.exists(output+'.pkl'):
            conf = dict()
            conf['smoothing_scales'] = np.array([8.2,13.1,21.0,33.6,54.,86.,138,221.]) 
            conf['nside'] =512
            conf['lmax'] = conf['nside']*2
            conf['verbose'] = False
            conf['output_folder'] =lab

            mcal_moments = moments_map(conf)

            tomo_bins = [0,1,2,3]
            for t in tomo_bins:
                mcal_moments.add_map(sources_maps[t]['EE'], field_label = 'kE', tomo_bin = t)
                mcal_moments.add_map(sources_maps[t]['EEn'], field_label = 'kN', tomo_bin = t)
                if t == 3:
                    mcal_moments.mask = sources_maps[t]['EE']==sources_maps[t]['EE']

            if not os.path.exists(conf['output_folder']):
                try:
                    os.mkdir(conf['output_folder'])
                except:
                    pass

            mcal_moments.transform_and_smooth('convergence','kE',None, shear = False, tomo_bins = tomo_bins, overwrite = True , skip_loading_smoothed_maps = False)         
            mcal_moments.transform_and_smooth('noise','kN',None, shear = False, tomo_bins = tomo_bins, overwrite = True , skip_loading_smoothed_maps = False)         

            mcal_moments.compute_moments_gen( label_moments='kEkE', field_label1 ='convergence_kE', field_label2 ='convergence_kE',  tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins)
            mcal_moments.compute_moments_gen( label_moments='kEkN', field_label1 ='convergence_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins)
            mcal_moments.compute_moments_gen( label_moments='kNkN', field_label1 ='noise_kE', field_label2 = 'noise_kE', tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins)
            mcal_moments.compute_moments_gen( label_moments='kNkE', field_label2 ='convergence_kE', field_label1 = 'noise_kE',  tomo_bins1 = tomo_bins, tomo_bins2 = tomo_bins)


            # del mcal_moments.smoothed_maps
            # del mcal_moments.fields
            gc.collect()

            # save_obj(output,mcal_moments)
            saved = {}
            for key in mcal_moments.moments.keys():
                saved[key] = mcal_moments.moments[key]
            import dill
            dill.dump(saved, open(output+'.pkl', 'wb'))


  
    output_moments = '/global/cfs/cdirs/lsst/www/shivamp/gen_mom/measure_here'
    label_output1 = 'measure_cosmogrid_jsim_' + str(jsim) + '_rot_' + str(rot) + '_rotparent_' + str(rand_rotate_parent_sim) + '_jnrealize_' + str(jnrealize)
    compute_phmoments(sources_maps = sources_maps,
                          output=output_moments+'/moments_'+label_output1,lab='/global/cfs/cdirs/lsst/www/shivamp/gen_mom/temp_data2/'+'temp_'+label_output1 )




   
# output_moments= '/global/cfs/cdirs/des/mgatti/maps_shivam_mask_noise/'
# output = '/global/cfs/cdirs/des/darkgrid/DarkGrid/'
# nside = 512
# make_maps(0, rand_rotate_parent_sim=False, jnrealize=0)

# noise_rels = 1
# rot_num = 1
from mpi4py import MPI 
from mpi4py import MPI
run_count = 0
n_jobs = 400
# jr = 0
first = 400
last = 800
numbers = np.arange(first, last)

while run_count<n_jobs:
    comm = MPI.COMM_WORLD
    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
    if (run_count+comm.rank) < n_jobs:
        # make_maps(comm.rank)
        make_maps(numbers[run_count+comm.rank], rand_rotate_parent_sim=False, jnrealize=0)
    run_count+=comm.size
    comm.bcast(run_count,root = 0)
    comm.Barrier()     

# first = 0
# last = 800
# numbers = np.arange(first, last)
# # numbers = np.arange(0,5)
# run_count = 0
# number_of_jobs = len(numbers)
# comm = MPI.COMM_WORLD
# # print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
# # try:
# if comm.rank < 400:
#     make_maps(numbers[run_count+comm.rank], rand_rotate_parent_sim=False, jnrealize=0)
#     comm.bcast(run_count,root = 0)
#     comm.Barrier()


# from mpi4py import MPI 
# if __name__ == '__main__':
    
#     runstodo = [...]
   
#     run_count=0
#     while run_count<len(runstodo):
#         comm = MPI.COMM_WORLD
    
#         if (run_count+comm.rank)<len(runstodo):
#             make_maps(runstodo[run_count+comm.rank])

#         run_count+=comm.size
#         comm.bcast(run_count,root = 0)
#         comm.Barrier() 