import h5py as h5
import numpy as np
import healpy as hp
import pickle
from Moments_analysis import gk_inv
from Moments_analysis import apply_random_rotation,addSourceEllipticity,convert_to_pix_coord
from Moments_analysis import moments_map
import copy

import tqdm


nside_out = 512
nside = 512
lmax = nside_out*2


std_e = 0.28 # typical values for DES Y3 between 0.26-0.33, depending on the tomographic bin.
n_density = 5.51/4. # gal/arcmin^2; 5.51 is the full des y3 catalog,  divide by 4 because we have 4 tomographc bins.
n_gal_per_pixel = np.int(n_density*hp.nside2resol(nside_out, arcmin = True)**2)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def rotate_map_approx(mask,rot_angles, nside=512, flip=False):
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



mask = load_obj("./Covariance/mask_DES_y3_py2")

mask = hp.ud_grade(mask, nside_out = 512)
masks = [mask, rotate_map_approx(mask,[ 180 ,0 , 0], flip=False), rotate_map_approx(mask,[  90 ,0 , 0], flip=True), rotate_map_approx(mask,[ 270 ,0 , 0], flip=True)]

conf = dict()
# conf['smoothing_scales'] = np.array([8.2, 13.125, 21.0,33.6,54.,86., 137.6, 220.16]) # arcmin
conf['smoothing_scales'] = np.array([21.0,33.6,54.,86., 137.6, 220.16]) # arcmin
conf['nside'] = nside
conf['lmax'] = conf['nside']*2
conf['verbose'] = False
conf['output_folder'] = './simulated_moments'

mcal_moments = moments_map(conf)


momentsEE_2_2 = []
momentsEE_2_3 = []
momentsEE_3_3 = []

momentsEEE_2_2_2 = []
momentsEEE_3_2_2 = []
momentsEEE_2_3_3 = []
momentsEEE_3_3_3 = []


maps1 = load_obj('./maps/kappa_2ndbin_512_1batch')
maps2 = load_obj('./maps/kappa_3rdbin_512_1batch')
label = 1

for i in tqdm.tqdm(range(25), ascii=True):

    map1 = maps1[i]

    g1_map,g2_map = gk_inv(map1,map1*0.,conf['nside'],conf['nside']*2)

    e1_noisemap = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g1_map))
    e2_noisemap = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g2_map))

    e1_noisemap_ = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g1_map))
    e2_noisemap_ = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g2_map))

    # add noise
    e1_map,e2_map = addSourceEllipticity({'shear1':g1_map,'shear2':g2_map},{'e1':e1_noisemap,'e2':e2_noisemap},es_colnames=("e1","e2"))

    map2 = maps2[i]

    g1_map2,g2_map2 = gk_inv(map2,map2*0.,conf['nside'],conf['nside']*2)

    e1_noisemap2 = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g1_map))
    e2_noisemap2 = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g2_map))

    e1_noisemap_2 = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g1_map))
    e2_noisemap_2 = np.random.normal(0,std_e/np.sqrt(n_gal_per_pixel),len(g2_map))

    # add noise
    e1_map2,e2_map2 = addSourceEllipticity({'shear1':g1_map2,'shear2':g2_map2},{'e1':e1_noisemap2,'e2':e2_noisemap2},es_colnames=("e1","e2"))


    for rot in range(4):

        mask_r = masks[rot]

        mcal_moments.add_map(e1_map*mask_r, field_label = 'e1', tomo_bin = 2)
        mcal_moments.add_map(e2_map*mask_r, field_label = 'e2', tomo_bin = 2)
        mcal_moments.add_map(e1_noisemap_*mask_r, field_label = 'e1r', tomo_bin = 2)
        mcal_moments.add_map(e2_noisemap_*mask_r, field_label = 'e2r', tomo_bin = 2)

        mcal_moments.add_map(e1_map2*mask_r, field_label = 'e1', tomo_bin = 3)
        mcal_moments.add_map(e2_map2*mask_r, field_label = 'e2', tomo_bin = 3)
        mcal_moments.add_map(e1_noisemap_2*mask_r, field_label = 'e1r', tomo_bin = 3)
        mcal_moments.add_map(e2_noisemap_2*mask_r, field_label = 'e2r', tomo_bin = 3)


        mcal_moments.transform_and_smooth('convergence'+str(label),'e1' ,'e2' , shear = True, tomo_bins = [2, 3], overwrite = True , skip_conversion_toalm = False)
        mcal_moments.transform_and_smooth('noise'+str(label),      'e1r','e2r', shear = True, tomo_bins = [2, 3], overwrite = True , skip_conversion_toalm = False)

        mcal_moments.compute_moments_gen( label_moments='kEkE_', field_label1 ='convergence'+str(label)+'_kE', denoise1 = 'noise'+str(label)+'_kE',  tomo_bins1 = [2, 3])

        # needed to denoise 3rd moments
        mcal_moments.compute_moments_gen( label_moments='kEkN_', field_label1 ='convergence'+str(label)+'_kE', field_label2 = 'noise'+str(label)+'_kE', denoise1 = 'noise'+str(label)+'_kE',  tomo_bins1 = [2, 3])


        momentsEE_2_2.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['2_2']))
        momentsEE_2_3.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['2_3']))
        momentsEE_3_3.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['3_3']))

        # I think the transposes are right here, but think hard about how to transpose when you are denoising.
        # IE, which redshift-smoothing combination is the noise at any given transpose
        EEEdenoise = mcal_moments.moments["kEkN_"]['2_2_2']
        momentsEEE_2_2_2.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['2_2_2'] - EEEdenoise - np.transpose(EEEdenoise, [2,1,0]) - np.transpose(EEEdenoise, [1,0,2]) ))
        EEEdenoise = mcal_moments.moments["kEkN_"]['3_2_2']
        momentsEEE_3_2_2.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['3_2_2'] - EEEdenoise - np.transpose(EEEdenoise, [2,1,0]) - np.transpose(EEEdenoise, [1,0,2]) ))
        EEEdenoise = mcal_moments.moments["kEkN_"]['2_3_3']
        momentsEEE_2_3_3.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['2_3_3'] - EEEdenoise - np.transpose(EEEdenoise, [2,1,0]) - np.transpose(EEEdenoise, [1,0,2]) ))
        EEEdenoise = mcal_moments.moments["kEkN_"]['3_3_3']
        momentsEEE_3_3_3.append(copy.deepcopy(mcal_moments.moments["kEkE_"]['3_3_3'] - EEEdenoise - np.transpose(EEEdenoise, [2,1,0]) - np.transpose(EEEdenoise, [1,0,2]) ))





momentsEE_2_2 = np.array(momentsEE_2_2)
momentsEE_2_3 = np.array(momentsEE_2_3)
momentsEE_3_3 = np.array(momentsEE_3_3)

momentsEEE_2_2_2 = np.array(momentsEEE_2_2_2)
momentsEEE_3_2_2 = np.array(momentsEEE_3_2_2)
momentsEEE_2_3_3 = np.array(momentsEEE_2_3_3)
momentsEEE_3_3_3 = np.array(momentsEEE_3_3_3)



save_obj('./masked_noised_cov/EE_2_2_0', momentsEE_2_2)
save_obj('./masked_noised_cov/EE_2_3_0', momentsEE_2_3)
save_obj('./masked_noised_cov/EE_3_3_0', momentsEE_3_3)

save_obj('./masked_noised_cov/EEE_2_2_2_0', momentsEEE_2_2_2)
save_obj('./masked_noised_cov/EEE_3_2_2_0', momentsEEE_3_2_2)
save_obj('./masked_noised_cov/EEE_2_3_3_0', momentsEEE_2_3_3)
save_obj('./masked_noised_cov/EEE_3_3_3_0', momentsEEE_3_3_3)
