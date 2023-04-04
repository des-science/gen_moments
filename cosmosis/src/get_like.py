from distutils.file_util import copy_file
import sys, os
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import ast
import astropy.constants as const
import astropy.units as units
from cosmosis.datablock import names, option_section
from cosmosis.datablock.cosmosis_py import errors

sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/gen_moments/cosmosis/src/')
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/gen_moments/data/')
import get_unique_comb_kp2_kp3 as get_unique_ind
import numpy as np
# Finally we can now import camb
import camb

cosmo_sec = names.cosmological_parameters
IA_sec = names.intrinsic_alignment_parameters
import healpy as hp
import pickle as pk

sys.path.insert(0, '/global/cfs/cdirs/lsst/www/shivamp/env/CS2_cori/lib/python3.9/site-packages')
import dill


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, protocol=2)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)


def setup(options):

    measure_fname = options.get_string(
        option_section,
        "measure_fname",
        default=
        '/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/src/measurements_auto_smbin_gtsc.pk'
        )
    probe_comb = options.get_string(option_section, "probe_comb", default='kp2_kp3')
    dv_data_type = options.get_string(option_section, "dv_data_type", default='theory')

    df_measure = dill.load(open(measure_fname, 'rb'))

    if probe_comb == 'kp2':
        if dv_data_type == 'theory':
            dv_measure = df_measure['kp2_fid_normalised_pcaonly']
        if dv_data_type == 'sims_mean':
            dv_measure = df_measure['mean_kp2sims_pca']
        cov_mat = df_measure['cov_mat_kp2sims_pca']
        ind_sel = df_measure['ind_filter_kp2']
        norm_func_probe = df_measure['scaler_kp2']
        pca_func_probe = df_measure['pca_kp2']

    if probe_comb == 'kp3':
        if dv_data_type == 'theory':
            dv_measure = df_measure['kp3_fid_normalised_pcaonly']
        if dv_data_type == 'sims_mean':
            dv_measure = df_measure['mean_kp3sims_pca']
        cov_mat = df_measure['cov_mat_kp3sims_pca']
        ind_sel = df_measure['ind_filter_kp3']
        norm_func_probe = df_measure['scaler_kp3']
        pca_func_probe = df_measure['pca_kp3']

    if probe_comb == 'kp2_kp3':
        if dv_data_type == 'theory':
            dv_measure = df_measure['kp2_kp3_fid_normalised_pcaonly']
        if dv_data_type == 'sims_mean':
            dv_measure = df_measure['mean_kp2kp3sims_pca']
        cov_mat = df_measure['cov_mat_kp2kp3sims_pca']
        ind_sel = df_measure['ind_filter_kp2_kp3']
        norm_func_probe = df_measure['scaler_kp2_kp3']
        pca_func_probe = df_measure['pca_kp2_kp3']

    config_out = {}
    config_out['dv_measure'] = dv_measure
    config_out['cov_mat'] = cov_mat
    config_out['inv_cov_mat'] = np.linalg.inv(cov_mat)
    config_out['ind_sel'] = ind_sel
    config_out['norm_func_probe'] = norm_func_probe
    config_out['pca_func_probe'] = pca_func_probe
    config_out['probe_comb'] = probe_comb
    return config_out


def execute(block, config):

    if config['probe_comb'] == 'kp2':
        dv_theory_inp = block[('gen_moments_theory', 'kp2_all_array')][config['ind_sel']]

    if config['probe_comb'] == 'kp3':
        dv_theory_inp = block[('gen_moments_theory', 'kp3_all_array')][config['ind_sel']]

    if config['probe_comb'] == 'kp2_kp3':
        dv_theory_inp = block[('gen_moments_theory', 'kp2_kp3_all_array')][config['ind_sel']]

    dv_theory_scaled = config['norm_func_probe'].transform(np.array([dv_theory_inp]))
    dv_theory_pca = config['pca_func_probe'].transform(dv_theory_scaled)

    delta_dv = dv_theory_pca - config['dv_measure']
    chi2 = np.dot(np.dot(delta_dv, config['inv_cov_mat']), delta_dv.T)[0][0]
    like_val = -0.5 * chi2
    likes = names.likelihoods
    block[likes, 'GM_LIKE'] = like_val
    block['data_vector', 'GM_chi2'] = chi2
    return 0
