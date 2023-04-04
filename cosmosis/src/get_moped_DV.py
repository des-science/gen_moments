# %%
import os
from selectors import EpollSelector
from tkinter import E
os.chdir('/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/')
os.environ["COSMOSIS_SRC_DIR"] = "/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/"
import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
# import camb
# from camb import model
# from jupyterthemes import jtplot
import math
import time
import h5py
import sys 
moments_path = os.path.realpath(os.path.join('/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/Moments_analysis/'))
sys.path.insert(0, moments_path)
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

# jtplot.reset()
from tqdm import tqdm




# %%
import cosmosis
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline
from scipy.stats import norm
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits




# %%
# old_stdout = sys.stdout
# sys.stdout = open(os.devnull, 'w')
# pipeline = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/saveDV_LHS_params.ini", 
#     ))

# start_bz_y3fid = pipeline.start_vector()    
# results_bz_all = pipeline.run_results(start_bz_y3fid)
# sys.stdout = old_stdout




# %%
def QR_inverse(matrix):
    _Q, _R = np.linalg.qr(matrix)
    return np.dot(_Q, np.linalg.inv(_R.T))





# %%
# # old_stdout = sys.stdout
# # sys.stdout = open(os.devnull, 'w')
# pipeline = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
#     ))

# start_vec = pipeline.start_vector()    
# results_dict = pipeline.run_results(start_vec)
# # sys.stdout = old_stdout

# kp2_fid = results_dict.block[('gen_moments_theory','kp2_all_array')][indsel_kp2]
# kp3_fid = results_dict.block[('gen_moments_theory','kp3_all_array')][indsel_kp3]
# kp2_kp3_fid = results_dict.block[('gen_moments_theory','kp2_kp3_all_array')][indsel_kp2_kp3]




# %%

# results_dict.block
# kp2_fid


# %%
from tqdm import tqdm
def data_compression(num_params, delta_parm_vals_percentages, indsel_kp2, indsel_kp3, indsel_kp2_kp3, inv_cov_kp2, inv_cov_kp3, inv_cov_kp2_kp3):     
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    pipeline = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
        ))

    start_vec = pipeline.start_vector()    
    results_dict = pipeline.run_results(start_vec)
    kp2_fid = results_dict.block[('gen_moments_theory','kp2_all_array')][indsel_kp2]
    kp3_fid = results_dict.block[('gen_moments_theory','kp3_all_array')][indsel_kp3]
    kp2_kp3_fid = results_dict.block[('gen_moments_theory','kp2_kp3_all_array')][indsel_kp2_kp3]
    sys.stdout = old_stdout
    transf_matrix_kp2 = np.zeros((num_params,len(kp2_fid)))
    transf_matrix_kp3 = np.zeros((num_params,len(kp3_fid)))
    transf_matrix_kp2_kp3 = np.zeros((num_params,len(kp2_kp3_fid)))    

    for ji in tqdm(range(num_params)):
        
        ddh_kp2 = np.zeros((len(kp2_fid),4))
        ddh_kp3 = np.zeros((len(kp3_fid),4))
        ddh_kp2_kp3 = np.zeros((len(kp2_kp3_fid),4))        
        
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        pipeline_ji = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
            ))

        start_vec_ji = pipeline_ji.start_vector()  
        param_fid_ji = start_vec_ji[ji]
        if param_fid_ji != 0:
            start_vec_ji[ji] = param_fid_ji*(1 - 2*delta_parm_vals_percentages[ji])
        else:
            start_vec_ji[ji] = param_fid_ji - 2*0.001
        results_dict_ji = pipeline_ji.run_results(start_vec_ji)
        ddh_kp2[:, 0] = results_dict_ji.block[('gen_moments_theory','kp2_all_array')][indsel_kp2]
        ddh_kp3[:, 0] = results_dict_ji.block[('gen_moments_theory','kp3_all_array')][indsel_kp3]        
        ddh_kp2_kp3[:, 0] = results_dict_ji.block[('gen_moments_theory','kp2_kp3_all_array')][indsel_kp2_kp3]        
        sys.stdout = old_stdout    

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        pipeline_ji = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
            ))

        start_vec_ji = pipeline_ji.start_vector()  
        param_fid_ji = start_vec_ji[ji]
        if param_fid_ji != 0:
            start_vec_ji[ji] = param_fid_ji*(1 - delta_parm_vals_percentages[ji])
        else:
            start_vec_ji[ji] = param_fid_ji - 0.001
        results_dict_ji = pipeline_ji.run_results(start_vec_ji)       
        ddh_kp2[:, 1] = results_dict_ji.block[('gen_moments_theory','kp2_all_array')][indsel_kp2]
        ddh_kp3[:, 1] = results_dict_ji.block[('gen_moments_theory','kp3_all_array')][indsel_kp3]        
        ddh_kp2_kp3[:, 1] = results_dict_ji.block[('gen_moments_theory','kp2_kp3_all_array')][indsel_kp2_kp3]        
        sys.stdout = old_stdout 

        
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        pipeline_ji = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
            ))
        start_vec_ji = pipeline_ji.start_vector()  
        param_fid_ji = start_vec_ji[ji]
        if param_fid_ji != 0:
            start_vec_ji[ji] = param_fid_ji*(1 + delta_parm_vals_percentages[ji])
        else:
            start_vec_ji[ji] = param_fid_ji + 0.001
        results_dict_ji = pipeline_ji.run_results(start_vec_ji)       
        ddh_kp2[:, 2] = results_dict_ji.block[('gen_moments_theory','kp2_all_array')][indsel_kp2]
        ddh_kp3[:, 2] = results_dict_ji.block[('gen_moments_theory','kp3_all_array')][indsel_kp3]        
        ddh_kp2_kp3[:, 2] = results_dict_ji.block[('gen_moments_theory','kp2_kp3_all_array')][indsel_kp2_kp3]        
        sys.stdout = old_stdout 
        
        
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        pipeline_ji = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
            ))        
        start_vec_ji = pipeline_ji.start_vector()  
        param_fid_ji = start_vec_ji[ji]
        if param_fid_ji != 0:
            start_vec_ji[ji] = param_fid_ji*(1 + 2*delta_parm_vals_percentages[ji])
        else:
            start_vec_ji[ji] = param_fid_ji + 2*0.001

        results_dict_ji = pipeline_ji.run_results(start_vec_ji)     
        ddh_kp2[:, 3] = results_dict_ji.block[('gen_moments_theory','kp2_all_array')][indsel_kp2]
        ddh_kp3[:, 3] = results_dict_ji.block[('gen_moments_theory','kp3_all_array')][indsel_kp3]        
        ddh_kp2_kp3[:, 3] = results_dict_ji.block[('gen_moments_theory','kp2_kp3_all_array')][indsel_kp2_kp3]        
        sys.stdout = old_stdout   
        
        if param_fid_ji != 0:
            der_kp2 = -(- ddh_kp2[:,0] +8*ddh_kp2[:,1]- 8*ddh_kp2[:,2]+ ddh_kp2[:,3])/(12*param_fid_ji*delta_parm_vals_percentages[ji])
            der_kp3 = -(- ddh_kp3[:,0] +8*ddh_kp3[:,1]- 8*ddh_kp3[:,2]+ ddh_kp3[:,3])/(12*param_fid_ji*delta_parm_vals_percentages[ji])
            der_kp2_kp3 = -(- ddh_kp2_kp3[:,0] +8*ddh_kp2_kp3[:,1]- 8*ddh_kp2_kp3[:,2]+ ddh_kp2_kp3[:,3])/(12*param_fid_ji*delta_parm_vals_percentages[ji])        
        else:
            der_kp2 = -(- ddh_kp2[:,0] +8*ddh_kp2[:,1]- 8*ddh_kp2[:,2]+ ddh_kp2[:,3])/(12*0.001)
            der_kp3 = -(- ddh_kp3[:,0] +8*ddh_kp3[:,1]- 8*ddh_kp3[:,2]+ ddh_kp3[:,3])/(12*0.001)
            der_kp2_kp3 = -(- ddh_kp2_kp3[:,0] +8*ddh_kp2_kp3[:,1]- 8*ddh_kp2_kp3[:,2]+ ddh_kp2_kp3[:,3])/(12*0.001)        

        transf_matrix_kp2[ji,:]= np.matmul(der_kp2,inv_cov_kp2).T
        transf_matrix_kp3[ji,:]= np.matmul(der_kp3,inv_cov_kp3).T
        transf_matrix_kp2_kp3[ji,:]= np.matmul(der_kp2_kp3,inv_cov_kp2_kp3).T        
        
    return transf_matrix_kp2, transf_matrix_kp3, transf_matrix_kp2_kp3 





# %%
def filter_full_data(data_full, params):
    filter_type = params['type']
    
    sm_all = data_full['sm_all']
    id_kp2_all_finite = data_full['id_kp2_all_finite'][0,:,:].T
    ind_filter_kp2 = []
    for jid in range(len(id_kp2_all_finite)):
        jz1, jz2, i, j = id_kp2_all_finite[jid]
        
        if filter_type == 'auto_zbin':
            if jz1 == jz2:
                ind_filter_kp2.append(jid)
                
        if filter_type == 'simple_test':
            if (jz1 == jz2) and (jz2==3) and (i==j):
                ind_filter_kp2.append(jid)
                

        if filter_type == 'auto_smbin':
            if i == j:
                ind_filter_kp2.append(jid)

        if filter_type == 'auto_zbin_gtsc':
            sc_all = params['sc_all']
            if jz1 == jz2:
                sc_jz_all = np.array([sc_all[jz1], sc_all[jz2]])
                sc_jz_max = np.amax(sc_jz_all)
                # if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]):
                if (sm_all[i] > sc_jz_max) & (sm_all[j] > sc_jz_max):                
                    ind_filter_kp2.append(jid)

        if filter_type == 'auto_smbin_gtsc':
            sc_all = params['sc_all']            
            if i == j:
                sc_jz_all = np.array([sc_all[jz1], sc_all[jz2]])
                sc_jz_max = np.amax(sc_jz_all)
                # if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]):
                if (sm_all[i] > sc_jz_max) & (sm_all[j] > sc_jz_max):                
                    ind_filter_kp2.append(jid)

        if filter_type == 'all_gtsc':
            sc_all = params['sc_all']    
            sc_jz_all = np.array([sc_all[jz1], sc_all[jz2]])
            sc_jz_max = np.amax(sc_jz_all)
            # if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]):
            if (sm_all[i] > sc_jz_max) & (sm_all[j] > sc_jz_max):
                ind_filter_kp2.append(jid)
                
        if filter_type == 'all':
            ind_filter_kp2.append(jid)
        
    ind_filter_kp2 = np.array(ind_filter_kp2)
    # data_kp2_lhs_filter = data_full['kp2_all_lhs'][:, ind_filter_kp2]

    id_kp3_all_finite = data_full['id_kp3_all_finite'][0,:,:].T
    ind_filter_kp3 = []
    for jid in range(len(id_kp3_all_finite)):
        jz1, jz2, jz3, i, j, k = id_kp3_all_finite[jid]
        if filter_type == 'auto_zbin':
            if (jz1 == jz2) and (jz2 == jz3):
                ind_filter_kp3.append(jid)
                
        if filter_type == 'simple_test':
            if (jz1 == jz2) and (jz2==jz3) and (jz2==3) and (i==j) and (j==k):
                ind_filter_kp3.append(jid)
                

        if filter_type == 'auto_smbin':
            if (i == j) and (j == k):
                ind_filter_kp3.append(jid)

        if filter_type == 'auto_zbin_gtsc':
            sc_all = params['sc_all']
            if (jz1 == jz2) and (jz2 == jz3):
                sc_jz_all = np.array([sc_all[jz1], sc_all[jz2], sc_all[jz3]])
                sc_jz_max = np.amax(sc_jz_all)
                # if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]) & (sm_all[k] > sc_all[jz3]):
                if (sm_all[i] > sc_jz_max) & (sm_all[j] > sc_jz_max) & (sm_all[k] > sc_jz_max):                    
                    ind_filter_kp3.append(jid)

        if filter_type == 'auto_smbin_gtsc':
            sc_all = params['sc_all']            
            if (i == j) and (j == k):
                sc_jz_all = np.array([sc_all[jz1], sc_all[jz2], sc_all[jz3]])
                sc_jz_max = np.amax(sc_jz_all)
                # if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]) & (sm_all[k] > sc_all[jz3]):
                if (sm_all[i] > sc_jz_max) & (sm_all[j] > sc_jz_max) & (sm_all[k] > sc_jz_max):                    
                    ind_filter_kp3.append(jid)

        if filter_type == 'all_gtsc':
            sc_all = params['sc_all']         
            sc_jz_all = np.array([sc_all[jz1], sc_all[jz2], sc_all[jz3]])
            sc_jz_max = np.amax(sc_jz_all)
            # if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]) & (sm_all[k] > sc_all[jz3]):
            if (sm_all[i] > sc_jz_max) & (sm_all[j] > sc_jz_max) & (sm_all[k] > sc_jz_max):                    
                ind_filter_kp3.append(jid)
                
        if filter_type == 'all':
            ind_filter_kp3.append(jid)
        
    ind_filter_kp3 = np.array(ind_filter_kp3)
    # data_kp3_lhs_filter = data_full['kp3_all_lhs'][:, ind_filter_kp3]

    return ind_filter_kp2, ind_filter_kp3
    



# %%
# _, _, ind_filter_kp2, ind_filter_kp3 = filter_full_data(data_full, params_filter)




# %%
jr_lhs = 0
nsamp_lhs = 20000
sdir = '/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/saveDVs/'
# data_full = pk.load(open(sdir + 'newIA/' + 'COMBINED_lhs_n' + str(nsamp_lhs) + '_jr' + str(jr_lhs) + '.pk','rb'))
# fname = sdir + 'OmAsonly/' + 'COMBINED_lhs_n' + str(nsamp_lhs) + '_jr' + str(jr_lhs) + '.pk'
# fname = sdir + 'OmAsonly/' + 'COMBINED_lhs_n' + str(nsamp_lhs) + '_jr' + str(jr_lhs) + '.pk'
fname = sdir + 'OmAsnarrow/' + 'COMBINED_lhs_n' + str(nsamp_lhs) + '_jr' + str(jr_lhs) + '.pk'
# np.save(fname, saved, allow_pickle=True)
import h5py
hf = h5py.File(fname, 'r')




# %%
data_full = {'kp2_all_lhs':hf['kp2_all_lhs'], 'kp3_all_lhs':hf['kp3_all_lhs'], 
         'id_kp2_all_finite':hf['id_kp2_all_finite'],
        'id_kp3_all_finite':hf['id_kp3_all_finite'],
         'sm_all':hf['sm_all']
        }




# %%
sc_all = [45.0, 28.0, 18.0, 18.0]

params_filter = {'type':'auto_smbin_gtsc', 'sc_all':sc_all}
# params_filter = {'type':'all_gtsc', 'sc_all':sc_all}
# params_filter = {'type':'all', 'sc_all':sc_all}
# params_filter = {'type':'simple_test', 'sc_all':sc_all}
# var_th = 0.9999999
df = pk.load(open('/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/fid_sim_measurements.pk','rb'))  

# saved = {'kp2_all_fid':kp2_all_fid, 'kp3_all_fid':kp3_all_fid, 'kp3_all_data':kp3_all_data, 'kp2_all_data':kp2_all_data}
kp2_all_fid = df['kp2_all_fid']
kp3_all_fid = df['kp3_all_fid']
kp3_all_data = df['kp3_all_data']
kp2_all_data = df['kp2_all_data']
kp2_kp3_all_data = np.hstack((kp2_all_data, kp3_all_data))
kp2_kp3_all_fid = np.concatenate((kp2_all_fid, kp3_all_fid))

ind_filter_kp2, ind_filter_kp3 = filter_full_data(data_full, params_filter)
ind_filter_kp2_kp3 = np.concatenate((ind_filter_kp2, hf['kp2_all_lhs'].shape[1] + ind_filter_kp3))

mean_kp2sims = np.mean(kp2_all_data.T, axis=1)
mean_kp3sims = np.mean(kp3_all_data.T, axis=1)
mean_kp2kp3sims = np.mean(kp2_kp3_all_data.T, axis=1)





# %%
len(ind_filter_kp2), len(ind_filter_kp3)


# %%
# cov_mat_kp2sims = np.cov(kp2_all_data[:,ind_filter_kp2].T)
# invcovmat_kp2sims = np.linalg.inv(cov_mat_kp2sims)
# snr_kp2_orig = np.sqrt(np.matmul(mean_kp2sims[ind_filter_kp2],np.matmul(invcovmat_kp2sims, mean_kp2sims[ind_filter_kp2].T)))
# print(snr_kp2_orig)



# %%
cov_mat_kp2sims = np.cov(kp2_all_data[:,ind_filter_kp2].T) + 0.15*np.diag(np.diag(np.cov(kp2_all_data[:,ind_filter_kp2].T)))
invcovmat_kp2sims = QR_inverse(cov_mat_kp2sims)
snr_kp2_orig = np.sqrt(np.matmul(mean_kp2sims[ind_filter_kp2],np.matmul(invcovmat_kp2sims, mean_kp2sims[ind_filter_kp2].T)))
print(snr_kp2_orig)




# %%
# snr_kp2_orig 


# %%
# cov_mat_kp3sims = np.cov(kp3_all_data[:,ind_filter_kp3].T)
# invcovmat_kp3sims = np.linalg.inv(cov_mat_kp3sims)
# snr_kp3_orig = np.sqrt(np.matmul(mean_kp3sims[ind_filter_kp3],np.matmul(invcovmat_kp3sims, mean_kp3sims[ind_filter_kp3].T)))
# print(snr_kp3_orig)



# %%
cov_mat_kp3sims = np.cov(kp3_all_data[:,ind_filter_kp3].T) + 0.15*np.diag(np.diag(np.cov(kp3_all_data[:,ind_filter_kp3].T)))
invcovmat_kp3sims = QR_inverse(cov_mat_kp3sims)
snr_kp3_orig = np.sqrt(np.matmul(mean_kp3sims[ind_filter_kp3],np.matmul(invcovmat_kp3sims, mean_kp3sims[ind_filter_kp3].T)))
print(snr_kp3_orig)




# %%
# np.amin(np.diag(cov_mat_kp3sims))


# %%
# cov_mat_kp2kp3sims = np.cov(kp2_kp3_all_data[:,ind_filter_kp2_kp3].T)
# invcovmat_kp2kp3sims = np.linalg.inv(cov_mat_kp2kp3sims)
# snr_kp2kp3_orig = np.sqrt(np.matmul(mean_kp2kp3sims[ind_filter_kp2_kp3],np.matmul(invcovmat_kp2kp3sims, mean_kp2kp3sims[ind_filter_kp2_kp3].T)))
# print(snr_kp2kp3_orig)



# %%
# cov_mat_kp2kp3sims = np.cov(kp2_kp3_all_data[:,ind_filter_kp2_kp3].T)
cov_mat_kp2kp3sims = np.cov(kp2_kp3_all_data[:,ind_filter_kp2_kp3].T) + 0.15*np.diag(np.diag(np.cov(kp2_kp3_all_data[:,ind_filter_kp2_kp3].T)))
invcovmat_kp2kp3sims = QR_inverse(cov_mat_kp2kp3sims)
snr_kp2kp3_orig = np.sqrt(np.matmul(mean_kp2kp3sims[ind_filter_kp2_kp3],np.matmul(invcovmat_kp2kp3sims, mean_kp2kp3sims[ind_filter_kp2_kp3].T)))
print(snr_kp2kp3_orig)




# %%
# num_params = 2
# delta_parm_vals_percentages = [0.005, 0.005]

num_params = 16
delta_parm_vals_percentages = 0.005*np.ones(num_params)

transf_matrix_kp2, transf_matrix_kp3, transf_matrix_kp2_kp3  = \
data_compression(num_params, delta_parm_vals_percentages, ind_filter_kp2, ind_filter_kp3, ind_filter_kp2_kp3, invcovmat_kp2sims, invcovmat_kp3sims, invcovmat_kp2kp3sims)







# %%
# transf_matrix_kp2.shape
kp2_transf_mat = np.dot(transf_matrix_kp2,kp2_all_data[:,ind_filter_kp2].T)
kp3_transf_mat = np.dot(transf_matrix_kp3,kp3_all_data[:,ind_filter_kp3].T)
kp2_kp3_transf_mat = np.dot(transf_matrix_kp2_kp3,kp2_kp3_all_data[:,ind_filter_kp2_kp3].T)




# %%
# kp2_transf_mat.shape
kp2_transf_mean = np.mean(kp2_transf_mat, axis=1)
kp3_transf_mean = np.mean(kp3_transf_mat, axis=1)
kp2_kp3_transf_mean = np.mean(kp2_kp3_transf_mat, axis=1)



# %%
cov_kp2_tranf_mat = np.cov(kp2_transf_mat)
invcov_kp2_tranf_mat = np.linalg.inv(cov_kp2_tranf_mat)
np.sqrt(np.matmul(kp2_transf_mean,np.matmul(invcov_kp2_tranf_mat, kp2_transf_mean.T)))




# %%
cov_kp3_tranf_mat = np.cov(kp3_transf_mat)
invcov_kp3_tranf_mat = np.linalg.inv(cov_kp3_tranf_mat)
np.sqrt(np.matmul(kp3_transf_mean,np.matmul(invcov_kp3_tranf_mat, kp3_transf_mean.T)))




# %%
cov_kp2kp3_tranf_mat = np.cov(kp2_kp3_transf_mat)
invcov_kp2kp3_tranf_mat = QR_inverse(cov_kp2kp3_tranf_mat)
np.sqrt(np.matmul(kp2_kp3_transf_mean,np.matmul(invcov_kp2kp3_tranf_mat, kp2_kp3_transf_mean.T)))




# %%
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
pipeline = LikelihoodPipeline(Inifile("/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/cosmosis/ini/MOPED_params.ini", 
    ))

start_vec = pipeline.start_vector()    
results_dict = pipeline.run_results(start_vec)
kp2_fid = results_dict.block[('gen_moments_theory','kp2_all_array')][ind_filter_kp2]
kp3_fid = results_dict.block[('gen_moments_theory','kp3_all_array')][ind_filter_kp3]
kp2_kp3_fid = results_dict.block[('gen_moments_theory','kp2_kp3_all_array')][ind_filter_kp2_kp3]
sys.stdout = old_stdout



# %%
# transf_matrix_kp2.shape
kp2_transf_fid = np.dot(transf_matrix_kp2,kp2_fid.T)
kp3_transf_fid = np.dot(transf_matrix_kp3,kp3_fid.T)
kp2_kp3_transf_fid = np.dot(transf_matrix_kp2_kp3,kp2_kp3_fid.T)






# %%
# (kp2_transf_fid - kp2_transf_mean)/kp2_transf_fid
# kp2_transf_fid
import pickle as pk

# %%
saved = {'transf_matrix_kp2':transf_matrix_kp2, 'transf_matrix_kp3':transf_matrix_kp3, 'transf_matrix_kp2_kp3':transf_matrix_kp2_kp3,
        'kp2_transf_mean':kp2_transf_mean, 'kp3_transf_mean':kp3_transf_mean, 'kp2_kp3_transf_mean':kp2_kp3_transf_mean,
         'kp2_transf_fid':kp2_transf_fid, 'kp3_transf_fid':kp3_transf_fid, 'kp2_kp3_transf_fid':kp2_kp3_transf_fid,
        'cov_kp2_tranf_mat':cov_kp2_tranf_mat, 'cov_kp3_tranf_mat':cov_kp3_tranf_mat, 'cov_kp2kp3_tranf_mat':cov_kp2kp3_tranf_mat,
         'ind_filter_kp2':ind_filter_kp2, 'ind_filter_kp3':ind_filter_kp3, 'ind_filter_kp2_kp3':ind_filter_kp2_kp3
        }

pk.dump(saved, open('measurements_allparams_auto_smbin_gtsc_MOPED_inf15p.pk','wb'))
# pk.dump(saved, open('measurements_OmAsonly_auto_smbin_gtsc_MOPED_inf15p.pk','wb'))
# pk.dump(saved, open('measurements_OmAsonly_all_gtsc_MOPED_inf15p.pk','wb'))
# pk.dump(saved, open('measurements_allparams_all_gtsc_MOPED_inf15p.pk','wb'))
# dill.dump(saved, open('measurements_OmAsonly_all_ncomp60.pk','wb'))


import pdb; pdb.set_trace()

