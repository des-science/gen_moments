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
import h5py
import sys
moments_path = os.path.realpath(os.path.join(os.getcwd(), '../Moments_analysis/'))
sys.path.insert(0, moments_path)
import h5py as h5
import healpy as hp
import pickle
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


import time
t0_init = time.time()
# test LCDM model
bestfit = {}
h=0.6736
bestfit["ombh2"] = 0.0493*h**2
bestfit["omch2"] = (0.26-0.0493-0.0014)*h**2
bestfit["As"] = 3.0775467136912062e-09 #rootfound for this for sig8 = 0.84
bestfit["H0"] = h*100
bestfit["tau"] = 0.5617335E-01
bestfit["ns"] = 0.9649
bestfit["mnu"] = 0.0014*h**2*93.14
bestfit["nnu"] = 3.046

pars_LCDM = camb.set_params(**bestfit, DoLateRadTruncation=True)
pars_LCDM.WantTransfer = True
results_LCDM = camb.get_results(pars_LCDM)
sig8 = results_LCDM.get_sigma8_0()
PK = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4,nonlinear=True, extrap_kmax= 10**10)
PK_L = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4,nonlinear=False, extrap_kmax= 10**10)
chitoz = results_LCDM.redshift_at_comoving_radial_distance
ztochi = results_LCDM.comoving_radial_distance


try:
    import pickle as pk
    df = pk.load(open('namaster_stuff.pk','rb'))
    M = df['M']
    ME = df['ME']
    mask = df['mask']
    lmax = 1024
    nside = 512

except:
    fname = '/global/cfs/cdirs/des/shivamp/gen_moments/Moments_analysis_minsu/Covariance/mask_DES_y3_py2'
    mask = load_obj(fname)
    print ('f_sky: ', 1./(len(mask)*1./len(mask[mask])))
    mask_sm =  hp.sphtfunc.smoothing(mask, (13./60.)*np.pi/180.  )
    mask_sm[mask] = 1.
    mask = copy.copy(mask_sm)
    # computes Cl.
    
    
    import pymaster as nmt
    print('loaded')
    # Read healpix maps and initialize a spin-0 and spin-2 field
    f_0 = nmt.NmtField(mask, [mask])
    f_2 = nmt.NmtField(mask, [mask,mask])
    bins = nmt.bins.NmtBin.from_lmax_linear(1024, 1, is_Dell=False)#nmt.bins.NmtBin(nside=1024, ells=2048)


    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_2, f_2, bins, is_teb=False)
    M = w.get_coupling_matrix()
    ME = (M[::4,:][:,::4])
    
    
def compute_Plz_mat(PK, zbin, lmax = 1024, mask = None):
    '''
    It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (k=l/chi(z)).
    '''
    nz = len(zbin)
    nell = lmax
    chi_zbin = results_LCDM.comoving_radial_distance(zbin)
    ell = np.arange(lmax)
    z_mat = np.tile(zbin.reshape(nz, 1), (1, nell))
    chi_mat = np.tile(chi_zbin.reshape(nz, 1), (1, nell))
    ell_mat = np.tile(ell.reshape(1, nell), (nz, 1))
    k_mat = ell_mat/chi_mat
    P_lz_mat = np.exp(PK.ev(z_mat, np.log(k_mat + 1e-6)))
    F_l =  hp.sphtfunc.pixwin(512, lmax = lmax)[:lmax]
    F_l_mat = np.tile(F_l.reshape(1, nell), (nz, 1))
    P_lz_mat *= (F_l_mat)**2
    if mask is not None:
        f_l = (ell+2)*(ell-1)/(ell*(ell+1))
        f_l[0:2] = 0
        f_l_mat = np.tile(f_l[:lmax].reshape(1, nell), (nz, 1))
        P_lz_mat *= f_l_mat
        
        P_lz_mat_maskv =  np.zeros((nz, nell))
        for jz in range(nz):
            P_lz_mat_maskv[jz, :] = mask[:lmax,:lmax]@P_lz_mat[jz, :lmax]
        f_linv = (ell*(ell+1))/((ell+2)*(ell-1))
        f_linv[0:2] = 0
        f_linv_mat = np.tile(f_linv.reshape(1, nell), (nz, 1))
        P_lz_mat = P_lz_mat_maskv*f_linv_mat  
    return P_lz_mat


def get_Dz_knletc(PK_L, zbin, lmax=1024):
    nz = len(zbin)
    nell = lmax
    knl_nz = np.zeros(nz)
    Dz_nz = np.zeros(nz)
    ns_mat = np.zeros((nz, nell))
    for jz in range(nz):
        z = zbin[jz]
        def helper(k):
            return k**3*PK_L.P(z, k)/(2*np.pi**2)-1
        knl = scipy.optimize.root(helper, 0.5).x[0]
        knl_nz[jz] = knl
        Dz_nz[jz] = np.sqrt(PK_L.P(z, 1)/PK_L.P(0, 1))
        kay = np.arange(lmax)/results_LCDM.comoving_radial_distance(z)
        ns = (kay/PK_L.P(0,kay))*(PK_L.P(0,kay+0.001)-PK_L.P(0,kay))/(0.001)
        ns_mat[jz, :] = ns
    return Dz_nz, knl_nz, ns_mat

def compute_factosum(zbin, smoothing_scales1, smoothing_scales2, lmax = 1024):
    '''
    It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (k=l/chi(z)).
    '''
    nz = len(zbin)
    nell = lmax
    ell = np.arange(lmax)   
    
    fac_to_sum = np.zeros((nz, nell, len(smoothing_scales1)))
    for i, sm in enumerate(zip(smoothing_scales1,smoothing_scales2)):
        # convert scale to radians ***
        sm_rad1 =(sm[0]/60.)*np.pi/180.
        sm_rad2 =(sm[1]/60.)*np.pi/180.

        # smoothing kernel (top-hat)
        A1 = 1./(2*np.pi*(1-np.cos(sm_rad1)))
        A2 = 1./(2*np.pi*(1-np.cos(sm_rad2)))
        B = np.sqrt(np.pi/(2.*ell+1.0))
        fact1 = -B*(scipy.special.eval_legendre(ell+1,np.cos(sm_rad1))-scipy.special.eval_legendre(ell-1,np.cos(sm_rad1)))*A1
        fact1[0] = 1/(4*np.pi) # MINSUP THIS WAS 1, I THINK FACT1*FACT2 SHOULD BE 1/4PI WHEN l=0? check this by plotting and making it smooth
        fact2 = -B*(scipy.special.eval_legendre(ell+1,np.cos(sm_rad2))-scipy.special.eval_legendre(ell-1,np.cos(sm_rad2)))*A2
        fact2[0] = 1.0
        fact1_mat = np.tile(fact1.reshape(1, nell), (nz, 1))
        fact2_mat = np.tile(fact2.reshape(1, nell), (nz, 1))
        fac_to_sum[:, :, i] = fact1_mat * fact2_mat
    return fac_to_sum


def compute_fact_dfact_kappa3(zbin, sm1, lmax = 1024):
    '''
    It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (k=l/chi(z)).
    '''
    nz = len(zbin)
    nell = lmax
    ell = np.arange(lmax)   
    fact_dfact_kappa3_to_sum = np.zeros((nz, nell, 2))
    sm_rad1 =(sm1/60.)*np.pi/180.
    fact1 = (scipy.special.eval_legendre(ell-1,np.cos(sm_rad1))-scipy.special.eval_legendre(ell+1,np.cos(sm_rad1)))/(4*np.pi*(1-np.cos(sm_rad1)))
    fact1[0] = 1./4*np.pi
    fact_dfact_kappa3_to_sum[:,:,0] = np.tile(fact1.reshape(1, nell), (nz, 1))
    d_fact1 =  scipy.special.eval_legendre(ell,np.cos(sm_rad1))*np.sin(sm_rad1)/(1-np.cos(sm_rad1))
    d_fact1 -= fact1*(4*np.pi/(2*ell+1))*np.sin(sm_rad1)/(1-np.cos(sm_rad1))
    d_fact1[0] = 0  #0th Wl = 1

    fact_dfact_kappa3_to_sum[:,:,1] = np.tile(d_fact1.reshape(1, nell), (nz, 1))
        
    return fact_dfact_kappa3_to_sum
        

def compute_abc_kappa3(zbin, Dz_nz, knl_nz, ns_mat, scheme='SC'):
    nz = len(zbin)
    nell = lmax
    ell = np.arange(lmax)  
    knl_mat = np.tile(knl_nz.reshape(nz, 1), (1, nell))
    Dz_mat = np.tile(Dz_nz.reshape(nz, 1), (1, nell))    
    
    chi_zbin = results_LCDM.comoving_radial_distance(zbin)
    ell = np.arange(lmax)
    z_mat = np.tile(zbin.reshape(nz, 1), (1, nell))
    chi_mat = np.tile(chi_zbin.reshape(nz, 1), (1, nell))
    ell_mat = np.tile(ell.reshape(1, nell), (nz, 1))
    k_mat = ell_mat/chi_mat    
    q_mat = k_mat/knl_mat
        # Initialise coefficients small-scales fitting formulae.
    if scheme == 'SC':
        coeff = [0.25,3.5,2.,1.,2.,-0.2,1.,0.,0.]
    elif scheme == 'GM':
        coeff = [0.484,3.740,-0.849,0.392,1.013,-0.575,0.128,-0.722,-0.926]

    a = (1. + ((sig8*Dz_mat)**coeff[5])*(0.7*(4.-2.**ns_mat)/(1.+2.**(2.*ns_mat+1)))**0.5*(q_mat*coeff[0])**(ns_mat+coeff[1]))/(1.+(q_mat*coeff[0])**(ns_mat+coeff[1]))
    b = (1. + 0.2*coeff[2]*(ns_mat+3)*(q_mat*coeff[6])**(ns_mat+coeff[7]+3))/(1.+(q_mat*coeff[6])**(ns_mat+coeff[7]+3.5));
    c = (1. + 4.5*coeff[3]/(1.5+(ns_mat+3)**4)*(q_mat*coeff[4])**(ns_mat+3+coeff[8]))/(1+(q_mat*coeff[4])**(ns_mat+3.5+coeff[8]));
    a[:,0] = 1.
    b[:,0] = 1.
    c[:,0] = 1.
    return a, b, c
        


def compute_masked_m12_from_factosum(zbin, P_lz_mat, fac_to_sum, abc=1.0):  
    nz = len(zbin)
    moments = np.zeros((nz, 1))
    # for i, sm in enumerate(zip(smoothing_scales1,smoothing_scales2)):
    moments[:, 0] = np.sum(abc*fac_to_sum[:,:,0]*P_lz_mat, axis=1)
    if zbin[0] == 0:
        moments[0,:] = 0.0*P_lz_mat.shape[1]
    return moments


def compute_masked_m123_vec(P_lz_mat, zbin, a, b, c, smoothing_scales1, smoothing_scales2, smoothing_scales3, lmax = 1024, scheme = 'LIN', mask = None):
    '''
    It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (k=l/chi(z)).
    '''
    nz = len(zbin)
    nell = lmax
    
    moments12a = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales1, smoothing_scales2)], abc=a)
    moments13a = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales1, smoothing_scales3)], abc=a)
    moments23a = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales2, smoothing_scales3)], abc=a)        

    moments12b = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales1, smoothing_scales2)], abc=b)
    moments13b = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales1, smoothing_scales3)], abc=b)
    moments23b = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales2, smoothing_scales3)], abc=b)        

    moments12c = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales1, smoothing_scales2)], abc=c)
    moments13c = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales1, smoothing_scales3)], abc=c)
    moments23c = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(smoothing_scales2, smoothing_scales3)], abc=c)        
    
    smoothing_scales1 = np.array([smoothing_scales1])
    smoothing_scales2 = np.array([smoothing_scales2])
    smoothing_scales3 = np.array([smoothing_scales3])    
    d_moments12_d_ln1b = np.zeros((nz, len(smoothing_scales1)))
    d_moments12_d_ln2b = np.zeros((nz, len(smoothing_scales1)))
    d_moments13_d_ln1b = np.zeros((nz, len(smoothing_scales1)))
    d_moments13_d_ln3b = np.zeros((nz, len(smoothing_scales1)))
    d_moments23_d_ln2b = np.zeros((nz, len(smoothing_scales1)))
    d_moments23_d_ln3b = np.zeros((nz, len(smoothing_scales1)))

    for i, sm in enumerate(zip(smoothing_scales1,smoothing_scales2,smoothing_scales3)):

        # convert scale to radians ***
        sm_rad1 =(sm[0]/60.)*np.pi/180.
        sm_rad2 =(sm[1]/60.)*np.pi/180.
        sm_rad3 =(sm[2]/60.)*np.pi/180.

        fact1 = fact_dfact_kappa3_to_sum_all[sm[0]][:,:,0]
        fact2 = fact_dfact_kappa3_to_sum_all[sm[1]][:,:,0]
        fact3 = fact_dfact_kappa3_to_sum_all[sm[2]][:,:,0]        

        d_fact1 = fact_dfact_kappa3_to_sum_all[sm[0]][:,:,1]
        d_fact2 = fact_dfact_kappa3_to_sum_all[sm[1]][:,:,1]
        d_fact3 = fact_dfact_kappa3_to_sum_all[sm[2]][:,:,1]        
        
        d_moments12_d_ln1b[:,i] = (sm_rad1*np.sum(b*fact2*d_fact1*P_lz_mat, axis=1))
        d_moments12_d_ln2b[:,i] = (sm_rad2*np.sum(b*fact1*d_fact2*P_lz_mat, axis=1))

        d_moments13_d_ln1b[:,i] = (sm_rad1*np.sum(b*fact3*d_fact1*P_lz_mat, axis=1))
        d_moments13_d_ln3b[:,i] = (sm_rad3*np.sum(b*fact1*d_fact3*P_lz_mat, axis=1))

        d_moments23_d_ln2b[:,i] = (sm_rad2*np.sum(b*fact3*d_fact2*P_lz_mat, axis=1))
        d_moments23_d_ln3b[:,i] = (sm_rad3*np.sum(b*fact2*d_fact3*P_lz_mat, axis=1))
    mu = 5/7

    moments  = 2*mu*moments13a*moments23a + (1-mu)*moments13c*moments23c
    moments += 0.5*(moments13b*d_moments23_d_ln3b + moments23b*d_moments13_d_ln3b)
    moments += 2*mu*moments12a*moments23a + (1-mu)*moments12c*moments23c
    moments += 0.5*(moments12b*d_moments23_d_ln2b + moments23b*d_moments12_d_ln2b)
    moments += 2*mu*moments13a*moments12a + (1-mu)*moments13c*moments12c
    moments += 0.5*(moments13b*d_moments12_d_ln1b + moments12b*d_moments13_d_ln1b)
    if zbin[0] == 0:
        moments[0,:] = 0.0*P_lz_mat.shape[1]
    
    return np.array(moments)


def kappa2_vec(corr_all, zbin, chibin, qi1, qi2, sm1, sm2, lmax = 1024, mask = None):
    corr = corr_all[(sm1, sm2)]
    intgnd = qi1*qi2*corr.T/chibin**2
    intgnd[:,0] = 0
    res = np.trapz(intgnd , chibin, axis = 1)
    return res


def kappa123_vec(corr_all, zbin, chibin, qi1, qi2, qi3, sm1, sm2, sm3, lmax = 1024, scheme = "LIN", mask = None):
    corr = corr_all[(sm1, sm2, sm3)]
    intgnd = qi1*qi2*qi3*corr.T/chibin**4
    intgnd[:,0] = 0
    res = np.trapz(intgnd, chibin, axis = 1)
    return res




            
#get relevant redshift distribution
nzbin2 = np.genfromtxt("./nzbins/FLASK_2.txt")
nzbin3 = np.genfromtxt("./nzbins/FLASK_3.txt")
zbin = nzbin2[:,0]
nz2 = nzbin2[:,1]
nz3 = nzbin3[:,1]
dz = zbin[1]-zbin[0]
sm = np.array([21.0,33.6,54.,86., 137.6, 220.16])
scheme='SC'

    
chibin = results_LCDM.comoving_radial_distance(zbin)
qi_b2 = np.zeros(len(zbin))
for i in range(len(zbin)):
    foo = nz2[i:]*(1-chibin[i]/chibin[i:])
    qi_b2[i] = np.trapz(foo, zbin[i:])
qi_b2 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi_b2*chibin
qi_b2[0] = 0


qi_b3 = np.zeros(len(zbin))
for i in range(len(zbin)):
    foo = nz3[i:]*(1-chibin[i]/chibin[i:])
    qi_b3[i] = np.trapz(foo, zbin[i:])
qi_b3 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi_b3*chibin
qi_b3[0] = 0


Plz_mat = compute_Plz_mat(PK, zbin, lmax = lmax, mask = ME)

Dz_nz, knl_nz, ns_mat = get_Dz_knletc(PK_L, zbin, lmax=lmax)

a_k3, b_k3, c_k3 = compute_abc_kappa3(zbin, Dz_nz, knl_nz, ns_mat, scheme=scheme)

fac_to_sum_all = {}
for i in range(len(sm)):
    for j in range(len(sm)):
        fac_to_sum_all[(sm[i], sm[j])] = compute_factosum(zbin, np.array([sm[i]]), np.array([sm[j]]), lmax = lmax)
        

fact_dfact_kappa3_to_sum_all = {}
for i in range(len(sm)):
    fact_dfact_kappa3_to_sum_all[(sm[i])] = compute_fact_dfact_kappa3(zbin, sm[i], lmax = lmax)  


corr2_all = {}
for i in range(len(sm)):
    for j in range(len(sm)):
        if i <= j:
            corr2_all[(sm[i], sm[j])] = compute_masked_m12_from_factosum(zbin, Plz_mat, fac_to_sum_all[(sm[i], sm[j])])  
        else:
            corr2_all[(sm[i], sm[j])] = corr2_all[(sm[j], sm[i])]
            

kp2_2_2 = np.zeros((len(sm),len(sm)))
kp2_2_3 = np.zeros((len(sm),len(sm)))
kp2_3_3 = np.zeros((len(sm),len(sm)))

for k in range(3):
    if k == 0:
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    kp2_2_2[i,j] = kappa2_vec(corr2_all, zbin, chibin, qi_b2, qi_b2, sm[i], sm[j], mask = ME)
                else:
                    kp2_2_2[i,j] = kp2_2_2[j,i]
                    
    elif k == 1:
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    kp2_2_3[i,j] = kappa2_vec(corr2_all, zbin, chibin, qi_b2, qi_b3, sm[i], sm[j], mask = ME)
                else:
                    kp2_2_3[i,j] = kp2_2_3[j,i]
    elif k == 2:
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    kp2_3_3[i,j] = kappa2_vec(corr2_all, zbin, chibin, qi_b3, qi_b3, sm[i], sm[j], mask = ME)
                else:
                    kp2_3_3[i,j] = kp2_3_3[j,i]
save_obj('./fisher_new/kp2param0_2_2', kp2_2_2)
save_obj('./fisher_new/kp2param0_2_3', kp2_2_3)
save_obj('./fisher_new/kp2param0_3_3', kp2_3_3)



corr3_all = {}
for i in range(len(sm)):
    for j in range(len(sm)):
        for k in range(len(sm)):
            if (k >= j and j >= i):
                corr3_all[(sm[i], sm[j], sm[k])] = compute_masked_m123_vec(Plz_mat, zbin, a_k3, b_k3, c_k3, sm[i], sm[j], sm[k], lmax = lmax, scheme=scheme, mask=mask)    
            else:
                foo = np.sort([i, j, k])
                corr3_all[(sm[i], sm[j], sm[k])] = corr3_all[(sm[foo[0]], sm[foo[1]], sm[foo[2]])]



kp3_2_2_2 = np.zeros((len(sm),len(sm),len(sm)))
kp3_3_2_2 = np.zeros((len(sm),len(sm),len(sm)))
kp3_2_3_3 = np.zeros((len(sm),len(sm),len(sm)))
kp3_3_3_3 = np.zeros((len(sm),len(sm),len(sm)))

for l in range(4):
    if l == 0:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    kp3_2_2_2[i,j,k] = kappa123_vec(corr3_all, zbin, chibin, qi_b2, qi_b2, qi_b2, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)
    elif l == 1:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    kp3_3_2_2[i,j,k] = kappa123_vec(corr3_all, zbin, chibin, qi_b3, qi_b2, qi_b2, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)

    elif l == 2:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    kp3_2_3_3[i,j,k] = kappa123_vec(corr3_all, zbin, chibin, qi_b2, qi_b3, qi_b3, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)

    elif l == 3:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    kp3_3_3_3[i,j,k] = kappa123_vec(corr3_all, zbin, chibin, qi_b3, qi_b3, qi_b3, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)

save_obj('./fisher_new/kp3param0SC_2_2_2', kp3_2_2_2)
save_obj('./fisher_new/kp3param0SC_3_2_2', kp3_3_2_2)
save_obj('./fisher_new/kp3param0SC_2_3_3', kp3_2_3_3)
save_obj('./fisher_new/kp3param0SC_3_3_3', kp3_3_3_3)
print(time.time() - t0_init)


    


# In[ ]:





