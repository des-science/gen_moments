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
from tqdm import tqdm
import astropy.constants as const
import astropy.units as units
from cosmosis.datablock import names, option_section as opt
from cosmosis.datablock.cosmosis_py import errors
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import warnings
# Finally we can now import camb
import camb
cosmo = names.cosmological_parameters
import h5py as h5
import healpy as hp
import pickle
import copy
import pickle as pk
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')

jtplot.reset()


class theory_setup():

    def __init__(self, options):
        self.z_array = np.linspace(zmin, zmax, nz)


    def get_Pk_camb(self, block):
        Om, h, Ob, ns, lg10As = block[cosmo, "omega_m"], block[cosmo, "h0"], block[cosmo, "omega_b"], block[cosmo, "n_s"], (block[cosmo, "lg10A_s"])
        mnu = block[cosmo, "mnu"]
        As = 10**lg10As
        Onu = mnu/((h**2)*93.14)
        nnu = block[cosmo, "nnu"]
        tau = block[cosmo, "tau"]
        bestfit = {}
        bestfit["ombh2"] = Ob*h**2
        bestfit["omch2"] = (Om-Ob-Onu)*h**2
        # bestfit["As"] = 3.0775467136912062e-09 #rootfound for this for sig8 = 0.84
        bestfit["As"] = As
        bestfit["H0"] = h*100
        bestfit["tau"] = tau
        bestfit["ns"] = ns
        bestfit["mnu"] = mnu
        bestfit["nnu"] = nnu

        pars_LCDM = camb.set_params(**bestfit, DoLateRadTruncation=True)
        pars_LCDM.WantTransfer = True
        results_LCDM = camb.get_results(pars_LCDM)
        self.sig8 = results_LCDM.get_sigma8_0()
        block[cosmo, 'sigma8'] = self.sig8
        self.PK = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4,nonlinear=True, extrap_kmax= 10**10)
        self.PK_L = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4,nonlinear=False, extrap_kmax= 10**10)
        self.chitoz = results_LCDM.redshift_at_comoving_radial_distance
        self.ztochi = results_LCDM.comoving_radial_distance
        self.dchi_dz = ((const.c.to(units.km / units.s)).value) / (results_LCDM.h_of_z) 
        self.chi_zbin = self.ztochi(self.z_array)

        nz = len(self.z_array)
        nell = self.lmax
        self.knl_nz = np.zeros(nz)
        self.Dz_nz = np.zeros(nz)
        self.ns_mat = np.zeros((nz, nell))
        for jz in range(nz):
            z = self.z_array[jz]
            def helper(k):
                return k**3*self.PK_L.P(z, k)/(2*np.pi**2)-1
            knl = scipy.optimize.root(helper, 0.5).x[0]
            self.knl_nz[jz] = knl
            self.Dz_nz[jz] = np.sqrt(self.PK_L.P(z, 1)/self.PK_L.P(0, 1))
            kay = np.arange(self.lmax)/self.ztochi(z)
            ns = (kay/self.PK_L.P(0,kay))*(self.PK_L.P(0,kay+0.001)-self.PK_L.P(0,kay))/(0.001)
            self.ns_mat[jz, :] = ns
        return 0

    def get_qi_nz(self, block):
        self.qi_gravonly = {}
        zarray_inp = block['nz_source', 'z']


        if ('nz_source', 'bin_' + str(binvs)) in block.keys():
            other_params_dict_bin['ng_value_source'] = block['nz_source', 'bin_' + str(binvs)]

        Om0, H0 = block[cosmo, "omega_m"], 100.*block[cosmo, "h0"]
        qi = np.zeros(len(self.z_array))
        for i in range(len(self.z_array)):
            foo = nz_source_bin[i:]*(1-self.chi_zbin[i]/self.chi_zbin[i:])
            qi[i] = np.trapz(foo, self.z_array[i:])
        qi *= (1.5*Om0*(H0/((const.c.to(units.km / units.s)).value))**2)*(1+self.z_array) * self.chi_zbin
        qi[0] = 0
        return qi

    def get_mask_stuff(self, fname='namaster_stuff.pk'):
        df = pk.load(open(fname,'rb'))
        M = df['M']
        self.ME = df['ME']
        self.mask = df['mask']
        self.lmax = df['lmax']
        return 0

    def compute_Plz_mat(self):
        '''
        It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
        3D power spectrum at fixed z. (k=l/chi(z)).
        '''
        nz = len(self.z_array)
        nell = self.lmax
        # chi_zbin = self.ztochi(self.z_array)
        ell = np.arange(self.lmax)
        z_mat = np.tile(self.z_array.reshape(nz, 1), (1, nell))
        chi_mat = np.tile(self.chi_zbin.reshape(nz, 1), (1, nell))
        ell_mat = np.tile(ell.reshape(1, nell), (nz, 1))
        k_mat = ell_mat/chi_mat
        P_lz_mat = np.exp(self.PK.ev(z_mat, np.log(k_mat + 1e-6)))
        F_l =  hp.sphtfunc.pixwin(512, lmax = self.lmax)[:self.lmax]
        F_l_mat = np.tile(F_l.reshape(1, nell), (nz, 1))
        self.P_lz_mat *= (F_l_mat)**2
        if self.mask is not None:
            f_l = (ell+2)*(ell-1)/(ell*(ell+1))
            f_l[0:2] = 0
            f_l_mat = np.tile(f_l[:self.lmax].reshape(1, nell), (nz, 1))
            P_lz_mat *= f_l_mat

            P_lz_mat_maskv =  np.zeros((nz, nell))
            for jz in range(nz):
                P_lz_mat_maskv[jz, :] = self.mask[:self.lmax,:self.lmax]@P_lz_mat[jz, :self.lmax]
            f_linv = (ell*(ell+1))/((ell+2)*(ell-1))
            f_linv[0:2] = 0
            f_linv_mat = np.tile(f_linv.reshape(1, nell), (nz, 1))
            self.P_lz_mat = P_lz_mat_maskv*f_linv_mat  
        return 0


    # def get_Dz_knletc(self):
    #     return 0


    def compute_factosum(self, smoothing_scales1, smoothing_scales2):
        '''
        It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
        3D power spectrum at fixed z. (k=l/chi(z)).
        '''
        nz = len(self.z_array)
        nell = self.lmax
        ell = np.arange(self.lmax)   

        self.fac_to_sum = np.zeros((nz, nell, len(smoothing_scales1)))
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
            self.fac_to_sum[:, :, i] = fact1_mat * fact2_mat
        return 0

    def compute_fact_dfact_kappa3(self, sm1):
        '''
        It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
        3D power spectrum at fixed z. (k=l/chi(z)).
        '''
        nz = len(self.z_array)
        nell = self.lmax
        ell = np.arange(self.lmax)   
        self.fact_dfact_kappa3_to_sum = np.zeros((nz, nell, 2))
        sm_rad1 =(sm1/60.)*np.pi/180.
        fact1 = (scipy.special.eval_legendre(ell-1,np.cos(sm_rad1))-scipy.special.eval_legendre(ell+1,np.cos(sm_rad1)))/(4*np.pi*(1-np.cos(sm_rad1)))
        fact1[0] = 1./4*np.pi
        self.fact_dfact_kappa3_to_sum[:,:,0] = np.tile(fact1.reshape(1, nell), (nz, 1))
        d_fact1 =  scipy.special.eval_legendre(ell,np.cos(sm_rad1))*np.sin(sm_rad1)/(1-np.cos(sm_rad1))
        d_fact1 -= fact1*(4*np.pi/(2*ell+1))*np.sin(sm_rad1)/(1-np.cos(sm_rad1))
        d_fact1[0] = 0  #0th Wl = 1
        self.fact_dfact_kappa3_to_sum[:,:,1] = np.tile(d_fact1.reshape(1, nell), (nz, 1))
        return 0   

    def compute_abc_kappa3(self):
        nz = len(self.z_array)
        nell = self.lmax
        ell = np.arange(self.lmax)  
        knl_mat = np.tile(self.knl_nz.reshape(nz, 1), (1, nell))
        Dz_mat = np.tile(self.Dz_nz.reshape(nz, 1), (1, nell))    

        # chi_zbin = ztochi(self.z_array)
        ell = np.arange(self.lmax)
        chi_mat = np.tile(self.chi_zbin.reshape(nz, 1), (1, nell))
        ell_mat = np.tile(ell.reshape(1, nell), (nz, 1))
        k_mat = ell_mat/chi_mat    
        q_mat = k_mat/knl_mat
            # Initialise coefficients small-scales fitting formulae.
        if self.scheme == 'SC':
            coeff = [0.25,3.5,2.,1.,2.,-0.2,1.,0.,0.]
        elif self.scheme == 'GM':
            coeff = [0.484,3.740,-0.849,0.392,1.013,-0.575,0.128,-0.722,-0.926]

        self.a = (1. + ((self.sig8*Dz_mat)**coeff[5])*(0.7*(4.-2.**self.ns_mat)/(1.+2.**(2.*self.ns_mat+1)))**0.5*(q_mat*coeff[0])**(self.ns_mat+coeff[1]))/(1.+(q_mat*coeff[0])**(self.ns_mat+coeff[1]))
        self.b = (1. + 0.2*coeff[2]*(self.ns_mat+3)*(q_mat*coeff[6])**(self.ns_mat+coeff[7]+3))/(1.+(q_mat*coeff[6])**(self.ns_mat+coeff[7]+3.5));
        self.c = (1. + 4.5*coeff[3]/(1.5+(self.ns_mat+3)**4)*(q_mat*coeff[4])**(self.ns_mat+3+coeff[8]))/(1+(q_mat*coeff[4])**(self.ns_mat+3.5+coeff[8]));
        self.a[:,0] = 1.
        self.b[:,0] = 1.
        self.c[:,0] = 1.
        return 0

    def compute_masked_m12_from_factosum(self, fac_to_sum, abc=1.0):  
        nz = len(self.z_array)
        moments = np.zeros((nz, 1))
        moments[:, 0] = np.sum(abc*fac_to_sum[:,:,0]*self.P_lz_mat, axis=1)
        if self.z_array[0] == 0:
            moments[0,:] = 0.0*self.P_lz_mat.shape[1]
        return moments


    def compute_masked_m123_vec(self, smoothing_scales_all3):
        '''
        It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
        3D power spectrum at fixed z. (k=l/chi(z)).
        '''
        # a, b, c = abc_all3
        a, b, c = self.a, self.b, self.c
        smoothing_scales1, smoothing_scales2, smoothing_scales3 = smoothing_scales_all3
        nz = len(self.z_array)

        moments12a = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales1, smoothing_scales2)], abc=a)
        moments13a = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales1, smoothing_scales3)], abc=a)
        moments23a = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales2, smoothing_scales3)], abc=a)        

        moments12b = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales1, smoothing_scales2)], abc=b)
        moments13b = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales1, smoothing_scales3)], abc=b)
        moments23b = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales2, smoothing_scales3)], abc=b)        

        moments12c = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales1, smoothing_scales2)], abc=c)
        moments13c = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales1, smoothing_scales3)], abc=c)
        moments23c = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(smoothing_scales2, smoothing_scales3)], abc=c)        

        smoothing_scales1, smoothing_scales2, smoothing_scales3 = np.array([smoothing_scales1]), np.array([smoothing_scales2]), np.array([smoothing_scales3]) 
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

            fact1 = self.fact_dfact_kappa3_to_sum_all[sm[0]][:,:,0]
            fact2 = self.fact_dfact_kappa3_to_sum_all[sm[1]][:,:,0]
            fact3 = self.fact_dfact_kappa3_to_sum_all[sm[2]][:,:,0]        

            d_fact1 = self.fact_dfact_kappa3_to_sum_all[sm[0]][:,:,1]
            d_fact2 = self.fact_dfact_kappa3_to_sum_all[sm[1]][:,:,1]
            d_fact3 = self.fact_dfact_kappa3_to_sum_all[sm[2]][:,:,1]        

            d_moments12_d_ln1b[:,i] = (sm_rad1*np.sum(b*fact2*d_fact1*Plz_mat, axis=1))
            d_moments12_d_ln2b[:,i] = (sm_rad2*np.sum(b*fact1*d_fact2*Plz_mat, axis=1))

            d_moments13_d_ln1b[:,i] = (sm_rad1*np.sum(b*fact3*d_fact1*Plz_mat, axis=1))
            d_moments13_d_ln3b[:,i] = (sm_rad3*np.sum(b*fact1*d_fact3*Plz_mat, axis=1))

            d_moments23_d_ln2b[:,i] = (sm_rad2*np.sum(b*fact3*d_fact2*Plz_mat, axis=1))
            d_moments23_d_ln3b[:,i] = (sm_rad3*np.sum(b*fact2*d_fact3*Plz_mat, axis=1))
        mu = 5/7

        moments  = 2*mu*moments13a*moments23a + (1-mu)*moments13c*moments23c
        moments += 0.5*(moments13b*d_moments23_d_ln3b + moments23b*d_moments13_d_ln3b)
        moments += 2*mu*moments12a*moments23a + (1-mu)*moments12c*moments23c
        moments += 0.5*(moments12b*d_moments23_d_ln2b + moments23b*d_moments12_d_ln2b)
        moments += 2*mu*moments13a*moments12a + (1-mu)*moments13c*moments12c
        moments += 0.5*(moments13b*d_moments12_d_ln1b + moments12b*d_moments13_d_ln1b)
        if self.z_array[0] == 0:
            moments[0,:] = 0.0*self.P_lz_mat.shape[1]

        return np.array(moments)


    def kappa2_vec(self, corr_all, qi1_qi2, sm1, sm2):
        corr = corr_all[(sm1, sm2)]
        intgnd = qi1_qi2*corr.T/self.chi_zbin**2
        intgnd[:,0] = 0
        res = np.trapz(intgnd , self.chi_zbin, axis = 1)
        return res


    def kappa123_vec(self, corr_all, qi1_qi2_qi3, sm1, sm2, sm3):
        corr = corr_all[(sm1, sm2, sm3)]
        intgnd = qi1_qi2_qi3*corr.T/self.chi_zbin**4
        intgnd[:,0] = 0
        res = np.trapz(intgnd, self.chi_zbin, axis = 1)
        return res



def get_qi3_qi2_forIA(params, qi_all, ni_all, self.z_array, Dz, dchi_dz):
    AIA0, z0, alpha_IA = params['A_IA'], params['z0_IA'], params['alpha_IA']
    IAz = AIA0*(((1+self.z_array)/(1+z0))**alpha_IA)*(0.0134/Dz)
    qi_new = {}
    for jz in range(ns_tot):
        qi_jz = qi_all[jz]
        ni_jz = ni_all[jz]
        qIAz = IAz * ni_jz * (1/dchi_dz)
        qi_new = qi_jz - qIAz
    return qi_new

# def get_qi3_qi2_forIA():

def saveDV(jr, nlhs=1000):
    df_cosmoparams = np.loadtxt('cosmo_samp_rs' + str(jr) + '_5cosmo_nsamp_' + str(nlhs) + '.txt')
    for jlhs in tqdm(range(df_cosmoparams.shape[0])):
        params = df_cosmoparams[jlhs, :]
        t0_init = time.time()
        # Ob, Om, As, h, ns = cosmo_params['Ob'], cosmo_params['Om'], cosmo_params['As'], cosmo_params['h'], cosmo_params['ns'] 
        PK, PK_L, chitoz, ztochi = get_Pk_camb(params)
        ME, mask, lmax, nside = get_mask_stuff(fname='namaster_stuff.pk')

        #get relevant redshift distribution
        nzbin2 = np.genfromtxt("./nzbins/FLASK_2.txt")
        nzbin3 = np.genfromtxt("./nzbins/FLASK_3.txt")
        self.z_array = nzbin2[:,0]
        nz2 = nzbin2[:,1]
        nz3 = nzbin3[:,1]
        dz = self.z_array[1]-self.z_array[0]
        sm = np.array([21.0,33.6,54.,86., 137.6, 220.16])
        scheme='SC'

        qi_all = {}
        for jz in range(nz):
            qi_all[jz] = get_qi(self.z_array, nz_source_bin_jz, chibin, cosmo_params)

        # chibin = results_LCDM.comoving_radial_distance(self.z_array)
        # qi_b2 = np.zeros(len(self.z_array))
        # for i in range(len(self.z_array)):
        #     foo = nz2[i:]*(1-chibin[i]/chibin[i:])
        #     qi_b2[i] = np.trapz(foo, self.z_array[i:])
        # qi_b2 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.z_array)*qi_b2*chibin
        # qi_b2[0] = 0


        # qi_b3 = np.zeros(len(self.z_array))
        # for i in range(len(self.z_array)):
        #     foo = nz3[i:]*(1-chibin[i]/chibin[i:])
        #     qi_b3[i] = np.trapz(foo, self.z_array[i:])
        # qi_b3 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.z_array)*qi_b3*chibin
        # qi_b3[0] = 0


        Plz_mat = compute_Plz_mat(PK, self.z_array, lmax = lmax, mask = ME)

        Dz_nz, knl_nz, ns_mat = get_Dz_knletc(PK_L, self.z_array, lmax=lmax)

        a_k3, b_k3, c_k3 = compute_abc_kappa3(self.z_array, Dz_nz, knl_nz, ns_mat, scheme=scheme)

        fac_to_sum_all = {}
        for i in range(len(sm)):
            for j in range(len(sm)):
                fac_to_sum_all[(sm[i], sm[j])] = compute_factosum(self.z_array, np.array([sm[i]]), np.array([sm[j]]), lmax = lmax)


        fact_dfact_kappa3_to_sum_all = {}
        for i in range(len(sm)):
            fact_dfact_kappa3_to_sum_all[(sm[i])] = compute_fact_dfact_kappa3(self.z_array, sm[i], lmax = lmax)  

        corr2_all = {}
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    corr2_all[(sm[i], sm[j])] = compute_masked_m12_from_factosum(self.z_array, Plz_mat, fac_to_sum_all[(sm[i], sm[j])])  
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
                            kp2_2_2[i,j] = kappa2_vec(corr2_all, self.z_array, chibin, qi_b2, qi_b2, sm[i], sm[j], mask = ME)
                        else:
                            kp2_2_2[i,j] = kp2_2_2[j,i]

            elif k == 1:
                for i in range(len(sm)):
                    for j in range(len(sm)):
                        if i <= j:
                            kp2_2_3[i,j] = kappa2_vec(corr2_all, self.z_array, chibin, qi_b2, qi_b3, sm[i], sm[j], mask = ME)
                        else:
                            kp2_2_3[i,j] = kp2_2_3[j,i]
            elif k == 2:
                for i in range(len(sm)):
                    for j in range(len(sm)):
                        if i <= j:
                            kp2_3_3[i,j] = kappa2_vec(corr2_all, self.z_array, chibin, qi_b3, qi_b3, sm[i], sm[j], mask = ME)
                        else:
                            kp2_3_3[i,j] = kp2_3_3[j,i]
        # save_obj('./fisher_new/kp2param0_2_2', kp2_2_2)
        # save_obj('./fisher_new/kp2param0_2_3', kp2_2_3)
        # save_obj('./fisher_new/kp2param0_3_3', kp2_3_3)
        
        kp2_all_theory = {'2_2':kp2_2_2, '2_3':kp2_2_3, '3_3':kp2_3_3}


        corr3_all = {}
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        corr3_all[(sm[i], sm[j], sm[k])] = compute_masked_m123_vec(Plz_mat, self.z_array, a_k3, b_k3, c_k3, sm[i], sm[j], sm[k], lmax = lmax, scheme=scheme, mask=mask)    
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
                            kp3_2_2_2[i,j,k] = kappa123_vec(corr3_all, self.z_array, chibin, qi_b2, qi_b2, qi_b2, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)
            elif l == 1:
                for i in range(len(sm)):
                    for j in range(len(sm)):
                        for k in range(len(sm)):
                            kp3_3_2_2[i,j,k] = kappa123_vec(corr3_all, self.z_array, chibin, qi_b3, qi_b2, qi_b2, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)

            elif l == 2:
                for i in range(len(sm)):
                    for j in range(len(sm)):
                        for k in range(len(sm)):
                            kp3_2_3_3[i,j,k] = kappa123_vec(corr3_all, self.z_array, chibin, qi_b2, qi_b3, qi_b3, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)

            elif l == 3:
                for i in range(len(sm)):
                    for j in range(len(sm)):
                        for k in range(len(sm)):
                            kp3_3_3_3[i,j,k] = kappa123_vec(corr3_all, self.z_array, chibin, qi_b3, qi_b3, qi_b3, sm[i], sm[j], sm[k], scheme ='SC', mask = ME)

        
        kp3_all_theory = {'2_2_2':kp3_2_2_2, '3_2_2':kp3_3_2_2, '2_3_3':kp3_2_3_3, '3_3_3':kp3_3_3_3} 
        
        kp_all_theory = {'kp2':kp2_all_theory, 'kp3':kp3_all_theory}
        fname = 'saved_DVs/lhs_n1000_jr' + str(jr) + '/kappa_all_jlhs' + str(jlhs)
        save_obj(fname, kp_all_theory)
    

import sys
jrv = sys.argv[1]
saveDV(jrv)




def setup(options):
    return theory_setup(options)


def execute(block, config):
    return config.execute(block)