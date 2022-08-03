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

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, protocol = 2)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)#, encoding='latin1')

class theory_setup():

    def __init__(self, options):
        nz = options.get_int(option_section, "nz", default=200)
        zmin = options.get_double(option_section, "zmin", default=0.0)
        zmax = options.get_double(option_section, "zmax", default=4.0)
        self.sm_all = ast.literal_eval(options.get_string(option_section, "sm_all", default='[21.0,33.6,54.,86., 137.6, 220.16]'))
        self.fname_mask_stuff = options.get_string(option_section, "fname_mask_stuff", default=os.environ['COSMOSIS_SRC_DIR'] + '/gen_moments/namaster_stuff.pk')
        self.scheme = options.get_string(option_section, "scheme", default='SC')
        self.do_save_DV = options.get_bool(option_section, "do_save_DV", default=False)
        self.isLHS_sample = options.get_bool(option_section, "isLHS_sample", default=False)        
        self.saveDV_dir = options.get_string(option_section, "saveDV_dir", default='DV_all')
        self.saveDV_prefix = options.get_string(option_section, "saveDV_prefix", default='')
        self.z_array = np.linspace(zmin, zmax, nz)
        
    def get_Pk_camb(self, block):
        Om, h, Ob, ns  = block[cosmo_sec, "omega_m"], block[cosmo_sec, "h0"], block[cosmo_sec, "omega_b"], block[cosmo_sec, "n_s"]

        # lg10As = block[cosmo_sec, "lg10A_s"]
        # As = 10**lg10As

        As = block[cosmo_sec, "A_s"]

        mnu = block[cosmo_sec, "mnu"]

        Onu = mnu/((h**2)*93.14)
        nnu = block[cosmo_sec, "nnu"]
        tau = block[cosmo_sec, "tau"]
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
        block[cosmo_sec, 'sigma_8'] = self.sig8
        self.PK = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4,nonlinear=True, extrap_kmax= 10**10)
        self.PK_L = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4,nonlinear=False, extrap_kmax= 10**10)
        self.chitoz = results_LCDM.redshift_at_comoving_radial_distance
        self.ztochi = results_LCDM.comoving_radial_distance
        self.dchi_dz = ((const.c.to(units.km / units.s)).value) / (results_LCDM.h_of_z(self.z_array)) 
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

    def get_qigrav_and_nz(self, block):
        zarray_inp = block['nz_source', 'z']
        self.nz_source_all = {}
        self.nbins_tot = 0
        for ji in range(1000):
            if ('nz_source', 'bin_' + str(ji+1)) in block.keys():
                nz_ji = block['nz_source', 'bin_' + str(ji+1)]
                nz_norm_ji = nz_ji/scipy.integrate.simps(nz_ji, zarray_inp)
                nz_norm_ji_interp = interpolate.interp1d(zarray_inp, nz_norm_ji, bounds_error=False, fill_value=0.0)
                nz_new_ji = nz_norm_ji_interp(self.z_array)
                nz_new_ji /= scipy.integrate.simps(nz_new_ji, self.z_array)
                self.nz_source_all[ji] = nz_new_ji
                self.nbins_tot += 1
            else:
                break
            

        Om0, H0 = block[cosmo_sec, "omega_m"], 100.*block[cosmo_sec, "h0"]
        self.qi_gravonly = {}
        for ji in range(self.nbins_tot):
            ng_array_source = self.nz_source_all[ji]
            chi_lmat = np.tile(self.chi_zbin.reshape(len(self.z_array), 1), (1, len(self.z_array)))
            chi_smat = np.tile(self.chi_zbin.reshape(1, len(self.z_array)), (len(self.z_array), 1))
            num = chi_smat - chi_lmat
            ind_lzero = np.where(num <= 0)
            num[ind_lzero] = 0
            ng_array_source_rep = np.tile(ng_array_source.reshape(1, len(self.z_array)), (len(self.z_array), 1))
            int_sourcez = scipy.integrate.simps(ng_array_source_rep * (num / chi_smat), self.z_array)
            coeff_ints = 3 * (H0 ** 2) * Om0 / (2. * ((const.c.to(units.km / units.s)).value)**2)
            qi_array = coeff_ints * self.chi_zbin * (1. + self.z_array) * int_sourcez
            qi_array[0] = 0.
            self.qi_gravonly[ji] = qi_array

        return 0

    def get_qi_after_IA(self, block):
        z0 = block[IA_sec,'z_piv']        
        AIA0 = block[IA_sec,'A1']
        alpha_IA = block[IA_sec,'alpha1']
        IAz = AIA0*(((1+self.z_array)/(1+z0))**alpha_IA)*(0.0134/self.Dz_nz)
        self.qi_wIA = {}
        for jz in range(self.nbins_tot):
            qi_jz = self.qi_gravonly[jz]
            ni_jz = self.nz_source_all[jz]
            qIAz = IAz * ni_jz * (1/self.dchi_dz)
            self.qi_wIA[jz] = qi_jz - qIAz
        return 0


    def get_mask_stuff(self):
        df = pk.load(open(self.fname_mask_stuff,'rb'))
        # M = df['M']
        # self.ME = df['ME']
        # self.mask = df['mask']        
        self.mask = df['ME']
        self.lmax = df['lmax']
        return 0

    def compute_Plz_mat(self):
        nz = len(self.z_array)
        nell = self.lmax        
        ell = np.arange(self.lmax)
        z_mat = np.tile(self.z_array.reshape(nz, 1), (1, nell))
        chi_mat = np.tile(self.chi_zbin.reshape(nz, 1), (1, nell))
        ell_mat = np.tile(ell.reshape(1, nell), (nz, 1))
        k_mat = ell_mat/chi_mat
        self.P_lz_mat = np.exp(self.PK.ev(z_mat, np.log(k_mat + 1e-6)))
        F_l =  hp.sphtfunc.pixwin(512, lmax = self.lmax)[:self.lmax]
        F_l_mat = np.tile(F_l.reshape(1, nell), (nz, 1))
        self.P_lz_mat *= (F_l_mat)**2
        if self.mask is not None:
            f_l = (ell+2)*(ell-1)/(ell*(ell+1))
            f_l[0:2] = 0
            f_l_mat = np.tile(f_l[:self.lmax].reshape(1, nell), (nz, 1))
            self.P_lz_mat *= f_l_mat

            P_lz_mat_maskv =  np.zeros((nz, nell))
            for jz in range(nz):
                P_lz_mat_maskv[jz, :] = self.mask[:self.lmax,:self.lmax]@self.P_lz_mat[jz, :self.lmax]
            f_linv = (ell*(ell+1))/((ell+2)*(ell-1))
            f_linv[0:2] = 0
            f_linv_mat = np.tile(f_linv.reshape(1, nell), (nz, 1))
            self.P_lz_mat = P_lz_mat_maskv*f_linv_mat  
        return 0


    def compute_factosum(self, smoothing_scales1, smoothing_scales2):
        nz = len(self.z_array)
        nell = self.lmax
        ell = np.arange(self.lmax)   

        fac_to_sum = np.zeros((nz, nell, len(smoothing_scales1)))
        for i, sm in enumerate(zip(smoothing_scales1,smoothing_scales2)):
            # convert scale to radians ***
            sm_rad1 =(sm[0]/60.)*np.pi/180.
            sm_rad2 =(sm[1]/60.)*np.pi/180.

            
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

    def compute_fact_dfact_kappa3(self, sm1):
        nz = len(self.z_array)
        nell = self.lmax
        ell = np.arange(self.lmax)   
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

    def compute_abc_kappa3(self):
        nz = len(self.z_array)
        nell = self.lmax
        ell = np.arange(self.lmax)  
        knl_mat = np.tile(self.knl_nz.reshape(nz, 1), (1, nell))
        Dz_mat = np.tile(self.Dz_nz.reshape(nz, 1), (1, nell))    

        ell = np.arange(self.lmax)
        chi_mat = np.tile(self.chi_zbin.reshape(nz, 1), (1, nell))
        ell_mat = np.tile(ell.reshape(1, nell), (nz, 1))
        k_mat = ell_mat/chi_mat    
        q_mat = k_mat/knl_mat
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

            d_moments12_d_ln1b[:,i] = (sm_rad1*np.sum(b*fact2*d_fact1*self.P_lz_mat, axis=1))
            d_moments12_d_ln2b[:,i] = (sm_rad2*np.sum(b*fact1*d_fact2*self.P_lz_mat, axis=1))

            d_moments13_d_ln1b[:,i] = (sm_rad1*np.sum(b*fact3*d_fact1*self.P_lz_mat, axis=1))
            d_moments13_d_ln3b[:,i] = (sm_rad3*np.sum(b*fact1*d_fact3*self.P_lz_mat, axis=1))

            d_moments23_d_ln2b[:,i] = (sm_rad2*np.sum(b*fact3*d_fact2*self.P_lz_mat, axis=1))
            d_moments23_d_ln3b[:,i] = (sm_rad3*np.sum(b*fact2*d_fact3*self.P_lz_mat, axis=1))
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


    def execute(self, block):

        self.get_mask_stuff()
        self.get_Pk_camb(block)
        self.get_qigrav_and_nz(block)
        self.get_qi_after_IA(block)
        self.compute_Plz_mat()
        self.compute_abc_kappa3()

        self.nsm = len(self.sm_all)

        self.fac_to_sum_all = {}
        for i in range((self.nsm)):
            for j in range((self.nsm)):
                self.fac_to_sum_all[(self.sm_all[i], self.sm_all[j])] = self.compute_factosum(np.array([self.sm_all[i]]), np.array([self.sm_all[j]]))


        self.fact_dfact_kappa3_to_sum_all = {}
        for i in range((self.nsm)):
            self.fact_dfact_kappa3_to_sum_all[(self.sm_all[i])] = self.compute_fact_dfact_kappa3(self.sm_all[i])  

        

        id_corr2_unique, id_kp2_unique, id_corr3_unique, id_kp3_unique = get_unique_ind.save_unique_combs(self.nsm, self.nbins_tot)

        self.corr2_all = {}
        for jid_c2 in range(len(id_corr2_unique)):
            i, j = id_corr2_unique[jid_c2]
            self.corr2_all[(self.sm_all[i], self.sm_all[j])] = self.compute_masked_m12_from_factosum(self.fac_to_sum_all[(self.sm_all[i], self.sm_all[j])])  


        self.kp2_all = {}
        for jid_kp2 in range(len(id_kp2_unique)):
            jz1, jz2, i, j = id_kp2_unique[jid_kp2]
            mjz1 = 1.+block['shear_calibration_parameters','m' + str(jz1+1)]
            mjz2 = 1.+block['shear_calibration_parameters','m' + str(jz2+1)]
            self.kp2_all[jz1, jz2, i, j] = (mjz1*mjz2)*self.kappa2_vec(self.corr2_all, self.qi_wIA[jz1]*self.qi_wIA[jz2], self.sm_all[i], self.sm_all[j])
        

        self.corr3_all = {}
        for jid_c3 in range(len(id_corr3_unique)):        
            i, j, k = id_corr3_unique[jid_c3]
            self.corr3_all[(self.sm_all[i], self.sm_all[j], self.sm_all[k])] = self.compute_masked_m123_vec((self.sm_all[i], self.sm_all[j], self.sm_all[k]))    

        self.kp3_all = {}
        for jid_kp3 in range(len(id_kp3_unique)):
            jz1, jz2, jz3, i, j, k = id_kp3_unique[jid_kp3]
            mjz1 = 1.+block['shear_calibration_parameters','m' + str(jz1+1)]
            mjz2 = 1.+block['shear_calibration_parameters','m' + str(jz2+1)]
            mjz3 = 1.+block['shear_calibration_parameters','m' + str(jz3+1)]
            self.kp3_all[jz1, jz2, jz3, i, j, k] = (mjz1*mjz2*mjz3)*self.kappa123_vec(self.corr3_all, self.qi_wIA[jz1]*self.qi_wIA[jz2]*self.qi_wIA[jz3], self.sm_all[i], self.sm_all[j], self.sm_all[k])

        kp_all_theory = {'kp2':self.kp2_all, 'kp3':self.kp3_all, 'id_kp2_unique':id_kp2_unique, 'id_kp3_unique':id_kp3_unique} 
        
        if self.do_save_DV:
            if self.isLHS_sample:
                lhs_val = block['lhs_id','lhs_val']
                fname = 'saved_DVs/lhs_n1000_jr' + str(jr) + '/kappa_all_jlhs' + str(jlhs)
                save_obj(fname, kp_all_theory)            
            else:
                fname = 'testDV'
                save_obj(fname, kp_all_theory)
    
        return 0


def setup(options):
    return theory_setup(options)


def execute(block, config):
    return config.execute(block)