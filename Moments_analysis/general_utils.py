import os
import numpy as np
import scipy
from scipy import integrate
import camb
from camb import model
import math
import sys
moments_path = os.path.realpath(os.path.join(os.getcwd(), '../Moments_analysis/'))
sys.path.insert(0, moments_path)
import h5py as h5
import healpy as hp
import pickle
from .healpy_utils import gk_inv
import copy
import pymaster as nmt

def save_obj(name, obj):
    with open(name + '.self.PKl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

def load_obj(name):
    with open(name + '.self.PKl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')


class moments_theory(object):
    def __init__(self, conf):
    '''
    Initialise the moments_theory object.
    conf should be a dictionary with
    'zbin' : array of redshifts for the redshift bin
    'parameters' : cosmological parameters for CAMB
        should be a dictionary with 'ombh2' 'omch2' 'As' 'H0' 'tau' 'ns' (optional) 'mnu' 'nnu'
        (note CAMB doesn't take sig8)
    'mask' : raw mask object
    '''

        self.zbin = conf['zbin']
        self.parameters = conf['parameters']
        self.mask = conf['mask']

        #compute cosmology from given parameters
        pars_LCDM = camb.set_params(**self.parameters, DoLateRadTruncation=True)
        pars_LCDM.WantTransfer = True
        results_LCDM = camb.get_results(pars_LCDM)

        self.PK   = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=1000.0, zmax=10,nonlinear=True, extrap_kmax= 10**10)

        self.PK_L = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=1000.0, zmax=10,nonlinear=False, extrap_kmax= 10**10)

        self.chitoz = results_LCDM.redshift_at_comoving_radial_distance
        self.ztochi = results_LCDM.comoving_radial_distance


        # import and smooth given mask
        mask_sm =  hp.sphtfunc.smoothing(mask, (13./60.)*np.pi/180.  )
        mask_sm[mask] = 1.
        mask = copy.copy(mask_sm)
        # computes Cl.
        lmax = 1024
        nside = 512

        # Read healpix maps and initialize a spin-0 and spin-2 field
        f_0 = nmt.NmtField(mask, [mask])
        f_2 = nmt.NmtField(mask, [mask,mask])
        bins = nmt.bins.NmtBin.from_lmax_linear(1024, 1, is_Dell=False)#nmt.bins.NmtBin(nside=1024, ells=2048)

        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f_2, f_2, bins, is_teb=False)
        M = w.get_coupling_matrix()
        self.ME = (M[::4,:][:,::4])

    def compute_masked_m12(z, smoothing_scales1, smoothing_scales2, lmax = 1024, abc = 1., mask = False):
        '''
        z needs to be a scalar here, but I think thats one of the key places for improvement
        smoothing scales can be arrays. But the output will be an array of results for (sm1[0] sm2[0]), (sm1[1] sm2[1]) and so on
        It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
        3D power spectrum at fixed z. (k=l/chi(z)).
        abc = 1 is the default behavior. With a defined 'scheme' for the 3rd moments, abc will become an array for the scheme
        mask is a boolean for if you want to mask or not
        '''
        if (z != 0.0):

            P_lz = self.PK.P(z, np.arange(lmax)/results_LCDM.comoving_radial_distance(z))
            F_l =  hp.sphtfunc.pixwin(512, lmax = lmax)
            ell = np.arange(lmax)

            # apply pixel kernel and mask
            P_lz = P_lz[:lmax]*F_l[:lmax]**2
            if mask:
                f_l = (ell+2)*(ell-1)/(ell*(ell+1))
                f_l[0:2] = 0
                P_lz = P_lz*f_l
                P_lz = self.ME[:lmax,:lmax]@P_lz[:lmax]
                f_linv = (ell*(ell+1))/((ell+2)*(ell-1))
                f_linv[0:2] = 0
                P_lz = P_lz*f_linv

            moments = np.zeros(len(smoothing_scales1))

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

                # integral
                moments[i] = np.sum(abc*fact1[:lmax]*fact2[:lmax]*P_lz[:lmax])
        else:
            moments = 0*smoothing_scales1
        return moments

    def compute_masked_m123(z, smoothing_scales1, smoothing_scales2, smoothing_scales3, lmax = 1024, scheme = 'LIN', mask = False):
        '''
        smoothing scales can be arrays. But the output will be an array of results for (sm1[0] sm2[0] sm3[0]), (sm1[1] sm2[1] sm3[1]) and so on
        'scheme': 'LIN' corresponds to not extended, "GM" and "SC" corresponds to different hyperextension schemes
        z needs to be a scalar here, but I think thats one of the key places for improvement
        It computes the smoothed (by a top-hat filter) 3rd moments of the density field given the
        3D power spectrum at fixed z. (k=l/chi(z)).
        mask is a boolean for if you want to mask or not
        '''
        if (z != 0):

            kay = np.arange(lmax)/results_LCDM.comoving_radial_distance(z)

            # NL mps as fn of ell
            P_lz = self.PK.P(z, kay)
            #pixel window function
            F_l =  hp.sphtfunc.pixwin(512, lmax = lmax)

            # growth factor at z defined as sqrt(P(z, k=1)/P(0,k=1)) bc k choice is irrelevant
            Dz = np.sqrt(self.PK_L.P(z, 1)/self.PK_L.P(0, 1))

            # k at which MPS becomes unity
            # this being called everytime might be very slow
            def helper(k):
                return k**3*self.PK_L.P(z, k)/(2*np.pi**2)-1

            knl = scipy.optimize.root(helper, 0.5).x[0]

            # d ln P/d ln k at relevant k values
            ns = (kay/self.PK_L.P(0,kay))*(self.PK_L.P(0,kay+0.001)-self.PK_L.P(0,kay))/(0.001)


                # Initialise coefficients small-scales fitting formulae.
            if scheme == 'SC':
                coeff = [0.25,3.5,2.,1.,2.,-0.2,1.,0.,0.]
            elif scheme == 'GM':
                coeff = [0.484,3.740,-0.849,0.392,1.013,-0.575,0.128,-0.722,-0.926]

            if (scheme == 'LIN'):
                a = 1.;
                b = 1.;
                c = 1.;

                moments12a =  compute_masked_m12(z, smoothing_scales1, smoothing_scales2, lmax, a, mask)
                moments13a =  compute_masked_m12(z, smoothing_scales1, smoothing_scales3, lmax, a, mask)
                moments23a =  compute_masked_m12(z, smoothing_scales2, smoothing_scales3, lmax, a, mask)

                moments12b =  moments12a
                moments13b =  moments13a
                moments23b =  moments23a

                moments12c =  moments12a
                moments13c =  moments13a
                moments23c =  moments23a

            else:
                # transition l from linear to non linear
                q= kay/knl

                a = (1. + ((sig8*Dz)**coeff[5])*(0.7*(4.-2.**ns)/(1.+2.**(2.*ns+1)))**0.5*(q*coeff[0])**(ns+coeff[1]))/(1.+(q*coeff[0])**(ns+coeff[1]))
                b = (1. + 0.2*coeff[2]*(ns+3)*(q*coeff[6])**(ns+coeff[7]+3))/(1.+(q*coeff[6])**(ns+coeff[7]+3.5));
                c = (1. + 4.5*coeff[3]/(1.5+(ns+3)**4)*(q*coeff[4])**(ns+3+coeff[8]))/(1+(q*coeff[4])**(ns+3.5+coeff[8]));
                a[0] =1
                b[0] =1
                c[0] =1

                moments12a =  compute_masked_m12(z, smoothing_scales1, smoothing_scales2, lmax, a, mask)
                moments13a =  compute_masked_m12(z, smoothing_scales1, smoothing_scales3, lmax, a, mask)
                moments23a =  compute_masked_m12(z, smoothing_scales2, smoothing_scales3, lmax, a, mask)

                moments12b =  compute_masked_m12(z, smoothing_scales1, smoothing_scales2, lmax, b, mask)
                moments13b =  compute_masked_m12(z, smoothing_scales1, smoothing_scales3, lmax, b, mask)
                moments23b =  compute_masked_m12(z, smoothing_scales2, smoothing_scales3, lmax, b, mask)

                moments12c =  compute_masked_m12(z, smoothing_scales1, smoothing_scales2, lmax, c, mask)
                moments13c =  compute_masked_m12(z, smoothing_scales1, smoothing_scales3, lmax, c, mask)
                moments23c =  compute_masked_m12(z, smoothing_scales2, smoothing_scales3, lmax, c, mask)


            ell = np.arange(lmax)

            P_lz = P_lz[:lmax]*F_l[:lmax]**2
            if mask:
                f_l = (ell+2)*(ell-1)/(ell*(ell+1))
                f_l[0:2] = 0
                P_lz = P_lz*f_l
                P_lz = self.ME[:lmax,:lmax]@P_lz[:lmax]
                f_linv = (ell*(ell+1))/((ell+2)*(ell-1))
                f_linv[0:2] = 0
                P_lz = P_lz*f_linv


            d_moments12_d_ln1b = np.zeros(len(smoothing_scales1))
            d_moments12_d_ln2b = np.zeros(len(smoothing_scales1))
            d_moments13_d_ln1b = np.zeros(len(smoothing_scales1))
            d_moments13_d_ln3b = np.zeros(len(smoothing_scales1))
            d_moments23_d_ln2b = np.zeros(len(smoothing_scales1))
            d_moments23_d_ln3b = np.zeros(len(smoothing_scales1))

            for i, sm in enumerate(zip(smoothing_scales1,smoothing_scales2,smoothing_scales3)):

                # convert scale to radians ***
                sm_rad1 =(sm[0]/60.)*np.pi/180.
                sm_rad2 =(sm[1]/60.)*np.pi/180.
                sm_rad3 =(sm[2]/60.)*np.pi/180.

                # smoothing kernel (top-hat)
                # first the (2l+1)Wl(theta)/4pi
                fact1 = (scipy.special.eval_legendre(ell-1,np.cos(sm_rad1))-scipy.special.eval_legendre(ell+1,np.cos(sm_rad1)))/(4*np.pi*(1-np.cos(sm_rad1)))
                fact1[0] = 1./4*np.pi
                fact2 = (scipy.special.eval_legendre(ell-1,np.cos(sm_rad2))-scipy.special.eval_legendre(ell+1,np.cos(sm_rad2)))/(4*np.pi*(1-np.cos(sm_rad2)))
                fact2[0] = 1./4*np.pi
                fact3 = (scipy.special.eval_legendre(ell-1,np.cos(sm_rad3))-scipy.special.eval_legendre(ell+1,np.cos(sm_rad3)))/(4*np.pi*(1-np.cos(sm_rad3)))
                fact3[0] = 1./(4*np.pi)

                # now the d_theta Wl(theta)
                d_fact1 =  scipy.special.eval_legendre(ell,np.cos(sm_rad1))*np.sin(sm_rad1)/(1-np.cos(sm_rad1))
                d_fact1 -= fact1*(4*np.pi/(2*ell+1))*np.sin(sm_rad1)/(1-np.cos(sm_rad1))
                d_fact1[0] = 0  #0th Wl = 1

                d_fact2 =  scipy.special.eval_legendre(ell,np.cos(sm_rad2))*np.sin(sm_rad2)/(1-np.cos(sm_rad2))
                d_fact2 -= fact2*(4*np.pi/(2*ell+1))*np.sin(sm_rad2)/(1-np.cos(sm_rad2))
                d_fact2[0] = 0

                d_fact3 =  scipy.special.eval_legendre(ell,np.cos(sm_rad3))*np.sin(sm_rad3)/(1-np.cos(sm_rad3))
                d_fact3 -= fact3*(4*np.pi/(2*ell+1))*np.sin(sm_rad3)/(1-np.cos(sm_rad3))
                d_fact3[0] = 0


                d_moments12_d_ln1b[i] = (sm_rad1*np.sum(b*fact2[:lmax]*d_fact1[:lmax]*P_lz[:lmax]))
                d_moments12_d_ln2b[i] = (sm_rad2*np.sum(b*fact1[:lmax]*d_fact2[:lmax]*P_lz[:lmax]))

                d_moments13_d_ln1b[i] = (sm_rad1*np.sum(b*fact3[:lmax]*d_fact1[:lmax]*P_lz[:lmax]))
                d_moments13_d_ln3b[i] = (sm_rad3*np.sum(b*fact1[:lmax]*d_fact3[:lmax]*P_lz[:lmax]))

                d_moments23_d_ln2b[i] = (sm_rad2*np.sum(b*fact3[:lmax]*d_fact2[:lmax]*P_lz[:lmax]))
                d_moments23_d_ln3b[i] = (sm_rad3*np.sum(b*fact2[:lmax]*d_fact3[:lmax]*P_lz[:lmax]))


            mu = 5/7

            moments  = 2*mu*moments13a*moments23a + (1-mu)*moments13c*moments23c
            moments += 0.5*(moments13b*d_moments23_d_ln3b + moments23b*d_moments13_d_ln3b)
            moments += 2*mu*moments12a*moments23a + (1-mu)*moments12c*moments23c
            moments += 0.5*(moments12b*d_moments23_d_ln2b + moments23b*d_moments12_d_ln2b)
            moments += 2*mu*moments13a*moments12a + (1-mu)*moments13c*moments12c
            moments += 0.5*(moments13b*d_moments12_d_ln1b + moments12b*d_moments13_d_ln1b)

        else:
            moments = 0*smoothing_scales1
        return np.array(moments)


    def kappa2(nz1, nz2, sm1, sm2, lmax = 1024, mask = False):

        #we'll be integrating in chi space
        chibin = results_LCDM.comoving_radial_distance(self.zbin)

        #lensing kernel 1
        qi1 = 0*self.zbin
        for i in range(len(self.zbin)):
            foo = nz1[i:]*(1-chibin[i]/chibin[i:])
            qi1[i] = np.trapz(foo, zbin[i:])
        qi1 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.zbin)*qi1*chibin
        qi1[0] = 0

        #lensing kernel 2
        qi2 = 0*self.zbin
        for i in range(len(self.zbin)):
            foo = nz2[i:]*(1-chibin[i]/chibin[i:])
            qi2[i] = np.trapz(foo, self.zbin[i:])
        qi2 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.zbin)*qi2*chibin
        qi2[0] = 0

        # compute the smoothed density field moment
        corr = np.array([compute_masked_m12(z, sm1, sm2, lmax, mask = mask) for z in self.zbin])

        # integrate over chi
        intgnd = qi1*qi2*corr.T/chibin**2
        intgnd[:,0] = 0
        res = np.trapz(intgnd , chibin, axis = 1)
        return res


    def kappa123(nz1, nz2, nz3, sm1, sm2, sm3, lmax = 1024, scheme = "LIN", mask = False):
        #we'll be integrating in chi space
        chibin = results_LCDM.comoving_radial_distance(self.zbin)

        #lensing kernel 1
        qi1 = np.zeros(len(self.zbin))
        for i in range(len(self.zbin)):
            foo = nz1[i:]*(1-chibin[i]/chibin[i:])
            qi1[i] = np.trapz(foo,  self.zbin[i:])
        qi1 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.zbin)*qi1*chibin
        qi1[0] = 0


        #lensing kernel 2
        qi2 = np.zeros(len(self.zbin))
        for i in range(len(self.zbin)):
            foo = nz2[i:]*(1-chibin[i]/chibin[i:])
            qi2[i] = np.trapz(foo, self.zbinv[i:])
        qi2 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.zbin)*qi2*chibin
        qi2[0] = 0

        #lensing kernel 3
        qi3 = np.zeros(len(self.zbin))
        for i in range(len(self.zbin)):
            foo = nz3[i:]*(1-chibin[i]/chibin[i:])
            qi3[i] =np.trapz(foo,  self.zbin[i:])
        qi3 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+self.zbin)*qi3*chibin
        qi3[0] = 0

        # compute the smoothed density field moment
        corr = np.array([compute_masked_m123(z, sm1, sm2, sm3, lmax, scheme, mask=mask) for z in self.zbin])

        # integrate over chi
        intgnd = qi1*qi2*qi3*corr.T/chibin**4
        intgnd[:,0] = 0
        res = np.trapz(intgnd, chibin, axis = 1)
        return res


def len_DV(n, zbin, s2, s3):
    '''
    n is # of smoothing scales
    zbin is # of redshift bins
    s2, s3 are "equi" or "all" where equi is Marco's, all is general
    this is SUPER nasty and hacky but it gets the combinatorics right for iterating over all
    possible moments. Use this as a template for building iterators over DV

    As it stands, returns the length of DV given # of sm and # of redshift bins and selection of moments
    '''

    bins_for_2 = []
    equibins_2 = []

    for i in range(zbin):
        for j in range(i+1):
            bins_for_2.append(str(i)+"_"+str(j))
            if i == j:
                equibins_2.append(str(i)+"_"+str(j))

    bins_for_3 = []
    isobins_3 = []
    equibins_3 = []
    rest_3 = []

    for i in range(zbin):
        for j in range(i+1):
            for k in range(j+1):
                bins_for_3.append(str(i)+"_"+str(j)+"_"+str(k))
                if (i == j) and (j == k):
                    equibins_3.append(str(i)+"_"+str(j)+"_"+str(k))
                elif ((i == j) or (j == k) or (i==k)) and not ((i == j) and (j == k) and (i==k)):
                    isobins_3.append(str(i)+"_"+str(j)+"_"+str(k))
                else:
                    rest_3.append(str(i)+"_"+str(j)+"_"+str(k))


    def pick_three(key, select3):
        if select3 == 'equi':
            foo = n

        if select3 == 'all':
            if (key in equibins_3):

                bar = 0
                count = int(n*(n+1)*(n+2)/6)
                ind1 = np.zeros(count, dtype = int)
                ind2 = np.zeros(count, dtype = int)
                ind3 = np.zeros(count, dtype = int)

                for i in range(n):
                    for j in range(i+1):
                        for k in range(j+1):
                            ind1[bar] = i
                            ind2[bar] = j
                            ind3[bar] = k
                            bar += 1
                foo = len(ind1)

            elif (key in isobins_3):
                ind1 = []
                ind2 = []
                ind3 = []
                for i in range(n):
                    for j in range(n):
                        for k in range(j+1):
                            ind1.append(i)
                            ind2.append(j)
                            ind3.append(k)
                foo = len(ind1)

            else:
                ind1 = []
                ind2 = []
                ind3 = []
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            ind1.append(i)
                            ind2.append(j)
                            ind3.append(k)

                foo = len(ind1)
        return foo

    def pick_two(key, select2):
        if select2 == 'equi':
            foo = n

        if select2 == 'all':
            if (key in equibins_2):

                foo = len(np.tril_indices(n)[0])

            else:
                ind1 = []
                ind2 = []
                for i in range(n):
                    for j in range(n):
                        ind1.append(i)
                        ind2.append(j)

                foo = len(ind1)
        return foo

    length = 0
    for key in bins_for_2:
        length += pick_two(key, s2)

    for key in bins_for_3:
        length += pick_three(key,s3)

    return length
