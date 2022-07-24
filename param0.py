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

PK = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=1000.0, zmax=10,nonlinear=True, extrap_kmax= 10**10)

PK_L = camb.get_matter_power_interpolator(pars_LCDM, hubble_units=False, k_hunit=False, kmax=1000.0, zmax=10,nonlinear=False, extrap_kmax= 10**10)

chitoz = results_LCDM.redshift_at_comoving_radial_distance
ztochi = results_LCDM.comoving_radial_distance


import h5py

# compute coupling matries with Namaster.

fname = '/global/cfs/cdirs/des/shivamp/gen_moments/Moments_analysis_minsu/Covariance/mask_DES_y3_py2'
mask = load_obj(fname)
# mask = load_obj("../Moments_analysis/Covariance/mask_DES_y3_py2")
#save_obj("mask_DES_y3_py2",mask)
print ('f_sky: ', 1./(len(mask)*1./len(mask[mask])))
mask_sm =  hp.sphtfunc.smoothing(mask, (13./60.)*np.pi/180.  )
mask_sm[mask] = 1.
mask = copy.copy(mask_sm)
# computes Cl.
lmax = 1024
nside = 512

try:
    import pickle as pk
    df = pk.load(open('namaster_stuff.pk','rb'))
    M = df['M']
    ME = df['ME']
except:
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

def compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales2, lmax = 1024, abc = 1., mask = None):
    '''
    It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (k=l/chi(z)).
    '''
    if (z != 0.0):
        P_lz = PK.P(z, np.arange(lmax)/results_LCDM.comoving_radial_distance(z))
        F_l =  hp.sphtfunc.pixwin(512, lmax = lmax)
        ell = np.arange(lmax)

        P_lz = P_lz[:lmax]*F_l[:lmax]**2
        if mask is not None:
            f_l = (ell+2)*(ell-1)/(ell*(ell+1))
            f_l[0:2] = 0
            P_lz = P_lz*f_l
            P_lz = mask[:lmax,:lmax]@P_lz[:lmax]
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

def compute_masked_m123(PK, PK_L, z, smoothing_scales1, smoothing_scales2, smoothing_scales3, lmax = 1024, scheme = 'LIN', mask = None):
    '''
    It computes the smoothed (by a top-hat filter) 2nd moments of the density field given the
    3D power spectrum at fixed z. (k=l/chi(z)).
    '''
    if (z != 0):

        kay = np.arange(lmax)/results_LCDM.comoving_radial_distance(z)

        # NL mps as fn of ell
        P_lz = PK.P(z, kay)
        #pixel window function
        F_l =  hp.sphtfunc.pixwin(512, lmax = lmax)

        # growth factor at z defined as sqrt(P(z, k=1)/P(0,k=1)) bc k choice is irrelevant
        Dz = np.sqrt(PK_L.P(z, 1)/PK_L.P(0, 1))

        # k at which MPS becomes unity
        def helper(k):
            return k**3*PK_L.P(z, k)/(2*np.pi**2)-1

        knl = scipy.optimize.root(helper, 0.5).x[0]

        # d ln P/d ln k at relevant k values
        ns = (kay/PK_L.P(0,kay))*(PK_L.P(0,kay+0.001)-PK_L.P(0,kay))/(0.001)


            # Initialise coefficients small-scales fitting formulae.
        if scheme == 'SC':
            coeff = [0.25,3.5,2.,1.,2.,-0.2,1.,0.,0.]
        elif scheme == 'GM':
            coeff = [0.484,3.740,-0.849,0.392,1.013,-0.575,0.128,-0.722,-0.926]

        if (scheme == 'LIN'):
            a = 1.;
            b = 1.;
            c = 1.;

            moments12a =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales2, lmax, a, mask)
            moments13a =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales3, lmax, a, mask)
            moments23a =  compute_masked_m12(PK, z, smoothing_scales2, smoothing_scales3, lmax, a, mask)

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

            moments12a =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales2, lmax, a, mask)
            moments13a =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales3, lmax, a, mask)
            moments23a =  compute_masked_m12(PK, z, smoothing_scales2, smoothing_scales3, lmax, a, mask)

            moments12b =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales2, lmax, b, mask)
            moments13b =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales3, lmax, b, mask)
            moments23b =  compute_masked_m12(PK, z, smoothing_scales2, smoothing_scales3, lmax, b, mask)

            moments12c =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales2, lmax, c, mask)
            moments13c =  compute_masked_m12(PK, z, smoothing_scales1, smoothing_scales3, lmax, c, mask)
            moments23c =  compute_masked_m12(PK, z, smoothing_scales2, smoothing_scales3, lmax, c, mask)


        ell = np.arange(lmax)

        P_lz = P_lz[:lmax]*F_l[:lmax]**2
        if mask is not None:
            f_l = (ell+2)*(ell-1)/(ell*(ell+1))
            f_l[0:2] = 0
            P_lz = P_lz*f_l
            P_lz = mask[:lmax,:lmax]@P_lz[:lmax]
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



def kappa2(PK, zbin, nz1, nz2, sm1, sm2, lmax = 1024, mask = None):
    chibin = results_LCDM.comoving_radial_distance(zbin)

    qi1 = np.zeros(len(zbin))
    for i in range(len(zbin)):
        foo = nz1[i:]*(1-chibin[i]/chibin[i:])
        qi1[i] = np.trapz(foo, zbin[i:])
    qi1 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi1*chibin
    qi1[0] = 0

    qi2 = np.zeros(len(zbin))
    for i in range(len(zbin)):
        foo = nz2[i:]*(1-chibin[i]/chibin[i:])
        qi2[i] = np.trapz(foo, zbin[i:])
    qi2 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi2*chibin
    qi2[0] = 0

    corr = np.array([compute_masked_m12(PK, z, sm1, sm2, lmax, mask = mask) for z in zbin])

    intgnd = qi1*qi2*corr.T/chibin**2
    intgnd[:,0] = 0
    res = np.trapz(intgnd , chibin, axis = 1)
    return res


def kappa123(PK, PK_L, zbin, nz1, nz2, nz3, sm1, sm2, sm3, lmax = 1024, scheme = "LIN", mask = None):
    chibin = results_LCDM.comoving_radial_distance(zbin)

    qi1 = np.zeros(len(zbin))
    for i in range(len(zbin)):
        foo = nz1[i:]*(1-chibin[i]/chibin[i:])
        qi1[i] = np.trapz(foo, zbin[i:])
    qi1 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi1*chibin
    qi1[0] = 0

    qi2 = np.zeros(len(zbin))
    for i in range(len(zbin)):
        foo = nz2[i:]*(1-chibin[i]/chibin[i:])
        qi2[i] = np.trapz(foo, zbin[i:])
    qi2 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi2*chibin
    qi2[0] = 0

    qi3 = np.zeros(len(zbin))
    for i in range(len(zbin)):
        foo = nz3[i:]*(1-chibin[i]/chibin[i:])
        qi3[i] =np.trapz(foo, zbin[i:])
    qi3 = (1.5*pars_LCDM.omegam*(pars_LCDM.H0/(camb.constants.c/1000.))**2)*(1+zbin)*qi3*chibin
    qi3[0] = 0

    corr = np.array([compute_masked_m123(PK, PK_L, z, sm1, sm2, sm3, lmax, scheme, mask) for z in zbin])

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


# sm = np.array([8.2, 13.125, 21.0,33.6,54.,86., 137.6, 220.16])
sm = np.array([21.0,33.6,54.,86., 137.6, 220.16])


kp2_2_2 = np.zeros((len(sm),len(sm)))
kp2_2_3 = np.zeros((len(sm),len(sm)))
kp2_3_3 = np.zeros((len(sm),len(sm)))

for k in range(3):
    if k == 0:
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    kp2_2_2[i,j] = kappa2(PK,zbin, nz2, nz2, np.array([sm[i]]), np.array([sm[j]]), mask = ME)
                else:
                    kp2_2_2[i,j] = kp2_2_2[j,i]
    elif k == 1:
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    kp2_2_3[i,j] = kappa2(PK,zbin, nz2, nz3, np.array([sm[i]]), np.array([sm[j]]), mask = ME)
                else:
                    kp2_2_3[i,j] = kp2_2_3[j,i]

    elif k == 2:
        for i in range(len(sm)):
            for j in range(len(sm)):
                if i <= j:
                    kp2_3_3[i,j] = kappa2(PK,zbin, nz3, nz3, np.array([sm[i]]), np.array([sm[j]]), mask = ME)
                else:
                    kp2_3_3[i,j] = kp2_3_3[j,i]

save_obj('./fisher/kp2param0_2_2', kp2_2_2)
save_obj('./fisher/kp2param0_2_3', kp2_2_3)
save_obj('./fisher/kp2param0_3_3', kp2_3_3)



kp3_2_2_2 = np.zeros((len(sm),len(sm),len(sm)))
kp3_3_2_2 = np.zeros((len(sm),len(sm),len(sm)))
kp3_2_3_3 = np.zeros((len(sm),len(sm),len(sm)))
kp3_3_3_3 = np.zeros((len(sm),len(sm),len(sm)))

for l in range(4):
    if l == 0:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_2_2_2[i,j,k] = kappa123(PK,PK_L, zbin, nz2, nz2, nz2, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='SC', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_2_2_2[i, j, k] = kp3_2_2_2[foo[0], foo[1], foo[2]]
    elif l == 1:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_3_2_2[i,j,k] = kappa123(PK,PK_L, zbin, nz3, nz2, nz2, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='SC', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_3_2_2[i, j, k] = kp3_3_2_2[foo[0], foo[1], foo[2]]

    elif l == 2:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_2_3_3[i,j,k] = kappa123(PK,PK_L, zbin, nz2, nz3, nz3, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='SC', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_2_3_3[i, j, k] = kp3_2_3_3[foo[0], foo[1], foo[2]]

    elif l == 3:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_3_3_3[i,j,k] = kappa123(PK,PK_L, zbin, nz3, nz3, nz3, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='SC', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_3_3_3[i, j, k] = kp3_3_3_3[foo[0], foo[1], foo[2]]

save_obj('./fisher/kp3param0SC_2_2_2', kp3_2_2_2)
save_obj('./fisher/kp3param0SC_3_2_2', kp3_3_2_2)
save_obj('./fisher/kp3param0SC_2_3_3', kp3_2_3_3)
save_obj('./fisher/kp3param0SC_3_3_3', kp3_3_3_3)



kp3_2_2_2 = np.zeros((len(sm),len(sm),len(sm)))
kp3_3_2_2 = np.zeros((len(sm),len(sm),len(sm)))
kp3_2_3_3 = np.zeros((len(sm),len(sm),len(sm)))
kp3_3_3_3 = np.zeros((len(sm),len(sm),len(sm)))

for l in range(4):
    if l == 0:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_2_2_2[i,j,k] = kappa123(PK,PK_L, zbin, nz2, nz2, nz2, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='GM', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_2_2_2[i, j, k] = kp3_2_2_2[foo[0], foo[1], foo[2]]
    elif l == 1:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_3_2_2[i,j,k] = kappa123(PK,PK_L, zbin, nz3, nz2, nz2, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='GM', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_3_2_2[i, j, k] = kp3_3_2_2[foo[0], foo[1], foo[2]]

    elif l == 2:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_2_3_3[i,j,k] = kappa123(PK,PK_L, zbin, nz2, nz3, nz3, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='GM', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_2_3_3[i, j, k] = kp3_2_3_3[foo[0], foo[1], foo[2]]

    elif l == 3:
        for i in range(len(sm)):
            for j in range(len(sm)):
                for k in range(len(sm)):
                    if (k >= j and j >= i):
                        kp3_3_3_3[i,j,k] = kappa123(PK,PK_L, zbin, nz3, nz3, nz3, np.array([sm[i]]), np.array([sm[j]]), np.array([sm[k]]), scheme ='GM', mask = ME)
                    else:
                        foo = np.sort([i, j, k])
                        kp3_3_3_3[i, j, k] = kp3_3_3_3[foo[0], foo[1], foo[2]]


save_obj('./fisher/kp3param0GM_2_2_2', kp3_2_2_2)
save_obj('./fisher/kp3param0GM_3_2_2', kp3_3_2_2)
save_obj('./fisher/kp3param0GM_2_3_3', kp3_2_3_3)
save_obj('./fisher/kp3param0GM_3_3_3', kp3_3_3_3)
