'''
This code takes FLASK output (convergence, shear, density fields) and add DES Y3 shape noise. It create maps of the shear field that needs to be used to compute the moments convariance.
'''
import gc
# import pyfits as pf
import pickle
import numpy as np

import healpy as hp
import os
import copy
# from Moments_analysis import convert_to_pix_coord, IndexToDeclRa, apply_random_rotation, addSourceEllipticity, gk_inv
# import healpy as hp
import scipy
from scipy.interpolate import interp1d
# import timeit
# import glob
import UFalcon
# import ekit
# from ekit import paths as path_tools
# import frogress
from tqdm import tqdm



import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from UFalcon import utils
from UFalcon import constants as constants_u




class Continuous:
    """
    Computes the lensing weights for a continuous, user-defined n(z) distribution.
    """

    def __init__(self, n_of_z, z_lim_low=0, z_lim_up=None, shift_nz=0.0, IA=0.0):
        """
        Constructor.
        :param n_of_z: either path to file containing n(z), assumed to be a text file readable with numpy.genfromtext
                        with the first column containing z and the second column containing n(z), or a callable that
                        is directly a redshift distribution
        :param z_lim_low: lower integration limit to use for n(z) normalization, default: 0
        :param z_lim_up: upper integration limit to use for n(z) normalization, default: last z-coordinate in n(z) file
        :param shift_nz: Can shift the n(z) function by some redshift (intended for easier implementation of photo z bias)
        :param IA: Intrinsic Alignment. If not None computes the lensing weights for IA component
                        (needs to be added to the weights without IA afterwards)
        """

        # we handle the redshift dist depending on its type
        if callable(n_of_z):
            if z_lim_up is None:
                raise ValueError("An upper bound of the redshift normalization has to be defined if n_of_z is not a "
                                 "tabulated function.")

            self.nz_intpt = n_of_z
            # set the integration limit and integration points
            self.lightcone_points = None
            self.limit = 1000
        else:
            # read from file
            nz = np.genfromtxt(n_of_z)

            # get the upper bound if necessary
            if z_lim_up is None:
                z_lim_up = nz[-1, 0]

            # get the callable function
            self.nz_intpt = interp1d(nz[:, 0] - shift_nz, nz[:, 1], bounds_error=False, fill_value=0.0)

            # points for integration
            self.lightcone_points = nz[np.logical_and(z_lim_low < nz[:, 0], nz[:, 0] < z_lim_up), 0]

            # check if there are any points remaining for the integration
            if len(self.lightcone_points) == 0:
                self.lightcone_points = None
                self.limit = 1000
            else:
                self.limit = 10 * len(self.lightcone_points)

        self.z_lim_up = z_lim_up
        self.z_lim_low = z_lim_low
        self.IA = IA
        # Normalization
        self.nz_norm = integrate.quad(lambda x: self.nz_intpt(x), z_lim_low, self.z_lim_up,
                                      points=self.lightcone_points, limit=self.limit)[0]

    def __call__(self, z_low, z_up, cosmo, lens = False):
        """
        Computes the lensing weights for the redshift interval [z_low, z_up].
        :param z_low: lower end of the redshift interval
        :param z_up: upper end of the redshift interval
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: lensing weight
        """
        if lens:
            norm = (z_up - z_low) * self.nz_norm
            # lensing weights
            def f(x):
                return (self.nz_intpt(x))

            if self.lightcone_points is not None:
                numerator = integrate.quad(f, z_low, z_up, points=self.lightcone_points[np.logical_and(z_low < self.lightcone_points, self.lightcone_points < z_up)])[0]
            else:
                numerator = integrate.quad(f, z_low, z_up)[0]
            return numerator / norm
        else:
            norm = utils.dimensionless_comoving_distance(z_low, z_up, cosmo) * self.nz_norm
            norm *= (utils.dimensionless_comoving_distance(0., (z_low + z_up)/2., cosmo) ** 2.)
            if abs(self.IA - 0.0) < 1e-10:
                # lensing weights without IA
                numerator = integrate.quad(self._integrand_1d, z_low, z_up, args=(cosmo,))[0]
            else:
                # lensing weights for IA
                numerator = (2.0/(3.0*cosmo.Om0)) * \
                            w_IA(self.IA, z_low, z_up, cosmo, self.nz_intpt, points=self.lightcone_points)

            return numerator / norm

    def _integrand_2d(self, y, x, cosmo):
        """
        The 2d integrant of the continous lensing weights
        :param y: redhsift that goes into the n(z)
        :param x: redshift for the Dirac part
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 2d integrand function
        """
        return self.nz_intpt(y) * \
               utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, y, cosmo) * \
               (1 + x) * \
               cosmo.inv_efunc(x) / \
               utils.dimensionless_comoving_distance(0, y, cosmo)

    def _integrand_1d(self, x, cosmo):
        """
        Function that integrates out y from the 2d integrand
        :param x: at which x (redshfit to eval)
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 1d integrant at x
        """
        if self.lightcone_points is not None:
            points = self.lightcone_points[np.logical_and(self.z_lim_low < self.lightcone_points,
                                                          self.lightcone_points < self.z_lim_up)]
            quad_y = lambda x: integrate.quad(lambda y: self._integrand_2d(y, x, cosmo), x, self.z_lim_up,
                                              limit=self.limit, points=points)[0]
        else:
            quad_y = lambda x: integrate.quad(lambda y: self._integrand_2d(y, x, cosmo), x, self.z_lim_up,
                                              limit=self.limit)[0]

        return quad_y(x)

def w_IA(IA, z_low, z_up, cosmo, nz_intpt, points=None):
    """
    Calculates the weight per slice for the NIA model given a
    distribution of source redshifts n(z).
    :param IA: Galaxy Intrinsic alignment amplitude
    :param z_low: Lower redshift limit of the shell
    :param z_up: Upper redshift limit of the shell
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param nz_intpt: nz function
    :param points: Points in redshift where integrand is evaluated (used for better numerical integration), can be None
    :return: Shell weight for NIA model

    """

    def f(x):
        return (F_NIA_model(x, IA, cosmo) * nz_intpt(x))

    if points is not None:
        dbl = integrate.quad(f, z_low, z_up, points=points[np.logical_and(z_low < points, points < z_up)])[0]
    else:
        dbl = integrate.quad(f, z_low, z_up)[0]

    return dbl


def density_prefactor(n_pix, n_particles, boxsize, cosmo):
    """
    Computes the prefactor to transform from number of particles to convergence, see https://arxiv.org/abs/0807.3651,
    eq. (A.1).
    :param n_pix: number of healpix pixels used
    :param n_particles: number of particles
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :return: convergence prefactor
    """
    convergence_factor = (n_pix / (4.0 * np.pi)) * \
                         (cosmo.H0.value / constants_u.c) ** 2 * \
                         (boxsize * 1000.0) ** 3 / n_particles
    return convergence_factor


def get_parameters_from_path(paths, suffix=True, fmt=None):
    """
    Given a list of standardised paths, or a single path created with
    create_path() this function reads the parameters in the paths.

    :param paths: Either a single string or a list of strings. The strings
                  should be paths in the create_path() format.
    :param suffix: If True assumes that the given paths have suffixes and
                   exclues them from the parsing
    :return: Returns a dictionary which contains the defined parameters and
             a list containing the undefined parameters.
    """
    # convert to list if needed
    if not isinstance(paths, list):
        paths = [paths]

    # use first path to initialize the dictionary and list for output
    defined_names = []
    undefined_count = 0
    path = paths[0]

    path = _prepare_path(path, suffix=suffix)

    # loop over parameters in first path to initialize dictionary
    for c in path:
        if isinstance(c, list):
            c = c[0]
        if '=' in c:
            b = c.split('=')
            defined_names.append(b[0])
        else:
            undefined_count += 1

    # initialize
    undefined = np.zeros((len(paths), undefined_count), dtype=object)
    defined = {}
    for d in defined_names:
        defined[d] = np.zeros(len(paths), dtype=object)

    # loop over files and get parameters
    for ii, path in enumerate(paths):
        path = _prepare_path(path, suffix=suffix)
        count = 0
        for idx_c, c in enumerate(path):
            if isinstance(c, list):
                c = c[0]
            if '=' in c:
                b = c.split('=')
                to_add = _check_type(b[1], fmt, idx_c)
                defined[b[0]][ii] = to_add
            else:
                to_add = _check_type(c, fmt, idx_c)
                undefined[ii, count] = to_add
                count += 1
    return defined, undefined





import yaml
def make_maps(jr):
    import healpy as hp
    from tqdm import tqdm
    ldir = '/global/cfs/cdirs/des/cosmogrid/raw/fiducial/cosmo_fiducial/run_' + str(jr) + '/'
    with open(ldir + 'params.yml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    Omegam, sig8 = data['Om'], data['s8']
    h = data['H0']/100.

    
    df = np.load(ldir + 'compressed_shells.npz')
    nsh = len(df['shell_info'])
    shid, sh_lz, sh_hz = np.zeros(nsh), np.zeros(nsh), np.zeros(nsh)
    for js in range(len(df['shell_info'])):
        shid[js], sh_lz[js], sh_hz[js] = df['shell_info'][js][1], df['shell_info'][js][2], df['shell_info'][js][3]
    z_bounds = {'z-low': sh_lz, 'z-high': sh_hz}
    # defines cosmology for PKDGRAV sims
    from astropy.cosmology import FlatLambdaCDM
    constants = FlatLambdaCDM(H0=float(h * 100.), Om0=float(Omegam))
    
    for jz in range(4):
        kappa = np.zeros((1, hp.nside2npix(2048)), dtype=np.float32)      
        ldir = '/global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/gen_moments/'      
        path_nz = ldir + 'nz_source_bin' + str(jz+1) + '.txt'
        lensing_weights = UFalcon.lensing_weights.Continuous(
                path_nz, z_lim_low=0., z_lim_up=4., shift_nz=0., IA=0.)
        
        shell_files = df['shells']
        for zz in tqdm(range(len(z_bounds['z-low']))):
            shell = shell_files[zz].astype(np.float64)
            # conversion from particle counts to convergence
            kappa_prefactor = UFalcon.lensing_weights.kappa_prefactor(
                            n_pix=shell.size, n_particles=832**3,
                            boxsize=1.33610451306, cosmo=constants)
            shell *= kappa_prefactor

            # Add the shell multiplied by the lensing weights
            weight = lensing_weights(
                z_bounds['z-low'][zz], z_bounds['z-high'][zz], constants)
            shell *= weight

            kappa += shell

        sdir = '/global/cfs/cdirs/lsst/www/shivamp/gen_mom/cosmogrid_kappa/fiducial/ns_2048/'
        hp.write_map(sdir + 'kappa-jr-' + str(jr) + '-jz-' + str(jz) + '.fits', kappa)
        kappa_dg = hp.ud_grade(kappa, 512)
        sdir = '/global/cfs/cdirs/lsst/www/shivamp/gen_mom/cosmogrid_kappa/fiducial/ns_512/'
        hp.write_map(sdir + 'kappa-jr-' + str(jr) + '-jz-' + str(jz) + '.fits', kappa_dg)


from mpi4py import MPI
run_count = 0
n_jobs = 200
# jr = 0

while run_count<n_jobs:
    comm = MPI.COMM_WORLD
    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
    if (run_count+comm.rank) < n_jobs:
        make_maps(comm.rank)
    run_count+=comm.size
    comm.bcast(run_count,root = 0)
    comm.Barrier()     

# # from mpi4py import MPI 
# # if __name__ == '__main__':
# #     runstodo=[]
# #     for i in range(0,1):
# #         #if not os.path.exists(output+'seedomb_'+str(i+1)+'.pkl'):
# #             runstodo.append(i)
# #     run_count=0
# #     print (runstodo)
# #     while run_count<len(runstodo):
# #         comm = MPI.COMM_WORLD
# #         print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
# #         try:
# #             make_maps(runstodo[run_count+comm.rank])
# #         except:
# #             pass
# #         run_count+=comm.size
# #         comm.bcast(run_count,root = 0)
# #         comm.Barrier() 
        
# # salloc -N 57 -C haswell -q interactive -t 02:00:00 -L SCRATCH
# # srun --nodes=57 --tasks-per-node=1 --cpu-bind=cores python make_kappamap_pkdgrav.py