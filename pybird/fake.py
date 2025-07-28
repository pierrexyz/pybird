import os, sys
import numpy as np
import yaml, h5py
from copy import deepcopy
from collections import defaultdict

from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import trapezoid as trapz
from scipy.special import legendre

def get_cov_gauss(kk, dk, ipk_lin, b1, f1, Vs=3.e9, nbar=3.e-4, n_mult=3): 
    # dk = np.concatenate((kk[1:]-kk[:-1], np.array([kk[-1]-kk[-2]]))) # this is true for k >> kf
    Nmode = 4 * np.pi * kk**2 * dk * (Vs / (2*np.pi)**3)
    mu_arr = np.linspace(0., 1., 200)
    k_mesh, mu_mesh = np.meshgrid(kk, mu_arr, indexing='ij')
    legendre_mesh = np.array([legendre(2*l)(mu_mesh) for l in range(3)])
    legendre_ell_mesh = np.array([(2*(2*l)+1)*legendre(2*l)(mu_mesh) for l in range(3)])
    pkmu_mesh = (b1 + f1 * mu_mesh**2)**2 * ipk_lin(k_mesh)
    integrand_mu_mesh = np.einsum('k,km,lkm,pkm->lpkm', 1./Nmode, (pkmu_mesh + 1/nbar)**2, legendre_ell_mesh, legendre_ell_mesh)
    cov_diagonal = 2 * trapz(integrand_mu_mesh, x=mu_arr, axis=-1)
    return np.block([[np.diag(cov_diagonal[i,j]) for i in range(n_mult)] for j in range(n_mult)]) 

class Fake():
    """A class to generate fake data for testing PyBird calculations.
    
    The Fake class creates synthetic power spectrum data with realistic covariance matrices
    for testing and validation purposes. It handles survey-specific parameters, cosmological
    models, and EFT nuisance parameters to generate realistic test cases.
    
    Attributes:
        n_sky (int): Number of sky patches.
        cosmo (dict): Cosmological parameters.
        boltzmann (str): Boltzmann solver to use ('class', 'Symbolic', etc.).
        zmin (list): Minimum redshift for each sky patch.
        zmax (list): Maximum redshift for each sky patch.
        zeff (list): Effective redshift for each sky patch.
        Veff (list): Effective volume for each sky patch.
        degsq (list): Sky area in square degrees for each sky patch.
        P0 (list): Power spectrum normalization for each sky patch.
        Omega_m_fid (float): Fiducial matter density parameter.
        kmin (float): Minimum k value for power spectrum.
        kmax (float): Maximum k value for power spectrum.
        dk (float): k-bin width.
        path_to_data (str): Path to save fake data files.
        fake_data_filename (str): Base name for fake data files.
        path_to_config (str): Path to save configuration files.
        fake_likelihood_config_filename (str): Base name for configuration files.
        
        fiducial_nuisance (dict): Fiducial EFT nuisance parameters.
        c (dict): Configuration dictionary for correlator.
        e (list): List of Correlator instances for each sky patch.
        nbar (list): Number density for each sky patch.
        H_fid (list): Fiducial Hubble parameter for each sky patch.
        D_fid (list): Fiducial angular diameter distance for each sky patch.
        kd (ndarray): Array of k values for power spectrum.
    
    Methods:
        to_list(): Convert input to list format matching n_sky.
        set_survey_specific(): Set survey-specific parameters.
        set_correlator(): Initialize correlator with configuration.
        set_nuisance(): Set EFT nuisance parameters.
        set(): Generate and save fake data.
        test(): Test the generated fake data with PyBird likelihood.
    """

    def __init__(self, n_sky, zmin, zmax, zeff, Veff, degsq, P0, 
        fiducial_cosmo, likelihood_config=None, likelihood_config_template_file='.h5', fiducial_nuisance=None, boltzmann='class', 
        Omega_m_fid=0.310, kmin=0.005, kmax=0.4, dk=0.01, k_arr=None, cov=None, nbar=None, nbar_prior=None, 
        fake_data_filename='fake_', path_to_data='./', fake_likelihood_config_filename='fake_', path_to_config='./'):

        if likelihood_config is None:
            if not os.path.isfile(likelihood_config_template_file): raise Exception("no likelihood_config dict provided, and %s not found" % likelihood_config_template_file)
            else: likelihood_config = yaml.full_load(open(likelihood_config_template_file, 'r'))

        self.path, self.fake_data_filename, self.path_to_file, self.path_to_config = path_to_data, '%s.h5' % fake_data_filename, os.path.join(path_to_data, '%s.h5') % fake_data_filename, os.path.join(path_to_config, '%s.yaml') % fake_likelihood_config_filename # output names and paths
        self.n_sky, self.cosmo, self.boltzmann, = n_sky, fiducial_cosmo, boltzmann

        with_emu = likelihood_config['with_emu'] if 'with_emu' in likelihood_config else False
        if self.boltzmann == 'Symbolic' or with_emu: 
            self.jax_jit = True
            from pybird.config import set_jax_enabled
            set_jax_enabled(True)
        else:
            self.jax_jit = False

        from pybird.io_pb import ReadWrite, save_dict_to_hdf5
        from pybird.projection import Hubble, DA

        self.io = ReadWrite()
        self.save_dict_to_hdf5 = save_dict_to_hdf5
        self.Hubble, self.DA = Hubble, DA
        
        self.set_survey_specific(zmin, zmax, zeff, Veff, degsq, P0, Omega_m_fid=Omega_m_fid, kmin=kmin, kmax=kmax, dk=dk, k_arr=k_arr, cov=cov, nbar=nbar)
        # self.set_boltzmann(fiducial_cosmo, fiducial_nuisance, boltzmann=boltzmann)
        self.set_correlator(likelihood_config, nbar=nbar_prior)
        self.set_nuisance(fiducial_nuisance=fiducial_nuisance)

    def to_list(self, x, type=float, what=''): 
        if isinstance(x, (list, np.ndarray)) and len(x) == self.n_sky: return x 
        elif isinstance(x, type): return self.n_sky * [x]
        else: raise Exception('%s to be provided in %s (or a list of n_sky %s) format' % (what, type, type))
    
    def set_survey_specific(self, zmin, zmax, zeff, Veff=10.e9, degsq=14000., P0=9.e3, Omega_m_fid=0.310, kmin=0.005, kmax=0.4, dk=0.01, k_arr=None, cov=None, nbar=None): 

        self.zmin, self.zmax, self.zeff, self.Veff, self.degsq, self.P0 = self.to_list(zmin, type=float, what='zmin'), self.to_list(zmax, type=float, what='zmax'), self.to_list(zeff, type=float, what='zeff'), self.to_list(Veff, type=float, what='Effective volume [Mpc^3]'), self.to_list(degsq, type=(float, int), what='Sky area [deg^2]'), self.to_list(P0, type=float, what='P0 [(Mpc/h)^3]')
        self.Omega_m_fid, self.H_fid, self.D_fid = Omega_m_fid, [self.Hubble(Omega_m_fid, z) for z in self.zeff], [self.DA(Omega_m_fid, z) for z in self.zeff] # for AP effect
        if k_arr is None: self.dk, self.kd = dk, np.arange(kmin, kmax, dk)
        else: self.dk, self.kd = k_arr[-1]-k_arr[-2], k_arr
        self.cov = None if cov is None else self.to_list(cov, type=np.ndarray, what='cov')
        
        if nbar is None: 
            def comoving_distance(z, Omega_m=0.31, h=0.68, c_light=3.e5):
                return c_light / (h * 100.) * self.DA(Omega_m, z) * (1.+z) # in Mpc
            def degsq_to_sa(degsq): return degsq*(np.pi/180.)**2 # Convert square degree to solid angle
            def get_nbar(zmin, zmax, Veff, degsq, P0): # Estimate nbar in [h/Mpc]^3 (since P0 is in [Mpc/h]^3) from Veff and area
                fac = 3 * Veff/degsq_to_sa(degsq)/(comoving_distance(zmax)**3 - comoving_distance(zmin)**3)
                return (-fac**0.5*P0 - fac*P0)/(fac*P0**2-P0**2)
            self.nbar = [get_nbar(zmin, zmax, Veff, degsq, P0) for zmin, zmax, Veff, degsq, P0 in zip(self.zmin, self.zmax, self.Veff, self.degsq, self.P0)]
        else:
            self.nbar = self.to_list(nbar, type=float, what='nbar')

        return

    def set_correlator(self, likelihood_config, nbar=None):

        from pybird.correlator import Correlator

        self.c = likelihood_config
        for option in ['with_survey_mask', 'with_wedge', 'with_binning', 'with_redshift_bin']: 
            if option in self.c: 
                if self.c[option]: print ('Warning: setting %s to False in fake data generation' % option)
            self.c[option] = False 

        if nbar is None: nbar = self.nbar
        self.c['sky'] = {'sky_%s' % (i+1): {'max': self.c['multipole'] * [0.20], 'min': self.c['multipole'] * [0.01], 'nd': float(nbar[i])} for i in range(self.n_sky)}
        d_sky = [{'z': self.zeff[i], 'x': self.kd, 'x_arr': self.c['multipole'] * [self.kd], 'fid': {'H': self.H_fid[i], 'D': self.D_fid[i]}, 'binsize': self.dk} for i in range(self.n_sky)] # that's some extra data-dependent settings to be passed through correlator config
        if 'nd' in self.c: self.c.pop('nd') # to avoid conflict # PZ: to change the input of this option to list of nd instead of nd in skylist?
        self.c_sky = self.io.config(self.c, d_sky) # creating skylist of correlator config dict

        self.e = [Correlator(self.c_sky[i]) for i in range(self.n_sky)] # skylist of PyBird correlator engine
        return  

    def set_nuisance(self, fiducial_nuisance=None): # same set of eft parameters for all sky
        if fiducial_nuisance is None: 
            print ('No fiducial EFT parameters specified, using default ones')
            # fiducial_nuisance =  {'b1': 1.9542,'c2': 0.5902, 'c4': 0.0, 'b3': -0.3686, 'cct': 0.1843, 'cr1': -0.8477, 'cr2': -0.8141, 'ce0': 1.499, 'ce1': 0.0, 'ce2': -1.6279, 'cr4': 0.0, 'cr6': 0.0, 'b2': 0.4173, 'b4': 0.4173}
            fiducial_nuisance =  {'b1': 1.9542,'b2': 0.4173, 'b4': 0.4173,'c2': 0.5902, 'c4': 0.0, 'b3': -0.3686, 'cct': 0.1843, 'cr1': -0.8477, 'cr2': -0.8141, 'ce0': 0., 'ce1': 0.0, 'ce2': -1.6279, 'cr4': 0.0, 'cr6': 0.0}
        
        if self.c['eft_basis'] == 'westcoast': 
            for b in ['b2', 'b4']: 
                if b in fiducial_nuisance: fiducial_nuisance.pop(b)
        if self.c['eft_basis'] == 'eftoflss':
            for b in ['c2', 'c4']:
                if b in fiducial_nuisance: fiducial_nuisance.pop(b)
        if not self.c['with_nnlo_counterterm']: 
            for b in ['cr4', 'cr6']:
                if b in fiducial_nuisance: fiducial_nuisance.pop(b)

        self.fiducial_nuisance = self.to_list(fiducial_nuisance, type=dict, what='Fiducial EFT parameters')
        return 

    def set(self, prior_center_on_truth=True): 
        
        d = defaultdict(dict)
        for i in range(self.n_sky): 
            self.e[i].compute(self.cosmo, cosmo_module=self.boltzmann)
            print("bsky[i]", self.fiducial_nuisance[i])
            bpk = self.e[i].get(deepcopy(self.fiducial_nuisance[i]))
            if self.cov is None: 
                b1, f1, ipk_lin = self.fiducial_nuisance[i]['b1'], self.e[i].bird.f, interp1d(self.e[i].bird.kin, self.e[i].bird.Pin)
                cov = get_cov_gauss(self.kd, self.dk, ipk_lin, b1, f1, Vs=self.Veff[i], nbar=self.nbar[i], n_mult=self.c['multipole'])
            else: 
                cov = self.cov[i]
            self.io.write_common(d['sky_%s' % (i+1)], self.zmin[i], self.zmax[i], self.zeff[i], self.Omega_m_fid, self.H_fid[i], self.D_fid[i])
            self.io.write_pk(d['sky_' + str(i+1)], self.c['multipole'], self.kd, bpk, cov, nsims_cov_pk=0, survey_mask_arr_p=None, survey_mask_mat_kp=None, binsize=self.dk)

        with h5py.File(self.path_to_file, 'w') as hf: self.save_dict_to_hdf5(hf, d)
        
        self.c['data_path'], self.c['data_file'] = self.path, self.fake_data_filename
        if prior_center_on_truth: 
            for name in self.fiducial_nuisance[0].keys(): 
                self.c["eft_prior"][name]['mean'] = [self.fiducial_nuisance[i][name] for i in range(self.n_sky)]
                self.c["eft_prior"][name]['range'] = [self.c['eft_prior'][name]['range'][0]] * self.n_sky # PZ: for now this is slightly hardcoded

        with open(self.path_to_config, 'w') as file: yaml.dump(self.c, file)

        return 

    def test(self, run_config=None, plot=False): # PZ: plot for now doesn't work in jax, need debug

        from pybird.run import Run

        if run_config is None:
            print ('no run_config dict provided, using default testing option')
            run_config = {'free_cosmo_name': [], 'fiducial_cosmo': self.cosmo, 'jax_jit': self.jax_jit, 
                'boltzmann': self.boltzmann, 'minimizers': ['onestep']}

        self.R = Run(run_config, self.c, self.path, verbose=True)
        if plot: self.R.I.L.c['write']['plot'], self.R.I.L.c['write']['show'] = True, True
        self.R.run(minimizers=run_config['minimizers'], samplers=[], output=False, save_to_file=False, verbose=True)


        return 
