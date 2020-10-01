import numpy as np
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
import yaml
import os
from astropy.io import fits
from scipy.interpolate import interp1d
try:
    from . limber import Limber
except ImportError:
    raise Exception('Cannot find limber library')

from . extract import TwoPointExtraction

import pybird

class DESsynth(Likelihood):
    
    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Load config file
        self.config = yaml.full_load(open(os.path.join(self.data_directory, self.config_file), 'r'))

        # path to data .fits
        path_to_data = os.path.join(self.data_directory, self.data_file)
        path_to_cut = os.path.join(self.data_directory, self.config["cut_file"])

        # Load data, covariance and mask
        extract = TwoPointExtraction(path_to_data, path_to_cut)
        self.ydata = extract.data_y # ['xip', 'xim', 'gammat', 'wtheta']
        self.mask = np.concatenate(extract.two_point_data.masks)
        if self.config["synth"]: self.ydata = np.loadtxt(os.path.join(self.data_directory, self.config["synth_file"]))[self.mask] # synthetic data for testing
        self.cov = extract.cov 
        self.invcov = np.linalg.inv(self.cov)
        self.chi2data = np.dot(self.ydata, np.dot(self.invcov, self.ydata))
        self.invcovdata = np.dot(self.ydata, self.invcov)

        # for plotting 
        self.multimask = extract.two_point_data.masks
        self.indexdict = extract.indexdict
        self.indexbin = [list(zip(extract.two_point_data.spectra[i].bin1, extract.two_point_data.spectra[i].bin2)) for i in range(4)]
        self.binpairs = [extract.two_point_data.spectra[i].bin_pairs for i in range(4)]
        self.angle = extract.angle
        
        self.correlator_data = [spectrum.value for spectrum in extract.two_point_data.spectra]
        self.correlator_data_list = self.__create_list(self.correlator_data)
        self.correlator_error_list = self.__create_list(np.sqrt(np.diag(self.cov)))
        self.angle_list = self.__create_list(self.angle)
        self.angle_min_clustering = np.array([angle[0] for angle in self.angle_list[-1]])
        
        # Load angles theta radial selection functions nlens, nsource
        des = fits.open(path_to_data)

        tam = np.empty(shape=(20)) # angle theta in arcmin
        for i, line in enumerate(des['wtheta'].data):
            bin1, bin2, angbin, val, ang, npairs = line
            if i < 20: tam[i] = ang
        self.t = tam * np.pi/(60. * 180.) # angle theta in radians
        
        Nl = des['nz_lens'].data.shape[0]
        zl = np.empty(shape=(Nl))
        nl = np.empty(shape=(5,Nl))
        for i, line in enumerate(des['nz_lens'].data):
            zlow, zmid, zhigh, bin1, bin2, bin3, bin4, bin5 = line
            zl[i] = zmid
            for j in range(5): nl[j,i] = line[3+j]/(zhigh-zlow)
        for j in range(5): nl[j] /= np.trapz(nl[j], x=zl)
        
        Ns = des['nz_source'].data.shape[0]
        zs = np.empty(shape=(Ns))
        ns = np.empty(shape=(4,Ns))
        for i, line in enumerate(des['nz_source'].data):
            zlow, zmid, zhigh, bin1, bin2, bin3, bin4 = line
            zs[i] = zmid
            for j in range(4): ns[j,i] = line[3+j]/(zhigh-zlow)
        for j in range(4): ns[j] /= np.trapz(ns[j], x=zs)
        
        zz = zl[zl<2.] # zl[zl<zhigh]
        self.zz = zz[1:]
        self.nlens = interp1d(zl, nl, kind='cubic', axis=-1)(self.zz)
        self.nsource = interp1d(zs, ns, kind='cubic', axis=-1)(self.zz)
        
        # self.config = {}
        self.config["output"] = "w"
        self.config["skycut"] = 5
        self.config["km"] = 0.7
        
        self.fullgg = self.config["fullgg"]
        self.marg = self.config["marg"]
        self.marg_chi2 = self.config["marg_chi2"]
        self.with_derived_bias = self.config["with_derived_bias"]
        self.nnlo = self.config["nnlo"]
        if self.nnlo: self.nnlo_file = self.config["nnlo_file"]
        else: self.nnlo_file = None
        
        # Loading Limber
        self.limber = Limber(self.t, self.zz, self.nlens, self.nsource, km=self.config["km"], load=False, save=False, nnlo=self.nnlo)
        
        # Settings for Pybird
        Nz = 200
        zeff = np.array([0.24, 0.38, 0.525, 0.685, 0.83])
        zbird = np.empty(shape=(5, Nz))
        nbird = np.empty(shape=(5, Nz))
        for i in range(5):
            zbird[i] = np.linspace(zeff[i] - 0.13, zeff[i] + 0.13, Nz)
            nbird[i] = interp1d(self.zz, self.nlens[i], kind='cubic')(zbird[i])
        
        self.config["z"] = zeff
        self.config["zz"] = zbird
        self.config["nz"] = nbird
        self.config["xdata"] = self.t
        self.config["with_resum"] = False
        self.config["w_theta_min"] = self.angle_min_clustering
        
        # Loading Pybird
        self.clustering = pybird.Correlator()
        self.clustering.set(self.config)

        # CLASSy settings
        self.zmax = 2. 
        self.zfid = self.config["z"][2]
        self.kk = np.geomspace(1.5e-5, 1., 200)
        self.need_cosmo_arguments(data, {'output': 'mPk', 'z_max_pk': self.zmax, 'P_k_max_1/Mpc': 2., 'non linear': 'halofit'})
        
        # BBN prior?
        if self.config["with_bbn"] and self.config["omega_b_BBNcenter"] is not None and self.config["omega_b_BBNsigma"] is not None:
            print ('BBN prior on omega_b: on')
        else:
            self.config["with_bbn"] = False
            print ('BBN prior on omega_b: none')

    def set_limber(self, bval, bg=None):
        self.limber.setBias(self.__bias_for_limber(bval, bg))

    def set_limber_nnlo(self, bnnlo=None):
        if bnnlo is None: self.limber.setnnlo(np.ones(shape=(self.limber.Nbin))) # setting with bnnlo = 1 for marginalization
        else: self.limber.setnnlo(bnnlo) 

def get_correlator(self, bval, bg=None):
        self.set_limber(bval, bg)
        if self.fullgg: clustering = np.asarray( self.clustering.get(self.__bias_for_clustering(bval, bg)) )
        else: clustering = self.limber.Xgg
        correlator = np.concatenate([self.limber.Xssp.reshape(-1), self.limber.Xssm.reshape(-1), self.limber.Xgs.reshape(-1), clustering.reshape(-1)])
        if self.nnlo: 
            if self.marg: 
                if bg is None: self.set_limber_nnlo() # setting nnlo for marg
                else: self.set_limber_nnlo(bg[self.limber.Nbin+self.limber.Ngg:]) 
            else: self.set_limber_nnlo(bval[self.limber.Nbin+3*self.limber.Ngg:]) # bval[self.limber.Nbin+self.limber.Ngg:] = [bnnlo_ss_1, ..., bnnlo_gg_5]
            if bg is not None: correlator += np.concatenate([self.limber.X2Lssp.reshape(-1), self.limber.X2Lssm.reshape(-1), self.limber.X2Lgs.reshape(-1), self.limber.X2Lgg.reshape(-1)])
        return correlator[self.mask]

    def get_correlator_marg(self, b1):
        if self.fullgg: 
            clustering_marg = np.asarray(self.clustering.getmarg(b1))
            correlator_marg = self.limber.getmarg(b1, external_gg_counterterm=clustering_marg[:,1,:], external_gg_b3=clustering_marg[:,0,:], nnlo=self.nnlo)
        else: correlator_marg = self.limber.getmarg(b1, nnlo=self.nnlo)
        return correlator_marg[:,self.mask]

    def get_chi2_marg(self, bval):
        correlator = self.get_correlator(bval)
        correlator_marg = self.get_correlator_marg(bval[:5]) # bval[:5] = [b1_1, ..., b1_5]
        chi2, bg = self.__chi2_marg(correlator, correlator_marg)
        return chi2, bg
    
    def get_chi2(self, bval, bg=None):
        correlator = self.get_correlator(bval, bg)
        chi2 = self.__chi2(correlator)
        return chi2

    def loglkl(self, cosmo, data):
        if self.with_derived_bias: data.derived_lkl = {}
        if data.need_cosmo_update is True: 
            pk, pnl, rz, dz_by_dr, Dz, Dfid, h, Omega0_m , cosmo_for_clustering = self.__set_cosmo(cosmo)
            if self.fullgg: self.clustering.compute(cosmo_for_clustering)
            self.limber.computeXi(self.kk, pk, pnl, rz, dz_by_dr, Dz, Dfid, h, Omega0_m, A=0, alpha=0)
        else: pass
        bval = np.array([data.mcmc_parameters[k]['current'] * data.mcmc_parameters[k]['scale'] for k in self.use_nuisance])
        # bval = np.loadtxt(os.path.join(self.data_directory, 'synth_bng_random.dat')) # synth truth
        if self.marg:
            chi2, bg = self.get_chi2_marg(bval)
            if self.marg_chi2: chi2 = self.get_chi2(bval, bg)
            if self.with_derived_bias:
                for i, elem in enumerate(data.get_mcmc_parameters(['derived_lkl'])):
                    data.derived_lkl[elem] = bg[i]
        else: chi2 = self.get_chi2(bval)
        prior = 0.
        if self.config["with_bbn"]: prior += -0.5 * ((data.cosmo_arguments['omega_b'] - self.config["omega_b_BBNcenter"]) / self.config["omega_b_BBNsigma"])**2
        return -0.5 * chi2 + prior
    
    def bestfit(self, bval):
        _, bg = self.get_chi2_marg(bval) # Gaussian parameters
        correlator = self.get_correlator(bval, bg)
        return self.__create_list(correlator)
    
    def __chi2(self, modelX):
        chi2 = np.dot(modelX-self.ydata, np.dot(self.invcov, modelX-self.ydata))
        return chi2
    
    def __chi2_marg(self, modelX, Pi):
        Covbi = np.dot(Pi, np.dot(self.invcov, Pi.T)) + self.__set_prior(nnlo=self.nnlo)
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = np.dot(modelX, np.dot(self.invcov, Pi.T)) - np.dot(self.invcovdata, Pi.T)
        chi2nomar = np.dot(modelX, np.dot(self.invcov, modelX)) - 2. * np.dot(self.invcovdata, modelX) + self.chi2data
        chi2mar = - np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.abs(np.linalg.det(Covbi)))
        chi2tot = chi2mar + chi2nomar # - Covbi.shape[0] * np.log(2. * np.pi)
        bg = - np.dot(Cinvbi, vectorbi) # Gaussian parameters
        return chi2tot, bg
    
    def __bias_for_limber(self, bval, bg=None): # bval: b1, c2, b3 ; bg: Gaussian parameters
        return self.__format_bias(bval, bg)
    
    def __bias_for_clustering(self, bval, bg=None):
        bgg, cgg, _, _, _ = self.__format_bias(bval, bg)
        def __bias_array_to_dict(bs, cct): return {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3], "cct": cct}
        return np.array([__bias_array_to_dict(bs, c) for bs, c in zip(bgg.T, cgg)])
    
    def __format_bias(self, bval, bg=None):
        if bg is None: 
            if self.marg: bg = np.zeros(shape=(self.limber.Nbin+self.limber.Ngg))
            else: bg = bval[2*self.limber.Ngg:] # starting after (b1, c2)_i, i=1, ..., Ngg
        bgg = np.vstack([[bval[i] for i in range(self.limber.Ngg)], 
                         [bval[i]/np.sqrt(2) for i in np.arange(self.limber.Ngg, 2*self.limber.Ngg)], 
                         [bg[i] for i in np.arange(self.limber.Nbin, self.limber.Nbin+self.limber.Ngg)], 
                         [bval[i]/np.sqrt(2) for i in np.arange(self.limber.Ngg, 2*self.limber.Ngg)]])
        cgg = np.array([bg[i] for i in np.arange(2*self.limber.Nss+self.limber.Ngs, self.limber.Nbin)])
        cgs = np.array([bg[i] for i in np.arange(2*self.limber.Nss, 2*self.limber.Nss+self.limber.Ngs)])
        cssp = np.array([bg[i] for i in range(self.limber.Nss)])
        cssm = np.array([bg[i] for i in np.arange(self.limber.Nss, 2*self.limber.Nss)])
        return bgg, cgg, cgs, cssp, cssm
    
    def __set_prior(self, nnlo=False):
        priors = 2.*np.ones(self.limber.Nbin+self.limber.Ng)
        if nnlo: priors = np.concatenate(( priors, 1.*np.ones(self.limber.Nbin) ))
        priormat = np.diagflat(1. / priors**2)
        return priormat
    
    def __create_list(self, correlator):
        if len(correlator) != 4:
            correlator = [correlator[self.indexdict['2pt_xip_startind']:self.indexdict['2pt_xip_endind']+1],
                          correlator[self.indexdict['2pt_xim_startind']:self.indexdict['2pt_xim_endind']+1],
                          correlator[self.indexdict['2pt_gammat_startind']:self.indexdict['2pt_gammat_endind']+1],
                          correlator[self.indexdict['2pt_wtheta_startind']:self.indexdict['2pt_wtheta_endind']+1],
                          ]
        
        xssp = [ [correlator[0][s] for s, ij in enumerate(self.indexbin[0]) if ij == binpair] for binpair in self.binpairs[0] ]
        xssm = [ [correlator[1][s] for s, ij in enumerate(self.indexbin[1]) if ij == binpair] for binpair in self.binpairs[1] ]
        xgs = [ [correlator[2][s] for s, ij in enumerate(self.indexbin[2]) if ij == binpair] for binpair in self.binpairs[2] ]
        xgg = [ [correlator[3][s] for s, ij in enumerate(self.indexbin[3]) if ij == binpair] for binpair in self.binpairs[3] ]
        return [xssp, xssm, xgs, xgg]
    
    def __set_cosmo(self, M):        
        # for Limber
        pk = np.array([M.pk_lin(ki, self.zfid) for ki in self.kk]) # linear P(k) in (Mpc)**3
        pnl = np.array([M.pk(ki, self.zfid) for ki in self.kk]) # halofit Pnl(k) in (Mpc)**3
        def deriv(x, func, dx=0.001): return 0.5*(func(x+dx)-func(x-dx))/dx
        def comoving_distance(z): return M.angular_distance(z)*(1+z)
        rz = np.array([comoving_distance(z) for z in self.zz])
        dr_by_dz = np.array([deriv(z, comoving_distance) for z in self.zz])
        Dz = np.array([M.scale_independent_growth_factor(zi) for zi in self.zz])
        Dfid = M.scale_independent_growth_factor(self.zfid)
        h = M.h()
        Omega0_m = M.Omega0_m()
        
        # for PyBird clustering
        cosmo_for_clustering = {}
        cosmo_for_clustering["k11"] = self.kk # k in h/Mpc
        cosmo_for_clustering["P11"] = [M.pk_lin(k * M.h(), self.zfid) * M.h()**3 for k in self.kk]  # P(k) in (Mpc/h)**3
        cosmo_for_clustering["f"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["z"]])
        cosmo_for_clustering["D"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["z"]])
        cosmo_for_clustering["Dz"] = np.array([[M.scale_independent_growth_factor(z) for z in zz] for zz in self.config["zz"]])
        cosmo_for_clustering["fz"] = np.array([[M.scale_independent_growth_factor_f(z) for z in zz] for zz in self.config["zz"]])
        def comoving_distance(z): return M.angular_distance(z) * (1 + z) * M.h()
        cosmo_for_clustering["rz"] = np.array([[comoving_distance(z) for z in zz] for zz in self.config["zz"]])
        
        return pk, pnl, rz, 1/dr_by_dz, Dz, Dfid, h, Omega0_m, cosmo_for_clustering



    
