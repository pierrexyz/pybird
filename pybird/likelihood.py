from pybird.module import *
from pybird.correlator import Correlator
from pybird.io_pb import ReadWrite
import numpy as np
import time

class Likelihood(object):
    """EFT Likelihood for cosmological parameter inference.
    
    This class implements the likelihood calculation for the Effective Field Theory
    of Large Scale Structure, handling multiple sky patches with correlated priors.
    It supports both marginalized and non-marginalized approaches for EFT nuisance
    parameters, and can incorporate BAO reconstruction, loop priors, and various
    optimization techniques.
    
    Attributes:
        c (dict): Master configuration dictionary.
        io (ReadWrite): I/O handler for reading and writing data.
        c_sky (list): List of configuration dictionaries for each sky patch.
        d_sky (list): List of data dictionaries for each sky patch.
        nsky (int): Number of sky patches.
        m_sky (list): List of data masks for each sky patch.
        y_sky (list): List of data vectors for each sky patch.
        p_sky (list): List of precision matrices for each sky patch.
        y_all (ndarray): Concatenated data vector across all sky patches.
        p_all (ndarray): Block-diagonal precision matrix combining all sky patches.
        
        correlator_sky (list): List of PyBird correlator engines for each sky patch.
        alpha_sky (list): List of BAO reconstruction alphas if enabled.
        out (list): List of output dictionaries for each sky patch.
        Tng_k_cache (list): Cache for model power spectra per sky patch.
        
        b_name (list): Names of all EFT parameters.
        bg_name (list): Names of Gaussian-marginalized EFT parameters.
        bg_prior_mean (ndarray): Prior means for marginalized parameters.
        bg_prior_sigma (ndarray): Prior widths for marginalized parameters.
        bng_name (list): Names of non-marginalized Gaussian EFT parameters.
        bng_prior_mean (ndarray): Prior means for non-marginalized Gaussian parameters.
        bng_prior_sigma (ndarray): Prior widths for non-marginalized Gaussian parameters.
        bnlog_name (list): Names of log-normal EFT parameters.
        bnlog_prior_mean (ndarray): Prior means for log-normal parameters.
        bnlog_prior_sigma (ndarray): Prior widths for log-normal parameters.
        
        Ng (int): Number of marginalized parameters.
        marg_lkl (bool): Whether to use marginalized likelihood.
        F2_bg_prior_matrix (ndarray): Prior precision matrix for marginalized parameters.
        F1_bg_prior_mean (ndarray): Prior mean vector for marginalized parameters.
        chi2_bg_prior_mean (float): Prior chi-squared contribution.
        prior_inv_corr_matrix (ndarray): Inverse correlation matrix for correlated skies.
        
        class_settings (dict): Settings for Boltzmann solver.
        kin (ndarray): k-values for Boltzmann solver input.
        
        optipath_chi2 (list): Optimized einsum path for non-marginalized chi-squared.
        optipath_F2 (list): Optimized einsum path for F2 matrix in marginalization.
        optipath_F1 (list): Optimized einsum path for F1 vector in marginalization.
        optipath_bg (list): Optimized einsum path for marginalized parameter calculation.
    
    Methods:
        set_data(): Set up data vectors and precision matrices from input data.
        set_config(): Set configuration options with defaults.
        set_class_settings(): Configure CLASS or other Boltzmann solver settings.
        set_eft_parameters(): Set up EFT parameter priors and correlations.
        set_boost(): Set optimized einsum paths for likelihood calculations.
        set_model_cache(): Cache model power spectra for re-use or diagnostics.
        
        get_chi2_marg(): Compute chi-squared with analytical marginalization over linear parameters.
        get_chi2_non_marg(): Compute standard non-marginalized chi-squared.
        get_chi2_for_hessian(): Compute chi-squared specifically for Hessian calculation.
        get_prior(): Compute log-prior for non-marginalized EFT parameters.
        get_alpha_bao_rec(): Calculate alpha parameters for BAO reconstruction.
        set_bao_rec(): Add BAO reconstruction alphas to theory vector.
        get_loop(): Calculate loop corrections and create appropriate precision matrix.
        
        loglkl(): Main method to compute log-likelihood for given parameters.
        get_eft_parameters(): Retrieve current values of all EFT parameters.
        set_out(): Store output results for writing.
        write(): Write results to output files using io handler.
    """
    def __init__(self, config, verbose=True):
        self.c = config
        self.io = ReadWrite()
        _, self.c_sky, self.d_sky = self.io.read(self.c, verbose=verbose) # skylists of formatted config dict and data dict
        self.nsky = len(self.c['sky'])
        self.set_model_cache()
        self.set_data()
        self.set_config(verbose=verbose)
        self.set_class_settings()
        self.correlator_sky = [Correlator(self.c_sky[i]) for i in range(self.nsky)] # skylist of PyBird correlator engine
        self.set_eft_parameters()
        self.set_boost()
        if self.c["with_bao_rec"]: self.alpha_sky = [None] * self.nsky # skylist of bao recon alpha 

    def set_data(self):
        self.m_sky = [self.d_sky[i]['mask'] for i in range(self.nsky)] # mask for theory model
        self.y_sky = [self.d_sky[i]['y'] for i in range(self.nsky)]    # data to fit per sky
        self.p_sky = [self.d_sky[i]['p'] for i in range(self.nsky)]    # precision matrix per sky
        self.y_all = concatenate(self.y_sky)
        self.p_all = block_diag(*self.p_sky) 
        self.out = [{} for i in range(self.nsky)]

    def set_config(self, verbose=True):
        options = ['get_maxlkl', 'with_boss_correlated_skies_prior', 'with_rs_marg', 'drop_logdet', 'cache']
        if verbose: print ('-----------------------')
        for keys in options:
            if not keys in self.c: self.c[keys] = False
            if verbose: print (keys, ':', self.c[keys])
        if verbose: print ('-----------------------')

    def set_class_settings(self):
        log10kmax = 0
        #if self.c["with_nnlo_counterterm"] or self.c["with_rs_marg"]: log10kmax = 1 # slower, but needed for the wiggle-no-wiggle split
        self.class_settings = {'output': 'mPk', 'z_max_pk': max(array([self.d_sky[i]["z"] for i in range(self.nsky)])), 'P_k_max_h/Mpc': 10.**log10kmax}
        self.kin = logspace(-5, log10kmax, 500) 
        return

    def set_eft_parameters(self):
        self.b_name = [param for param in self.c["eft_prior"]]
        self.bg_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'marg_gauss']
        self.bg_prior_mean = array([self.c["eft_prior"][param]["mean"] for param in self.bg_name])[...,:self.nsky] 
        self.bg_prior_sigma = array([self.c["eft_prior"][param]["range"] for param in self.bg_name])[...,:self.nsky] 
        self.bng_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'gauss']
        self.bng_prior_mean = array([self.c["eft_prior"][param]["mean"] for param in self.bng_name])[...,:self.nsky]
        self.bng_prior_sigma = array([self.c["eft_prior"][param]["range"] for param in self.bng_name])[...,:self.nsky]
        self.bnlog_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'lognormal']
        self.bnlog_prior_mean = array([self.c["eft_prior"][param]["mean"] for param in self.bnlog_name])[...,:self.nsky]
        self.bnlog_prior_sigma = array([self.c["eft_prior"][param]["range"] for param in self.bnlog_name])[...,:self.nsky]

        # if self.c["read_gauss_prior_mean_from_file"]: 
        #     self.bg_centers = []
        #     for i in range(self.nsky):
        #         with open(filename) as f: data_file = f.read()
        #         eft_params_str = data_file.split(', \n')[1].replace("# ", "")
        #         eft_truth = {key: float(value) for key, value in (pair.split(': ') for pair in eft_params_str.split(', '))}
        #     self.bg_prior_mean = array(self.bg_centers).T

        # if self.c["fix_to_truth"]: self.bg_prior_sigma *= 1.e-6

        self.Ng = len(self.bg_name)
        if self.Ng > 0:
            self.marg_lkl = True
        else: self.marg_lkl = False

        if self.marg_lkl:
            if self.c['with_boss_correlated_skies_prior']: # BOSS skies i,j = {1, 2, 3, 4} = {CMASS NGC, CMASS SGC, LOWZ NGC, LOWZ SGC}
                self.F2_bg_prior_matrix = linalg.inv(get_corr(N=self.Ng)) # prior inverse covariance matrix for marginalization
                self.prior_inv_corr_matrix = linalg.inv(get_corr(N=1)) # inverse correlation matrix for non-marg EFT parameters : b1, c2, c4
            else:
                self.F2_bg_prior_matrix = eye(self.Ng*self.nsky)
                self.prior_inv_corr_matrix = eye(self.nsky)

            self.F2_bg_prior_matrix /= concatenate(self.bg_prior_sigma.T)**2

            bg_prior_mean = concatenate(self.bg_prior_mean.T)
            self.F1_bg_prior_mean = einsum('a,ab->b', bg_prior_mean, self.F2_bg_prior_matrix) 
            self.chi2_bg_prior_mean = einsum('a,b,ab->', bg_prior_mean, bg_prior_mean, self.F2_bg_prior_matrix)
        else:
            if self.c['with_boss_correlated_skies_prior']: # BOSS skies i,j = {1, 2, 3, 4} = {CMASS NGC, CMASS SGC, LOWZ NGC, LOWZ SGC}
                self.prior_inv_corr_matrix = linalg.inv(get_corr(N=1)) # inverse correlation matrix for non-marg EFT parameters
            else:
                self.prior_inv_corr_matrix = eye(self.nsky)

        return

    def set_boost(self):
        self.optipath_chi2 = einsum_path('a,b,ab->', self.y_all, self.y_all, self.p_all, optimize='optimal')[0]
        if self.marg_lkl:
            dummy_ak = zeros(shape=(self.Ng, self.y_all.shape[0]))
            self.optipath_F2 = einsum_path('ak,bp,kp->ab', dummy_ak, dummy_ak, self.p_all, optimize='optimal')[0]
            self.optipath_F1 = einsum_path('ak,p,kp->a', dummy_ak, self.y_all, self.p_all, optimize='optimal')[0]
            self.optipath_bg = einsum_path('a,ab->b', self.y_all, self.p_all, optimize='optimal')[0] 
        return

    # def get_chi2_marg(self, Tng_k, Tg_bk, P):
    #     """Marginalized chi2"""
    #     F2 = einsum('ak,bp,kp->ab', Tg_bk, Tg_bk, P, optimize=self.optipath_F2)
    #     F1 = einsum('ak,p,kp->a', Tg_bk, Tng_k, P, optimize=self.optipath_F1)
    #     F0 = self.get_chi2_non_marg(Tng_k, P)

    #     F1 -= self.F1_bg_prior_mean
    #     F2 += self.F2_bg_prior_matrix

    #     invF2 = linalg.inv(F2)
    #     chi2 = F0 - einsum('a,b,ab->', F1, F1, invF2) 
    #     if not self.c['drop_logdet']: chi2 += linalg.slogdet(F2)[1]
    #     chi2 += self.chi2_bg_prior_mean
    #     bg = - einsum('a,ab->b', F1, invF2, optimize=self.optipath_bg)

    #     return chi2, bg

    def get_chi2_marg(self, Tng_k, Tg_bk, P): 

        def get_F(Tng_k, Tg_bk, P):
            F2 = einsum('ak,bp,kp->ab', Tg_bk, Tg_bk, P, optimize=self.optipath_F2)
            F1 = einsum('ak,p,kp->a', Tg_bk, Tng_k, P, optimize=self.optipath_F1)
            F0 = self.get_chi2_non_marg(Tng_k, P) 
            return F2, F1, F0

        for i, (Tng_i, Tg_i, P_i) in enumerate(zip(Tng_k, Tg_bk, P)):
            F2_i, F1_i, F0_i = get_F(Tng_i, Tg_i, P_i)
            if i == 0: F2 = F2_i; F1 = F1_i; F0 = F0_i
            else: F2 += F2_i; F1 += F1_i; F0 += F0_i

        F1 -= self.F1_bg_prior_mean
        F2 += self.F2_bg_prior_matrix

        invF2 = linalg.inv(F2)
        chi2 = F0 - einsum('a,b,ab->', F1, F1, invF2) 
        if not self.c['drop_logdet']: chi2 += linalg.slogdet(F2)[1]
        chi2 += self.chi2_bg_prior_mean
        bg = - einsum('a,ab->b', F1, invF2, optimize=self.optipath_bg)
        
        return chi2, bg

    def get_chi2_non_marg(self, T_k, P, T_k_2=None):
        """Standard non-marginalized chi2"""
        if T_k_2 is None: chi2 = einsum('k,p,kp->', T_k, T_k, P, optimize=self.optipath_chi2)
        else: chi2 = einsum('k,p,kp->', T_k, T_k_2, P, optimize=self.optipath_chi2)
        return chi2

    def get_chi2_for_hessian(self, T_k, P, T_k_2=None, hessian_type='H'):
        if not is_jax: raise Exception('No support for hessian without jax')
        chi2 = 0.
        if 'F' in hessian_type: chi2 += self.get_chi2_non_marg(T_k, P) - 2.*self.get_chi2_non_marg(T_k, P, stop_gradient(T_k))  # taking the Hessian gives purely F = Fisher, with F^{-1}.grad(logdetF) on the best fit being the Jeffreys bias correction
        if 'H' in hessian_type: 
            if T_k_2 is None: T_k_2 = T_k # T_k_2 is supposed to be the posterior mode, previously found and saved; if not, 'H' will be equal to 'F'
            chi2 += self.get_chi2_non_marg(T_k-stop_gradient(T_k_2), P)  # = 0, but taking the Hessian gives H = Fisher on the best fit, with H^{-1}.grad(logdetH) being the H bias correction
        if hessian_type == 'FH': chi2 /= 2. # that's the relevant measure for noiseless synthetic data
        return chi2 

    def get_prior(self, b_sky):
        """Prior"""
        def _get_prior(bs, prior_mean, prior_sigma, prior_inv_corr_mat=None, prior_type='gauss'):
            if prior_type == 'gauss':
                prior = - 0.5 * einsum( 'n,nm,m->', bs - prior_mean, prior_sigma**-2 * prior_inv_corr_mat, bs - prior_mean )
            elif prior_type == 'lognormal':
                # if any(b <= 0. for b in bs): prior = - 0.5 * inf
                # else: prior = - 0.5 * einsum( 'n,nm,m->', log(bs) - prior_mean, prior_sigma**-2 * prior_inv_corr_mat, log(bs) - prior_mean ) #- np.sum( np.log(np.einsum('n,nm->m', bs, np.linalg.inv(prior_inv_corr_mat))), axis=0 ) #- np.sum(np.log(bs), axis=0)
                prior = - 0.5 * nan_to_num(einsum( 'n,nm,m->', log(bs) - prior_mean, prior_sigma**-2 * prior_inv_corr_mat, log(bs) - prior_mean ) - sum(log(bs), axis=0), nan=inf)
            elif prior_type == 'single_gauss': 
                prior = - 0.5 * (bs - prior_mean)**2 * prior_sigma**-2
            return prior
        prior = 0.
        for i, param in enumerate(self.bng_name): prior += _get_prior(array([b_sky[j][param] for j in range(self.nsky)]), self.bng_prior_mean[i], self.bng_prior_sigma[i], self.prior_inv_corr_matrix, prior_type='gauss')
        for i, param in enumerate(self.bnlog_name): prior += _get_prior(array([b_sky[j][param] for j in range(self.nsky)]), self.bnlog_prior_mean[i], self.bnlog_prior_sigma[i], self.prior_inv_corr_matrix, prior_type='lognormal')
        if self.marg_lkl and self.c["get_maxlkl"]: 
            for i, param in enumerate(self.bg_name): 
                prior += _get_prior(array([b_sky[j][param] for j in range(self.nsky)]), self.bg_prior_mean[i], self.bg_prior_sigma[i], self.prior_inv_corr_matrix, prior_type='gauss')
        return prior

    def get_alpha_bao_rec(self, class_engine, i_sky=0):
        rd_by_rdfid = class_engine.rs_drag() / self.d_sky[i_sky]['bao_rec_fid']['rd']
        DM_by_DMfid = class_engine.angular_distance(self.d_sky[i_sky]['z']) / self.d_sky[i_sky]['bao_rec_fid']['D']
        H_by_Hfid = class_engine.Hubble(self.d_sky[i_sky]['z']) * c_light*1e-3 / self.d_sky[i_sky]['bao_rec_fid']['H']
        alpha_par = 1. / (rd_by_rdfid * H_by_Hfid)
        alpha_per = DM_by_DMfid / rd_by_rdfid
        return array([alpha_par, alpha_per])

    def set_bao_rec(self, alphas, Tng_k, Tg_bk=None):
        Tng_k = concatenate((Tng_k, alphas))
        if Tg_bk is not None: Tg_bk = pad(Tg_bk, [(0, 0), (0, 2)], mode='constant', constant_values=0)
        return Tng_k, Tg_bk

    def get_loop(self, b_sky, sky='sky', i_sky=0, marg=False): 
        Tng1_k = self.correlator_sky[i_sky].get(b_sky[i_sky], what="1loop").reshape(-1)[self.m_sky[i_sky]] # here we put c_{meas} = c_{EFT} + c_{sys} (in principle we should put c_EFT but as long as we are conservative on c_{sys,max} in the denominator, we can drop c_{sys} in the numerator)
        # b_fid = b_sky[i_sky].copy(); b_fid['b1'] = 2.
        # Tng0_k = self.correlator_sky[i_sky].get(b_fid, what="linear").reshape(-1)[self.m_sky[i_sky]] # this is just P11
        Tng0_k = self.correlator_sky[i_sky].get(b_sky[i_sky], what="linear")
        if is_jax: Tng0_k = Tng0_k.at[0].set(self.c['sky'][sky]['c_sys_max'] / self.c_sky[i_sky]["nd"])
        else: Tng0_k[0] += self.c['sky'][sky]['c_sys_max'] / self.c_sky[i_sky]["nd"] # adding c_{sys,max} , an estimate of the upper bound on systematic effects that are degenerate with k^0
        Tng0_k = Tng0_k.reshape(-1)[self.m_sky[i_sky]] 
        p0_kk = Tng0_k.shape[0]**-1 * diag(Tng0_k**-2) # we divide the chi2 (so we divide the precision matrix) by the number of bins
        if not marg: return Tng1_k, p0_kk
        else:
            Tg1_bk = self.correlator_sky[i_sky].getmarg(b_sky[i_sky], self.bg_name)[:, self.m_sky[i_sky]] # so for now, all marg parameters can only multiply one-loop term, so we don't need to ask anything special with respect to the standard case
            return Tng1_k, p0_kk, Tg1_bk
    
    def loglkl(self, free_b, free_b_name, cosmo_engine=None, cosmo_module='class', need_cosmo_update=True, cosmo_dict=None, hessian_type=None):
        """Compute the log-likelihood for a given set of EFT parameters.
        
        This method is the core function that calculates the log-likelihood given a set of
        EFT parameters while handling cosmological dependencies. It reconstructs theory predictions,
        calculates chi-squared values (marginalized or non-marginalized), and computes
        prior contributions.
        
        Args:
            free_b (ndarray): Array of free EFT parameters, potentially including rs_marg parameter
                at the beginning followed by parameters for each sky patch.
            free_b_name (list): List of names corresponding to free_b parameters.
            cosmo_engine (object, optional): Boltzmann solver instance (CLASS, CPJ, or Symbolic).
                Defaults to None.
            cosmo_module (str, optional): Type of cosmology calculation ('class', 'taylor', etc.).
                Defaults to 'class'.
            need_cosmo_update (bool, optional): Whether to update cosmological quantities.
                Defaults to True.
            cosmo_dict (dict, optional): Dictionary of precomputed cosmological quantities.
                Defaults to None.
            hessian_type (str, optional): Type of Hessian calculation ('H', 'F', or 'FH'). 
                Defaults to None.
            
        Returns:
            float: Log-likelihood value (including priors).
            
        Notes:
            - The free_b array is reshaped into per-sky parameters after extracting any global
              parameters like alpha_rs if with_rs_marg is True.
            - For marginalized likelihood, parameters specified in bg_name are analytically
              marginalized over using Gaussian priors.
            - When get_maxlkl is True, the marginalized parameters are calculated and stored.
            - BAO reconstruction alphas are included if with_bao_rec is True.
            - Loop corrections are included if with_loop_prior is True.
        """
        pad = 0
        if self.c["with_rs_marg"]:
            alpha_rs_marg = free_b[free_b_name.index("alpha_rs")]
            pad += 1

        free_b_sky = array(free_b[pad:]).reshape(self.nsky, -1)
        free_b_name_sky = np.array(free_b_name[pad:]).reshape(self.nsky, -1)

        self.b_sky = [] # skylist of EFT parameters dict

        for i in range(self.nsky):
            self.b_sky.append({bn: free_b_sky[i][n] for n, fbn in enumerate(free_b_name_sky[i]) for bn in self.b_name if fbn.split('_', 1)[0] == bn})
            self.b_sky[i].update({bn: 0. for bn in self.b_name if bn not in self.b_sky[i]})

        if need_cosmo_update:
            if cosmo_module == 'taylor' and cosmo_engine is not None: 
                cosmo_engine.set(self) # here cosmo_module is the Taylor expansion of the cosmology-dependent pieces in PyBird
            else:
                for i in range(self.nsky):
                    cosmo_dict_i = cosmo_dict[i] if cosmo_dict is not None else None
                    self.correlator_sky[i].compute(cosmo_dict=cosmo_dict_i, cosmo_engine=cosmo_engine, cosmo_module=cosmo_module) 
                    if self.c["with_bao_rec"]: self.alpha_sky[i] = self.get_alpha_bao_rec(cosmo_engine, i_sky=i)

        if self.marg_lkl:
            if True: 
                Tng_k, Tg_bk = [], []
                for i in range(self.nsky):
                    Tng_k.append( self.correlator_sky[i].get(self.b_sky[i]).reshape(-1)[self.m_sky[i]] )
                    Tg_bk.append( self.correlator_sky[i].getmarg(self.b_sky[i], self.bg_name)[:, self.m_sky[i]] )
                    if self.c["with_bao_rec"]: Tng_k[i], Tg_bk[i] = self.set_bao_rec(self.alpha_sky[i], Tng_k[i], Tg_bk[i])
                _Tng_k, _Tg_bk, _p = [concatenate(Tng_k)-self.y_all], [block_diag(*Tg_bk)], [self.p_all]
            if self.c["with_loop_prior"]: 
                Tng1_k, Tg1_bk, p0 = [], [], []
                for i, sky in enumerate(self.c['sky']):
                    Tng1_k_i, p0_i, Tg1_bk_i = self.get_loop(self.b_sky, sky=sky, i_sky=i, marg=True) # 1loop, bin-weighted diagonal precision matrix of inverse linear squared 
                    Tng1_k.append(Tng1_k_i); p0.append(p0_i); Tg1_bk.append(Tg1_bk_i)
                _Tng_k.append(concatenate(Tng1_k)); _Tg_bk.append(block_diag(*Tg1_bk)); _p.append(block_diag(*p0))
            chi2, bg = self.get_chi2_marg(_Tng_k, _Tg_bk, _p) 
            if self.c["get_maxlkl"]:
                bg_resh = bg.reshape(self.nsky,-1)
                for i in range(self.nsky): 
                    self.b_sky[i].update({p: b for p, b in zip(self.bg_name, bg_resh[i])})

        if not self.marg_lkl or self.c["get_maxlkl"]:
            chi2 = 0.
            for i, sky in enumerate(self.c['sky']):
                Tng_k_i = self.correlator_sky[i].get(self.b_sky[i]).reshape(-1)[self.m_sky[i]] 
                if self.c["with_bao_rec"]: Tng_k_i, _ = self.set_bao_rec(self.alpha_sky[i], Tng_k_i, None)
                if hessian_type is not None: chi2_i = self.get_chi2_for_hessian(Tng_k_i, self.p_sky[i], T_k_2=self.Tng_k_cache[i], hessian_type=hessian_type)
                else:
                    chi2_i = self.get_chi2_non_marg(Tng_k_i-self.y_sky[i], self.p_sky[i])
                if self.c["cache"] or self.c["write"]["fake"] or self.c["write"]["save"] or self.c['write']['plot']: self.set_out(self.correlator_sky[i].get(self.b_sky[i]), chi2_i, self.b_sky[i], i_sky=i)
                chi2 += chi2_i
                if self.c["with_loop_prior"]:
                    T1_k, p0 = self.get_loop(self.b_sky, sky=sky, i_sky=i) # 1loop, bin-weighted diagonal precision matrix of inverse linear squared 
                    chi2 += self.get_chi2_non_marg(T1_k, p0)

        prior = self.get_prior(self.b_sky) 
        lkl = - 0.5 * chi2 + prior
        return lkl

    def get_eft_parameters(self): # all values eft parameters including the marginalised ones (such that the likelihood is maximised)
        eft_param_key = list(self.b_sky[0].keys())
        eft_param_val = array([list(b_sky.values()) for b_sky in self.b_sky]).T.tolist()
        eft_param_dict = {key: val for key, val in zip(eft_param_key, eft_param_val)}
        return eft_param_dict

    def set_model_cache(self):
        self.Tng_k_cache = [self.out[i]['y'] for i in range(self.nsky)] if hasattr(self, 'out') else self.nsky * [None]
        return

    def set_out(self, y_arr, chi2, eft_parameters, M=None, i_sky=0):
        xmask = self.d_sky[i_sky]['mask_arr']
        self.out[i_sky]['y_arr'] = [y_arr[i, xmask_i] for i, xmask_i in enumerate(xmask)]
        self.out[i_sky]['y'] = y_arr.reshape(-1)[self.m_sky[i_sky]]
        self.out[i_sky]['x_unmasked'] = self.d_sky[i_sky]['x']
        self.out[i_sky]['y_arr_unmasked'] = y_arr
        if self.c["with_bao_rec"]: self.out[i_sky]['alpha'] = self.alpha_sky[i_sky]
        self.out[i_sky]['chi2'] = chi2
        self.out[i_sky]['eft_parameters'] = eft_parameters
        if M is not None: # class engine
            self.out[i_sky]['cosmo'] = {'omega_b': M.omega_b(), 'omega_cdm': M.Omega0_cdm() * M.h()**2, 'Omega_k': M.Omega0_k(), 'Omega_nu': M.Omega_nu} 
            self.out[i_sky]['cosmo'].update(M.get_current_derived_parameters(["Omega_m", "h", "A_s", "n_s", "sigma8"]))

    def write(self):
        self.io.write(self.c, self.d_sky, self.out)

# correlation matrix for BOSS four skies
def get_corr(N=1, eps_12=0.1, eps_13=0.2): # correlation rho_ij = 1 - eps_ij^2 / 2, eps_12 = 10% diff between NGC / SGC | eps_13 = 20% diff between CMASS / LOWZ
    rho_12, rho_13 = diag(N * [1 - 0.5 * eps_12**2]), diag(N * [1 - 0.5 * eps_13**2])
    corr = block([   [eye(N), rho_12, rho_13, rho_12 * rho_13], 
                        [rho_12, eye(N), rho_12 * rho_13, rho_13],
                        [rho_13, rho_12 * rho_13, eye(N), rho_12], 
                        [rho_12 * rho_13, rho_13, rho_12, eye(N)]    ])
    return corr
