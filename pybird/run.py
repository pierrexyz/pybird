import os, sys
from numpy import savetxt, printoptions, ndarray
from time import perf_counter as tic
from scipy import stats
import h5py
from pybird.io_pb import save_dict_to_hdf5
def pvalue(minchi2, dof): return 1. - stats.chi2.cdf(minchi2, dof)

class Run():
    """A class for running cosmological parameter inference with PyBird.
    
    The Run class provides a high-level interface for cosmological parameter inference,
    handling both minimization and MCMC sampling. It can be initialized either with
    paths to YAML configuration files or directly with configuration dictionaries.
    
    The class manages:
    - Cosmological parameter setup and priors
    - EFT nuisance parameters
    - Multiple minimization algorithms
    - Various MCMC samplers
    - Output file generation and saving
    - Model debiasing and measurement
    - Taylor expansion around fiducial cosmology
    - JAX acceleration if enabled
    
    Attributes:
        path_to_output (str): Directory path for saving output files.
        likelihood_config (dict): Configuration for likelihood calculation.
        c (dict): Configuration dictionary with run settings including:
            free_cosmo_name (list): Names of varying cosmological parameters.
            fiducial_cosmo (dict): Fiducial cosmological parameters.
            measure (bool): Whether to compute Fisher matrices.
            debiasing (bool): Whether to compute debiasing terms.
            hessian_type (str): Type of Hessian to compute.
            vectorize (bool): Whether to vectorize calculations.
            taylor (bool): Whether to use Taylor expansion.
            jax_jit (bool): Whether to use JAX JIT compilation.
            order (int): Order of Taylor expansion.
        
        kwargs (dict): Additional keyword arguments including:
            boltzmann (str): Choice of Boltzmann solver.
            free_nuisance_name (list): Names of varying nuisance parameters.
            fiducial_nuisance (dict): Fiducial nuisance parameters.
        
        I (Inference): Inference instance for parameter estimation.
    
    Methods:
        run(minimizers=None, samplers=None, initial_pos=None, samplers_options=None,
            set_fake=False, sample_fake=False, output=True, save_to_file=False,
            hash_file='runs_output', verbose=True): 
            Run parameter inference with specified minimizers and/or samplers.
            
        set_header(free_param_name, elapse_time): 
            Create header for output files with run information.
    
    Examples:
        # Initialize with YAML config files
        run = Run('path/to/run.yaml', 'path/to/likelihood.yaml', 'path/to/output')
        
        # Initialize with dictionaries
        run = Run(run_config_dict, likelihood_config_dict, 'path/to/output')
        
        # Run minimization and sampling
        results = run.run(minimizers=['minuit'], samplers=['emcee'])
    """
    def __init__(self, *args, verbose=True):

        if isinstance(args[0], str): # if providing paths to config yaml file
            path_to_run_config, path_to_likelihood_config, self.path_to_output = args[:3]

            if not os.path.exists(path_to_output): raise Exception("path to output directory: %s not found" % path_to_output)
            self.path_to_output = path_to_output # to save output files: samples, best-fits, etc.

            def read_config(path_to_config):
                if not os.path.isfile(path_to_config): raise Exception("path to config file: %s not found" % path_to_config)
                return yaml.full_load(open(path_to_config, 'r'))
            run_config, self.likelihood_config = read_config(path_to_run_config), read_config(path_to_likelihood_config)

        elif isinstance(args[0], dict): # if providing directly the config dicts
            run_config, self.likelihood_config, self.path_to_output = args[:3]

        self.c = {}
        def set_args(option, default):
            if option not in run_config: self.c[option] = default
            else: self.c[option] = run_config[option]
            if verbose: print ('%s: %s' % (option, self.c[option]))

        set_args('free_cosmo_name', ['omega_cdm', 'h', 'ln10^10A_s'])
        set_args('fiducial_cosmo', {'omega_b': 0.02235, 'omega_cdm': 0.120, 'h': 0.675, 'ln10^{10}A_s': 3.044, 'n_s': 0.965})
        set_args('cosmo_prior', False)
        set_args('ext_probe', False)
        set_args('ext_loglkl', None)
        set_args('measure', False)
        set_args('taylor_measure', False)
        set_args('debiasing', False)
        set_args('hessian_type', None)
        set_args('vectorize', False)
        set_args('emulate', None)
        set_args('taylor', False)
        set_args('jax_jit', False)
        set_args('order', 3)

        self.kwargs = {}
        for option in ['boltzmann', 'free_nuisance_name', 'fiducial_nuisance', 'cosmo_prior_config']:
            if option in run_config:
                self.kwargs[option] = run_config[option]
                if verbose: print ('%s: %s' % (option, self.kwargs[option]))

        if self.c['cosmo_prior'] and 'cosmo_prior_config' not in self.kwargs: raise Exception('cosmo_prior: provide a dict \'cosmo_prior_config\'')

        # setting jax?
        if self.c['vectorize'] or self.c['taylor']: 
            print ('\'vectorize\' or \'taylor\' is True, setting \'jax_jit\' to True')
            self.c['jax_jit'] = True
        if self.c['jax_jit']:
            from pybird.config import set_jax_enabled
            set_jax_enabled(True)

        from pybird.inference import Inference
        self.I = Inference(self.c['free_cosmo_name'], self.c['fiducial_cosmo'], self.likelihood_config, **self.kwargs, verbose=verbose)

    def run(self, minimizers=None, samplers=None, initial_pos=None, samplers_options=None, return_extras=False, plot_bestfit=False, set_fake=False, sample_fake=False, output=True, save_to_file=False, hash_file='runs_output', verbose=True):
        outdict = {}
        if minimizers is not None: 
            if type(minimizers) == str: minimizers = [minimizers]
            for minimizer in minimizers: 
                self.I.set_minimizer(minimizer=minimizer, cosmo_prior=self.c['cosmo_prior'], ext_probe=self.c['ext_probe'], ext_loglkl=self.c['ext_loglkl'], jax_jit=self.c['jax_jit'], emulate=self.c['emulate'], taylor=self.c['taylor'], order=self.c['order'], verbose=verbose)
                if verbose: print('minimisation starts...')
                toc = tic()                 # timing starts...
                chi2, bestfit, free_param_name = self.I.get_maxp(initial_pos=initial_pos)
                elapse_time = tic() - toc   # ... timing ends
                if verbose: print ('minimisation done in %.3f sec.' % elapse_time)
                ndata = self.I.L.y_all.shape[0]
                dof = ndata - len(free_param_name) - self.I.L.Ng # number of data points - free parameters (cosmo + non-marg nuisance) - marg nuisance
                pval = pvalue(chi2, dof)
                if verbose: 
                    print ('min chi2: %.3f, ndata: %.0f, dof: %.0f, p-value: %.3f' % (chi2, ndata, dof, pval))
                    with printoptions(precision=3): print ('bestfit %s: %s' % (free_param_name, bestfit))
                # if save_to_file:
                #     filename = os.path.join(self.path_to_output, 'bestfit_from-%s_%s.txt') % (minimizer, hash_file)
                #     header = "min chi2: %.3f, ndata: %.0f, dof: %.0f, p-value: %.3f \n" % (chi2, ndata, dof, pval)
                #     for key in free_param_name: header += "%s, " % key
                #     savetxt(filename, bestfit, header=header, fmt='%.5e')
                #     if verbose: print ("best-fit saved at %s" % filename)
                outdict[minimizer] = {'min chi2': chi2, 'ndata': ndata, 'dof': dof, 'p-value': pval, 'bestfit': bestfit, 'free parameters': free_param_name, 'elapse time (sec.)': elapse_time}
                initial_pos = bestfit # useful to start the sampling at a good place (although in principle also done internally in Inference())
            if self.c['measure'] or self.c['debiasing']: self.I.set_model_cache(verbose=verbose)
            if plot_bestfit:
                self.I.plot_bestfit(verbose=verbose)
            if set_fake: 
                self.I.set_fake(sample_fake=sample_fake, verbose=verbose)
                outdict[minimizer]['eft parameters'] = self.I.L.get_eft_parameters()
        if samplers is not None: 
            if type(samplers) == str: samplers = [samplers]
            if samplers_options is None: samplers_options = len(samplers) * [{}]
            if type(samplers_options) == dict: samplers_options = [samplers_options]
            for sampler, options in zip(samplers, samplers_options):
                self.I.set_sampler(sampler=sampler, cosmo_prior=self.c['cosmo_prior'], ext_probe=self.c['ext_probe'], ext_loglkl=self.c['ext_loglkl'], measure=self.c['measure'], taylor_measure=self.c['taylor_measure'], debiasing=self.c['debiasing'], hessian_type=self.c['hessian_type'], jax_jit=self.c['jax_jit'], vectorize=self.c['vectorize'], emulate=self.c['emulate'], taylor=self.c['taylor'], return_extras=return_extras, options=options, verbose=verbose)
                if verbose: print('sampling starts...')
                toc = tic()                 # timing starts...
                if return_extras: samples, free_param_name, extras = self.I.get_p(initial_pos=initial_pos, verbose=verbose)
                else: samples, free_param_name = self.I.get_p(initial_pos=initial_pos, verbose=verbose)
                elapse_time = tic() - toc   # ... timing ends
                if verbose: print ('sampling done in %.3f sec.' % elapse_time)
                # if save_to_file: 
                #     filename = os.path.join(self.path_to_output, 'samples_from-%s_%s.txt') % (sampler, hash_file)
                #     header = self.set_header(free_param_name, elapse_time)
                #     savetxt(filename, samples, header=header)
                #     if verbose: print ("samples saved at %s" % filename)
                outdict[sampler] = {'samples': samples, 'free parameters': free_param_name, 'elapse time (sec.)': elapse_time}
                if return_extras: outdict[sampler]['extras'] = extras
        if save_to_file: 
            with h5py.File(os.path.join(self.path_to_output, '%s.h5') % hash_file, 'w') as hf: save_dict_to_hdf5(hf, outdict)
        if verbose: print ('-' * 32)
        if output: return outdict
        else: return

    def set_header(self, free_param_name, elapse_time): 
        header = "Samples from %s, generated with PyBird (jax_jit: %s), Boltzmann: %s, sampler: %s (vectorize: %s), in %.2f sec. \n" % (
            self.likelihood_config['data_file'], self.c['jax_jit'], self.kwargs['boltzmann'], self.kwargs['sampler'], self.kwargs['vectorize'], elapse_time)
        header = 'fixed cosmology: '
        for key, value in self.c['fiducial_cosmo'].items(): 
            if key not in self.c['free_cosmo_name']: header += "%s: %.5e, " % (key, value)
        header += "\n"
        # header = 'fixed nuisance: '
        # for key, value in self.c['fiducial_nuisance'].items(): 
        #     if key not in self.L.l['free_nuisance_name']: header += "%s: %.4e, " % (key, value)
        # header += "\n"
        header = 'varying parameters: '
        for key in free_param_name: header += '%s, ' % key
        return header



