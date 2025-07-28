from pybird.module import *
from pybird.utils import diff_all
from pybird.correlator import Correlator
from pybird.likelihood import Likelihood
from time import perf_counter as tic

class Inference():
    """A class for cosmological parameter inference using EFT of LSS.
    
    The Inference class implements parameter inference for cosmological models using
    the Effective Field Theory of Large Scale Structure. It handles likelihood calculations,
    parameter sampling, minimization, and various inference techniques.
    
    Attributes:
        cosmo_prior_covmat (ndarray): Covariance matrix for cosmological parameter priors.
        cosmo_prior_dict (dict): Dictionary of cosmological parameter priors.
        likelihood_config (dict): Configuration for the likelihood calculation.
        L (Likelihood): Likelihood instance for calculations.
        l (dict): Dictionary of free parameters and their values.
        M (object): Boltzmann solver instance (CLASS, Symbolic, or CPJ).
        need_cosmo_update (bool): Whether cosmological parameters need updating.
        T (Taylor): Taylor expansion instance if used.
        bias (ndarray): Debiasing terms if computed.
    
    Methods:
        set_nuisance(): Set nuisance parameters for the likelihood.
        set_config_and_boltzmann(): Configure cosmological parameters and Boltzmann solver.
        set_need_cosmo_update(): Determine if cosmological parameters need updating.
        get_param_name_and_pos(): Get parameter names and positions.
        init(): Initialize inference setup.
        set_sampler(): Configure parameter sampler (emcee, nuts, mclmc, etc.).
        set_minimizer(): Configure parameter minimization.
        set_debiasing(): Compute debiasing terms.
        set_model_cache(): Cache model at best-fit point.
        set_fake(): Generate fake data from best-fit model.
        set_taylor(): Set up Taylor expansion around fiducial cosmology.
        _set_taylor(): Internal method to compute Taylor expansion.
        update_boltzmann(): Update Boltzmann solver with new parameters.
        _loglkl(): Compute log-likelihood.
        _logm(): Compute log of marginalization term.
        _logp(): Compute log-posterior.
        set_logp(): Configure log-posterior calculation.
    """
    def __init__(self, free_cosmo_name, fiducial_cosmo, likelihood_config, 
        cosmo_prior_config=None, boltzmann='class', free_nuisance_name=None, fiducial_nuisance=None, verbose=True):
        
        self.cosmo_prior_config = cosmo_prior_config

        self.likelihood_config = likelihood_config
        self.L = Likelihood(likelihood_config, verbose=verbose)
        
        free_nuisance_name, fiducial_nuisance = self.set_nuisance(likelihood_config, free_nuisance_name, fiducial_nuisance)
        self.set_config_and_boltzmann(free_cosmo_name, free_nuisance_name, fiducial_cosmo, fiducial_nuisance, boltzmann=boltzmann)

        self.is_fisher = False

    def set_nuisance(self, likelihood_config, free_nuisance_name=None, fiducial_nuisance=None):
        nsky, nuisance_prior = len(likelihood_config['sky']), likelihood_config['eft_prior']
        if isinstance(fiducial_nuisance, dict): 
            print("fid nuisance", fiducial_nuisance)
            if free_nuisance_name is None: free_nuisance_name = list(fiducial_nuisance.keys())
        else: 
            if nsky == 1: # single sky
                if free_nuisance_name is None: # if which nuisance to vary is not specified, 
                    free_nuisance_name = [param for param, prior in nuisance_prior.items() if 'marg' not in prior["type"] and prior['type'] != 'unvaried'] # find which ones vary from the definition of the prior
                if fiducial_nuisance is None: # if not fiducial values for the nuisances are not specified, 
                    fiducial_nuisance = {}
                    # fiducial_nuisance.update({param: 2. for param, prior in nuisance_prior.items() if prior["type"] != 'gauss' and prior['type'] != 'unvaried' and 'marg' not in prior["type"]}) # or to a random value (here 2), if the prior is e.g., flat
                    fiducial_nuisance.update({param: prior['mean'][0] if 'mean' in prior else 2. for param, prior in nuisance_prior.items() if prior["type"] in ['gauss', 'flat']}) # set the fiducial to the mean of the Gaussian prior
            else: # multiskies
                if free_nuisance_name is None: 
                    free_nuisance_name = []
                    for i in range(nsky): # looping over skies to give indexes to the parameters name
                        free_nuisance_name.extend(['%s_%s' % (param, i+1) for param, prior in nuisance_prior.items() if 'marg' not in prior["type"] and prior['type'] != 'unvaried'])
                if fiducial_nuisance is None: 
                    fiducial_nuisance = {}
                    for i in range(nsky): # same reason here
                        # fiducial_nuisance.update({'%s_%s' % (param, i+1): 2. for param, prior in nuisance_prior.items() if prior["type"] != 'gauss' and prior['type'] != 'unvaried' and 'marg' not in prior["type"]})
                        fiducial_nuisance.update({'%s_%s' % (param, i+1): prior['mean'][i] if 'mean' in prior else 2. for param, prior in nuisance_prior.items() if prior["type"] in ['gauss', 'flat']})

        # print("fiducial nuisance", fiducial_nuisance)
        return free_nuisance_name, fiducial_nuisance
    
    def set_config_and_boltzmann(self, free_cosmo_name, free_nuisance_name, fiducial_cosmo, fiducial_nuisance, boltzmann='class'):
        self.l = {
            'free_cosmo_name': [key for key in fiducial_cosmo.keys() if key in free_cosmo_name], # sorted as in the same order in the cosmo dict
            'free_nuisance_name': free_nuisance_name, 
            'cosmo': deepcopy(fiducial_cosmo),  # Ensuring a unique copy per instance- otherwise get issues when making two instances of Inference in the same file
            'nuisance': deepcopy(fiducial_nuisance),
            'boltzmann': boltzmann,
        }

        if self.l['boltzmann'] == 'class':
            from classy import Class
            self.M = Class()
            self.M.set(self.L.class_settings)
        elif self.l['boltzmann'] == 'Symbolic':
            from pybird.symbolic import Symbolic
            self.M = Symbolic()
        elif self.l['boltzmann'] == 'CPJ':
            from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
            from pybird.integrated_model_jax import IntegratedModel
            class CosmoModule():
                def __init__(self, cosmo):
                    self.cosmo = cosmo # dictionary of cosmological parameters
            self.M = CosmoModule(self.l['cosmo'])
            self.M.CPJ = CPJ(probe='mpk_lin')
            self.M.growth = IntegratedModel(None, None, None)
            if "emu_path" in self.L.c: self.M.growth.restore(self.L.c["emu_path"] + "/growth_model_new.h5") # change path only if provided otherwise use default
        else:
            raise Exception('Boltzmann %s not recognized, please choose between class, Symbolic, or CPJ' % self.l['boltzmann'])

        self.set_need_cosmo_update()

    def set_need_cosmo_update(self): 
        self.need_cosmo_update = True
        self.l['|'] = len(self.l['free_cosmo_name'])
        if self.l['|'] == 0: # if no free cosmo, run cosmo module only once and for all
            free_nuisance = array([self.l['nuisance'][key] for key in self.l['free_nuisance_name']])
            init = self._loglkl(free_nuisance) 
            self.need_cosmo_update = False
        return 

    def get_param_name_and_pos(self, verbose=True):
        if 'maxp_pos' in self.l: 
            if verbose: print ('starting from previously found best-fit')
            name, pos = list(self.l['maxp_pos'].keys()), array(list(self.l['maxp_pos'].values())) # if finds final positions from e.g., previous minimization, use it as initial positions for the next run
        else: 
            if 'pos' in self.l: 
                name, pos = list(self.l['pos'].keys()), array(list(self.l['pos'].values())) 
            else: 
                name = self.l['free_cosmo_name'] + self.l['free_nuisance_name']
                # pos = array([val for (key, val) in self.l['cosmo'].items() if key in self.l['free_cosmo_name']] + [val for (key, val) in self.l['nuisance'].items() if key in self.l['free_nuisance_name']]) # this messes things badly as it does not put things in the right order if 'free_cosmo_name' and 'cosmo' are not in the same order
                pos = array([self.l['cosmo'][p] for p in self.l['free_cosmo_name']] + [self.l['nuisance'][p] for p in self.l['free_nuisance_name']]) 
                self.l['pos'] = {key: val for key, val in zip(name, pos)} # can be useful for another run (without tracing leakage from jitting over self.l['cosmo'] / self.l['nuisance'])
        return name, pos

    def init(self, minimize=False, cosmo_prior=False, ext_probe=False, ext_loglkl=None, jax_jit=False, measure=False, taylor_measure=False, debiasing=False, hessian_type=None, vectorize=False, taylor=False, order=3, verbose=True):
        if vectorize or measure or taylor: jax_jit = True
        if not is_jax and jax_jit: raise Exception('To jit, switch to jax-mode!')
        if measure and debiasing: raise Exception('Can\'t apply measure and debiasing at the same time, choose one!')
        if (measure or debiasing) and hessian_type is None: raise Exception('Asking \'measure\' or \'debiasing\', please choose between \'hessian_type\' = \'H\', \'F\', or \'FH\'')
        if taylor: self.set_taylor(self._get_bird_correlator, bird_correlator=True, log_measure=False, order=order, verbose=verbose)
        if self.L.marg_lkl and (minimize or measure or debiasing): self.L.c["get_maxlkl"] = True # equivalent as dropping the logdet on the nuisance subspace, however simplifying the implementation of the residual measure / debiasing on the nuisance-projected cosmo subspace
        free_param_name, initial_pos = self.get_param_name_and_pos(verbose=verbose)
        get_logp = self.set_logp(cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, jax_jit=jax_jit, measure=measure, hessian_type=hessian_type, taylor_measure=taylor_measure, vectorize=vectorize, taylor=taylor)
        if jax_jit and not vectorize: get_logp(initial_pos) # jitting except if vectorize for which the jit is performed by the sampler
        if (measure or debiasing) and hessian_type is not None: self.set_model_cache(verbose=verbose) # caching model at the mode for H-measure evaluations
        if debiasing: self.set_debiasing(hessian_type=hessian_type, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, taylor=taylor, verbose=verbose)
        if taylor_measure: 
            def _logm(params): return self._logm(params, hessian_type=hessian_type, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, taylor=taylor)
            self.set_taylor(_logm, bird_correlator=False, log_measure=True, order=1, verbose=verbose)
        return get_logp, initial_pos, free_param_name 

    def set_sampler(self, sampler='emcee', cosmo_prior=False, ext_probe=False, ext_loglkl=None, jax_jit=False, measure=False, taylor_measure=False, debiasing=False, hessian_type=None, vectorize=False, taylor=False, return_extras=False, options={}, verbose=True):
        
        if verbose: print("----- sampling with %s -----" % sampler)
        if vectorize and sampler not in ['emcee', 'zeus']: 
            if verbose: print ('warning: no vectorization in %s; switching off' % sampler)
            vectorize = False
        if taylor and not self.need_cosmo_update:
            if verbose: print ('warning: fix cosmo: taylor irrelevant; switching off')
            taylor = False
        if sampler == 'fisher': 
            if measure: 
                if verbose: print ('warning: measure irrelevant in fisher; switching off')
                measure = False
            if taylor_measure and not hasattr(self, 'T_logm'):
                if verbose: print ('warning: taylor_measure irrelevant in fisher; switching off')
                taylor_measure = False
            if taylor and self.l['boltzmann'] in ['Symbolic', 'CPJ'] and not hasattr(self, 'T'): # no need to Taylor expand for Fisher, so let's switch off - except if Taylor already exists (the results will be anyway the same, this is only for timing consideration)
                if verbose: print ('warning: %s Boltzmann solver is JAX-differentiable --- switching off Taylor' % self.l['boltzmann'])
                taylor = False
            if not taylor and self.l['boltzmann'] in ['class']: # if Boltzmann is not differentiable, we need to switch to Taylor expansion to perform Fisher
                if verbose: print ('warning: %s Boltzmann solver not JAX-differentiable --- switching on Taylor' % self.l['boltzmann'])
                taylor = True
            if debiasing: 
                if verbose: print ('warning: debiasing irrelevant in fisher; switching off')
                debiasing = False

        get_logp, _initial_pos, free_param_name = self.init(minimize=False, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, jax_jit=jax_jit, measure=measure, taylor_measure=taylor_measure, debiasing=debiasing, hessian_type=hessian_type, vectorize=vectorize, taylor=taylor, verbose=verbose)
        n = len(free_param_name)

        if sampler == 'fisher':
            from numpy.random import multivariate_normal
            from numpy.linalg import inv
            def _get_p(initial_pos=None, size=1000*n, verbose=verbose):
                if initial_pos is None: initial_pos = _initial_pos
                F = -hessian(get_logp)(initial_pos) 
                self.is_fisher, self.C_fisher = True, inv(F)
                samples = multivariate_normal(initial_pos, self.C_fisher, size=size)
                extras = {'Fisher': F}
                return samples, extras

        elif sampler == 'emcee':
            import emcee
            from numpy.random import randn, multivariate_normal
            if verbose and vectorize: print("emcee: vectorized")
            def _get_p(initial_pos=None, num_samples=4000*n, discard=1000*n//4, thin=20*n, n_walkers=4*n, verbose=verbose):
                if initial_pos is None: initial_pos = _initial_pos
                if self.is_fisher: 
                    if verbose: print ('Fisher matrix found: drawing initial conditions from multivariate normal')
                    self.pos = multivariate_normal(initial_pos, self.C_fisher, size=n_walkers)
                else: self.pos = initial_pos + 1e-4 * array(randn(n_walkers, len(initial_pos)))
                self.nwalkers, self.ndim = self.pos.shape
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, get_logp, vectorize=vectorize)
                sampler.run_mcmc(self.pos, num_samples, progress = verbose)
                extras = {'emcee_sampler': sampler}
                try: 
                    tau = array(sampler.get_autocorr_time())
                    if verbose: 
                        with printoptions(precision=0): print('autocorr time: ', tau)
                    extras['tau'] = tau 
                except: 
                    if verbose: print ('autocorr might be too short, beware...')
                flat_samples = sampler.get_chain(discard = discard, thin = thin, flat = True)
                
                return flat_samples, extras

        elif sampler == 'zeus':
            import zeus
            from numpy.random import randn, multivariate_normal
            if verbose and vectorize: print("zeus: vectorized")
            def _get_p(initial_pos=None, num_samples=4000*n, discard=1000*n//4, thin=20*n, n_walkers=4*n, verbose=verbose):
                if initial_pos is None: initial_pos = _initial_pos
                if self.is_fisher: 
                    if verbose: print ('Fisher matrix found: drawing initial conditions from multivariate normal')
                    self.pos = multivariate_normal(initial_pos, self.C_fisher, size=n_walkers)
                else: self.pos = initial_pos + 1e-4 * array(randn(n_walkers, len(initial_pos)))
                self.nwalkers, self.ndim = self.pos.shape
                sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, get_logp, verbose=verbose, vectorize=vectorize)
                sampler.run_mcmc(self.pos, num_samples)
                chain = sampler.get_chain(discard = discard, thin = thin, flat = True)
                extras = {'zeus_sampler': sampler}
                return chain, extras

        # elif sampler == 'nuts':
        #     import numpyro
        #     import numpyro.distributions as dist
        #     from numpyro.infer import NUTS, init_to_value
        #     from numpyro.infer import MCMC as nuts_MCMC
        #     from jax import random 
        #     def _get_p(initial_pos=None, num_samples=1000*n, num_warmup=10*n, prior_dict=None, verbose=verbose):
        #         if initial_pos is None: initial_pos = _initial_pos
        #         param_dict = {}
        #         for i, name in enumerate(free_param_name):
        #             param_dict[name] = initial_pos[i]

        #         def model():
        #             if prior_dict is None:
        #                 print("Warning: using default prior of [-5, 5] for all parameters")
        #                 params = [numpyro.sample(name, dist.Uniform(-5, 5)) for name in free_param_name]
        #             else:
        #                 params = [numpyro.sample(name, dist.Uniform(prior_dict[name][0], prior_dict[name][1])) for name in free_param_name]
        #             likelihood_value = get_logp(params)
        #             numpyro.factor("likelihood", likelihood_value)

        #         nuts_kernel = NUTS(model, init_strategy = init_to_value(values = param_dict))
        #         # chain_method = 'vectorized' if vectorize else 'parallel'
        #         mcmc = nuts_MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, progress_bar=verbose)# , chain_method=chain_method)
        #         mcmc.run(random.PRNGKey(0))
        #         samples = mcmc.get_samples()
        #         samples_combined = vstack([samples[key] for key in free_param_name]).T
        #         return samples_combined

        elif sampler == 'nuts':
            # from https://blackjax-devs.github.io/blackjax/examples/quickstart.html#nuts
            import jax, blackjax
            from datetime import date
            def _get_p(
                initial_pos=None,
                num_samples=2500,  # post-warm-up draws
                num_warmup=None,             # warm-up steps (defaults to 10 % of draws)
                target_accept=0.8,
                max_num_doublings=10,
                verbose=verbose,
            ):
                """Run a single-chain BlackJAX NUTS sampler (JIT-compiled).

                Parameters
                ----------
                initial_pos : array-like
                    Starting point of the Markov chain.
                num_samples : int
                    Number of *post-warm-up* samples to draw.
                num_warmup : int | None
                    Number of warm-up (adaptation) steps.  Defaults to 10 % of ``num_samples``.
                target_accept : float
                    Dual-averaging target acceptance probability.
                max_num_doublings : int
                    Maximum tree depth used by NUTS *after* warm-up (controls max leapfrog steps).
                verbose : bool
                    Print basic diagnostics if ``True``.
                """

                if initial_pos is None:
                    initial_pos = _initial_pos

                if num_warmup is None:
                    num_warmup = num_samples // 10

                # ── RNG setup ──────────────────────────────────────────────
                rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
                rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

                # ── Warm-up / adaptation ──────────────────────────────────
                toc = tic() 
                warmup = blackjax.window_adaptation(
                    blackjax.nuts,
                    get_logp,
                    target_acceptance_rate=target_accept,
                    max_num_doublings=max_num_doublings,
                )

                (state, parameters), _ = warmup.run(
                    warmup_key,
                    initial_pos,
                    num_steps=num_warmup,
                )

                # include the user-selected tree depth *only* in the final kernel
                kernel = blackjax.nuts(
                    get_logp,
                    **parameters,
                ).step
                walltime_warmup = tic() - toc
                if verbose: print ('warmup done in %.2s sec.' % walltime_warmup)

                # ── JIT-compiled sampling loop ────────────────────────────
                @jax.jit
                def inference_loop(key, init_state):
                    def one_step(carry, k):
                        new_state, info = kernel(k, carry)
                        return new_state, (
                            new_state.position,
                            info.acceptance_rate,
                            info.is_divergent,
                        )
                    keys = jax.random.split(key, num_samples)
                    _, (positions, accepts, divergences) = jax.lax.scan(
                        one_step, init_state, keys
                    )
                    return positions, accepts, divergences

                toc = tic()
                positions, accepts, divergences = inference_loop(sample_key, state)
                walltime_sampling = tic() - toc

                if verbose:
                    acc = jax.numpy.mean(accepts).item()
                    div = jax.numpy.mean(divergences).item()
                    print(
                        f"avg accept rate: {acc:0.2f}, frac divergent: {div:0.3f}"
                    )
                    print ('sampling done in %.2s sec.' % walltime_sampling)

                extras = {'walltime_warmup': walltime_warmup, 'walltime_sampling': walltime_sampling, 'acceptance_rate': acc, 'divergences': div}
                return positions, extras

        elif sampler == 'mclmc':
            import jax, blackjax
            from datetime import date
            
            def _get_p(
                initial_pos=None, 
                num_samples=25000*n, 
                num_adaptation=10000*n,
                num_chains=1,
                target_acceptance_rate=0.65,
                verbose=verbose
            ):
                if initial_pos is None: 
                    initial_pos = _initial_pos
                
                rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
                rng_key, init_key, tune_key, sample_key = jax.random.split(rng_key, 4)
                
                if verbose:
                    print(f"MCLMC: Running adaptation for {num_adaptation} steps...")
                
                try:
                    initial_state = blackjax.mcmc.mclmc.init(
                        position=initial_pos, 
                        logdensity_fn=get_logp, 
                        rng_key=init_key
                    )
                    kernel = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
                        logdensity_fn=get_logp,
                        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                        inverse_mass_matrix=sqrt_diag_cov,
                    )
                    
                    # Find L and step_size using the adaptation function
                    adaptation_result = blackjax.mclmc_find_L_and_step_size(
                        mclmc_kernel=kernel,
                        num_steps=num_adaptation,
                        state=initial_state,
                        rng_key=tune_key,
                        diagonal_preconditioning=False,
                        desired_energy_var=5e-4
                    )
                    
                    # Handle different return structures from different BlackJAX versions
                    if verbose:
                        print(f"MCLMC: Adaptation result has {len(adaptation_result)} elements")
                    
                    if len(adaptation_result) == 2:
                        adapted_state, mclmc_params = adaptation_result
                    elif len(adaptation_result) == 3:
                        adapted_state, mclmc_params, _ = adaptation_result
                    else:
                        # If we get more than 3 values, take the first two
                        adapted_state, mclmc_params = adaptation_result[0], adaptation_result[1]
                    
                    L_adapted = mclmc_params.L
                    step_size_adapted = mclmc_params.step_size
                    
                    if verbose:
                        print(f"MCLMC: Adaptation successful...")
                
                except Exception as e:
                    if verbose:
                        print(f"MCLMC: Adaptation failed: {e}")
                        print("Try upgrading BlackJAX to 1.2.5 or later")
                    raise e
                
                # Create the final MCLMC kernel with adapted parameters
                final_mclmc = blackjax.mclmc(
                    get_logp,
                    L=L_adapted,
                    step_size=step_size_adapted,
                )
                
                if verbose:
                    print(f"MCLMC: Running sampling for {num_samples} steps...")
                
                # Generate keys outside of JIT-compiled function
                keys = jax.random.split(sample_key, num_samples)
                
                @jax.jit
                def sampling_loop(keys, initial_state):
                    def one_step(state, rng_key):
                        new_state, info = final_mclmc.step(rng_key, state)
                        return new_state, (new_state.position, info)
                    
                    final_state, (positions, infos) = jax.lax.scan(
                        one_step, initial_state, keys
                    )
                    return positions, infos
                
                positions, infos = sampling_loop(keys, adapted_state)
                
                if verbose:
                    print(f"MCLMC: Sampling completed...")
                
                return positions, _

        elif sampler == 'nautilus':
            from nautilus import Prior, Sampler
            from random import choices
            def _get_p(initial_pos=None, max_calls=None, prior_dict=None, verbose=verbose):
                if initial_pos is None: initial_pos = _initial_pos
                prior_flat = Prior()
                for name in free_param_name:
                    if prior_dict is not None:
                        prior_flat.add_parameter(name, dist=prior_dict[name])
                    else:
                        print("Warning: using default prior of [-5, 5] for all parameters - note for Nautilus this is not recommended and will likely lead to NaNs")
                        prior_flat.add_parameter(name, dist=(-5,5))

                def loglkl_wrapper(param_dict):
                    param_pos = [param_dict[key] for key in free_param_name]
                    return get_logp(param_pos)
            
                sampler = Sampler(prior_flat, loglkl_wrapper)
                if max_calls:
                    sampler.run(verbose=verbose, n_like_max = max_calls)
                else:
                    sampler.run(verbose=verbose)
                samples, log_w, log_l = sampler.posterior()

                random_idx = choices(range(len(log_w)), weights=exp(log_w), k=500*n)
                samples = samples[random_idx]
                
                return samples, sampler
        else:
            raise Exception('sampler %s not recognized, please choose between emcee, nuts, mclmc, fisher or nautilus' % sampler)

        def get_p(**kwargs):
            samples, extras = _get_p(**kwargs, **options)
            if debiasing: samples = samples + self.bias
            if is_jax: clear_caches() # useful if we want to do consecutive runs
            if return_extras: return samples, free_param_name, extras
            else: return samples, free_param_name

        self.get_p = get_p

        return

    def set_minimizer(self, minimizer='', cosmo_prior=False, ext_probe=False, ext_loglkl=None, jax_jit=False, taylor=False, options={},  order=3, verbose=True):
        # currently we are not jitting the minimizer itself. could eventually also consider that. 
        
        if verbose: print("----- minimisation with %s -----" % minimizer)
        if taylor and not self.need_cosmo_update:
            if verbose: print ('warning: fix cosmo: taylor irrelevant; switching off')
            taylor = False

        get_logp, _initial_pos, free_param_name = self.init(minimize=True, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, jax_jit=jax_jit, taylor=taylor, order=order, verbose=verbose)
        n = len(free_param_name)

        def chi2(params): return -2. * get_logp(params)

        if minimizer == 'onestep': # simply return the global chi2 on the provided initial position
            print ('... just computing the chi2 on initial position')
            def _get_maxp(initial_pos=None): 
                if initial_pos is None: initial_pos = _initial_pos
                return chi2(initial_pos), initial_pos
        
        if minimizer == 'scipy.optimize':
            if is_jax: from jax.scipy.optimize import minimize
            else: from scipy.optimize import minimize
            def _get_maxp(initial_pos=None): 
                if initial_pos is None: initial_pos = _initial_pos
                m = minimize(chi2, initial_pos, method="BFGS", tol=1e-5)
                return m.fun, m.x

        if minimizer == 'minuit':
            from iminuit import minimize
            def _get_maxp(initial_pos=None): 
                if initial_pos is None: initial_pos = _initial_pos
                m = minimize(chi2, initial_pos)
                return m['fun'], m['x']

        if minimizer == 'adam':
            if not is_jax: raise Exception('To use optax, switch to jax-mode!')
            import optax
            def _get_maxp(initial_pos=None, steps=1000*(n+10), lr=1e-3, tolerance=1e-4, patience=10*(n+10)):
                if initial_pos is None: initial_pos = _initial_pos
                opt = optax.adam(lr)
                params = initial_pos
                opt_state = opt.init(params)
                prev_chi2 = chi2(params)
                no_improvement_steps = 0
                for step in range(steps):
                    grads = grad(chi2)(params)
                    updates, opt_state = opt.update(grads, opt_state)
                    params = optax.apply_updates(params, updates)

                    current_chi2 = chi2(params)
                    chi2_change = abs(prev_chi2 - current_chi2)  

                    if chi2_change < tolerance:
                        no_improvement_steps += 1
                        if no_improvement_steps >= patience:
                            print(f"Stopping at step {step} due to chi2 plateauing.")
                            break
                    else:
                        no_improvement_steps = 0  # Reset if improvement is seen

                    prev_chi2 = current_chi2

                return current_chi2, params
        
        def get_maxp(initial_pos=None):
            chi2, maxp = _get_maxp(initial_pos=initial_pos)
            self.l['maxp_pos'] = {key: val for key, val in zip(free_param_name, maxp)} # useful to store for a next run 
            if is_jax: clear_caches() # if we want to do consecutive runs
            return chi2, maxp, free_param_name
        
        self.get_maxp = get_maxp

        return 
    
    def set_debiasing(self, hessian_type='H', cosmo_prior=False, ext_probe=False, ext_loglkl=None, taylor=False, verbose=True):
        if verbose: print ('Computing debiasing...')
        if 'maxp_pos' not in self.l: raise Exception('No best-fit values found to compute debiasing. Please first perform a minimization. ')
        _, p = self.get_param_name_and_pos()
        # if 'H' in hessian_type: self.set_model_cache(verbose=verbose)

        def _H(params): 
            return - hessian(self._logp)(params, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, taylor=taylor, hessian_type=hessian_type) 
        def _logdetH(params):
            H = _H(params)
            sign, logdetH = slogdet(H)
            return sign * logdetH

        toc = tic() 
        self.bias = 0.5 * einsum('ar,r->a', linalg.inv(_H(p)), jacfwd(_logdetH)(p)) ### This is just first order the taylor expansion of log(sqrt(det H))
        elapse_time = tic() - toc  
        if verbose: print ('Debiasing computed in %.3f sec.' % (elapse_time))
        return 

    def set_model_cache(self, verbose=True):
        if verbose: print ('Caching model best-fit')
        if 'maxp_pos' not in self.l: raise Exception('No best-fit values found to compute H-measure. Please first perform a minimization. ')
        self.L.c['cache'] = True
        get_logp = self.set_logp(jax_jit=False, measure=False, vectorize=False, taylor=False, verbose=False) # important to reset the likelihood in no-jax to avoid tracing issue
        _ = get_logp(array(list(self.l['maxp_pos'].values()))) # setting Likelihood on the mode
        self.L.set_model_cache() # storing the theory model evaluated on the mode for potential future usage
        self.L.c['cache'] = False # turning off in future calls of self.L
        return 

    def set_fake(self, sample_fake=False, options={}, verbose=True):
        if verbose: print ('Generating noiseless fake data from model best-fit')
        if 'maxp_pos' not in self.l: raise Exception('No best-fit values found to generate fake data. Please first perform a minimization. ')
        if self.L.marg_lkl: self.L.c["get_maxlkl"] = True
        self.L.c['write']['fake'] = True 
        get_logp = self.set_logp(jax_jit=False, measure=False, vectorize=False, taylor=False, verbose=False) # important to reset the likelihood in no-jax to avoid tracing issue
        _ = get_logp(array(list(self.l['maxp_pos'].values()))) # setting Likelihood on the mode
        self.L.write() # writing to file the fake data
        self.L.c['write']['fake'] = False # turning off in future calls of self.L
        if sample_fake:
            if verbose: print ('Future sampling set on fake data from best-fit.')
            self.likelihood_config['data_file'] = 'fake_%s.h5' % self.L.c['write']['out_name'] # replacing original data to fake data 
            self.likelihood_config['data_path'] = self.L.c['write']['out_path']
            self.L = Likelihood(self.likelihood_config, verbose=verbose) # resetting likelihood
            if not self.need_cosmo_update: self.set_need_cosmo_update() # when not varying the cosmology, computing at least once the cosmology-dependent pieces 
        return

    def set_taylor(self, f, bird_correlator=True, log_measure=False, order=3, verbose=True): 
        to_set = False
        if bird_correlator: T_engine = 'T'
        elif log_measure: T_engine = 'T_logm' 
        if hasattr(self, T_engine):
            # fiducial_cosmo = array([val for (key, val) in self.l['maxp_pos'].items() if key in self.l['free_cosmo_name']])
            if bird_correlator: 
                x0 = array([self.l['maxp_pos'][key] for key in self.l['free_cosmo_name']])
                if verbose and array_equal(self.T.x0, x0): print ('Taylor expansion found, with no update in fiducials: using existing.' )
            elif log_measure: 
                x0 = array(list(self.l['maxp_pos'].values()))
                if verbose and array_equal(self.T_logm.x0, x0): print ('Taylor expansion found, with no update in fiducials: using existing.' )
            else: to_set = True
        else: to_set = True
        if to_set: self._set_taylor(f, bird_correlator=bird_correlator, log_measure=log_measure, order=order, verbose=verbose)
        return 

    def _set_taylor(self, f, bird_correlator=True, log_measure=False, order=3, verbose=True): # order 3 ok, order 2 mainly for testing (= Fisher)
        if verbose: 
            if bird_correlator: words = 'cosmology-dependent correlator pieces'
            elif log_measure: words = 'log-measure'
            else: words = ''
            print ('Taylor: expanding %s...' % words)
        if 'maxp_pos' not in self.l: 
            # raise Exception('No best-fit values found to Taylor-expand around. Please first perform a minimization. ')
            if verbose: print ("Taylor: no best-fit values found --- using provided fiducial values instead")
            # fiducial_cosmo = array([val for (key, val) in self.l['cosmo'].items() if key in self.l['free_cosmo_name']])
            self.l['maxp_pos'] = {key: val for (key, val) in self.l['cosmo'].items() if key in self.l['free_cosmo_name']}
            self.l['maxp_pos'].update({key: val for (key, val) in self.l['nuisance'].items() if key in self.l['free_nuisance_name']})
        else: 
            if verbose: print ("Taylor: best-fit values found --- using them as fiducials")
        # fiducial_cosmo = array([val for (key, val) in self.l['maxp_pos'].items() if key in self.l['free_cosmo_name']])
        if bird_correlator: x0 = array([self.l['maxp_pos'][key] for key in self.l['free_cosmo_name']])
        elif log_measure: x0 = array(list(self.l['maxp_pos'].values()))

        toc = tic() 
        if bird_correlator: _, size_per_sky = f(x0, return_size_per_sky=True)
        if self.l['boltzmann'] in ['Symbolic', 'CPJ']: # JAX-differentiable Boltzmann code
            def make_jacfwd_chain(f, max_order):
                chain = [f]
                for _ in range(max_order): chain.append(jacfwd(chain[-1]))
                return chain
            def compute_taylor_terms_jax(f, x, max_order):
                assert max_order >= 0, "Order must be ≥ 0"
                chain = make_jacfwd_chain(f, max_order) # PZ: this could be cached to be re-used for another x
                return [fn(x) for fn in chain]
            fn = compute_taylor_terms_jax(f, x0, order)
        else: # Using finite difference
            fn = diff_all(f, x0, max_order=order) # , epsilon=array([3., 4., 5.])*1e-3) #2.e-2 * fiducial_cosmo) # stable stepsize
        elapse_time = tic() - toc   # ... timing ends
        if verbose: print ('Taylor: derivatives up to order %s computed in %.3f sec.' % (order, elapse_time))

        class Taylor():
            def __init__(self, x0, fn):
                
                self.x0, self.fn = x0, fn
                self.order = len(fn)-1
                
                self.op1 = einsum_path('a,...a->...', x0, fn[1], optimize='optimal')[0]
                if self.order >= 2: self.op2 = einsum_path('a,b,...ab->...', x0, x0, fn[2], optimize='optimal')[0]
                if self.order >= 3: self.op3 = einsum_path('a,b,c,...abc->...', x0, x0, x0, fn[3], optimize='optimal')[0]
                if self.order >= 4: self.op4 = einsum_path('a,b,c,d,...abcd->...', x0, x0, x0, x0, fn[4], optimize='optimal')[0]

            def compute(self, x): 
                da = x - self.x0
                if self.order == 1: 
                    f0, f1 = self.fn
                    f = f0 + einsum('a,...a->...', da, f1, optimize=self.op1) 
                elif self.order == 2: 
                    f0, f1, f2 = self.fn
                    f = f0 + einsum('a,...a->...', da, f1, optimize=self.op1) + .5 * einsum('a,b,...ab->...', da, da, f2, optimize=self.op2) 
                elif self.order == 3: 
                    f0, f1, f2, f3 = self.fn
                    f = f0 + einsum('a,...a->...', da, f1, optimize=self.op1) + .5 * einsum('a,b,...ab->...', da, da, f2, optimize=self.op2) + 1/6. * einsum('a,b,c,...abc->...', da, da, da, f3, optimize=self.op3)
                elif self.order == 4:
                    f0, f1, f2, f3, f4 = self.fn
                    f = f0 + einsum('a,...a->...', da, f1, optimize=self.op1) + .5 * einsum('a,b,...ab->...', da, da, f2, optimize=self.op2) + 1/6. * einsum('a,b,c,...abc->...', da, da, da, f3, optimize=self.op3) + 1/24. * einsum('a,b,c,d,...abcd->...', da, da, da, da, f4, optimize=self.op4)    
                return f

        class TaylorBird(Taylor):
            def __init__(self, x0, fn, size_per_sky):
                super().__init__(x0, fn)
                self.size_per_sky = size_per_sky

            def compute(self, x):
                self.bird_1D = super().compute(x)
                return

            def set(self, L):
                idx = 0
                for i, s in enumerate(self.size_per_sky): 
                    L.correlator_sky[i].bird.unravel(self.bird_1D[idx:idx + s])
                    idx += s
                return

        if bird_correlator: self.T = TaylorBird(x0, fn, size_per_sky)
        elif log_measure: self.T_logm = Taylor(x0, fn)
        return

    def _get_bird_correlator(self, free_cosmo, return_size_per_sky=False):
        self.l['cosmo'].update({key: val for key, val in zip(self.l['free_cosmo_name'], free_cosmo)}) 
        self.update_boltzmann(self.l['cosmo'])
        size_per_sky, bird_1D_sky = [], []
        for i in range(self.L.nsky): 
            self.L.correlator_sky[i].compute(cosmo_engine=self.M, cosmo_module=self.l['boltzmann'])
            bird_1D_sky.append(self.L.correlator_sky[i].bird.concatenate())
            size_per_sky.append(len(bird_1D_sky[i]))
        if return_size_per_sky: return concatenate((bird_1D_sky)), size_per_sky # the concatanated birds to Taylor expand, size of each bird_1D per sky to unravel them later on
        else: return concatenate((bird_1D_sky)) # the concatanated birds to Taylor expand

    def update_boltzmann(self, cosmo):
        if self.l['boltzmann'] == 'class': 
            if 'w0_fld' in cosmo.keys():
                if 'Omega_Lambda' not in cosmo.keys(): cosmo['Omega_Lambda'] = 0
            self.M.set(cosmo)
            self.M.compute()
        elif self.l['boltzmann'] == 'Symbolic':
            self.M.set(cosmo)
        elif self.l['boltzmann'] == 'CPJ': 
            self.M.cosmo = cosmo 
        else: 
            pass
        return

    def _loglkl(self, params, taylor=False, hessian_type=None):
        free_cosmo, free_nuisance = params[:self.l['|']], params[self.l['|']:] # arrays of values ordered as keys
        if taylor: 
            self.T.compute(array(free_cosmo))
            loglkl = self.L.loglkl(free_nuisance, self.l['free_nuisance_name'], cosmo_engine=self.T, 
                need_cosmo_update=self.need_cosmo_update, cosmo_dict=None, cosmo_module='taylor', hessian_type=hessian_type)
        else: 
            if self.need_cosmo_update: 
                self.l['cosmo'].update({key: val for key, val in zip(self.l['free_cosmo_name'], free_cosmo)}) 
                self.update_boltzmann(self.l['cosmo'])
            loglkl = self.L.loglkl(free_nuisance, self.l['free_nuisance_name'], cosmo_engine=self.M, 
                need_cosmo_update=self.need_cosmo_update, cosmo_dict=None, cosmo_module=self.l['boltzmann'], hessian_type=hessian_type)
        return self._safe(loglkl)

    def _safe(self, loglkl):
        # Use vectorised masking; works for both JAX & NumPy thanks to symbols
        # brought into scope by `pybird.module`.
        return where(isfinite(loglkl), loglkl, -inf)

    def _logpr(self, params):
        free_cosmo = params[:self.l['|']]
        delta_cosmo = array([free_cosmo[self.l['free_cosmo_name'].index(k)] - mu for k, mu in zip(self.cosmo_prior_config['name'], self.cosmo_prior_config['mean'])])
        return -.5 * einsum('a,ab,b->', delta_cosmo, array(self.cosmo_prior_config['inv_mat']), delta_cosmo)

    def _logm(self, params, hessian_type='H', taylor_measure=False, cosmo_prior=False, ext_probe=False, ext_loglkl=None, taylor=False):
        if taylor_measure: 
            return self.T_logm.compute(params)
        else: 
            H = - hessian(self._logp)(params, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, taylor=taylor, hessian_type=hessian_type) 
            sign, logdetH = slogdet(H)
            return .5 * sign * logdetH

    def _logp(self, params, cosmo_prior=False, ext_probe=False, ext_loglkl=None, taylor=False, hessian_type=None):
        loglkl = self._loglkl(params, taylor=taylor, hessian_type=hessian_type)
        if cosmo_prior: loglkl += self._logpr(params)
        if ext_probe: loglkl += ext_loglkl(params)
        return self._safe(loglkl)

    def _logpm(self, params, cosmo_prior=False, ext_probe=False, ext_loglkl=None, measure=False, taylor_measure=False, hessian_type=None, taylor=False): 
        loglkl = self._logp(params, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, taylor=taylor)
        if measure: loglkl += self._logm(params, hessian_type=hessian_type, taylor_measure=taylor_measure, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, taylor=taylor)
        return self._safe(loglkl)

    def set_logp(self, cosmo_prior=False, ext_probe=False, ext_loglkl=None, jax_jit=False, measure=False, taylor_measure=False, hessian_type=None, vectorize=False, taylor=False, verbose=True):   
            
        # cosmo_prior, taylor, measure?
        def _logp_0(a): return self._logpm(a, cosmo_prior=cosmo_prior, ext_probe=ext_probe, ext_loglkl=ext_loglkl, measure=measure, taylor_measure=taylor_measure, hessian_type=hessian_type, taylor=taylor)

        # vectorize?
        def _logp_1():
            if vectorize: return vmap(_logp_0, in_axes=(0))
            else: return _logp_0

        # jit?
        get_logp = jit(_logp_1()) if jax_jit else _logp_1()
        return get_logp


