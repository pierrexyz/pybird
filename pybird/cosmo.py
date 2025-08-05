from pybird.module import * 
import numpy as np
import jax


class Cosmo():
    """A Class to compute the linear power spectrum and growth factors from cosmological parameters.
    
    This class provides an interface to various linear cosmology computation engines,
    including CLASS, CosmoPower emulator, and symbolic solvers. It handles
    the calculation of linear matter power spectra, scale-independent growth factors and rates,
    and other cosmological quantities needed for perturbation theory calculations.
    
    Attributes:
        c (dict): Configuration dictionary holding parameters for calculations.
        
    Methods:
        set_cosmo(): Compute linear cosmological quantities from input parameters using 
            a specified backend module (class, CPJ, or Symbolic). Returns a dictionary containing 
            computed quantities like the linear matter power spectrum, growth factors and rates,
            and parameters for AP effect.
    """ 
    def __init__(self, config):
        self.c = config

    def _is(self, module, module_name): 
        return module.casefold() == module_name.casefold()

    def set_cosmo(self, cosmo_dict, module='class', engine=None):
        
        # Handle None cosmo_dict
        if cosmo_dict is None:
            cosmo_dict = {}

        # checking redshift
        if self.c["z"] is None:
            if "z" not in cosmo_dict: raise Exception("Please provide a 'z' in cosmo_dict or when better, when providing options to correlator() through .set({\'z\': z, ...})")
            else:
                self.c["z"] = cosmo_dict["z"]
                cosmo_dict.pop("z")
        else: 
            if "z" in cosmo_dict: 
                if cosmo_dict["z"] != self.c["z"]: raise Exception("The provided z in cosmo_dict is different than the one set in correlator?")
                else: cosmo_dict.pop("z")

        cosmo = {}

        log10kmax = 0.
        #if self.c["with_nnlo_counterterm"]: log10kmax = 1 # slower, but required for the wiggle-no-wiggle split scheme
        cosmo["kk"] = logspace(-5, log10kmax, 512)  # k in h/Mpc

        if self._is(module, 'class') or self._is(module, 'classy'):

            if not engine:
                from classy import Class
                cosmo_dict_local = cosmo_dict.copy()
                if self.c["with_bias"] and "bias" in cosmo_dict: del cosmo_dict_local["bias"] # remove to not pass it to classy that otherwise complains
                if not self.c["with_time"] and "A" in cosmo_dict: del cosmo_dict_local["A"] # same as above
                if self.c["with_redshift_bin"]: zmax = max(self.c["redshift_bin_zz"])
                else: 
                    zmax = self.c["z"]
                M = Class()
                M.set(cosmo_dict_local)
                M.set({'output': 'mPk', 'P_k_max_h/Mpc': 10.**log10kmax, 'z_max_pk': zmax, })
                #     'tol_perturbations_integration': 1.e-6, 'tol_background_integration': 1.e-5, 'k_per_decade_for_pk': 200, 'k_per_decade_for_bao': 200})
                M.compute()
            else: M = engine

            cosmo["pk_lin"] = array([M.pk_lin(k*M.h(), self.c["z"])*M.h()**3 for k in cosmo["kk"]]) # P(k) in (Mpc/h)**3

            if self.c["multipole"] > 0: 
                cosmo["f"] = M.scale_independent_growth_factor_f(self.c["z"])
            if not self.c["with_time"]:
                cosmo["D"] = M.scale_independent_growth_factor(self.c["z"])
            if self.c["with_nonequal_time"]:
                cosmo["D1"] = M.scale_independent_growth_factor(self.c["z1"])
                cosmo["D2"] = M.scale_independent_growth_factor(self.c["z2"])
                cosmo["f1"] = M.scale_independent_growth_factor_f(self.c["z1"])
                cosmo["f2"] = M.scale_independent_growth_factor_f(self.c["z2"])
            if self.c["with_exact_time"] or self.c["with_quintessence"]:
                cosmo["z"] = self.c["z"]
                cosmo["Omega0_m"] = M.Omega0_m()
            # if "w0_fld" in cosmo_dict:
            #     cosmo["w0_fld"] = cosmo_dict["w0_fld"]
            if self.c["with_ap"]:
                cosmo["H"], cosmo["DA"] = M.Hubble(self.c["z"]) / M.Hubble(0.), M.angular_distance(self.c["z"]) * M.Hubble(0.)

            if self.c["with_redshift_bin"]:
                def comoving_distance(z): return M.angular_distance(z) * (1+z) * M.h()
                cosmo["Dz"] = array([M.scale_independent_growth_factor(z) for z in self.c["redshift_bin_zz"]])
                cosmo["fz"] = array([M.scale_independent_growth_factor_f(z) for z in self.c["redshift_bin_zz"]])
                cosmo["rz"] = array([comoving_distance(z) for z in self.c["redshift_bin_zz"]])

            if self.c["with_quintessence"]:
                # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum.
                # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
                # This does not work for 'with_redshift_bin': True. # eventually to code up
                zm = 5. # z in matter domination
                def scale_factor(z): return 1/(1.+z)
                Omega0_m = cosmo["Omega0_m"]
                w = cosmo["w0_fld"]
                GF = GreenFunction(Omega0_m, w=w, quintessence=True)
                Dq = GF.D(scale_factor(zfid)) / GF.D(scale_factor(zm))
                Dm = M.scale_independent_growth_factor(self.c["z"]) / M.scale_independent_growth_factor(zm)
                cosmo["pk_lin"] *= Dq**2 / Dm**2 * ( 1 + (1+w)/(1.-3*w) * (1-Omega0_m)/Omega0_m * (1+zm)**(3*w) )**2 # 1611.07966 eq. (4.15)
                cosmo["f"] = GF.fplus(1/(1.+self.c["z"]))

            # wiggle-no-wiggle split # algo: 1003.3999; details: 2004.10607
            def get_smooth_wiggle_resc(kk, pk, alpha_rs=1.): # k [h/Mpc], pk [(Mpc/h)**3]
                kp = linspace(1.e-7, 7, 2**16)   # 1/Mpc
                ilogpk = interp1d(log(kk * M.h()), log(pk / M.h()**3), fill_value="extrapolate") # Mpc**3
                lnkpk = log(kp) + ilogpk(log(kp))
                harmonics = dst(lnkpk, type=2, norm='ortho')
                odd, even = harmonics[::2], harmonics[1::2]
                nn = arange(0, odd.shape[0], 1)
                nobao = delete(nn, arange(120, 240,1))
                smooth_odd = interp1d(nn, odd, kind='cubic')(nobao)
                smooth_even = interp1d(nn, even, kind='cubic')(nobao)
                smooth_odd = interp1d(nobao, smooth_odd, kind='cubic')(nn)
                smooth_even = interp1d(nobao, smooth_even, kind='cubic')(nn)
                smooth_harmonics =  array([[o, e] for (o, e) in zip(smooth_odd, smooth_even)]).reshape(-1)
                smooth_lnkpk = dst(smooth_harmonics, type=3, norm='ortho')
                smooth_pk = exp(smooth_lnkpk) / kp
                wiggle_pk = exp(ilogpk(log(kp))) - smooth_pk
                spk = interp1d(kp, smooth_pk, bounds_error=False)(kk * M.h()) * M.h()**3 # (Mpc/h)**3
                wpk_resc = interp1d(kp, wiggle_pk, bounds_error=False)(alpha_rs * kk * M.h()) * M.h()**3 # (Mpc/h)**3 # wiggle rescaling
                kmask = where(kk < 1.02)[0]
                return kk[kmask], spk[kmask], pk[kmask] #spk[kmask]+wpk_resc[kmask]

            #if self.c["with_nnlo_counterterm"]: cosmo["kk"], cosmo["Psmooth"], cosmo["pk_lin"] = get_smooth_wiggle_resc(cosmo["kk"], cosmo["pk_lin"])

            return cosmo

        elif self._is(module, 'Symbolic'):

            if not engine: 
                from pybird.symbolic import Symbolic
                M = Symbolic(); M.set(cosmo_dict)
            else: 
                M = engine
            
            M.compute(cosmo["kk"], self.c['z'])
            
            cosmo['pk_lin'] = M.pk_lin
            cosmo['D'], cosmo['f'] = M.D, M.f
            if self.c["with_ap"]: cosmo['H'], cosmo['DA'] = M.H, M.DA

            return cosmo

        elif self._is(module, 'CPJ'):

            def to_Mpc_per_h_jax(_pk, _kk, h):
                ilogpk_ = interp1d(log(_kk), log(_pk), fill_value='extrapolate')
                return exp(ilogpk_(log(_kk*h))) * h**3

            if not engine:
                from pybird.integrated_model_jax import IntegratedModel
                from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
                cosmo_dict_local = cosmo_dict.copy()

                M = CPJ(probe='mpk_lin')
                M_growth = IntegratedModel(None, None, None)
                M_growth.restore(self.c["emu_path"] + "/growth_model.h5") 

                if self.c["with_bias"] and "bias" in cosmo_dict: del cosmo_dict_local["bias"] # remove to not pass it to classy that otherwise complains
                if not self.c["with_time"] and "A" in cosmo_dict: del cosmo_dict_local["A"] # same as above
                if self.c["with_redshift_bin"]: zmax = max(self.c["redshift_bin_zz"])
                else: zmax = self.c["z"]  

            else:
                M, M_growth, cosmo_dict_local = engine.CPJ, engine.growth, engine.cosmo

            try: 
                input_dict_pk = {key: array([cosmo_dict_local[key]]) for key in ["omega_b", "omega_cdm", "n_s", "ln10^{10}A_s", "h"]}
            
            except Exception(e):
                print("the input dict did not build... probably you are missing some of the required cosmo inputs for the emu")
                print("exception:", e) 
            
            input_dict_pk["z"] = array([self.c["z"]]) 
            
            cosmo["pk_lin"] = array(to_Mpc_per_h_jax(M.predict(input_dict_pk), M.modes, cosmo_dict_local["h"]))
            cosmo["kk"] = array(M.modes)

            if self.c["multipole"] > 0: 
                sigma_8s = array([0.8]) # this has no impact on growth- I mistakenly included in training so this is a dummy value
                emulator_growth_input = stack([
                    input_dict_pk["omega_b"],
                    input_dict_pk["omega_cdm"],
                    input_dict_pk["n_s"],
                    sigma_8s,
                    input_dict_pk["h"],
                    array([self.c["z"]])
                ], axis=1)

                D, f, H, DA = M_growth.predict(emulator_growth_input)[0]
                cosmo["f"] = f
            if not self.c["with_time"]:
                cosmo["D"] = D
            if self.c["with_ap"]:
                cosmo["H"], cosmo["DA"] = H, DA


        elif module is None: 
            # no cosmo module -assume you have already input your required pk_lin 
            cosmo = cosmo_dict

        return cosmo
