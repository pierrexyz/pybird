import os
import numpy as np
from copy import deepcopy
from scipy.interpolate import splrep, splev, interp1d, RegularGridInterpolator, LinearNDInterpolator, InterpolatedUnivariateSpline
import time
from scipy.special import loggamma
from scipy.misc import derivative
from abc import ABC
from scipy.stats import linregress

# from . common import Common, co
# from . bird import Bird
# from . nonlinear import NonLinear
# from . resum import Resum
# from . projection import Projection
# from . greenfunction import GreenFunction
# from . fourier import FourierTransform

# import importlib, sys
# importlib.reload(sys.modules['common'])
# importlib.reload(sys.modules['bird'])
# importlib.reload(sys.modules['nonlinear'])
# importlib.reload(sys.modules['nnlo'])
# importlib.reload(sys.modules['resum'])
# importlib.reload(sys.modules['projection'])
# importlib.reload(sys.modules['greenfunction'])
# importlib.reload(sys.modules['fourier'])
# importlib.reload(sys.modules['eisensteinhu'])

from .common import Common, co
from .bird import Bird
from .nonlinear import NonLinear
from .nnlo import NNLO_higher_derivative, NNLO_counterterm
from .resum import Resum
from .projection import Projection
from .greenfunction import GreenFunction
from .eisensteinhu import EisensteinHu
from .matching import Matching

# # import pdb; pdb.set_trace()
# ################

class Correlator(object):
    def __init__(self, config_dict=None, load_engines=True):

        self.cosmo_catalog = {
            "P11": Option(
                "P11", (list, np.ndarray), description="Linear matter power spectrum in [Mpc/h]^3", default=None
            ),
            "k11": Option(
                "k11", (list, np.ndarray), description="k-array in [h/Mpc] on which P11 is evaluated", default=None
            ),
            "D": Option(
                "D",
                (float, list, np.ndarray),
                description="Scale independent growth function. To specify if 'skycut' > 1 or 'with_nonequal_time' / 'with_redshift_bin' is True.",
                default=None,
            ),
            "f": Option(
                "f",
                (float, list, np.ndarray),
                description="Scale independent growth rate (for RSD). Automatically set to 0 for 'output': 'm__'.",
                default=None,
            ),
            "bias": Option(
                "bias",
                (dict, list, np.ndarray),
                description="EFT parameters in dict = \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'cr1', 'cr2', 'ce0', 'ce1', 'ce2' \}.",
                default=None,
            ),
            "Omega0_m": Option(
                "Omega0_m",
                float,
                description="Fractional matter abundance at present time. To specify for exact time dependence.",
                default=None,
            ),
            "w0_fld": Option(
                "w0_fld",
                float,
                description="Dark energy equation of state parameter. To specify for exact time dependence if varied (otherwise w0 = -1).",
                default=None,
            ),
            "z": Option(
                "z",
                (float, list, np.ndarray),
                description="Effective redshift(s). Should match the number of skycuts. To specify for exact time dependence.",
                default=None,
            ),
            "DA": Option(
                "DA",
                (float, list, np.ndarray),
                description="Angular distance times H_0. To specify if 'with_AP' is True.",
                default=None,
            ),
            "H": Option(
                "H",
                (float, list, np.ndarray),
                description="Hubble parameter by H_0. To specify if 'with_AP' is True.",
                default=None,
            ),
            "Dz": Option(
                "Dz",
                (list, np.ndarray),
                description="Scale independent growth function over redshift bin. To specify if 'with_redshift_bin' is True.",
                default=None,
            ),
            "fz": Option(
                "fz",
                (list, np.ndarray),
                description="Scale independent growth rate over redshift bin. To specify if 'with_redshift_bin' is True.",
                default=None,
            ),
            "rz": Option(
                "rz",
                (list, np.ndarray),
                description="Comoving distance in [Mpc/h] over redshift bin. To specify if 'with_redshift_bin' or if 'output':'w'.",
                default=None,
            ),
            "D1": Option(
                "D1",
                float,
                description="Scale independent growth function at redshift z1. To specify if 'with_nonequal_time' is True.",
                default=None,
            ),
            "D2": Option(
                "D2",
                float,
                description="Scale independent growth function at redshift z2. To specify if 'with_nonequal_time' is True.",
                default=None,
            ),
            "f1": Option(
                "f1",
                float,
                description="Scale independent growth rate at redshift z1. To specify if 'with_nonequal_time' is True.",
                default=None,
            ),
            "f2": Option(
                "f2",
                float,
                description="Scale independent growth rate at redshift z2. To specify if 'with_nonequal_time' is True.",
                default=None,
            ),
            "EH": Option(
                "EH",
                dict,
                description="Cosmological parameters for Eisenstein-Hu power spectrum. To specify if 'with_nnlo_counterterm' is True.",
                default=None,
            ),
            "Psmooth": Option("Psmooth", (list, np.ndarray),
                description="Smooth power spectrum. To specify if \'with_nnlo_counterterm\' is True.",
                default=None) ,
            "pk_lin_2": Option("pk_lin_2", (list, np.ndarray),
                description="Alternative linear matter power spectrum in [Mpc/h]^3 replacing \'pk_lin\' in the internal loop integrals (and resummation)",
                default=None) ,
        }

        self.config_catalog = {
            "output": Option(
                "output",
                str,
                ["bPk", "bCf", "mPk", "mCf", "bmPk", "bmCf"],
                description="Correlator: biased tracers / matter / biased tracers-matter -- power spectrum / correlation function.",
                default="bPk",
            ),
            "multipole": Option(
                "multipole",
                int,
                [0, 2, 3],
                description="Number of multipoles. 0: real space. 2: monopole + quadrupole. 3: monopole + quadrupole + hexadecapole.",
                default=2,
            ),
            "wedge": Option("wedge", int, description="Number of wedges. 0: compute multipole instead. ", default=0),
            "wedges_bounds": Option(
                "wedges_bounds",
                (list, np.ndarray),
                description="Wedges bounds: [0, a_1, ..., a_{n-1}, 1], n: number of wedges, such that 0 < mu < a_1, ..., a_{n-1} < mu < 1. Default: equi-spaced between 0 and 1.",
                default=None,
            ),
            "skycut": Option("skycut", int, description="Number of skycuts.", default=1),
            "with_time": Option(
                "with_time",
                bool,
                description="Time (in)dependent evaluation (for multi skycuts / redshift bin). For 'with_redshift_bin': True, or 'skycut' > 1, automatically set to False.",
                default=True,
            ),
            "z": Option(
                "z",
                (float, list, np.ndarray),
                description="Effective redshift(s). Should match the number of skycuts.",
                default=None,
            ),
            "km": Option("km", float, description="Inverse galaxy spatial extension scale in [h/Mpc].", default=1.0),
            "nd": Option("nd", float, description="Mean galaxy density", default=3e-4),
            "with_stoch": Option("with_stoch", bool, description="With stochastic terms.", default=False),
            "with_bias": Option(
                "with_bias",
                bool,
                description="Bias (in)dependent evalution. Automatically set to False for 'with_time': False.",
                default=False,
            ),
            "with_exact_time": Option(
                "with_exact_time", bool, description="Exact time dependence or EdS approximation.", default=False
            ),
            "with_redshift_bin": Option(
                "with_redshift_bin",
                bool,
                description="Account for the galaxy count distribution over a redshift bin.",
                default=False,
            ),
            "zz": Option(
                "zz",
                (list, np.ndarray),
                description="Array of redshift points inside a redshift bin. For multi skycuts, a list of arrays should be provided.",
                default=None,
            ),
            "nz": Option(
                "nz",
                (list, np.ndarray),
                description="Galaxy counts distribution over a redshift bin. For multi skycuts, a list of arrays should be provided.",
                default=None,
            ),
            "kmax": Option("kmax", float, description="kmax in [h/Mpc] for 'output': '_Pk'", default=0.25),
            #             "smin": Option("smin", float,
            #                 description="smin in [Mpc/h] for \'output\': \'_Cf\'",
            #                 default=1.) ,
            "accboost": Option("accboost", int, [1, 2, 3], description="Sampling accuracy boost.", default=1),
            "optiresum": Option(
                "optiresum",
                bool,
                description="True: Resumming only with the BAO peak. False: Resummation on the full correlation function.",
                default=False,
            ),
            "xdata": Option("xdata", (np.ndarray, list), description="Array of data points.", default=None),
            "with_resum": Option("with_resum", bool, description="Apply IR-resummation.", default=True),
            "with_AP": Option(
                "with_AP",
                bool,
                description="Apply Alcock Paczynski effect. Automatically set to False for 'output': 'w'.",
                default=False,
            ),
            "DA_AP": Option(
                "DA_AP",
                (float, list, np.ndarray),
                description="Fiducial angular diameter distance times H_0. A list can be provided for multi skycuts. If only one value is passed, use it for all skycuts.",
                default=None,
            ),
            "H_AP": Option(
                "H_AP",
                (float, list, np.ndarray),
                description="Fiducial Hubble parameter by H_0. A list can be provided for multi skycuts. If only one value is passed, use it for all skycuts.",
                default=None,
            ),
            "with_window": Option(
                "window",
                bool,
                description="Apply mask. Automatically set to False for 'output': 'w' or '_Cf'.",
                default=False,
            ),
            "windowPk": Option(
                "windowPk",
                (str, list),
                description="Path to Fourier convolution window file for 'output': '_Pk'. If not provided, read 'windowCf', precompute the Fourier one and save it here.",
                default=None,
            ),
            "windowCf": Option(
                "windowCf",
                (str, list),
                description="Path to configuration space window file with columns: s [Mpc/h], Q0, Q2, Q4. A list can be provided for multi skycuts. Put 'None' for each skycut without window.",
                default=None,
            ),
            "with_binning": Option(
                "with_binning", bool, description="Apply binning for linear-spaced data bins.", default=False
            ),
            "with_fibercol": Option(
                "with_fibercol", bool, description="Apply fiber collision effective window corrections.", default=False
            ),
            "with_nnlo_higher_derivative": Option(
                "with_nnlo_higher_derivative",
                bool,
                description="With next-to-next-to-leading estimate k^2 P1loop.",
                default=False,
            ),
            "with_nnlo_counterterm": Option(
                "with_nnlo_counterterm",
                bool,
                description="With next-to-next-to-leading counterterm k^4 P11.",
                default=False,
            ),
            "with_tidal_alignments": Option(
                "with_tidal_alignments",
                bool,
                description="With tidal alignements: bq * (\mu^2 - 1/3) \delta_m ",
                default=False,
            ),
            "with_quintessence": Option(
                "with_quintessence", bool, description="Clustering quintessence.", default=False
            ),
            "with_nonequal_time": Option(
                "with_nonequal_time",
                bool,
                description="Non equal time correlator. Automatically set to 'with_time' to False ",
                default=False,
            ),
            "z1": Option(
                "z1", (float, list, np.ndarray), description="Redshift z_1 for non equal time correlator.", default=None
            ),
            "z2": Option(
                "z2", (float, list, np.ndarray), description="Redshift z_2 for non equal time correlator.", default=None
            ),
            "corr_convert": Option(
                "corr_convert", bool, description = "Hankel transform the power spectrum to correlation function.", default=False),
            "kr": Option("kr", float,
                description="Inverse velocity product renormalization scale in [h/Mpc].",
                default=0.25) ,
            "eft_basis": Option("eft_basis", str,
                description="Basis of EFT parameters: \'eftoflss\' (default), \'westcoast\', or \'eastcoast\'. See cosmology command \'bias\' for more details.",
                default="eftoflss") ,
            "keep_loop_pieces_independent": Option("keep_loop_pieces_independent", bool,
                description="keep the loop pieces 13 and 22 independent (mainly for debugging)",
                default=False) ,
            "with_uvmatch_2": Option("with_uvmatch_2", bool,
                description="In case two linear power spectra \`pk_lin\` and \`pk_lin_2\` are provided (see description in cosmo_catalog), match the UV as in the case if only \`pk_lin\` would be provided. Implemented only for output=\`Pk\`. ",
                default=False) ,
            "fftbias": Option("fftbias", float,
                description="real power bias for fftlog decomposition of pk_lin (usually to keep to default value)",
                default=-1.6) ,
            "fftaccboost": Option("fftaccboost", int, [1, 2, 3],
                description="FFTLog accuracy boost factor. Default FFTLog sampling : NFFT ~ 256. ",
                default=1) ,
        }

        if config_dict is not None:
            self.set(config_dict, load_engines=load_engines)

    def info(self, description=True):

        for on in ["config", "cosmo"]:

            print("\n")
            if on is "config":
                print("Configuration commands [.set(config_dict)]")
                print("----------------------")
                catalog = self.config_catalog
            elif on is "cosmo":
                print("Cosmology commands [.compute(cosmo_dict)]")
                print("------------------")
                catalog = self.cosmo_catalog

            for (name, config) in zip(catalog, catalog.values()):
                if config.list is None:
                    print("'%s': %s" % (name, typename(config.type)))
                else:
                    print("'%s': %s ; options: %s" % (name, typename(config.type), config.list))
                if description:
                    print("    - %s" % config.description)
                    print("    * default: %s" % config.default)

    def set(self, config_dict, load_engines=True):

        # Reading config provided by user
        self.__read_config(config_dict)

        # Setting no-optional config
        self.config["smin"] = 1.0
        self.config["smax"] = 1000.0
        self.config["kmin"] = 0.001

        # Checking for config conflict
        self.__is_config_conflict()
        
        # print(self.config['with_cf'])
        
        # Setting list of EFT parameters required by the user to provide later
        self.__set_eft_parameters_list()
        
        # Loading PyBird engines
        self.__load_engines(load_engines=load_engines)

    def compute(self, cosmo_dict, module=None, Templatefit = False, corr_convert = False):

        cosmo_dict_local = cosmo_dict.copy()
        

        if module == "class":
            cosmo_dict_class = self.setcosmo(cosmo_dict, module="class")
            cosmo_dict_local.update(cosmo_dict_class)

        self.__read_cosmo(cosmo_dict_local)
        self.__is_cosmo_conflict()
        
        if (corr_convert == True):
            self.kmode = np.logspace(np.log10(self.co.kmin), np.log10(40.0), 10000)
            # self.dist = np.logspace(0.0, 3.0, 5000)
            self.pk2xi_0 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=0)
            self.pk2xi_2 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=2)
            self.pk2xi_4 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=4)
            self.kmode_in = self.kmode[np.where(self.kmode <= 0.5)[0]]
            self.kmode_out = self.kmode[np.where(self.kmode > 0.5)[0]]
        
        # if Shapefit == True:
        #     self.setShapefit_full(self.cosmo['P11'], kmode = self.cosmo['k11'], init = True)
        # else:
        if self.config["skycut"] == 1:
            self.bird = Bird(
                self.cosmo,
                with_bias=self.config["with_bias"],
                with_stoch=self.config["with_stoch"],
                with_nnlo_counterterm=self.config["with_nnlo_counterterm"],
                eft_basis=self.config["eft_basis"],
                co=self.co,
            )
            # if (
            #     self.config["with_nnlo_higher_derivative"] or self.config["with_nnlo_counterterm"]
            # ):  # we use Eisenstein-Hu power spectrum because we don't want spurious BAO signals
            #     EH = EisensteinHu(self.cosmo["EH"])
            #     PEH = EH.__call__(self.bird.kin)
            #     PEH_interp = interp1d(np.log(self.bird.kin), np.log(PEH), fill_value="extrapolate")
            #     if self.config[
            #         "with_nnlo_higher_derivative"
            #     ]:  # we get a bird correlator on EH PS. Most of the corrections are not applied since they are negligible on the nnlo
            #         self.birdEH = deepcopy(self.bird)
            #         self.birdEH.Pin = PEH
            #         self.nonlinear.PsCf(self.birdEH)
            #         if self.config["with_bias"]:
            #             self.birdEH.setPsCf(self.bias)
            #         else:
            #             self.birdEH.setPsCfl()
            #         if self.config["wedge"] != 0:
            #             self.projection.Wedges(self.birdEH)  # This has not been checked
            #         if self.config["with_binning"]:
            #             self.projection.xbinning(self.birdEH)
            #         else:
            #             self.projection.xdata(self.birdEH)
            #     if self.config["with_nnlo_counterterm"]:
            #         if self.config["with_cf"]:
            #             self.nnlo_counterterm.Cf(self.bird, PEH_interp)
            #         else:
            #             self.nnlo_counterterm.Ps(self.bird, PEH_interp)
            if self.config["with_nnlo_counterterm"]: # we use smooth power spectrum since we don't want spurious BAO signals
                ilogPsmooth = interp1d(np.log(self.bird.kin), np.log(self.cosmo["Psmooth"]), fill_value='extrapolate')
                if self.config["with_cf"]: self.nnlo_counterterm.Cf(self.bird, ilogPsmooth)
                else: self.nnlo_counterterm.Ps(self.bird, ilogPsmooth)
            self.nonlinear.PsCf(self.bird)
            if self.config["with_uvmatch_2"]: self.matching.Ps(self.bird) 
            if self.config["with_bias"]:
                self.bird.setPsCf(self.bias)
            else:
                self.bird.setPsCfl()
            if self.config["with_nonequal_time"]:
                self.bird.settime(self.cosmo)  # set D1*D2 / D1**2*D2**2 / 0.5 (D1**2*D2 + D2**2*D1) on 11 / 22 / 13
            if self.config["with_resum"]:
                if self.config["with_cf"]:
                    self.resum.PsCf(self.bird)
                else:
                    if Templatefit == False:
                        self.resum.Ps(self.bird)
                        # np.save('IRPs11_8.npy', self.bird.IRPs11)
                        # np.save('IRPsct_8.npy', self.bird.IRPsct)
                        # np.save('IRPsloop_8.npy', self.bird.IRPsloop)
                    else:
                        # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.bird.setShapefit(0.0, xdata=self.co.k)
                        # # print(np.min(IRPs11), np.max(IRPs11), np.min(IRPsct), np.max(IRPsct), np.min(IRPsloop), np.max(IRPsloop))
                        # self.IRPs11_new, self.IRPsct_new, self.IRPsloop_new = self.resum.IRPs(self.bird, IRPs_all=[IRPs11, IRPsct, IRPsloop])
                        self.resum.Ps(self.bird, setPs=False, init=True, makeQ = False)
                        
                        # ratioes = np.linspace(-0.15, 0.15, 101)
                        # self.bird.IRPs11_interp = interp1d(ratioes, np.load('IRPs11_interp.npy'), axis = 0, bounds_error=True, kind = 'cubic')
                        # self.bird.IRPsct_interp = interp1d(ratioes, np.load('IRPsct_interp.npy'), axis = 0, bounds_error=True, kind = 'cubic')
                        # self.bird.IRPsloop_interp = interp1d(ratioes, np.load('IRPsloop_interp.npy'), axis = 0, bounds_error=True, kind = 'cubic')
                        # print(np.max(self.bird.IRPs11/self.bird.IRPs11_interp(0.0)))
                        # print(np.max(self.bird.IRPsct/self.bird.IRPsct_interp(0.0)))
                        # print(np.max(self.bird.IRPsloop/self.bird.IRPsloop_interp(0.0)))
                        # np.save('IRPs11_fid.npy', self.bird.IRPs11)
                        # np.save('IRPsct_fid.npy', self.bird.IRPsct)
                        # np.save('IRPsloop_fid.npy', self.bird.IRPsloop)
                        # raise ValueError('Test completed.')
                        
                        # self.resum.IRPs(self.bird)
                        # P11l, Pctl, Ploopl, self.IRPs11_new, self.IRPsct_new, self.IRPsloop_new = self.bird.setShapefit(0.0, xdata=self.co.k)
                        # print("New Shapefit")
                        
                        
                        # self.resum.Ps(self.bird, setPs=False)
                        # self.resum.IRPs(self.bird)
                        
            if self.config["with_redshift_bin"]:
                self.projection.redshift(
                    self.bird, self.cosmo["rz"], self.cosmo["Dz"], self.cosmo["fz"], pk=self.config["output"]
                )
            # print(np.shape(self.bird.P11l), np.shape(self.bird.Ploopl), np.shape(self.bird.Pctl), np.shape(self.bird.Pstl))
            if (self.config["with_AP"] == True and Templatefit == False):
                self.projection.AP(self.bird)
                # print(np.shape(self.bird.P11l), np.shape(self.bird.Ploopl), np.shape(self.bird.Pctl), np.shape(self.bird.Pstl))
            if (self.config["with_window"] == True and Templatefit == False):
                self.projection.Window(self.bird)
            if self.config["with_fibercol"]:
                self.projection.fibcolWindow(self.bird)
            if self.config["wedge"] != 0:
                self.projection.Wedges(self.bird)
            if (corr_convert == True and Templatefit == False):
                # # np.save('P11l.npy', self.bird.P11l)
                # # np.save('Ploopl.npy', self.bird.Ploopl)
                # # np.save('Pctl.npy', self.bird.Pctl)
                # # np.save('Pstl.npy', self.bird.Pstl)
                # # np.save('k.npy', self.co.k)
                
                # P11l_interp = interp1d(self.co.k, self.bird.P11l, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                # Ploopl_interp = interp1d(self.co.k, self.bird.Ploopl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                # Pctl_interp = interp1d(self.co.k, self.bird.Pctl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                # Pstl_interp = interp1d(self.co.k, self.bird.Pstl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                
                # # np.save('P11l.npy', P11l_interp)
                # # np.save('Ploopl.npy', Ploopl_interp)
                # # np.save('Pctl.npy', Pctl_interp)
                # # np.save('Pstl.npy', Pstl_interp)
                # # np.save('k.npy', self.kmode)
                
                # damping = 3.5
                
                # P11l_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, P11l_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
                # P11l_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, P11l_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
                # P11l_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, P11l_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
                # self.bird.C11l = np.concatenate((P11l_mono_new, P11l_quad_new, P11l_hexa_new), axis = 0)
                
                # Ploopl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Ploopl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
                # Ploopl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Ploopl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
                # Ploopl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Ploopl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
                # self.bird.Cloopl = np.concatenate((Ploopl_mono_new, Ploopl_quad_new, Ploopl_hexa_new), axis = 0)
                
                # Pctl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Pctl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
                # Pctl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Pctl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
                # Pctl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Pctl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
                # self.bird.Cctl = np.concatenate((Pctl_mono_new, Pctl_quad_new, Pctl_hexa_new), axis = 0)
                
                # Pstl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Pstl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
                # Pstl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Pstl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
                # Pstl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Pstl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
                # self.bird.Cstl = np.concatenate((Pstl_mono_new, Pstl_quad_new, Pstl_hexa_new), axis = 0)
                self.pk2xi_fun(bird=[self.bird.P11l, self.bird.Ploopl, self.bird.Pctl, self.bird.Pstl])
            
            # np.save('Plin_long.npy', self.bird.P11l)
            # np.save('Ploop_long.npy', self.bird.Ploopl)
            if (self.config["with_binning"] == True and Templatefit == False):
                self.projection.xbinning(self.bird)
            else:
                if Templatefit == False and corr_convert == False:
                    self.projection.xdata(self.bird)
                    
            # print(np.shape(self.bird.P11l), np.shape(self.bird.Ploopl), np.shape(self.bird.Pctl), np.shape(self.bird.Pstl))

        elif self.config["skycut"] > 1:
            if self.config["with_time"]:  # if all skycuts have same redshift
                cosmoi = deepcopy(self.cosmo)
                cosmoi["f"] = self.cosmo["f"][0]
                cosmoi["D"] = self.cosmo["D"][0]
                cosmoi["z"] = self.config["z"][0]
                if self.config["with_AP"]:
                    cosmoi["DA"] = self.cosmo["DA"][0]
                    cosmoi["H"] = self.cosmo["H"][0]
                self.bird = Bird(
                    cosmoi,
                    with_bias=False,
                    with_stoch=self.config["with_stoch"],
                    with_nnlo_counterterm=self.config["with_nnlo_counterterm"],
                    co=self.co,
                )
                if (
                    self.config["with_nnlo_higher_derivative"] or self.config["with_nnlo_counterterm"]
                ):  # this works only if the skycut has same redshift
                    EH = EisensteinHu(self.cosmo["EH"])
                    PEH = EH.__call__(self.bird.kin)
                    PEH_interp = interp1d(np.log(self.bird.kin), np.log(PEH), fill_value="extrapolate")
                    if self.config["with_nnlo_higher_derivative"]:
                        self.birdEH = deepcopy(self.bird)
                        self.birdEH.Pin = PEH
                        self.nonlinear.PsCf(self.birdEH)
                        self.birdEH.setPsCfl()
                        self.birdsEH = []
                        for i in range(self.config["skycut"]):
                            birdEH_local = deepcopy(self.birdEH)
                            if self.config["wedge"] != 0:
                                self.projection[i].Wedges(birdEH_local)  # This has not been checked
                            if self.config["with_binning"]:
                                self.projection[i].xbinning(birdEH_local)
                            else:
                                self.projection[i].xdata(birdEH_local)
                            self.birdsEH.append(birdEH_local)
                    if self.config["with_nnlo_counterterm"]:
                        if self.config["with_cf"]:
                            self.nnlo_counterterm.Cf(self.bird, PEH_interp)
                        else:
                            self.nnlo_counterterm.Ps(self.bird, PEH_interp)
                self.nonlinear.PsCf(self.bird)
                self.bird.setPsCfl()
                if self.config["with_resum"]:
                    if self.config["with_cf"]:
                        self.resum.PsCf(self.bird)
                    else:
                        if Templatefit == False:
                            self.resum.Ps(self.bird)
                        else:
                            # self.resum.Ps(self.bird, setPs=False)
                            
                            P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.bird.setShapefit(0.0, xdata=self.co.k)
                            self.bird.IRPs11_new, self.bird.IRPsct_new, self.bird.IRPsloop_new = self.resum.IRPs(self.bird, IRPs_all=[IRPs11, IRPsct, IRPsloop])
                            
                self.birds = [deepcopy(self.bird) for i in range(self.config["skycut"])]

            else:
                self.birds = []
                cosmoi = deepcopy(self.cosmo)

                def mycycle(skycut, first=0, L=None):
                    if L is None:
                        L = [i for i in range(skycut)]
                    if (skycut % 2) == 0:
                        first = skycut // 2
                    else:
                        if first == 0:
                            first = (skycut + 1) // 2
                        else:
                            first = skycut // 2
                    return [item for i, item in enumerate(L + L) if i < skycut + first and first <= i]

                zbins = mycycle(self.config["skycut"], first=2)  # cycle to get the middle redshift

                for i in zbins:
                    cosmoi["f"], cosmoi["D"], cosmoi["z"] = self.cosmo["f"][i], self.cosmo["D"][i], self.config["z"][i]
                    if self.config["with_AP"]:
                        cosmoi["DA"], cosmoi["H"] = self.cosmo["DA"][i], self.cosmo["H"][i]

                    if i == zbins[0]:
                        self.bird = Bird(
                            cosmoi,
                            with_bias=False,
                            with_stoch=self.config["with_stoch"],
                            with_nnlo_counterterm=self.config["with_nnlo_counterterm"],
                            co=self.co,
                        )
                        self.nonlinear.PsCf(self.bird)
                        self.bird.setPsCfl()
                        if self.config["with_resum"]:
                            if Templatefit == False:
                                self.resum.Ps(self.bird, makeIR=True, makeQ=True, setPs=False)
                        self.birds.append(self.bird)
                    else:
                        birdi = deepcopy(self.bird)
                        birdi.settime(cosmoi)  # set new cosmo (in particular, f), and rescale by (Dnew/Dold)**(2p)
                        if self.config["with_resum"]:
                            if self.config["with_cf"]:
                                self.resum.PsCf(birdi, makeIR=False, makeQ=True, setPs=False, setCf=True)
                            # else:
                            #     self.resum.Ps(birdi, makeIR=False, makeQ=True, setPs=True)
                            else:
                                if Templatefit == False:
                                    self.resum.Ps(birdi, makeIR=False, makeQ=True, setPs=True)
                                else:
                                    # self.resum.Ps(birdi, makeIR=False, makeQ=True, setPs=False)
                                    P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = birdi.setShapefit(0.0, xdata=self.co.k)
                                    birdi.IRPs11_new, birdi.IRPsct_new, birdi.IRPsloop_new = self.resum.IRPs(birdi, IRPs_all=[IRPs11, IRPsct, IRPsloop])
                                    # self.resum.IRPs(self.bird)
                        self.birds.append(birdi)

                if self.config["with_resum"]:
                    if self.config["with_cf"]:
                        self.resum.PsCf(self.birds[0], makeIR=False, makeQ=False, setPs=False, setCf=True)
                    # else:
                    #     self.resum.Ps(self.birds[0], makeIR=False, makeQ=False, setPs=True)
                    else:
                        if Templatefit == False:
                            self.resum.Ps(self.birds[0], makeIR=False, makeQ=False, setPs=True)
                        else:
                            # self.resum.Ps(self.birds[0], makeIR=False, makeQ=False, setPs=False)
                            P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.birds[0].setShapefit(0.0, xdata=self.co.k)
                            self.birds[0].IRPs11_new, self.birds[0].IRPsct_new, self.birds[0].IRPsloop_new = self.resum.IRPs(self.birds[0], IRPs_all=[IRPs11, IRPsct, IRPsloop])
                            # self.resum.IRPs(self.bird)

                self.birds = mycycle(self.config["skycut"], first=0, L=self.birds)  # cycle back the birds

            for i in range(self.config["skycut"]):
                if self.config["with_redshift_bin"] and self.config["nz"][i] is not None:
                    self.projection[i].redshift(
                        self.birds[i],
                        self.cosmo["rz"][i],
                        self.cosmo["Dz"][i],
                        self.cosmo["fz"][i],
                        pk=self.config["output"],
                    )
                if (self.config["with_AP"] == True and Templatefit == False):
                    self.projection[i].AP(self.birds[i])
                if (self.config["with_window"] == True and Templatefit == False):
                    self.projection[i].Window(self.birds[i])
                if self.config["with_fibercol"]:
                    self.projection[i].fibcolWindow(self.birds[i])
                if self.config["wedge"] != 0:
                    self.projection[i].Wedges(self.birds[i])
                if (corr_convert == True and Templatefit == False):
                    # P11l_interp = interp1d(self.co.k, self.bird[i].P11l, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                    # Ploopl_interp = interp1d(self.co.k, self.bird[i].Ploopl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                    # Pctl_interp = interp1d(self.co.k, self.bird[i].Pctl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                    # Pstl_interp = interp1d(self.co.k, self.bird[i].Pstl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
                    
                    # damping = 0.25
                    
                    # P11l_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, P11l_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
                    # P11l_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, P11l_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
                    # P11l_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, P11l_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
                    # self.bird.C11l[i] = np.concatenate((P11l_mono_new, P11l_quad_new, P11l_hexa_new), axis = 0)
                    
                    # Ploopl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Ploopl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
                    # Ploopl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Ploopl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
                    # Ploopl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Ploopl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
                    # self.bird.Cloopl[i] = np.concatenate((Ploopl_mono_new, Ploopl_quad_new, Ploopl_hexa_new), axis = 0)
                    
                    # Pctl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Pctl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
                    # Pctl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Pctl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
                    # Pctl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Pctl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
                    # self.bird.Cctl[i] = np.concatenate((Pctl_mono_new, Pctl_quad_new, Pctl_hexa_new), axis = 0)
                    
                    # Pstl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Pstl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
                    # Pstl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Pstl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
                    # Pstl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Pstl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
                    # self.bird.Cstl[i] = np.concatenate((Pstl_mono_new, Pstl_quad_new, Pstl_hexa_new), axis = 0)
                    self.pk2xi_fun(bird = [self.birds[i].P11l, self.birds[i].Ploopl, self.birds[i].Pctl, self.birds[i].Pstl], index=i)
                    
                
                if (self.config["with_binning"] == True and Templatefit == False):
                    if corr_convert == False:
                        self.projection[i].xbinning(self.birds[i])
                else:
                    # self.projection[i].xdata(self.birds[i])
                    if Templatefit == False and corr_convert == False:
                        self.projection.xdata(self.bird)
                        
    def pk2xi_fun(self, bird, damping = 0.25, index = None, output = False):
        P11l, Ploopl, Pctl, Pstl = bird
        # print(np.shape(P11l), np.shape(Ploopl), np.shape(Pctl), np.shape(Pstl))
        # np.save('Plin_cf.npy', P11l)
        # np.save('Ploop_cf.npy', Ploopl)
        # np.save('Pct_cf.npy', Pctl)
        # np.save('Pst_cf.npy', Pstl)
        # np.save('k_mode.npy', self.co.k)
        # raise ValueError('Test completed')
        
        P11l_mono, P11l_quad, P11l_hexa = P11l
        Ploopl_mono, Ploopl_quad, Ploopl_hexa = Ploopl
        Pctl_mono, Pctl_quad, Pctl_hexa = Pctl
        # Pstl_mono, Pstl_quad, Pstl_hexa = Pstl
        
        # print(np.log10(P11l_mono[:, -20:]))
        
        # power_0, scale_0 = np.polyfit(np.log10(self.co.k[-20:]), np.log10(P11l_mono[0][-20:]), 1)
        # power_1, scale_1 = np.polyfit(np.log10(self.co.k[-20:]), np.log10(P11l_mono[1][-20:]), 1)
        # power_2, scale_2 = np.polyfit(np.log10(self.co.k[-20:]), np.log10(P11l_mono[2][-20:]), 1)
        
        # print(power_0, power_1, power_2)
        # print(scale_0, scale_1, scale_2)
        
        power_0, scale_0, r_value_0, p_value_0, std_err = linregress(np.log10(self.co.k[-20:]), np.log10(P11l_mono[0][-20:]))
        power_1, scale_1, r_value_1, p_value_1, std_err = linregress(np.log10(self.co.k[-20:]), np.log10(P11l_mono[1][-20:]))
        power_2, scale_2, r_value_2, p_value_2, std_err = linregress(np.log10(self.co.k[-20:]), np.log10(P11l_mono[2][-20:]))
        
        print(power_0, scale_0, r_value_0, p_value_0)
        print(power_1, scale_1, r_value_1, p_value_1)
        print(power_2, scale_2, r_value_2, p_value_2)
        # np.save('P11l_mono.npy', P11l_mono)
        
        P11l_mono_interp = np.array([np.concatenate((interp1d(self.co.k, P11l_mono[0], fill_value = 'extrapolate', kind = 'cubic')(self.kmode_in), 
                                    10.0**scale_0*self.kmode_out**power_0)), 
                                    np.concatenate((interp1d(self.co.k, P11l_mono[1], fill_value = 'extrapolate', kind = 'cubic')(self.kmode_in), 
                                                                  10.0**scale_1*self.kmode_out**power_1)),
                                    np.concatenate((interp1d(self.co.k, P11l_mono[2], fill_value = 'extrapolate', kind = 'cubic')(self.kmode_in), 
                                                                10.0**scale_2*self.kmode_out**power_2))])
        # P11l_mono_interp = interp1d(self.co.k, P11l_mono, kind = 'linear', fill_value='extrapolate')(self.kmode)
        
        P11l_quad_interp = interp1d(self.co.k, P11l_quad, kind = 'linear', fill_value='extrapolate')(self.kmode)
        P11l_hexa_interp = interp1d(self.co.k, P11l_hexa, kind = 'nearest', fill_value = 'extrapolate')(self.kmode)
        
        
        
        P11l_interp = np.array([P11l_mono_interp, P11l_quad_interp, P11l_hexa_interp])
        
        Ploopl_mono_interp = interp1d(self.co.k, Ploopl_mono, kind = 'linear', fill_value = 'extrapolate')(self.kmode)
        Ploopl_quad_interp = interp1d(self.co.k, Ploopl_quad, kind = 'linear', fill_value = 'extrapolate')(self.kmode)
        Ploopl_hexa_interp = interp1d(self.co.k, Ploopl_hexa, kind = 'nearest', fill_value = 'extrapolate')(self.kmode)
        Ploopl_interp = np.array([Ploopl_mono_interp, Ploopl_quad_interp, Ploopl_hexa_interp])
        
        Pctl_mono_interp = interp1d(self.co.k, Pctl_mono, kind = 'linear', fill_value = 'extrapolate')(self.kmode)
        Pctl_quad_interp = interp1d(self.co.k, Pctl_quad, kind = 'linear', fill_value = 'extrapolate')(self.kmode)
        Pctl_hexa_interp = interp1d(self.co.k, Pctl_hexa, kind = 'nearest', fill_value = 'extrapolate')(self.kmode)
        Pctl_interp = np.array([Pctl_mono_interp, Pctl_quad_interp, Pctl_hexa_interp])
        
        Pstl_interp = np.zeros(shape=(self.co.Nl, self.co.Nst, len(self.kmode)))
        Pstl_interp[0, 0] = self.kmode ** 0 / self.co.nd
        # self.Pstl[1, 0] = self.co.k ** 0 / self.co.nd
        Pstl_interp[0, 1] = self.kmode ** 2 / self.co.km ** 2 / self.co.nd
        Pstl_interp[1, 2] = self.kmode ** 2 / self.co.km ** 2 / self.co.nd
        
        
        # P11l_interp = interp1d(self.co.k, P11l, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
        # Ploopl_interp = interp1d(self.co.k, Ploopl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
        # Pctl_interp = interp1d(self.co.k, Pctl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
        # Pstl_interp = interp1d(self.co.k, Pstl, kind = 'cubic', fill_value = 'extrapolate')(self.kmode)
        
        P11l_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, P11l_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
        P11l_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, P11l_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
        P11l_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, P11l_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.N11)]])
        
        Ploopl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Ploopl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
        Ploopl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Ploopl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])
        Ploopl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Ploopl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nloop)]])        
        
        Pctl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Pctl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
        Pctl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Pctl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
        Pctl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Pctl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nct)]])
        
        Pstl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Pstl_interp[0, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
        Pstl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Pstl_interp[1, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
        Pstl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Pstl_interp[2, i], self.co.dist, damping=damping) for i in range(self.co.Nst)]])
        
        if output == False:
            if index is None:
                self.bird.C11l = np.concatenate((P11l_mono_new, P11l_quad_new, P11l_hexa_new), axis = 0)
                self.bird.Cloopl = np.concatenate((Ploopl_mono_new, Ploopl_quad_new, Ploopl_hexa_new), axis = 0)
                self.bird.Cctl = np.concatenate((Pctl_mono_new, Pctl_quad_new, Pctl_hexa_new), axis = 0)
                self.bird.Cstl = np.concatenate((Pstl_mono_new, Pstl_quad_new, Pstl_hexa_new), axis = 0)
            else:
                self.birds[index].C11l = np.concatenate((P11l_mono_new, P11l_quad_new, P11l_hexa_new), axis = 0)
                self.birds[index].Cloopl = np.concatenate((Ploopl_mono_new, Ploopl_quad_new, Ploopl_hexa_new), axis = 0)
                self.birds[index].Cctl = np.concatenate((Pctl_mono_new, Pctl_quad_new, Pctl_hexa_new), axis = 0)
                self.birds[index].Cstl = np.concatenate((Pstl_mono_new, Pstl_quad_new, Pstl_hexa_new), axis = 0)
        else:
            C11l = np.concatenate((P11l_mono_new, P11l_quad_new, P11l_hexa_new), axis = 0)
            Cloopl = np.concatenate((Ploopl_mono_new, Ploopl_quad_new, Ploopl_hexa_new), axis = 0)
            Cctl = np.concatenate((Pctl_mono_new, Pctl_quad_new, Pctl_hexa_new), axis = 0)
            Cstl = np.concatenate((Pstl_mono_new, Pstl_quad_new, Pstl_hexa_new), axis = 0)
            
            return C11l, Cloopl, Cctl, Cstl

    def get(self, bias=None):
        
        # for p in marg_gauss_eft_parameters_list:
        #     if p not in self.gauss_eft_parameters_list:
        #         raise Exception("The parameter %s specified in getmarg() is not an available Gaussian EFT parameter to marginalize. Check your options. " % p)

        if self.config["skycut"] == 1:
            if not self.config["with_bias"]:
                self.__is_bias_conflict(bias)
                if "Pk" in self.config["output"]:
                    self.bird.setreducePslb(self.bias)
                elif "Cf" in self.config["output"]:
                    self.bird.setreduceCflb(self.bias)
            if "Pk" in self.config["output"]:
                return self.bird.fullPs
            elif "Cf" in self.config["output"]:
                return self.bird.fullCf

        elif self.config["skycut"] > 1:
            if not isinstance(bias, (list, np.ndarray)) or len(bias) != self.config["skycut"]:
                raise Exception("Please specify bias (in a list of dicts) for each corresponding skycuts. ")
            for i in range(self.config["skycut"]):
                self.__is_bias_conflict(bias[i])
                if "Cf" in self.config["output"]:
                    self.birds[i].setreduceCflb(self.bias)
                elif "Pk" in self.config["output"]:
                    self.birds[i].setreducePslb(self.bias)
            if "Pk" in self.config["output"]:
                return [self.birds[i].fullPs for i in range(self.config["skycut"])]
            elif "Cf" in self.config["output"]:
                return [self.birds[i].fullCf for i in range(self.config["skycut"])]

    def getmarg(self, bias, model=1):
        def marg(loop, ct, b1, f, Pst=None, bq=0):

            if "m" in self.config["output"]:
                return np.array([ct[0].reshape(-1) / self.config["km"] ** 2])

            elif "b" in self.config["output"]:

                if loop.ndim is 3:
                    loop = np.swapaxes(loop, axis1=0, axis2=1)
                    ct = np.swapaxes(ct, axis1=0, axis2=1)
                    if Pst is not None:
                        Pst = np.swapaxes(Pst, axis1=0, axis2=1)

                if self.co.Nloop is 12:
                    Pb3 = loop[3] + b1 * loop[7]  # config["with_time"] = True
                elif self.co.Nloop is 18:
                    Pb3 = (
                        loop[3] + b1 * loop[7] + bq * loop[16]
                    )  # config["with_time"] = True, config["with_tidal_alignments"] = True
                elif self.co.Nloop is 22:
                    Pb3 = f * loop[8] + b1 * loop[16]  # config["with_time"] = False, config["with_exact_time"] = False
                elif self.co.Nloop is 35:
                    Pb3 = f * loop[18] + b1 * loop[29]  # config["with_time"] = False, config["with_exact_time"] = True

                m = np.array(
                    [Pb3.reshape(-1), 2 * (f * ct[0 + 3] + b1 * ct[0]).reshape(-1) / self.config["km"] ** 2]
                )  # b3, cct

                if self.config["multipole"] >= 2:
                    m = np.vstack([m, 2 * (f * ct[1 + 3] + b1 * ct[1]).reshape(-1) / self.config["km"] ** 2])  # cr1
                if self.config["multipole"] >= 3:
                    m = np.vstack([m, 2 * (f * ct[2 + 3] + b1 * ct[2]).reshape(-1) / self.config["km"] ** 2])  # cr2

                if self.config["with_stoch"]:
                    if model <= 4:
                        m = np.vstack([m, Pst[2].reshape(-1)])  # k^2 quad
                    if model == 1:
                        m = np.vstack([m, Pst[0].reshape(-1)])  # k^0 mono
                    if model == 3:
                        m = np.vstack([m, Pst[1].reshape(-1)])  # k^2 mono
                    if model == 4:
                        m = np.vstack([m, Pst[1].reshape(-1), Pst[0].reshape(-1)])  # k^2 mono, k^0 mono

            return m

        def marg_from_bird(bird, bias_local):
            self.__is_bias_conflict(bias_local)
            
            if self.config["with_tidal_alignments"]: bq = self.bias["bq"]
            else: bq = 0.
            
            if "Pk" in self.config["output"]:
                return marg(bird.Ploopl, bird.Pctl, self.bias["b1"], bird.f, Pst=bird.Pstl, bq=bq)
            elif "Cf" in self.config["output"]:
                return marg(bird.Cloopl, bird.Cctl, self.bias["b1"], bird.f, Pst=bird.Cstl, bq=bq)

        if self.config["skycut"] == 1:
            return marg_from_bird(self.bird, bias)
        elif self.config["skycut"] > 1:
            return [marg_from_bird(bird_i, bias_i) for (bird_i, bias_i) in zip(self.birds, bias)]

    def getnnlo(self, bias):

        if self.config["skycut"] == 1:
            if not self.config["with_bias"]:
                self.__is_bias_conflict(bias)
                bias_local = deepcopy(self.bias)  # we remove the counterterms and the stochastic terms, if any.
                bias_local["cct"] = 0.0
                bias_local["cr1"] = 0.0
                bias_local["cr2"] = 0.0
                bias_local["ce0"] = 0.0
                bias_local["ce1"] = 0.0
                bias_local["ce2"] = 0.0
                self.birdEH.setreducePslb(bias_local)
                bnnlo = np.array([bias_local["bnnlo_l%s" % (2 * i)] for i in range(self.config["multipole"])])
                if "Pk" in self.config["output"]:
                    nnlo = self.nnlo_higher_derivative.Ps(self.birdEH)  # k^2 P1Loop
                elif "Cf" in self.config["output"]:
                    nnlo = self.nnlo_higher_derivative.Cf(self.birdEH)  # FT[k^2 P1Loop]
                return np.einsum("l,lx->lx", bnnlo, nnlo)
        elif self.config["skycut"] > 1:
            if not self.config["with_bias"]:
                if not isinstance(bias, (list, np.ndarray)) or len(bias) != self.config["skycut"]:
                    raise Exception("Please specify bias (in a list of dicts) for each corresponding skycuts. ")
                nnlo_higher_derivative = []
                for i in range(self.config["skycut"]):
                    self.__is_bias_conflict(bias[i])
                    bias_local = deepcopy(self.bias)  # we remove the counterterms and the stochastic terms, if any.
                    bias_local["cct"] = 0.0
                    bias_local["cr1"] = 0.0
                    bias_local["cr2"] = 0.0
                    bias_local["ce0"] = 0.0
                    bias_local["ce1"] = 0.0
                    bias_local["ce2"] = 0.0
                    self.birdsEH[i].setreducePslb(bias_local)
                    bnnlo = np.array([bias_local["bnnlo_l%s" % (2 * l)] for l in range(self.config["multipole"])])
                    if "Pk" in self.config["output"]:
                        nnlo = self.nnlo_higher_derivative[i].Ps(self.birdsEH[i])  # k^2 P1Loop
                    elif "Cf" in self.config["output"]:
                        nnlo = self.nnlo_higher_derivative[i].Cf(self.birdsEH[i])  # FT[k^2 P1Loop]
                    nnlo_higher_derivative.append(np.einsum("l,lx->lx", bnnlo, nnlo))
                return nnlo_higher_derivative

    def __load_engines(self, load_engines=True):

        self.co = Common(
            Nl=self.config["multipole"],
            kmax=self.config["kmax"],
            km=self.config["km"],
            nd=self.config["nd"],
            halohalo=self.config["halohalo"],
            with_cf=self.config["with_cf"],
            with_time=self.config["with_time"],
            optiresum=self.config["optiresum"],
            exact_time=self.config["with_exact_time"],
            quintessence=self.config["with_quintessence"],
            with_tidal_alignments=self.config["with_tidal_alignments"],
            nonequaltime=self.config["with_common_nonequal_time"],
            corr_convert = self.config["corr_convert"],
            kr=self.config["kr"],
            with_uvmatch=self.config["with_uvmatch_2"],
            keep_loop_pieces_independent=self.config["keep_loop_pieces_independent"],
        )

        if load_engines:
            self.nonlinear = NonLinear(load=True, save=True, NFFT=256*self.config["fftaccboost"], co=self.co)
            self.resum = Resum(co=self.co)
            
            if self.config["with_uvmatch_2"]: self.matching = Matching(self.nonlinear, co=self.co)

            if self.config["with_nnlo_counterterm"]:
                self.nnlo_counterterm = NNLO_counterterm(co=self.co)
            if self.config["with_nnlo_higher_derivative"]:
                if self.config["skycut"] == 1:
                    self.nnlo_higher_derivative = NNLO_higher_derivative(
                        self.config["xdata"], with_cf=self.config["with_cf"], co=self.co
                    )
                elif self.config["skycut"] > 1:
                    self.nnlo_higher_derivative = [
                        NNLO_higher_derivative(self.config["xdata"][i], with_cf=self.config["with_cf"], co=self.co)
                        for i in range(self.config["skycut"])
                    ]

            if self.config["skycut"] == 1:
                self.projection = Projection(
                    self.config["xdata"],
                    with_AP=self.config["with_AP"],
                    D_fid = self.config["DA_AP"],
                    H_fid = self.config["H_AP"],
                    # DA_AP=self.config["DA_AP"],
                    # H_AP=self.config["H_AP"],
                    window_fourier_name=self.config["windowPk"],
                    path_to_window="",
                    window_configspace_file=self.config["windowCf"],
                    binning=self.config["with_binning"],
                    fibcol=self.config["with_fibercol"],
                    Nwedges=self.config["wedge"],
                    wedges_bounds=self.config["wedges_bounds"],
                    zz=self.config["zz"],
                    nz=self.config["nz"],
                    co=self.co,
                )
            elif self.config["skycut"] > 1:
                self.projection = []
                for i in range(self.config["skycut"]):
                    if len(self.config["xdata"]) == 1:
                        xdata = self.config["xdata"][i]
                    elif len(self.config["xdata"]) == self.config["skycut"]:
                        xdata = self.config["xdata"][i]
                    else:
                        xdata = self.config["xdata"]
                    if self.config["with_window"]:
                        windowPk = self.config["windowPk"][i]
                        windowCf = self.config["windowCf"][i]
                    else:
                        windowPk = None
                        windowCf = None
                    if self.config["with_AP"]:
                        if isinstance(self.config["DA_AP"], float):
                            DA_AP = self.config["DA_AP"]
                        elif len(self.config["DA_AP"]) is self.config["skycut"]:
                            DA_AP = self.config["DA_AP"][i]
                        if isinstance(self.config["H_AP"], float):
                            H_AP = self.config["H_AP"]
                        elif len(self.config["H_AP"]) is self.config["skycut"]:
                            H_AP = self.config["H_AP"][i]
                    else:
                        DA_AP = None
                        H_AP = None
                    if self.config["with_redshift_bin"]:
                        zz = self.config["zz"][i]
                        nz = self.config["nz"][i]
                    else:
                        zz = None
                        nz = None
                    self.projection.append(
                        Projection(
                            xdata,
                            with_AP = self.config['with_AP'],
                            D_fid=DA_AP,
                            H_fid=H_AP,
                            # DA_AP=DA_AP,
                            # H_AP=H_AP,
                            window_fourier_name=windowPk,
                            path_to_window="",
                            window_configspace_file=windowCf,
                            binning=self.config["with_binning"],
                            fibcol=self.config["with_fibercol"],
                            Nwedges=self.config["wedge"],
                            wedges_bounds=self.config["wedges_bounds"],
                            zz=zz,
                            nz=nz,
                            co=self.co,
                        )
                    )

    def __read_cosmo(self, cosmo_dict):

        # Checking if the inputs are consistent with the options
        for (name, cosmo) in zip(self.cosmo_catalog, self.cosmo_catalog.values()):
            for cosmo_key in cosmo_dict:
                if cosmo_key is name:
                    cosmo.check(cosmo_key, cosmo_dict[cosmo_key])

        # Setting unspecified configs to default value
        for (name, cosmo) in zip(self.cosmo_catalog, self.cosmo_catalog.values()):
            if cosmo.value is None:
                cosmo.value = cosmo.default

        # Translating the catalog to a dict
        self.cosmo = translate_catalog_to_dict(self.cosmo_catalog)

    def __is_cosmo_conflict(self):

        if self.cosmo["k11"] is None or self.cosmo["P11"] is None:
            raise Exception("Please provide a linear matter power spectrum 'P11' and the corresponding 'k11'. ")

        if len(self.cosmo["k11"]) != len(self.cosmo["P11"]):
            raise Exception(
                "Please provide a linear matter power spectrum 'P11' and the corresponding 'k11' of same length."
            )

        if self.cosmo["k11"][0] > 1e-4 or self.cosmo["k11"][-1] < 1.0:
            raise Exception(
                "Please provide a linear matter spectrum 'P11' and the corresponding 'k11' with min(k11) < 1e-4 and max(k11) > 1."
            )

        if self.config["skycut"] > 1:
            if self.cosmo["D"] is None:
                raise Exception("You asked multi skycuts. Please specify the growth function 'D'. ")
            elif len(self.cosmo["D"]) is not self.config["skycut"]:
                raise Exception("Please specify (in a list) as many growth functions 'D' as the corresponding skycuts.")

        if self.config["multipole"] == 0:
            self.cosmo["f"] = 0.0
        elif not self.config["with_redshift_bin"]:
            if self.cosmo["f"] is None:
                raise Exception("Please specify the growth rate 'f'.")
            if self.config["skycut"] == 1:
                if not isinstance(self.cosmo["f"], float):
                    raise Exception("Please provide a single growth rate 'f'.")
            elif len(self.cosmo["f"]) != self.config["skycut"]:
                raise Exception("Please specify (in a list) as many 'f' as the corresponding skycuts.")

        if self.config["wedge"] > 0:
            if self.config["wedges_bounds"] is not None:
                if (
                    len(self.config["wedges_bounds"]) != self.config["wedge"] + 1
                    or self.config["wedges_bounds"][0] != 0
                    or self.config["wedges_bounds"][-1] != 1
                ):
                    raise Exception(
                        "If specifying 'wedges_bounds', specify them in a list as: [0, a_1, ..., a_{n-1}, 1], where n: number of wedges"
                    )

        if self.config["with_bias"]:
            self.__is_bias_conflict()

        if self.config["with_AP"]:
            if self.cosmo["DA"] is None or self.cosmo["H"] is None:
                raise Exception("You asked to apply the AP effect. Please specify 'DA' and 'H'. ")

            if self.config["skycut"] == 1:
                if not isinstance(self.cosmo["DA"], float) and not isinstance(self.cosmo["H"], float):
                    raise Exception("Please provide a single pair of 'DA' and 'H'.")
            elif len(self.cosmo["DA"]) != self.config["skycut"] or len(self.cosmo["H"]) != self.config["skycut"]:
                raise Exception("Please specify (in lists) as many 'DA' and 'H' as the corresponding skycuts.")

        if self.config["with_redshift_bin"]:
            if self.cosmo["Dz"] is None or self.cosmo["fz"] is None:
                raise Exception("You asked to account the galaxy counts distribution. Please specify 'Dz' and 'fz'. ")

            if self.config["skycut"] == 1:
                if len(self.cosmo["Dz"]) != len(self.config["zz"]) or len(self.cosmo["fz"]) != len(self.config["zz"]):
                    raise Exception("Please specify 'Dz' and 'fz' with same length as 'zz'. ")
            elif len(self.cosmo["Dz"]) != self.config["skycut"] or len(self.cosmo["fz"]) != self.config["skycut"]:
                raise Exception("Please specify (in lists) as many 'Dz' and 'fz' as the corresponding skycuts.")

        if self.config["with_nonequal_time"]:
            if (
                self.cosmo["D1"] is None
                or self.cosmo["D2"] is None
                or self.cosmo["f1"] is None
                or self.cosmo["f2"] is None
            ):
                raise Exception("You asked nonequal time correlator. Pleas specify: 'D1', 'D2', 'f1', 'f2'.  ")

    # def __is_bias_conflict(self, bias=None):  # rewrite this...

    #     ###raise Exception("Input error in \'%s\'; input configs: %s. Check Correlator.info() in any doubt." % ())

    #     if bias is not None:
    #         self.cosmo["bias"] = bias

    #     if self.cosmo["bias"] is None:
    #         raise Exception("Please specify 'bias'. ")
    #     if isinstance(self.cosmo["bias"], (list, np.ndarray)):
    #         self.cosmo["bias"] = self.cosmo["bias"][0]
    #     if not isinstance(self.cosmo["bias"], dict):
    #         raise Exception("Please specify bias in a dict. ")

    #     if "bm" in self.config["output"]:  # redshift halo - real-space matter
    #         if not self.config["with_stoch"]:
    #             if self.config["multipole"] == 0:
    #                 if len(self.cosmo["bias"]) is not 5:
    #                     raise Exception("Please specify a dict of 5 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct' \}. ")
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": 0.0,
    #                         "cr2": 0.0,
    #                         "ce0": 0.0,
    #                         "ce1": 0.0,
    #                         "ce2": 0.0,
    #                     }
    #             elif self.config["multipole"] == 2 or self.config["multipole"] == 3:
    #                 if len(self.cosmo["bias"]) is not 6:
    #                     raise Exception(
    #                         "Please specify a dict of 6 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'cr1' \} "
    #                     )
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": self.cosmo["bias"]["cr1"],
    #                         "cr2": 0.0,
    #                         "ce0": 0.0,
    #                         "ce1": 0.0,
    #                         "ce2": 0.0,
    #                     }

    #     # if "bm" in self.config["output"]: # redshift halo - redshift matter
    #     #     if not self.config["with_stoch"]:
    #     #         if self.config["multipole"] == 0:
    #     #             if len(self.cosmo["bias"]) is not 6: raise Exception("Please specify a dict of 6 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\' + matter counterterm: \'dct\' \}. ")
    #     #             else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": 0., "cr2": 0., "ce0": 0., "ce1": 0., "ce2": 0., "dct": self.cosmo["bias"]["dct"], "dr1": 0., "dr2": 0. }
    #     #         elif self.config["multipole"] == 2:
    #     #             if len(self.cosmo["bias"]) is not 8: raise Exception("Please specify a dict of 8 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\' + matter counterterms: \'dct\', \'dr1\' \}. ")
    #     #             else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": 0., "ce0": 0., "ce1": 0., "ce2": 0., "dct": self.cosmo["bias"]["dct"], "dr1": self.cosmo["bias"]["dr1"], "dr2": 0. }
    #     #         elif self.config["multipole"] == 3:
    #     #             if len(self.cosmo["bias"]) is not 10: raise Exception("Please specify a dict of 10 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\', \'cr2\' + matter counterterms: \'dct\', \'dr1\', \'dr2\' \}. ")
    #     #             else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": self.cosmo["bias"]["cr2"], "ce0": 0., "ce1": 0., "ce2": 0., "dct": self.cosmo["bias"]["dct"], "dr1": self.cosmo["bias"]["dr1"], "dr2": self.cosmo["bias"]["dr2"] }
    #     #     else:
    #     #         pass # to code up

    #     elif "m" in self.config["output"]:
    #         if self.config["multipole"] == 0:
    #             if len(self.cosmo["bias"]) is not 1:
    #                 raise Exception("Please specify a dict of 1 bias: \{ 'cct' \}. ")
    #             else:
    #                 self.bias = {
    #                     "b1": 1.0,
    #                     "b2": 1.0,
    #                     "b3": 1.0,
    #                     "b4": 0.0,
    #                     "cct": self.cosmo["bias"]["cct"],
    #                     "cr1": 0.0,
    #                     "cr2": 0.0,
    #                     "ce0": 0.0,
    #                     "ce1": 0.0,
    #                     "ce2": 0.0,
    #                 }
    #         elif self.config["multipole"] == 2:
    #             if len(self.cosmo["bias"]) is not 2:
    #                 raise Exception("Please specify a dict of 2 biases: \{ 'cct', 'cr1' \}. ")
    #             else:
    #                 self.bias = {
    #                     "b1": 1.0,
    #                     "b2": 1.0,
    #                     "b3": 1.0,
    #                     "b4": 0.0,
    #                     "cct": self.cosmo["bias"]["cct"],
    #                     "cr1": self.cosmo["bias"]["cr1"],
    #                     "cr2": 0.0,
    #                     "ce0": 0.0,
    #                     "ce1": 0.0,
    #                     "ce2": 0.0,
    #                 }
    #         elif self.config["multipole"] == 3:
    #             if len(self.cosmo["bias"]) is not 3:
    #                 raise Exception("Please specify a dict of 3 biases: \{ 'cct', 'cr1', 'cr2' \}. ")
    #             else:
    #                 self.bias = {
    #                     "b1": 1.0,
    #                     "b2": 1.0,
    #                     "b3": 1.0,
    #                     "b4": 0.0,
    #                     "cct": self.cosmo["bias"]["cct"],
    #                     "cr1": self.cosmo["bias"]["cr1"],
    #                     "cr2": self.cosmo["bias"]["cr2"],
    #                     "ce0": 0.0,
    #                     "ce1": 0.0,
    #                     "ce2": 0.0,
    #                 }

    #     else:

    #         Nextra = 0
    #         if self.config["with_nnlo_counterterm"]:
    #             Nextra += self.config["multipole"]
    #         if self.config["with_nnlo_higher_derivative"]:
    #             Nextra += self.config["multipole"]
    #         if self.config["with_tidal_alignments"]:
    #             Nextra += 1

    #         if not self.config["with_stoch"]:
    #             if self.config["multipole"] == 0:
    #                 if len(self.cosmo["bias"]) is not 5 + Nextra:
    #                     raise Exception("Please specify a dict of 5 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct' \}. ")
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": 0.0,
    #                         "cr2": 0.0,
    #                         "ce0": 0.0,
    #                         "ce1": 0.0,
    #                         "ce2": 0.0,
    #                     }
    #             elif self.config["multipole"] == 2:
    #                 if len(self.cosmo["bias"]) is not 6 + Nextra:
    #                     raise Exception(
    #                         "Please specify a dict of 6 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'cr1' \}. "
    #                     )
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": self.cosmo["bias"]["cr1"],
    #                         "cr2": 0.0,
    #                         "ce0": 0.0,
    #                         "ce1": 0.0,
    #                         "ce2": 0.0,
    #                     }
    #             elif self.config["multipole"] == 3:
    #                 if len(self.cosmo["bias"]) is not 7 + Nextra:
    #                     raise Exception(
    #                         "Please specify a dict of 7 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'cr1', 'cr2' \}. "
    #                     )
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": self.cosmo["bias"]["cr1"],
    #                         "cr2": self.cosmo["bias"]["cr2"],
    #                         "ce0": 0.0,
    #                         "ce1": 0.0,
    #                         "ce2": 0.0,
    #                     }
    #         else:
    #             if self.config["multipole"] == 0:
    #                 if len(self.cosmo["bias"]) is not 6 + Nextra:
    #                     raise Exception(
    #                         "Please specify a dict of 6 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'ce0' \}. "
    #                     )
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": 0.0,
    #                         "cr2": 0.0,
    #                         "ce0": self.cosmo["bias"]["ce0"],
    #                         "ce1": 0.0,
    #                         "ce2": 0.0,
    #                     }
    #             elif self.config["multipole"] == 2:
    #                 if len(self.cosmo["bias"]) is not 9 + Nextra:
    #                     raise Exception(
    #                         "Please specify a dict of 9 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'cr1', 'ce0', 'ce1', 'ce2'  \}. "
    #                     )
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": self.cosmo["bias"]["cr1"],
    #                         "cr2": 0.0,
    #                         "ce0": self.cosmo["bias"]["ce0"],
    #                         "ce1": self.cosmo["bias"]["ce1"],
    #                         "ce2": self.cosmo["bias"]["ce2"],
    #                     }
    #             elif self.config["multipole"] == 3:
    #                 if len(self.cosmo["bias"]) is not 10 + Nextra:
    #                     raise Exception(
    #                         "Please specify a dict of 10 biases: \{ 'b1', 'b2', 'b3', 'b4', 'cct', 'cr1', 'cr2', 'ce0', 'ce1', 'ce2' \}. "
    #                     )
    #                 else:
    #                     self.bias = {
    #                         "b1": self.cosmo["bias"]["b1"],
    #                         "b2": self.cosmo["bias"]["b2"],
    #                         "b3": self.cosmo["bias"]["b3"],
    #                         "b4": self.cosmo["bias"]["b4"],
    #                         "cct": self.cosmo["bias"]["cct"],
    #                         "cr1": self.cosmo["bias"]["cr1"],
    #                         "cr2": self.cosmo["bias"]["cr2"],
    #                         "ce0": self.cosmo["bias"]["ce0"],
    #                         "ce1": self.cosmo["bias"]["ce1"],
    #                         "ce2": self.cosmo["bias"]["ce2"],
    #                     }

    #         if self.config["with_nnlo_counterterm"]:
    #             try:
    #                 self.bias["cnnlo_l0"] = self.cosmo["bias"]["cnnlo_l0"]
    #                 self.bias["cnnlo_l2"] = self.cosmo["bias"]["cnnlo_l2"]
    #                 if self.config["multipole"] == 3:
    #                     self.bias["cnnlo_l4"] = self.cosmo["bias"]["cnnlo_l4"]
    #             except:
    #                 raise Exception(
    #                     "Please specify the next-to-next-to-leading counterterm coefficients 'cnnlo_l0', 'cnnlo_l2', ...  "
    #                 )

    #         if self.config["with_nnlo_higher_derivative"]:
    #             try:
    #                 self.bias["bnnlo_l0"] = self.cosmo["bias"]["bnnlo_l0"]
    #                 self.bias["bnnlo_l2"] = self.cosmo["bias"]["bnnlo_l2"]
    #                 if self.config["multipole"] == 3:
    #                     self.bias["bnnlo_l4"] = self.cosmo["bias"]["bnnlo_l4"]
    #             except:
    #                 raise Exception(
    #                     "Please specify the next-to-next-to-leading higher-derivative biases 'bnnlo_l0', 'bnnlo_l2', ...  "
    #                 )

    #         if self.config["with_tidal_alignments"]:
    #             try:
    #                 self.bias["bq"] = self.cosmo["bias"]["bq"]
    #             except:
    #                 raise Exception("Please specify the tidal alignments bias 'bq'.  ")
    #         else:
    #             self.bias["bq"] = 0.0  # enforced for marg
    
    def __is_bias_conflict(self, bias=None):
        if bias is not None: self.cosmo["bias"] = bias
        if self.cosmo["bias"] is None: raise Exception("Please specify \'bias\'. ")
        if isinstance(self.cosmo["bias"], (list, np.ndarray)): self.cosmo["bias"] = self.cosmo["bias"][0]
        if not isinstance(self.cosmo["bias"], dict): raise Exception("Please specify bias in a dict. ")
    
        for p in self.eft_parameters_list:
            if p not in self.cosmo["bias"]:
                raise Exception ("%s not found, please provide (given command \'eft_basis\': \'%s\') %s" % (p, self.config["eft_basis"], self.eft_parameters_list))
    
        # PZ: here I should auto-fill the EFT parameters for all output options!!!
    
        self.bias = self.cosmo["bias"]
    
        if "b" in self.config["output"]:
            if "westcoast" in self.config["eft_basis"]:
                self.bias["b2"] = 2.**-.5 * (self.bias["c2"] + self.bias["c4"])
                self.bias["b4"] = 2.**-.5 * (self.bias["c2"] - self.bias["c4"])
            elif "eastcoast" in self.config["eft_basis"]:
                self.bias["b2"] = self.bias["b1"] + 7/2. * self.bias["bG2"]
                self.bias["b3"] = self.bias["b1"] + 15. * self.bias["bG2"] + 6. * self.bias["bGamma3"]
                self.bias["b4"] = 1/2. * self.bias["bt2"] - 7/2. * self.bias["bG2"]
        elif "m" in self.config["output"]: self.bias.update({"b1": 1., "b2": 1., "b3": 1., "b4": 0.})
        if self.config["multipole"] == 0: self.bias.update({"cr1": 0., "cr2": 0.})

    def __set_eft_parameters_list(self):
    
        if self.config["eft_basis"] in ["eftoflss", "westcoast"]:
            self.gauss_eft_parameters_list = ['cct']
            if self.config["multipole"] >= 2: self.gauss_eft_parameters_list.extend(['cr1', 'cr2'])
        elif self.config["eft_basis"] == "eastcoast":
            self.gauss_eft_parameters_list = ['c0']
            if self.config["multipole"] >= 2: self.gauss_eft_parameters_list.extend(['c2', 'c4'])
        if self.config["with_stoch"]: self.gauss_eft_parameters_list.extend(['ce0', 'ce1', 'ce2'])
        if self.config["with_nnlo_counterterm"]:
            if self.config["eft_basis"] in ["eftoflss", "westcoast"]: self.gauss_eft_parameters_list.extend(['cr4', 'cr6'])
            elif self.config["eft_basis"] == "eastcoast": self.gauss_eft_parameters_list.append('ct')
        self.eft_parameters_list = deepcopy(self.gauss_eft_parameters_list)
        if "b" in self.config["output"]:
            if self.config["eft_basis"] in ["eftoflss", "westcoast"]: self.gauss_eft_parameters_list.append('b3')
            elif self.config["eft_basis"] == "eastcoast": self.gauss_eft_parameters_list.append('bGamma3')
            if self.config["eft_basis"] == "eftoflss": self.eft_parameters_list.extend(['b1', 'b2', 'b3', 'b4'])
            elif self.config["eft_basis"] == "westcoast": self.eft_parameters_list.extend(['b1', 'c2', 'b3', 'c4'])
            elif self.config["eft_basis"] == "eastcoast": self.eft_parameters_list.extend(['b1', 'bt2', 'bG2', 'bGamma3'])
            
        if self.config["with_tidal_alignments"]: self.eft_parameters_list.append('bq')

    def __read_config(self, config_dict):

        # Checking if the inputs are consistent with the options
        for (name, config) in zip(self.config_catalog, self.config_catalog.values()):
            for config_key in config_dict:
                if config_key == name:
                    config.check(config_key, config_dict[config_key])
                    is_config = True
            ### Keep this warning for typos in options that are then unread...
            if not is_config:
                raise Exception("%s is not an available configuration option. Please check correlator.info() for help. " % config_key)

        # Setting unspecified configs to default value
        for (name, config) in zip(self.config_catalog, self.config_catalog.values()):
            if config.value is None:
                config.value = config.default

        # Translating the catalog to a dict
        self.config = translate_catalog_to_dict(self.config_catalog)

        self.config["accboost"] = float(self.config["accboost"])

    def __is_config_conflict(self):
        
        # print(self.config["output"])

        if "Cf" in self.config["output"]:
            self.config["with_window"] = False
            self.config["with_cf"] = True
        else:
            self.config["with_cf"] = False

        # if self.config["with_cf"]: self.config["with_stoch"] = False
        # if self.config["wedge"] is not 0: self.config["multipole"] = 3 # enforced

        if "bm" in self.config["output"]:
            self.config["halohalo"] = False
        else:
            self.config["halohalo"] = True

        if self.config["skycut"] > 1:
            self.config["with_bias"] = False
            if len(self.config["z"]) != self.config["skycut"]:
                raise Exception("Please specify as many redshifts 'z' as the number of skycuts.")
                self.config["z"] = np.asarray(self.config["z"])

            def checkEqual(lst):
                return lst[1:] == lst[:-1]  ### TO CHANGE

            if np.all(checkEqual(self.config["z"])):  # if same redshift
                self.config["with_time"] = True
                # self.config["z"] = self.config["z"][0]
            else:
                self.config["with_time"] = False

        if self.config["xdata"] is None:
            raise Exception("Please specify a data point array 'xdata'.")
        if len(self.config["xdata"]) == 1 and isinstance(self.config["xdata"][0], (list, np.ndarray)):
            self.config["xdata"] = self.config["xdata"][0]
        # else:
        #     self.config["xdata"] = np.asarray(self.config["xdata"])

        #     def is_conflict_xdata(xdata):
        #         if "Cf" in self.config["output"]:
        #             if xdata[0] < self.config["smin"] or xdata[-1] > self.config["smax"]:
        #                 raise Exception("Please specify a data point array \'xdata\' in: (%s, %s)." % (self.config["smin"], self.config["smax"]))
        #         else:
        #             if xdata[0] < self.config["kmin"] or xdata[-1] > self.config["kmax"]:
        #                 raise Exception("Please specify a data point array \'xdata\' in: (%s, %s) or increase the kmax." % (self.config["kmin"], self.config["kmax"]))

        #     if self.config["skycut"] == 1:
        #         is_conflict_xdata(self.config["xdata"])
        #     elif self.config["skycut"] > 1:
        #         if len(self.config["xdata"]) == 1: is_conflict_xdata(self.config["xdata"])
        #         if len(self.config["xdata"]) is self.config["skycut"]:
        #             for xi in self.config["xdata"]: is_conflict_xdata(xi)
        #         else:
        #             raise Exception("Please provide a commmon data point array \'xdata\' or as many arrays (in a list) as the corresponding skycuts.")

        self.config[
            "with_common_nonequal_time"
        ] = False  # this is to pass for the common Class to setup the numbers of loops (22 and 13 gathered by default)

        if self.config["with_nonequal_time"]:

            self.config[
                "with_common_nonequal_time"
            ] = True  # this is to pass for the common Class to setup the numbers of loops (22 and 13 seperated since they have different time dependence)

            if self.config["skycut"] > 1:
                raise Exception("Nonequal time correlator available only for skycut = 1. ")
            try:
                self.config["z1"]
                self.config["z2"]
            except:
                print("Please specify 'z1' and 'z2' for nonequaltime correlator. ")

            self.config["with_time"] = False
            self.config["with_bias"] = False

        if self.config["with_AP"]:
            if self.config["DA_AP"] is None:
                raise Exception("You asked to apply the AP effect. Please specify 'DA_AP'.")
            if self.config["H_AP"] is None:
                raise Exception("You asked to apply the AP effect. Please specify 'H_AP'.")
            if self.config["skycut"] == 1:
                if isinstance(self.config["DA_AP"], list):
                    self.config["DA_AP"] = self.config["DA_AP"][0]
                if isinstance(self.config["H_AP"], list):
                    self.config["H_AP"] = self.config["H_AP"][0]

        if self.config["with_window"]:

            def is_conflict_window(windowCf):
                try:
                    test = np.loadtxt(windowCf)
                    if self.config["with_cf"]:
                        self.windowPk = None
                except IOError:
                    print(
                        "You asked to apply a mask. Please specify a correct path to the configuration space window file."
                    )
                    raise

            if self.config["skycut"] == 1:
                if isinstance(self.config["windowCf"], list):
                    self.config["windowCf"] = self.config["windowCf"][0]
                if isinstance(self.config["windowPk"], list):
                    self.config["windowPk"] = self.config["windowPk"][0]
                is_conflict_window(self.config["windowCf"])
            elif self.config["skycut"] > 1:
                for windowCf in self.config["windowCf"]:
                    if windowCf is not None:
                        is_conflict_window(windowCf)
        else:
            self.config["windowPk"] = None
            self.config["windowCf"] = None

        if self.config["with_redshift_bin"]:
            self.config["with_bias"] = False
            self.config["with_time"] = False
            self.config[
                "with_cf"
            ] = True  # even for the Pk, we first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            # self.config["with_common_nonequal_time"] = True # approximating 13 and 22 loop time to be the same, see Projection.redshift()

            def is_conflict_zz(zz, nz):
                if zz is None or nz is None:
                    pass  # raise Exception("You asked to account for the galaxy counts distribution over a redshift bins. Please provide a distribution \'nz\' and corresponding \'zz\'. ")
                elif len(zz) != len(nz):
                    raise Exception("Please provide 'nz' and corresponding 'zz' of the same length. ")

            if self.config["skycut"] == 1:
                is_conflict_zz(self.config["zz"], self.config["nz"])
            elif self.config["skycut"] > 1:
                self.config["zz"] = np.asarray(self.config["zz"])
                self.config["nz"] = np.asarray(self.config["nz"])
                # print (len(self.config["zz"]), len(self.config["nz"]))
                if len(self.config["zz"]) == self.config["skycut"] and len(self.config["nz"]) == self.config["skycut"]:
                    for zz, nz in zip(self.config["zz"], self.config["nz"]):
                        is_conflict_zz(zz, nz)
                else:
                    raise Exception(
                        "Please provide as many 'nz' with corresponding 'zz' (in a list) as the corresponding skycuts. "
                    )
        else:
            self.config["zz"] = None
            self.config["nz"] = None

        # if self.config["with_quintessence"]: self.config["with_exact_time"] = True

    def setcosmo(self, cosmo_dict, module="class"):

        if module is "class":

            from classy import Class

            # Not sure this is useful: does class read z_max_pk?
            if self.config["skycut"] == 1:
                if self.config["with_redshift_bin"]:
                    zmax = max(self.config["zz"])
                else:
                    zmax = self.config["z"]
            elif self.config["skycut"] > 1:
                if self.config["with_redshift_bin"]:
                    maxbin = np.argmax(self.config["z"])
                    zmax = max(self.config["zz"][maxbin])
                else:
                    zmax = max(self.config["z"])

            cosmo_dict_local = cosmo_dict.copy()
            if self.config["with_bias"]:
                del cosmo_dict_local["bias"]  # Class does not like dictionary with keys other than the ones it reads...

            M = Class()
            M.set(cosmo_dict_local)
            M.set({"output": "mPk", "P_k_max_h/Mpc": 100.0, "z_max_pk": zmax})
            M.compute()

            cosmo = {}

            if self.config["with_bias"]:
                try:
                    cosmo["bias"] = cosmo_dict["bias"]
                except:
                    print("Please specify 'bias'.")
                    raise

            if self.config["skycut"] == 1:
                zfid = self.config["z"]
            elif self.config["skycut"] > 1:
                zfid = self.config["z"][self.config["skycut"] // 2]

            cosmo["k11"] = np.logspace(-5, 100.0, 2000)  # k in h/Mpc
            cosmo["P11"] = np.array([M.pk_lin(k * M.h(), zfid) * M.h() ** 3 for k in cosmo["k11"]])  # P(k) in (Mpc/h)**3

            if self.config["skycut"] == 1:
                if self.config["multipole"] is not 0:
                    cosmo["f"] = M.scale_independent_growth_factor_f(self.config["z"])
                if self.config["with_nonequal_time"]:
                    cosmo["D"] = M.scale_independent_growth_factor(self.config["z"])
                    cosmo["D1"] = M.scale_independent_growth_factor(self.config["z1"])
                    cosmo["D2"] = M.scale_independent_growth_factor(self.config["z2"])
                    cosmo["f1"] = M.scale_independent_growth_factor_f(self.config["z1"])
                    cosmo["f2"] = M.scale_independent_growth_factor_f(self.config["z2"])
                if self.config["with_exact_time"] or self.config["with_quintessence"]:
                    cosmo["z"] = self.config["z"]
                    cosmo["Omega0_m"] = M.Omega0_m()
                    try:
                        cosmo["w0_fld"] = cosmo_dict["w0_fld"]
                    except:
                        pass
                if self.config["with_AP"]:
                    cosmo["DA"] = M.angular_distance(self.config["z"]) * M.Hubble(0.0)
                    cosmo["H"] = M.Hubble(self.config["z"]) / M.Hubble(0.0)

            elif self.config["skycut"] > 1:
                if self.config["multipole"] is not 0:
                    cosmo["f"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["z"]])
                cosmo["D"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["z"]])
                if self.config["with_AP"]:
                    cosmo["DA"] = np.array([M.angular_distance(z) * M.Hubble(0.0) for z in self.config["z"]])
                    cosmo["H"] = np.array([M.Hubble(z) / M.Hubble(0.0) for z in self.config["z"]])

            if self.config["with_redshift_bin"]:

                def comoving_distance(z):
                    return M.angular_distance(z) * (1 + z) * M.h()

                if self.config["skycut"] == 1:
                    cosmo["D"] = M.scale_independent_growth_factor(self.config["z"])
                    cosmo["Dz"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["zz"]])
                    cosmo["fz"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["zz"]])
                    cosmo["rz"] = np.array([comoving_distance(z) for z in self.config["zz"]])

                elif self.config["skycut"] > 1:
                    cosmo["Dz"] = np.array(
                        [[M.scale_independent_growth_factor(z) for z in zz] for zz in self.config["zz"]]
                    )
                    cosmo["fz"] = np.array(
                        [[M.scale_independent_growth_factor_f(z) for z in zz] for zz in self.config["zz"]]
                    )
                    cosmo["rz"] = np.array([[comoving_distance(z) for z in zz] for zz in self.config["zz"]])

            if self.config["with_quintessence"]:
                # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum.
                # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
                # This does not work for multi skycuts nor 'with_redshift_bin': True. # eventually to code up
                zm = 5.0  # z in matter domination

                def scale_factor(z):
                    return 1 / (1.0 + z)

                Omega0_m = cosmo["Omega0_m"]
                w = cosmo["w0_fld"]
                GF = GreenFunction(Omega0_m, w=w, quintessence=True)
                Dq = GF.D(scale_factor(zfid)) / GF.D(scale_factor(zm))
                Dm = M.scale_independent_growth_factor(zfid) / M.scale_independent_growth_factor(zm)
                cosmo["P11"] *= (
                    Dq ** 2
                    / Dm ** 2
                    * (1 + (1 + w) / (1.0 - 3 * w) * (1 - Omega0_m) / Omega0_m * (1 + zm) ** (3 * w)) ** 2
                )  # 1611.07966 eq. (4.15)
                cosmo["f"] = GF.fplus(1 / (1.0 + cosmo["z"]))

            if self.config["with_nnlo_counterterm"] or self.config["with_nnlo_higher_derivative"]:
                EH_dict = {
                    "Omega0_b": M.Omega_b(),
                    "Omega0_m": M.Omega0_m(),
                    "h": M.h(),
                    "A_s": M.get_current_derived_parameters(["A_s"]),
                    "n_s": M.n_s(),
                    "T_cmb": M.T_cmb(),
                    "D": M.scale_independent_growth_factor(self.config["z"]),
                }
                cosmo["EH"] = EH_dict
            def get_smooth_wiggle_resc(kk, pk, alpha_rs=1.): # k [h/Mpc], pk [(Mpc/h)**3]
                from scipy.fftpack import dst
                kp = np.linspace(1.e-7, 7, 2**16)   # 1/Mpc
                ilogpk = interp1d(np.log(kk * M.h()), np.log(pk / M.h()**3), fill_value="extrapolate") # Mpc**3
                lnkpk = np.log(kp) + ilogpk(np.log(kp))
                harmonics = dst(lnkpk, type=2, norm='ortho')
                odd, even = harmonics[::2], harmonics[1::2]
                nn = np.arange(0, odd.shape[0], 1)
                nobao = np.delete(nn, np.arange(120, 240,1))
                smooth_odd = interp1d(nn, odd, kind='cubic')(nobao)
                smooth_even = interp1d(nn, even, kind='cubic')(nobao)
                smooth_odd = interp1d(nobao, smooth_odd, kind='cubic')(nn)
                smooth_even = interp1d(nobao, smooth_even, kind='cubic')(nn)
                smooth_harmonics =  np.array([[o, e] for (o, e) in zip(smooth_odd, smooth_even)]).reshape(-1)
                smooth_lnkpk = dst(smooth_harmonics, type=3, norm='ortho')
                smooth_pk = np.exp(smooth_lnkpk) / kp
                wiggle_pk = np.exp(ilogpk(np.log(kp))) - smooth_pk
                spk = interp1d(kp, smooth_pk, bounds_error=False)(kk * M.h()) * M.h()**3 # (Mpc/h)**3
                wpk_resc = interp1d(kp, wiggle_pk, bounds_error=False)(alpha_rs * kk * M.h()) * M.h()**3 # (Mpc/h)**3 # wiggle rescaling
                kmask = np.where(kk < 1.02)[0]
                return kk[kmask], spk[kmask], pk[kmask] #spk[kmask]+wpk_resc[kmask]
            # print(cosmo["Omega0_m"], cosmo["z"])
            return cosmo
        
        # wiggle-no-wiggle split # algo: 1003.3999; details: 2004.10607
        
    
        if self.c["with_nnlo_counterterm"]: cosmo["kk"], cosmo["Psmooth"], cosmo["pk_lin"] = get_smooth_wiggle_resc(cosmo["kk"], cosmo["pk_lin"])
    
        return cosmo
        
    def setPS_SF(self, init=False, factor_m=None):
        
        if self.config["skycut"] == 1:
            # start = time.time()
            self.bird = Bird(
                self.cosmo,
                with_bias=self.config["with_bias"],
                with_stoch=self.config["with_stoch"],
                with_nnlo_counterterm=self.config["with_nnlo_counterterm"],
                co=self.co,
            )
            self.nonlinear.PsCf(self.bird)
            if self.config["with_bias"]:
                self.bird.setPsCf(self.bias)
            else:
                self.bird.setPsCfl()
            if self.config["with_nonequal_time"]:
                self.bird.settime(self.cosmo)  # set D1*D2 / D1**2*D2**2 / 0.5 (D1**2*D2 + D2**2*D1) on 11 / 2    2 / 13   
            # end = time.time()
            # print(end - start)
            # start = time.time()
            if self.config["with_resum"]:
                if self.config["with_cf"]:
                    self.resum.PsCf(self.bird)
                else:
                    self.resum.Ps(self.bird, makeIR=True, makeQ=False, setPs=False, init=True)
                        
            # end = time.time()
            # print(end - start)
                        
        elif self.config["skycut"] > 1:
            if self.config["with_time"]:  # if all skycuts have same redshift
                cosmoi = deepcopy(self.cosmo)
                cosmoi["f"] = self.cosmo["f"][0]
                cosmoi["D"] = self.cosmo["D"][0]
                cosmoi["z"] = self.config["z"][0]
                if self.config["with_AP"]:
                    cosmoi["DA"] = self.cosmo["DA"][0]
                    cosmoi["H"] = self.cosmo["H"][0]
                self.bird = Bird(
                    cosmoi,
                    with_bias=False,
                    with_stoch=self.config["with_stoch"],
                    with_nnlo_counterterm=self.config["with_nnlo_counterterm"],
                    co=self.co,
                )
                self.nonlinear.PsCf(self.bird)
                self.bird.setPsCfl()
                if self.config["with_resum"]:
                    if self.config["with_cf"]:
                        self.resum.PsCf(self.bird)
                    else:
                        self.resum.Ps(self.bird, makeIR=True, makeQ=False, setPs=False, init=True)
        else:
            self.birds = []
            cosmoi = deepcopy(self.cosmo)
        
            def mycycle(skycut, first=0, L=None):
                if L is None:
                    L = [i for i in range(skycut)]
                if (skycut % 2) == 0:
                    first = skycut // 2
                else:
                    if first == 0:
                        first = (skycut + 1) // 2
                    else:
                        first = skycut // 2
                return [item for i, item in enumerate(L + L) if i < skycut + first and first <= i]
        
            zbins = mycycle(self.config["skycut"], first=2)  # cycle to get the middle redshift
        
            for i in zbins:
                cosmoi["f"], cosmoi["D"], cosmoi["z"] = self.cosmo["f"][i], self.cosmo["D"][i], self.config["z"][i]
                if self.config["with_AP"]:
                    cosmoi["DA"], cosmoi["H"] = self.cosmo["DA"][i], self.cosmo["H"][i]
        
                if i == zbins[0]:
                    self.bird = Bird(
                        cosmoi,
                        with_bias=False,
                        with_stoch=self.config["with_stoch"],
                        with_nnlo_counterterm=self.config["with_nnlo_counterterm"],
                        co=self.co,
                    )
                    self.nonlinear.PsCf(self.bird)
                    self.bird.setPsCfl()
                    if self.config["with_resum"]:
                        self.resum.Ps(self.bird, makeIR=True, makeQ=False, setPs=False, init=True)
                    self.birds.append(self.bird)
                else:
                    birdi = deepcopy(self.bird)
                    birdi.settime(cosmoi)  # set new cosmo (in particular, f), and rescale by (Dnew/Dold)**(2    p)
                    if self.config["with_resum"]:
                        if self.config["with_cf"]:
                            self.resum.PsCf(birdi, makeIR=False, makeQ=True, setPs=False, setCf=True)
            if self.config["with_resum"]:
                if self.config["with_cf"]:
                    self.resum.PsCf(self.birds[0], makeIR=False, makeQ=False, setPs=False, setCf=True)
        
            self.birds = mycycle(self.config["skycut"], first=0, L=self.birds)  # cycle back the birds
                
    def setShapefit_full(self, Plin, factor_m = None, kmode=None, factor_a = 0.6, factor_kp = 0.03, init = False, redindex = 0, sigma8_ratio = None):
        if kmode is None:
            kmode = self.correlator.co.k
        if sigma8_ratio is None:
            sigma8_ratio = 1.0
            
        if init == True:
            num = 21
            factor_m = np.linspace(-0.15, 0.15, num)
            # template = np.load('../fitting_codes/template.npy')
            # num_f = 3
            # num_a = 3
            # num_kp = 3
            # num_skycut = np.int32(self.config['skycut'])
            # factor_m = np.linspace(-0.15, 0.15, num_f)
            # factor_a = np.linspace(0.1, 0.9, num_a)
            # factor_kp = np.linspace(0.01, 0.09, num_kp)
            
            # grid = np.concatenate((factor_m.reshape(1, -1), factor_a.reshape(1, -1), factor_kp.reshape(1, -1)), axis = 0)
            
            IRPs11_all = []
            IRPsct_all = []
            IRPsloop_all = []
            P11l_all = []
            Pctl_all = []
            Ploopl_all = []
            # scale = []
            # sigma8_fid = np.trapz(Plin*kmode**2*(3*(np.sin(kmode*8)-kmode*8*np.cos(kmode*8))/(kmode*8)**3)**2/(2.0*np.pi**2), x=kmode)
            if self.config['skycut'] > 1:
                for i in range(len(self.config['skycut'])):
                    IRPs11_all.append([])
                    IRPsct_all.append([])
                    IRPsloop_all.append([])
                    P11l_all.append([])
                    Pctl_all.append([])
                    Ploopl_all.append([])
                    # scale.append([])
            
            # index_f = 0
            # index_a = 0
            # index_kp = 0
            # index_skycut = 0
            # IRPs11_all = np.zeros(shape=(num_skycut, num_f, num_a, num_kp, self.co.Nl, self.co.Nn, self.co.Nk))
            # IRPsct_all = np.zeros(shape=(num_skycut, num_f, num_a, num_kp, self.co.Nl, self.co.Nn, self.co.Nk))
            # IRPsloop_all = np.zeros(shape=(num_skycut, num_f, num_a, num_kp, self.co.Nl, self.co.Nloop, self.co.Nn, self.co.Nk))
            # Ploopl_all = np.zeros(shape=(num_skycut, num_f, num_a, num_kp, self.co.Nl, self.co.Nloop, self.co.Nk))
            # P11l_all = np.empty(shape=(num_skycut, num_f, num_a, num_kp, self.co.Nl, self.co.N11, self.co.Nk))
            # Pctl_all = np.empty(shape=(num_skycut, num_f, num_a, num_kp, self.co.Nl, self.co.Nct, self.co.Nk))
            
            for i in range(num):
                m = factor_m[i]
            # for i in range(num_f*num_a*num_kp*num_skycut):
                # m = factor_m[index_f]
                # a = factor_a[index_a]
                # kp = factor_kp[index_kp]
                
                # factor_a = -0.294*m+0.591
                # factor_kp = 0.024*m+0.038
                # factor_a = -0.1889*m+0.5006
                # factor_kp = 0.0259*m+0.0364
                ratio = np.exp(m/factor_a*np.tanh(factor_a*np.log(kmode/factor_kp)))
                # ratio = np.exp(m/a*np.tanh(a*np.log(kmode/kp)))
                # sigma8_new = np.trapz(Plin*ratio*kmode**2*(3*(np.sin(kmode*8)-kmode*8*np.cos(kmode*8))/(kmode*8)**3)**2/(2.0*np.pi**2), x=kmode)
                # ratio = np.exp(m/(factor_a)*np.tanh(factor_a*np.log(kmode/factor_kp)))*np.heaviside(factor_kp-kmode, 0.5) + np.exp(m/(factor_a)*np.arctan(factor_a*np.log(kmode/factor_kp)))*np.heaviside(kmode - factor_kp, 0.5)
                self.cosmo["P11"] = Plin*ratio
                # print(m, np.trapz(self.cosmo["P11"]*kmode**2*(3*(np.sin(kmode*8)-kmode*8*np.cos(kmode*8))/(kmode*8)**3)**2/(2.0*np.pi**2), x=kmode))
                self.setPS_SF(init = init)
                
                # IRPs11_all[index_skycut, index_f, index_a, index_kp] = self.bird.IRPs11
                # IRPsct_all[index_skycut, index_f, index_a, index_kp] = self.bird.IRPsct
                # IRPsloop_all[index_skycut, index_f, index_a, index_kp] = self.bird.IRPsloop
                # P11l_all[index_skycut, index_f, index_a, index_kp] = self.bird.P11l
                # Pctl_all[index_skycut, index_f, index_a, index_kp] = self.bird.Pctl
                # Ploopl_all[index_skycut, index_f, index_a, index_kp] = self.bird.Ploopl
                
                # print(index_skycut, index_kp, index_a, index_f, kp, a, m)
                
                # index_skycut += 1
                # if index_skycut == num_skycut:
                #     index_skycut = 0
                #     index_kp += 1
                
                # if index_kp == num_kp:
                #     index_kp = 0
                #     index_a += 1
                    
                # if index_a == num_a:
                #     index_a = 0
                #     index_f += 1
                    
                if self.config['skycut'] == 1:
                    
                    # scale.append(np.trapz(self.cosmo["P11"]*kmode**2*(3*(np.sin(kmode*8)-kmode*8*np.cos(kmode*8))/(kmode*8)**3)**2/(2.0*np.pi**2), x=kmode))
                    # scale2n = np.concatenate((2 * [self.co.Na * [((scale[i]/sigma8_fid)**2) ** (n + 1)] for n in range(self.co.NIR)]))
                    IRPs11_all.append(self.bird.IRPs11)
                    IRPsct_all.append(self.bird.IRPsct)
                    IRPsloop_all.append(self.bird.IRPsloop)
                    # print(np.min(self.bird.C11), np.max(self.bird.C11), np.min(self.bird.Pin), np.max(self.bird.Pin))
                    # IRPs11_all.append(np.einsum('n, lnk-> lnk', (scale[i]/sigma8_fid)**2*scale2n, self.bird.IRPs11))
                    # IRPsct_all.append(np.einsum('n, lnk-> lnk', (scale[i]/sigma8_fid)**2*scale2n, self.bird.IRPsct))
                    # IRPsloop_all.append(np.einsum('n, lmnk-> lmnk', (scale[i]/sigma8_fid)**4*scale2n, self.bird.IRPsloop))
                    P11l_all.append(self.bird.P11l)
                    Ploopl_all.append(self.bird.Ploopl)
                    Pctl_all.append(self.bird.Pctl)
                    # print(np.sqrt(scale[i]))
                else:
                    for i in range(len(self.config['skycut'])):
                        IRPs11_all[i].append(self.birds[i].IRPs11)
                        IRPsct_all[i].append(self.birds[i].IRPsct)
                        IRPsloop_all[i].append(self.birds[i].IRPsloop)
                        P11l_all[i].append(self.birds[i].P11l)
                        Pctl_all[i].append(self.birds[i].Pctl)
                        Ploopl_all[i].append(self.birds[i].Ploopl)
                        
                print('Finish ' + str(i+1) + ' iteration(s) for m = ' + str(m) + '.')
                
            self.IRPs11_interp = []
            self.IRPsct_interp = []
            self.IRPsloop_interp = []
            self.P11l_interp = []
            self.Pctl_interp = []
            self.Ploopl_interp = []
            if self.config['skycut'] > 1:
                for i in range(len(self.config['skycut'])):
                    self.IRPs11_interp.append([])
                    self.IRPsct_interp.append([])
                    self.IRPsloop_interp.append([])
                    self.P11l_interp.append([])
                    self.Pctl_interp.append([])
                    self.Ploopl_interp.append([])
            
            # for i in range(num_skycut):
                # self.IRPs11_interp.append(RegularGridInterpolator(grid, np.array(IRPs11_all[i]), bounds_error=True))
                # self.IRPsct_interp.append(RegularGridInterpolator(grid, np.array(IRPsct_all[i]), bounds_error=True))
                # self.IRPsloop_interp.append(RegularGridInterpolator(grid, np.array(IRPsloop_all[i]), bounds_error=True))
                # self.P11l_interp.append(RegularGridInterpolator(grid, np.array(P11l_all[i]), bounds_error=True))
                # self.Pctl_interp.append(RegularGridInterpolator(grid, np.array(Pctl_all[i]), bounds_error=True))
                # self.Ploopl_interp.append(RegularGridInterpolator(grid, np.array(Ploopl_all[i]), bounds_error=True))
                
                #         scale[i].append(np.trapz(self.cosmo["P11"]*kmode**2*(3*(np.sin(kmode*8)-kmode*8*np.cos(kmode*8))/(kmode*8)**3)**2/(2.0*np.pi**2), x=kmode))
            
            if self.config['skycut'] == 1:
                # print(np.shape(np.array(IRPs11_all)))
                self.IRPs11_interp = interp1d(factor_m, np.array(IRPs11_all), axis = 0, bounds_error=True)
                self.IRPsct_interp = interp1d(factor_m, np.array(IRPsct_all), axis = 0, bounds_error=True)
                self.IRPsloop_interp = interp1d(factor_m, np.array(IRPsloop_all), axis = 0, bounds_error=True)
                self.P11l_interp = interp1d(factor_m, np.array(P11l_all), axis = 0, bounds_error=True)
                self.Pctl_interp = interp1d(factor_m, np.array(Pctl_all), axis = 0, bounds_error=True)
                self.Ploopl_interp = interp1d(factor_m, np.array(Ploopl_all), axis = 0, bounds_error=True)
            #     self.scale_interp = interp1d(factor_m, np.sqrt(np.array(scale)/scale[np.int32((num-1)/2)]), axis = 0, bounds_error=True)
            else:
                # self.IRPs11_interp = []
                # self.IRPsct_interp = []
                # self.IRPsloop_interp = []
                for i in range(len(self.config['skycut'])):
                    self.IRPs11_interp[i] = interp1d(factor_m, np.array(IRPs11_all[i]), axis = 0, bounds_error=True)
                    self.IRPsct_interp[i] = interp1d(factor_m, np.array(IRPsct_all[i]), axis=0, bounds_error=True)
                    self.IRPsloop_interp[i] = interp1d(factor_m, np.array(IRPsloop_all[i]), axis=0, bounds_error=True)
                    self.P11l_interp[i] = interp1d(factor_m, np.array(P11l_all[i]), axis = 0, bounds_error=True)
                    self.Pctl_interp[i] = interp1d(factor_m, np.array(Pctl_all[i]), axis = 0, bounds_error=True)
                    self.Ploopl_interp[i] = interp1d(factor_m, np.array(Ploopl_all[i]), axis = 0, bounds_error=True)
            #         self.scale_interp[i] = interp1d(factor_m, np.sqrt(np.array(scale[i])/scale[i][np.int32((num-1)/2)]), axis = 0, bounds_error=True)
        else:
            # param = np.array([factor_m, factor_a[0], factor_kp[0]])
            # print(param)
            
            # self.bird.IRPs11 = np.float64(self.IRPs11_interp[redindex](param)[0])
            # self.bird.IRPsct = np.float64(self.IRPsct_interp[redindex](param)[0])
            # self.bird.IRPsloop = np.float64(self.IRPsloop_interp[redindex](param)[0])
            # self.bird.P11l = np.float64(self.P11l_interp[redindex](param)[0])
            # self.bird.Pctl = np.float64(self.Pctl_interp[redindex](param)[0])
            # self.bird.Ploopl = np.float64(self.Ploopl_interp[redindex](param)[0])
            # self.resum.Ps(self.bird, makeIR=False, makeQ=True, setPs=True, init=False)
            
            
            # ratio = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(kmode/factor_kp)))
            # self.setPS_SF(init=False, factor_m = factor_m)
            if self.config['skycut'] == 1:
                self.bird.IRPs11 = self.IRPs11_interp(factor_m)
                self.bird.IRPsct = self.IRPsct_interp(factor_m)
                self.bird.IRPsloop = self.IRPsloop_interp(factor_m)
                
                sigma2n = np.concatenate((2 * [self.co.Na * [sigma8_ratio ** (n + 1)] for n in range(self.co.NIR)]))
                self.bird.IRPs11 = np.einsum("n,lnk->lnk", sigma8_ratio * sigma2n, self.bird.IRPs11)
                self.bird.IRPsct = np.einsum("n,lnk->lnk", sigma8_ratio * sigma2n, self.bird.IRPsct)
                self.bird.IRPsloop = np.einsum("n,lmnk->lmnk", sigma8_ratio ** 2 * sigma2n, self.bird.IRPsloop)
                
                self.bird.P11l = self.P11l_interp(factor_m)
                self.bird.Pctl = self.Pctl_interp(factor_m)
                self.bird.Ploopl = self.Ploopl_interp(factor_m)
                
                self.bird.P11l = self.bird.P11l*sigma8_ratio
                self.bird.Pctl = self.bird.Pctl*sigma8_ratio
                self.bird.Ploopl = self.bird.Ploopl*sigma8_ratio**2
                
                self.resum.Ps(self.bird, makeIR=False, makeQ=True, setPs=True, init=False)
            else:
                self.birds[redindex].IRPs11 = self.IRPs11_interp[redindex](factor_m)
                self.birds[redindex].IRPsct = self.IRPsct_interp[redindex](factor_m)
                self.birds[redindex].IRPsloop = self.IRPsloop_interp[redindex](factor_m)
                self.birds[redindex].P11l = self.P11l_interp(factor_m)
                self.birds[redindex].Pctl = self.Pctl_interp(factor_m)
                self.birds[redindex].Ploopl = self.Ploopl_interp(factor_m)
                self.resum.Ps(self.birds[redindex], makeIR=False, makeQ=True, setPs=True, init=False)
    


class BiasCorrelator(Correlator):
    """
    Class to load pre-computed correlator
    """

    def __init__(self, config_dict=None, load_engines=False):
        Correlator.__init__(self, config_dict, load_engines=load_engines)


def translate_catalog_to_dict(catalog):
    newdict = dict.fromkeys(catalog)
    for key, option in zip(catalog, catalog.values()):
        newdict[key] = option.value
    return newdict


def typename(onetype):
    if isinstance(onetype, tuple):
        return [t.__name__ for t in onetype]
    else:
        return [onetype.__name__]
    


class Option(object):
    def __init__(self, config_name, config_type, config_list=None, description="", default=None, verbose=False):

        self.verbose = verbose
        self.name = config_name
        self.type = config_type
        self.list = config_list
        self.description = description
        self.default = default
        self.value = None

    def check(self, config_key, config_value):
        is_config = False
        if self.verbose:
            print("'%s': '%s'" % (config_key, config_value))
        if isinstance(config_value, self.type):
            if self.list is None:
                is_config = True
            elif isinstance(config_value, str):
                if any(config_value in o for o in self.list):
                    is_config = True
            elif isinstance(config_value, (int, float, bool)):
                if any(config_value == o for o in self.list):
                    is_config = True
        if is_config:
            self.value = config_value
        else:
            self.error()
        return is_config

    def error(self):
        if self.list is None:
            try:
                raise Exception(
                    "Input error in '%s'; input configs: %s. Check Correlator.info() in any doubt."
                    % (self.name, typename(self.type))
                )
            except Exception as e:
                print(e)
        else:
            try:
                raise Exception(
                    "Input error in '%s'; input configs: %s. Check Correlator.info() in any doubt."
                    % (self.name, self.list)
                )
            except Exception as e:
                print(e)
                
class PowerToCorrelation(ABC):
    """Generic class for converting power spectra to correlation functions
    Using a class based method as there might be multiple implementations and
    some of the implementations have state.
    """

    def __init__(self, ell=0):
        self.ell = ell

    def __call__(self, ks, pk, ss):
        """Generates the correlation function
        Parameters
        ----------
        ks : np.ndarray
            The k values for the power spectrum data. *Assumed to be in log space*
        pk : np.ndarray
            The P(k) values
        ss : np.nparray
            The distances to calculate xi(s) at.
        Returns
        -------
        xi : np.ndarray
            The correlation function at the specified distances
        """
        raise NotImplementedError()
    
class PowerToCorrelationSphericalBessel(PowerToCorrelation):
    def __init__(self, qs=None, ell=15, low_ring=True, fourier=True):

        """
        From Stephen Chen. Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
        the untransformed coordinate, up to a given order L in bessel functions (j_l for l
        less than or equal to L. The point is to save time by evaluating the Mellin transforms
        u_m in advance.
        Does not use fftw as in spherical_bessel_transform_fftw.py, which makes it convenient
        to evaluate the generalized correlation functions in qfuncfft, as there aren't as many
        ffts as in LPT modules so time saved by fftw is minimal when accounting for the
        startup time of pyFFTW.
        Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
        with the above modifications.
        Taken from velocileptors.
        """

        if qs is None:
            qs = np.logspace(-4, np.log(5.0), 2000)

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2 * np.pi**2)

        self.q = qs
        self.ell = ell

        self.Nx = len(qs)
        self.Delta = np.log(qs[-1] / qs[0]) / (self.Nx - 1)

        self.N = 2 ** (int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.pads = np.zeros((self.N - self.Nx) // 2)
        self.pad_iis = np.arange(self.Npad - self.Npad // 2, self.N - self.Npad // 2)

        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N // 2 + 1)
        self.ydict = {}
        self.udict = {}
        self.qdict = {}

        if low_ring:
            for ll in range(self.ell + 1):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta / np.pi * np.angle(self.UK(ll, q + 1j * np.pi / self.Delta))  # ln(xmin*ymax)
                ys = np.exp(lnxy - self.Delta) * qs / (qs[0] * qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

                self.ydict[ll] = ys
                self.udict[ll] = us
                self.qdict[ll] = q

        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(self.ell + 1):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0] * qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)

                self.ydict[ll] = ys
                self.udict[ll] = us
                self.qdict[ll] = q

    def __call__(self, ks, fq, ss, damping=0.25, nu=None):
        """
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        """
        if nu is None:
            nu = self.ell

        fq = fq * np.exp(-(ks**2) * damping**2)

        q = self.qdict[nu]
        y = self.ydict[nu]
        f = np.concatenate((self.pads, self.q ** (3 - q) * fq, self.pads))

        fks = np.fft.rfft(f)
        gks = self.udict[nu] * fks
        gs = np.fft.hfft(gks) / self.N

        return np.real((1j) ** nu * splev(ss, splrep(y, y ** (-q) * gs[self.pad_iis])))

    def UK(self, nu, z):
        """
        The Mellin transform of the spherical bessel transform.
        """
        return self.sqrtpi * np.exp(np.log(2) * (z - 2) + loggamma(0.5 * (nu + z)) - loggamma(0.5 * (3 + nu - z)))

    def update_tilt(self, nu, tilt):
        """
        Update the tilt for a particular nu. Assume low ring coordinates.
        """
        q = tilt
        ll = nu

        ms = np.arange(0, self.N // 2 + 1)
        lnxy = self.Delta / np.pi * np.angle(self.UK(ll, q + 1j * np.pi / self.Delta))  # ln(xmin*ymax)
        ys = np.exp(lnxy - self.Delta) * self.q / (self.q[0] * self.q[-1])
        us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

        self.ydict[ll] = ys
        self.udict[ll] = us
        self.qdict[ll] = q

    def loginterp(
        x,
        y,
        yint=None,
        side="both",
        lorder=9,
        rorder=9,
        lp=1,
        rp=-2,
        ldx=1e-6,
        rdx=1e-6,
        interp_min=-12,
        interp_max=12,
        Nint=10**5,
        verbose=False,
        option="B",
    ):
        """
        Extrapolate function by evaluating a log-index of left & right side.
        From Chirag Modi's CLEFT code at
        https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
        The warning for divergent power laws on both ends is turned off. To turn back on uncomment lines 26-33.
        """

        if yint is None:
            yint = InterpolatedUnivariateSpline(x, y, k=5)
        if side == "both":
            side = "lr"

        # Make sure there is no zero crossing between the edge points
        # If so assume there can't be another crossing nearby

        if np.sign(y[lp]) == np.sign(y[lp - 1]) and np.sign(y[lp]) == np.sign(y[lp + 1]):
            l = lp
        else:
            l = lp + 2

        if np.sign(y[rp]) == np.sign(y[rp - 1]) and np.sign(y[rp]) == np.sign(y[rp + 1]):
            r = rp
        else:
            r = rp - 2

        lneff = derivative(yint, x[l], dx=x[l] * ldx, order=lorder) * x[l] / y[l]
        rneff = derivative(yint, x[r], dx=x[r] * rdx, order=rorder) * x[r] / y[r]

        # print(lneff, rneff)

        # uncomment if you like warnings.
        # if verbose:
        #    if lneff < 0:
        #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        #        print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
        #    if rneff > 0:
        #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        #        print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

        if option == "A":

            xl = np.logspace(interp_min, np.log10(x[l]), Nint)
            xr = np.logspace(np.log10(x[r]), interp_max, Nint)
            yl = y[l] * (xl / x[l]) ** lneff
            yr = y[r] * (xr / x[r]) ** rneff
            # print(xr/x[r])

            xint = x[l + 1 : r].copy()
            yint = y[l + 1 : r].copy()
            if side.find("l") > -1:
                xint = np.concatenate((xl, xint))
                yint = np.concatenate((yl, yint))
            if side.find("r") > -1:
                xint = np.concatenate((xint, xr))
                yint = np.concatenate((yint, yr))
            yint2 = InterpolatedUnivariateSpline(xint, yint, k=5, ext=3)

        else:
            # nan_to_numb is to prevent (xx/x[l/r])^lneff to go to nan on the other side
            # since this value should be zero on the wrong side anyway
            # yint2 = lambda xx: (xx <= x[l]) * y[l]*(xx/x[l])**lneff \
            #                 + (xx >= x[r]) * y[r]*(xx/x[r])**rneff \
            #                 + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k = 5, ext=3)(xx)
            yint2 = (
                lambda xx: (xx <= x[l]) * y[l] * np.nan_to_num((xx / x[l]) ** lneff)
                + (xx >= x[r]) * y[r] * np.nan_to_num((xx / x[r]) ** rneff)
                + (xx > x[l]) * (xx < x[r]) * InterpolatedUnivariateSpline(x, y, k=5, ext=3)(xx)
            )

        return yint2
    