import os
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d

# from . common import Common, co
# from . bird import Bird
# from . nonlinear import NonLinear
# from . resum import Resum
# from . projection import Projection
# from . greenfunction import GreenFunction
# from . fourier import FourierTransform

from common import Common, co
from bird import Bird
from nonlinear import NonLinear
from nnlo import NNLO_higher_derivative, NNLO_counterterm
from resum import Resum
from projection import Projection
from greenfunction import GreenFunction
from fourier import FourierTransform
from eisensteinhu import EisensteinHu

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

# from common import Common, co
# from bird import Bird
# from nonlinear import NonLinear
# from nnlo import NNLO_higher_derivative, NNLO_counterterm
# from resum import Resum
# from projection import Projection
# from greenfunction import GreenFunction
# from fourier import FourierTransform
# from eisensteinhu import EisensteinHu

class Correlator(object):

    def __init__(self, config_dict=None, load_engines=True):

        self.cosmo_catalog = {
            "P11": Option("P11", (list, np.ndarray),
                description="Linear matter power spectrum in [Mpc/h]^3",
                default=None) ,
            "k11": Option("k11", (list, np.ndarray),
                description="k-array in [h/Mpc] on which P11 is evaluated",
                default=None) ,
            "D": Option("D", (float, list, np.ndarray),
                description="Scale independent growth function. To specify if \'skycut\' > 1 or \'with_nonequal_time\' / \'with_redshift_bin\' is True.", 
                default=None) ,
            "f": Option("f", (float, list, np.ndarray),
                description="Scale independent growth rate (for RSD). Automatically set to 0 for \'output\': \'m__\'.", 
                default=None) ,
            "bias": Option("bias", (dict, list, np.ndarray),
                description="EFT parameters in dictionary to specify as \
                    (\'eft_basis\': \'eftoflss\') \{ \'b1\'(a), \'b2\'(a), \'b3\'(a), \'b4\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'westcoast\') \{ \'b1\'(a), \'c2\'(a), \'c4\'(a), \'b3\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'eastcoast\') \{ \'b1\'(a), \'b2\'(a), \'bG2\'(a), \'bgamma3\'(a), \'c0\', \'c2\'(b), \'c4\'(c), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    if (a): \'b\' in \'output\'; (b): \'multipole\'>=2; (d): \'with_stoch\' is True ",
                default=None) ,
            "Omega0_m": Option("Omega0_m", float,
                description="Fractional matter abundance at present time. To specify for exact time dependence.", 
                default=None) ,
            "w0_fld": Option("w0_fld", float,
                description="Dark energy equation of state parameter. To specify for exact time dependence if varied (otherwise w0 = -1).", 
                default=None) ,
            "z": Option("z", (float, list, np.ndarray),
                description="Effective redshift(s). Should match the number of skycuts. To specify for exact time dependence.",
                default=None) ,
            "DA": Option("DA", (float, list, np.ndarray),
                description="Angular distance times H_0. To specify if \'with_AP\' is True.", 
                default=None) ,
            "H": Option("H", (float, list, np.ndarray),
                description="Hubble parameter by H_0. To specify if \'with_AP\' is True.", 
                default=None) ,
            "Dz": Option("Dz", (list, np.ndarray),
                description="Scale independent growth function over redshift bin. To specify if \'with_redshift_bin\' is True.", 
                default=None) ,
            "fz": Option("fz", (list, np.ndarray),
                description="Scale independent growth rate over redshift bin. To specify if \'with_redshift_bin\' is True.", 
                default=None) ,
            "rz": Option("rz", (list, np.ndarray),
                description="Comoving distance in [Mpc/h] over redshift bin. To specify if \'with_redshift_bin\' or if \'output\':\'w\'.", 
                default=None) ,
            "D1": Option("D1", float,
                description="Scale independent growth function at redshift z1. To specify if \'with_nonequal_time\' is True.", 
                default=None) ,
            "D2": Option("D2", float,
                description="Scale independent growth function at redshift z2. To specify if \'with_nonequal_time\' is True.", 
                default=None) ,
            "f1": Option("f1", float,
                description="Scale independent growth rate at redshift z1. To specify if \'with_nonequal_time\' is True.", 
                default=None) ,
            "f2": Option("f2", float,
                description="Scale independent growth rate at redshift z2. To specify if \'with_nonequal_time\' is True.", 
                default=None) ,
            "Psmooth": Option("Psmooth", (list, np.ndarray),
                description="Smooth power spectrum. To specify if \'with_nnlo_counterterm\' is True.", 
                default=None) ,
        }

        self.config_catalog = {
            "output": Option("output", str, ["bPk", "bCf", "mPk", "mCf", "bmPk", "bmCf"], 
                description="Correlator: biased tracers / matter / biased tracers-matter -- power spectrum / correlation function.", 
                default="bPk") ,
            "multipole": Option("multipole", int, [0, 2, 3], 
                description="Number of multipoles. 0: real space. 2: monopole + quadrupole. 3: monopole + quadrupole + hexadecapole.",
                default=2) ,
            "wedge": Option("wedge", int, 
                description="Number of wedges. 0: compute multipole instead. ",
                default=0) ,
            "wedges_bounds": Option("wedges_bounds", (list, np.ndarray), 
                description="Wedges bounds: [0, a_1, ..., a_{n-1}, 1], n: number of wedges, such that 0 < mu < a_1, ..., a_{n-1} < mu < 1. Default: equi-spaced between 0 and 1.",
                default=None) ,
            "skycut": Option("skycut", int, 
                description="Number of skycuts.",
                default=1) ,
            "z": Option("z", (float, list, np.ndarray),
                description="Effective redshift(s). Should match the number of skycuts.",
                default=None) ,
            "km": Option("km", float,
                description="Inverse tracer spatial extension scale in [h/Mpc].",
                default=0.7) ,
            "kr": Option("kr", float,
                description="Inverse velocity product renormalization scale in [h/Mpc].",
                default=0.25) ,
            "nd": Option("nd", float,
                description="Mean galaxy density",
                default=3e-4) ,
            "eft_basis": Option("eft_basis", str,
                description="Basis of EFT parameters: \'eftoflss\' (default), \'westcoast\', or \'eastcoast\'. See cosmology command \'bias\' for more details.",
                default="eftoflss") ,
            "with_stoch": Option("with_stoch", bool, 
                description="With stochastic terms.",
                   default=False) ,
            "with_bias": Option("with_bias", bool, 
                description="Bias (in)dependent evalution. Automatically set to False for \'with_time\': False.",
                   default=False) ,
            "with_time": Option("with_time", bool,
                description="Time (in)dependent evaluation (for multi skycuts / redshift bin). For \'with_redshift_bin\': True, or \'skycut\' > 1, automatically set to False.",
                default=True) ,
            "with_exact_time": Option("with_exact_time", bool,
                description="Exact time dependence or EdS approximation.",
                default=False) ,
            "with_redshift_bin": Option("with_redshift_bin", bool,
                description="Account for the galaxy count distribution over a redshift bin.",
                default=False) ,
            "zz": Option("zz", (list, np.ndarray),
                description="Array of redshift points inside a redshift bin. For multi skycuts, a list of arrays should be provided.",
                default=None) ,
            "nz": Option("nz", (list, np.ndarray),
                description="Galaxy counts distribution over a redshift bin. For multi skycuts, a list of arrays should be provided.",
                default=None) ,
            "kmax": Option("kmax", float,
                description="kmax in [h/Mpc] for \'output\': \'_Pk\'",
                default=0.25) ,
#             "smin": Option("smin", float,
#                 description="smin in [Mpc/h] for \'output\': \'_Cf\'",
#                 default=1.) ,
            "xdata": Option("xdata", (np.ndarray, list),
                description="Array of data points.",
                default=None) ,
            "with_resum": Option("with_resum", bool,
                description="Apply IR-resummation.",
                default=True) ,
            "optiresum": Option("optiresum", bool,
                description="True: Resumming only with the BAO peak. False: Resummation on the full correlation function.",
                default=False) ,
            "with_AP": Option("wity_AP", bool,
                description="Apply Alcock Paczynski effect. Automatically set to False for \'output\': \'w\'.",
                default=False) ,
            "z_AP": Option("z_AP", (float, list, np.ndarray),
                description="Fiducial redshift used to convert coordinates to distances. A list can be provided for multi skycuts. If only one value is passed, use it for all skycuts.",
                default=None) ,
            "Omega_m_AP": Option("Omega_m_AP", (float, list, np.ndarray),
                description="Fiducial matter abundance used to convert coordinates to distances. A list can be provided for multi skycuts. If only one value is passed, use it for all skycuts.",
                default=None) ,
            "with_window": Option("window", bool,
                description="Apply mask. Automatically set to False for \'output\': \'w\' or \'_Cf\'.",
                default=False) ,
            "windowPk": Option("windowPk", (str, list),
                description="Path to Fourier convolution window file for \'output\': \'_Pk\'. If not provided, read \'windowCf\', precompute the Fourier one and save it here.",
                default=None) ,
            "windowCf": Option("windowCf", (str, list),
                description="Path to configuration space window file with columns: s [Mpc/h], Q0, Q2, Q4. A list can be provided for multi skycuts. Put \'None\' for each skycut without window.",
                default=None) ,
            "with_binning": Option("with_binning", bool,
                description="Apply binning for linear-spaced data bins.",
                default=False) ,
            "with_fibercol": Option("with_fibercol", bool,
                description="Apply fiber collision effective window corrections.",
                default=False) ,
            "with_nnlo_counterterm": Option("with_nnlo_counterterm", bool,
                description="With next-to-next-to-leading counterterm k^4 P11.",
                default=False) ,
            "with_tidal_alignments": Option("with_tidal_alignments", bool,
                description="With tidal alignements: bq * (\mu^2 - 1/3) \delta_m ",
                default=False) ,
            "with_quintessence": Option("with_quintessence", bool,
                description="Clustering quintessence.",
                default=False) ,
            "with_nonequal_time": Option("with_nonequal_time", bool,
                description="Non equal time correlator. Automatically set to \'with_time\' to False ",
                default=False) ,
            "z1": Option("z1", (float, list, np.ndarray),
                description="Redshift z_1 for non equal time correlator.",
                default=None) ,
            "z2": Option("z2", (float, list, np.ndarray),
                description="Redshift z_2 for non equal time correlator.",
                default=None) ,
            "accboost": Option("accboost", int, [1, 2, 3],
                description="Sampling accuracy boost.",
                default=1) ,
            "fftbias": Option("fftbias", float,
                description="real power bias for fftlog decomposition of P11 (usually to keep to default value)",
                default=-1.6) ,
            "keep_loop_pieces_independent": Option("keep_loop_pieces_independent", bool,
                description="keep the loop pieces 13 and 22 independent (mainly for debugging)",
                default=False) ,
        }
        
        if config_dict is not None: self.set(config_dict, load_engines=load_engines)
    

    def info(self, description=True):

        for on in ['config', 'cosmo']:

            print ("\n")
            if on is 'config':
                print ("Configuration commands [.set(config_dict)]")
                print ("----------------------")
                catalog = self.config_catalog
            elif on is 'cosmo':
                print ("Cosmology commands [.compute(cosmo_dict)]")
                print ("------------------")
                catalog = self.cosmo_catalog

            for (name, config) in zip(catalog, catalog.values()):
                if config.list is None: print("\'%s\': %s" % (name, typename(config.type)))
                else: print("\'%s\': %s ; options: %s" % (name, typename(config.type), config.list))
                if description:
                    print ('    - %s' % config.description)
                    print ('    * default: %s' % config.default)
    

    def set(self, config_dict, load_engines=True):
        
        # Reading config provided by user
        self.__read_config(config_dict)
        
        # Setting no-optional config
        self.config["smin"] = 1.
        self.config["smax"] = 1000.
        self.config["kmin"] = 0.001
        
        # Checking for config conflict
        self.__is_config_conflict()

        # Setting list of EFT parameters required by the user to provide later
        self.__set_eft_parameters_list()
        
        # Loading PyBird engines
        self.__load_engines(load_engines=load_engines)


    def compute(self, cosmo_dict, module=None): 

        cosmo_dict_local = cosmo_dict.copy()
        
        if module is 'class': 
            cosmo_dict_class = self.setcosmo(cosmo_dict, module='class')
            cosmo_dict_local.update(cosmo_dict_class)
        
        self.__read_cosmo(cosmo_dict_local)
        self.__is_cosmo_conflict()

        if self.config["skycut"] == 1:
            self.bird = Bird(self.cosmo, with_bias=self.config["with_bias"], eft_basis=self.config["eft_basis"], with_stoch=self.config["with_stoch"], with_nnlo_counterterm=self.config["with_nnlo_counterterm"], co=self.co)
            if self.config["with_nnlo_counterterm"]: # we use smooth power spectrum since we don't want spurious BAO signals
                ilogPsmooth = interp1d(np.log(self.bird.kin), np.log(self.cosmo["Psmooth"]), fill_value='extrapolate')
                if self.config["with_cf"]: self.nnlo_counterterm.Cf(self.bird, ilogPsmooth)
                else: self.nnlo_counterterm.Ps(self.bird, ilogPsmooth)
            self.nonlinear.PsCf(self.bird)
            if self.config["with_bias"]: self.bird.setPsCf(self.bias)
            else: self.bird.setPsCfl()
            if self.config["with_nonequal_time"]: self.bird.settime(self.cosmo) # set D1*D2 / D1**2*D2**2 / 0.5 (D1**2*D2 + D2**2*D1) on 11 / 22 / 13
            if self.config["with_resum"]:
                if self.config["with_cf"]: self.resum.PsCf(self.bird)
                else: self.resum.Ps(self.bird)
            if self.config["with_redshift_bin"]: self.projection.redshift(self.bird, self.cosmo["rz"], self.cosmo["Dz"], self.cosmo["fz"], pk=self.config["output"])
            if self.config["with_AP"]: self.projection.AP(self.bird)
            if self.config["with_window"]: self.projection.Window(self.bird)
            if self.config["with_fibercol"]: self.projection.fibcolWindow(self.bird)
            if self.config["wedge"] != 0: self.projection.Wedges(self.bird)
            if self.config["with_binning"]: self.projection.xbinning(self.bird)
            else: self.projection.xdata(self.bird)

        elif self.config["skycut"] > 1:
            if self.config["with_time"]: # if all skycuts have same redshift
                cosmoi = deepcopy(self.cosmo)
                cosmoi["f"] = self.cosmo["f"][0]
                cosmoi["D"] = self.cosmo["D"][0]
                cosmoi["z"] = self.config["z"][0]
                if self.config["with_AP"]: 
                    cosmoi["DA"] = self.cosmo["DA"][0]
                    cosmoi["H"] = self.cosmo["H"][0]
                self.bird = Bird(cosmoi, with_bias=False, eft_basis=self.config["eft_basis"], with_stoch=self.config["with_stoch"], with_nnlo_counterterm=self.config["with_nnlo_counterterm"], co=self.co)
                if self.config["with_nnlo_counterterm"]: # this works only if the skycut has same redshift
                    ilogPsmooth = interp1d(np.log(self.bird.kin), np.log(self.cosmo["Psmooth"]), fill_value='extrapolate')
                    if self.config["with_cf"]: self.nnlo_counterterm.Cf(self.bird, ilogPsmooth)
                    else: self.nnlo_counterterm.Ps(self.bird, ilogPsmooth)
                self.nonlinear.PsCf(self.bird)
                self.bird.setPsCfl()
                if self.config["with_resum"]:
                    if self.config["with_cf"]: self.resum.PsCf(self.bird)
                    else: self.resum.Ps(self.bird)
                self.birds = [deepcopy(self.bird) for i in range(self.config["skycut"])]

            else:
                self.birds = []
                cosmoi = deepcopy(self.cosmo)

                def mycycle(skycut, first=0, L=None):
                    if L is None: L = [i for i in range(skycut)]
                    if (skycut % 2) == 0: 
                        first = skycut//2
                    else:
                        if first == 0: first = (skycut+1)//2
                        else: first = skycut//2
                    return [item for i, item in enumerate(L+L) if i < skycut+first and first <= i]

                zbins = mycycle(self.config["skycut"], first=2) # cycle to get the middle redshift

                for i in zbins:
                    cosmoi["f"], cosmoi["D"], cosmoi["z"] = self.cosmo["f"][i], self.cosmo["D"][i], self.config["z"][i]
                    if self.config["with_AP"]: cosmoi["DA"], cosmoi["H"] = self.cosmo["DA"][i], self.cosmo["H"][i]

                    if i == zbins[0]:
                        self.bird = Bird(cosmoi, with_bias=False, with_stoch=self.config["with_stoch"], with_nnlo_counterterm=self.config["with_nnlo_counterterm"], co=self.co)
                        self.nonlinear.PsCf(self.bird)
                        self.bird.setPsCfl()
                        if self.config["with_resum"]: self.resum.Ps(self.bird, makeIR=True, makeQ=True, setPs=False)
                        self.birds.append(self.bird)
                    else:
                        birdi = deepcopy(self.bird)
                        birdi.settime(cosmoi) # set new cosmo (in particular, f), and rescale by (Dnew/Dold)**(2p)
                        if self.config["with_resum"]:
                            if self.config["with_cf"]: self.resum.PsCf(birdi, makeIR=False, makeQ=True, setPs=False, setCf=True)
                            else: self.resum.Ps(birdi, makeIR=False, makeQ=True, setPs=True)
                        self.birds.append(birdi)
                        
                if self.config["with_resum"]:
                    if self.config["with_cf"]: self.resum.PsCf(self.birds[0], makeIR=False, makeQ=False, setPs=False, setCf=True)
                    else: self.resum.Ps(self.birds[0], makeIR=False, makeQ=False, setPs=True)

                self.birds = mycycle(self.config["skycut"], first=0, L=self.birds) # cycle back the birds 

            for i in range(self.config["skycut"]):
                if self.config["with_redshift_bin"] and self.config["nz"][i] is not None: self.projection[i].redshift(self.birds[i], self.cosmo["rz"][i], self.cosmo["Dz"][i], self.cosmo["fz"][i], pk=self.config["output"])
                if self.config["with_AP"]: self.projection[i].AP(self.birds[i]) 
                if self.config["with_window"]: self.projection[i].Window(self.birds[i])
                if self.config["with_fibercol"]: self.projection[i].fibcolWindow(self.birds[i])
                if self.config["wedge"] != 0: self.projection[i].Wedges(self.birds[i]) 
                if self.config["with_binning"]: self.projection[i].xbinning(self.birds[i])
                else: self.projection[i].xdata(self.birds[i])

    def get(self, bias=None, what="full"):

        if "full" not in what and not self.config["keep_loop_pieces_independent"]:
            raise Exception("If you want to get something else than the full correlator, please set keep_loop_pieces_independent: True")

        if self.config["skycut"] == 1:
            if not self.config["with_bias"]: 
                self.__is_bias_conflict(bias)
                if "Pk" in self.config["output"]: self.bird.setreducePslb(self.bias, what=what)
                elif "Cf" in self.config["output"]: self.bird.setreduceCflb(self.bias, what=what)
            if "Pk" in self.config["output"]: return self.bird.fullPs
            elif "Cf" in self.config["output"]: return self.bird.fullCf  

        elif self.config["skycut"] > 1:
            if not isinstance(bias, (list, np.ndarray)) or len(bias) != self.config["skycut"]: 
                raise Exception("Please specify bias (in a list of dicts) for each corresponding skycuts. ")
            for i in range(self.config["skycut"]):
                self.__is_bias_conflict(bias[i])
                if "Cf" in self.config["output"]: self.birds[i].setreduceCflb(self.bias)
                elif "Pk" in self.config["output"]: self.birds[i].setreducePslb(self.bias)
            if "Pk" in self.config["output"]: return [self.birds[i].fullPs for i in range(self.config["skycut"])]
            elif "Cf" in self.config["output"]: return [self.birds[i].fullCf for i in range(self.config["skycut"])]

    def getmarg(self, bias, marg_gauss_eft_parameters_list):

        for p in marg_gauss_eft_parameters_list:
            if p not in self.gauss_eft_parameters_list: 
                raise Exception("The parameter %s specified in getmarg() is not an available Gaussian EFT parameter to marginalize. Check your options. " % p)

        def marg(loopl, ctl, b1, f, stl=None, nnlol=None, bq=0):

            # concatenating multipoles: loopl.shape = (Nl, Nloop, Nk) -> loop.shape = (Nloop, Nl * Nk)
            loop = np.swapaxes(loopl, axis1=0, axis2=1).reshape(loopl.shape[1],-1)
            ct = np.swapaxes(ctl, axis1=0, axis2=1).reshape(ctl.shape[1],-1)
            if stl is not None: st = np.swapaxes(stl, axis1=0, axis2=1).reshape(stl.shape[1],-1)
            if nnlol is not None: nnlo = np.swapaxes(nnlol, axis1=0, axis2=1).reshape(nnlol.shape[1],-1) 

            pg = np.empty(shape=(len(marg_gauss_eft_parameters_list), loop.shape[1]))
            for i, p in enumerate(marg_gauss_eft_parameters_list):
                if p in ['b3', 'bGamma3']: 
                    if self.co.Nloop == 12: pg[i] = loop[3] + b1 * loop[7]                          # config["with_time"] = True
                    elif self.co.Nloop == 18: pg[i] = loop[3] + b1 * loop[7] + bq * loop[16]        # config["with_time"] = True, config["with_tidal_alignments"] = True
                    elif self.co.Nloop == 22: pg[i] = f * loop[8] + b1 * loop[16]                   # config["with_time"] = False, config["with_exact_time"] = False
                    elif self.co.Nloop == 35: pg[i] = f * loop[18] + b1 * loop[29]                  # config["with_time"] = False, config["with_exact_time"] = True
                    if p == 'bGamma3': pg[i] *= 6. # b3 = bt1 + 15. * bG2 + 6. * bGamma3 : config["eft_basis"] = 'eastcoast'
                # counterterm : config["eft_basis"] = 'eftoflss' or 'westcoast'
                elif p == 'cct': pg[i] = 2 * (f * ct[0+3] + b1 * ct[0]) / self.config["km"]**2 # ~ 2 (b1 + f * mu^2) k^2/km^2 P11 
                elif p == 'cr1': pg[i] = 2 * (f * ct[1+3] + b1 * ct[1]) / self.config["kr"]**2 # ~ 2 (b1 mu^2 + f * mu^4) k^2/kr^2 P11 
                elif p == 'cr2': pg[i] = 2 * (f * ct[2+3] + b1 * ct[2]) / self.config["kr"]**2 # ~ 2 (b1 mu^4 + f * mu^6) k^2/kr^2 P11 
                # counterterm : config["eft_basis"] = 'eastcoast'                       # (2.15) and (2.23) of 2004.10607
                elif p in ['c0', 'c2', 'c4']:
                    ct0, ct2, ct4 = - 2 * ct[0], - 2 * f * ct[1], - 2 * f**2 * ct[2]    # - 2 ct0 k^2 P11 , - 2 ct2 f mu^2 k^2 P11 , - 2 ct4 f^2 mu^4 k^2 P11 
                    if p == 'c0':   pg[i] = ct0                                           
                    elif p == 'c2': pg[i] = - f/3. * ct0 + ct2                        
                    elif p == 'c4': pg[i] = 3/35. * f**2 * ct0 - 6/7. * f * ct2 + ct4                                      
                # stochastic term
                elif p == 'ce0': pg[i] = st[0] # k^0 / nd mono
                elif p == 'ce1': pg[i] = st[1] # k^2 / km^2 / nd mono
                elif p == 'ce2': pg[i] = st[2] # k^2 / km^2 / nd quad
                # nnlo term: config["eft_basis"] = 'eftoflss' or 'westcoast'
                elif p == 'cr4': pg[i] = 0.25 * b1**2 * nnlo[0] / self.config["kr"]**4 # ~ 1/4 b1^2 k^4/kr^4 mu^4 P11
                elif p == 'cr6': pg[i] = 0.25 * b1 * nnlo[1] / self.config["kr"]**4    # ~ 1/4 b1 k^4/kr^4 mu^6 P11
                # nnlo term: config["eft_basis"] = 'eastcoast'
                elif p == 'ct': pg[i] = - f**4 * (b1**2 * nnlo[0] + 2. * b1 * f * nnlo[1] + f**2 * nnlo[2]) # ~ k^4 mu^4 P11

            return pg

        def marg_from_bird(bird, bias_local):
            self.__is_bias_conflict(bias_local)
            if self.config["with_tidal_alignments"]: bq = self.bias["bq"]
            else: bq = 0.
            if "Pk" in self.config["output"]: return marg(bird.Ploopl, bird.Pctl, self.bias["b1"], bird.f, stl=bird.Pstl, nnlol=bird.Pnnlol, bq=bq)
            elif "Cf" in self.config["output"]: return marg(bird.Cloopl, bird.Cctl, self.bias["b1"], bird.f, stl=bird.Cstl, nnlol=bird.Cnnlol, bq=bq)

        if self.config["skycut"] == 1: return marg_from_bird(self.bird, bias)
        elif self.config["skycut"] > 1: return [ marg_from_bird(bird_i, bias_i) for (bird_i, bias_i) in zip(self.birds, bias) ]

    def __load_engines(self, load_engines=True):

        self.co = Common(Nl=self.config["multipole"], kmax=self.config["kmax"], km=self.config["km"], kr=self.config["kr"], nd=self.config["nd"], eft_basis=self.config["eft_basis"],
            halohalo=self.config["halohalo"], with_cf=self.config["with_cf"], with_time=self.config["with_time"], optiresum=self.config["optiresum"], 
            exact_time=self.config["with_exact_time"], quintessence=self.config["with_quintessence"], 
            with_tidal_alignments=self.config["with_tidal_alignments"], nonequaltime=self.config["with_common_nonequal_time"], keep_loop_pieces_independent=self.config["keep_loop_pieces_independent"])
        
        if load_engines:
            self.nonlinear = NonLinear(load=True, save=True, fftbias=self.config["fftbias"], co=self.co)
            self.resum = Resum(co=self.co)

            if self.config["with_nnlo_counterterm"]: self.nnlo_counterterm = NNLO_counterterm(co=self.co)

            if self.config["skycut"] == 1: 
                self.projection = Projection(self.config["xdata"], Om_AP=self.config["Omega_m_AP"], z_AP=self.config["z_AP"], 
                    window_fourier_name=self.config["windowPk"], path_to_window='', window_configspace_file=self.config["windowCf"], 
                    binning=self.config["with_binning"], fibcol=self.config["with_fibercol"], Nwedges=self.config["wedge"], wedges_bounds=self.config["wedges_bounds"],
                    zz=self.config["zz"], nz=self.config["nz"], co=self.co)
            elif self.config["skycut"] > 1:
                self.projection = []
                for i in range(self.config["skycut"]):
                    if len(self.config["xdata"]) == 1: xdata = self.config["xdata"][i]
                    elif len(self.config["xdata"]) == self.config["skycut"]: xdata = self.config["xdata"][i]
                    else: xdata = self.config["xdata"]
                    if self.config["with_window"]: windowPk = self.config["windowPk"][i] ; windowCf = self.config["windowCf"][i]
                    else: windowPk = None; windowCf = None
                    if self.config["with_AP"]:
                        if isinstance(self.config["Omega_m_AP"], float): Om_AP = self.config["Omega_m_AP"]
                        elif len(self.config["Omega_m_AP"]) is self.config["skycut"]: Om_AP = self.config["Omega_m_AP"][i]
                        if isinstance(self.config["z_AP"], float): z_AP = self.config["z_AP"]
                        elif len(self.config["z_AP"]) is self.config["skycut"]: z_AP = self.config["z_AP"][i]
                    else: Om_AP = None ; z_AP = None
                    if self.config["with_redshift_bin"]: zz = self.config["zz"][i] ; nz = self.config["nz"][i]
                    else: zz = None ; nz = None
                    self.projection.append( Projection(xdata, Om_AP=Om_AP, z_AP=z_AP, 
                        window_fourier_name=windowPk, path_to_window='', window_configspace_file=windowCf, 
                        binning=self.config["with_binning"], fibcol=self.config["with_fibercol"], Nwedges=self.config["wedge"], wedges_bounds=self.config["wedges_bounds"],
                        zz=zz, nz=nz, co=self.co) ) 

    def __read_cosmo(self, cosmo_dict):

        # Checking if the inputs are consistent with the options
        for (name, cosmo) in zip(self.cosmo_catalog, self.cosmo_catalog.values()):
                for cosmo_key in cosmo_dict:
                    if cosmo_key is name:
                        cosmo.check(cosmo_key, cosmo_dict[cosmo_key])

        # Setting unspecified configs to default value 
        for (name, cosmo) in zip(self.cosmo_catalog, self.cosmo_catalog.values()):
            if cosmo.value is None: cosmo.value = cosmo.default

        # Translating the catalog to a dict
        self.cosmo = translate_catalog_to_dict(self.cosmo_catalog)

    def __is_cosmo_conflict(self):

        if self.cosmo["k11"] is None or self.cosmo["P11"] is None:
            raise Exception("Please provide a linear matter power spectrum \'P11\' and the corresponding \'k11\'. ")
        
        if len(self.cosmo["k11"]) != len(self.cosmo["P11"]):
            raise Exception("Please provide a linear matter power spectrum \'P11\' and the corresponding \'k11\' of same length.")

        if self.cosmo["k11"][0] > 1e-4 or self.cosmo["k11"][-1] < 1.:
            raise Exception("Please provide a linear matter spectrum \'P11\' and the corresponding \'k11\' with min(k11) < 1e-4 and max(k11) > 1.")

        if self.config["skycut"] > 1:
            if self.cosmo["D"] is None:
                raise Exception("You asked multi skycuts. Please specify the growth function \'D\'. ")
            elif len(self.cosmo["D"]) is not self.config["skycut"]:
                raise Exception("Please specify (in a list) as many growth functions \'D\' as the corresponding skycuts.")

        if self.config["multipole"] == 0: self.cosmo["f"] = 0.
        elif not self.config["with_redshift_bin"]:
            if self.cosmo["f"] is None: 
                raise Exception("Please specify the growth rate \'f\'.")
            if self.config["skycut"] == 1:
                if not isinstance(self.cosmo["f"], float):
                    raise Exception("Please provide a single growth rate \'f\'.")
            elif len(self.cosmo["f"]) != self.config["skycut"]: 
                raise Exception("Please specify (in a list) as many \'f\' as the corresponding skycuts.")

        if self.config["wedge"] > 0:
            if self.config["wedges_bounds"] is not None:
                if len(self.config["wedges_bounds"]) != self.config["wedge"]+1 or self.config["wedges_bounds"][0] != 0 or self.config["wedges_bounds"][-1] != 1:
                    raise Exception("If specifying \'wedges_bounds\', specify them in a list as: [0, a_1, ..., a_{n-1}, 1], where n: number of wedges")
        
        if self.config["with_bias"]: self.__is_bias_conflict()

        if self.config["with_AP"]:
            if self.cosmo["DA"] is None or self.cosmo["H"] is None:
                raise Exception("You asked to apply the AP effect. Please specify \'DA\' and \'H\'. ")
            
            if self.config["skycut"] == 1:
                if not isinstance(self.cosmo["DA"], float) and not isinstance(self.cosmo["H"], float):
                    raise Exception("Please provide a single pair of \'DA\' and \'H\'.")
            elif len(self.cosmo["DA"]) != self.config["skycut"] or len(self.cosmo["H"]) != self.config["skycut"]:
                raise Exception("Please specify (in lists) as many \'DA\' and \'H\' as the corresponding skycuts.")

        if self.config["with_redshift_bin"]:
            if self.cosmo["Dz"] is None or self.cosmo["fz"] is None:
                raise Exception("You asked to account the galaxy counts distribution. Please specify \'Dz\' and \'fz\'. ")

            if self.config["skycut"] == 1:
                if len(self.cosmo["Dz"]) != len(self.config["zz"]) or len(self.cosmo["fz"]) != len(self.config["zz"]):
                    raise Exception("Please specify \'Dz\' and \'fz\' with same length as \'zz\'. ")
            elif len(self.cosmo["Dz"]) != self.config["skycut"] or len(self.cosmo["fz"]) != self.config["skycut"]:
                raise Exception("Please specify (in lists) as many \'Dz\' and \'fz\' as the corresponding skycuts.")

        if self.config["with_nonequal_time"]:
            if self.cosmo["D1"] is None or self.cosmo["D2"] is None or self.cosmo["f1"] is None or self.cosmo["f2"] is None:
                raise Exception("You asked nonequal time correlator. Pleas specify: \'D1\', \'D2\', \'f1\', \'f2\'.  ")

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
                self.bias["b1"] = self.bias["bt1"]
                self.bias["b2"] = self.bias["bt1"] + 7/2. * self.bias["bG2"]
                self.bias["b3"] = self.bias["bt1"] + 15. * self.bias["bG2"] + 6. * self.bias["bGamma3"]
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
            elif self.config["eft_basis"] == "eastcoast": self.eft_parameters_list.extend(['bt1', 'bt2', 'bG2', 'bGamma3'])
        if self.config["with_tidal_alignments"]: self.eft_parameters_list.append('bq')

    def __read_config(self, config_dict):

        # Checking if the inputs are consistent with the options
        for config_key in config_dict:
            is_config = False
            for (name, config) in zip(self.config_catalog, self.config_catalog.values()):
                if config_key == name:
                    config.check(config_key, config_dict[config_key])
                    is_config = True
            ### v1.2: we'll activate this later
            # if not is_config: 
            #     raise Exception("%s is not an available configuration option. Please check correlator.info() for help. " % config_key)
            
        # Setting unspecified configs to default value 
        for (name, config) in zip(self.config_catalog, self.config_catalog.values()):
            if config.value is None: config.value = config.default

        # Translating the catalog to a dict
        self.config = translate_catalog_to_dict(self.config_catalog)
        
        self.config["accboost"] = float(self.config["accboost"])

    def __is_config_conflict(self):

        if "Cf" in self.config["output"]:
            self.config["with_window"] = False
            self.config["with_cf"] = True
        else:
            self.config["with_cf"] = False

        # if self.config["with_cf"]: self.config["with_stoch"] = False
        # if self.config["wedge"] is not 0: self.config["multipole"] = 3 # enforced
        
        if "bm" in self.config["output"]: self.config["halohalo"] = False
        else: self.config["halohalo"] = True
        
        if self.config["skycut"] > 1:
            self.config["with_bias"] = False
            if len(self.config["z"]) != self.config["skycut"]:
                raise Exception("Please specify as many redshifts \'z\' as the number of skycuts.")
                self.config["z"] = np.asarray(self.config["z"])
            def checkEqual(lst): return lst[1:] == lst[:-1] ### TO CHANGE
            if checkEqual(self.config["z"]): # if same redshift
                self.config["with_time"] = True 
                #self.config["z"] = self.config["z"][0]
            else: self.config["with_time"] = False
        
        if self.config["xdata"] is None:
            raise Exception("Please specify a data point array \'xdata\'.")
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
        
        self.config["with_common_nonequal_time"] = False # this is to pass for the common Class to setup the numbers of loops (22 and 13 gathered by default)

        if self.config["with_nonequal_time"]:

            self.config["with_common_nonequal_time"] = True # this is to pass for the common Class to setup the numbers of loops (22 and 13 seperated since they have different time dependence)

            if self.config["skycut"] > 1: raise Exception("Nonequal time correlator available only for skycut = 1. ")
            try:
                self.config["z1"]
                self.config["z2"]
            except:
                print("Please specify \'z1\' and \'z2\' for nonequaltime correlator. ")

            self.config["with_time"] = False
            self.config["with_bias"] = False

        if self.config["with_AP"]:
            if self.config["Omega_m_AP"] is None: raise Exception("You asked to apply the AP effect. Please specify \'Omega_m_AP\'.")
            if self.config["z_AP"] is None: self.config["z_AP"] = self.config["z"]
            if self.config["skycut"] == 1:
                if isinstance(self.config["z_AP"], list): self.config["z_AP"] = self.config["z_AP"][0]
                
        if self.config["with_window"]:
            
            def is_conflict_window(windowCf):
                try: 
                    test = np.loadtxt(windowCf)
                    if self.config["with_cf"]: self.windowPk = None
                except IOError:
                    print("You asked to apply a mask. Please specify a correct path to the configuration space window file.")
                    raise
            
            if self.config["skycut"] == 1:
                if isinstance(self.config["windowCf"], list): self.config["windowCf"] = self.config["windowCf"][0]
                if isinstance(self.config["windowPk"], list): self.config["windowPk"] = self.config["windowPk"][0]
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
            self.config["with_cf"] = True # even for the Pk, we first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            # self.config["with_common_nonequal_time"] = True # approximating 13 and 22 loop time to be the same, see Projection.redshift()

            def is_conflict_zz(zz, nz):
                if zz is None or nz is None:
                    pass#raise Exception("You asked to account for the galaxy counts distribution over a redshift bins. Please provide a distribution \'nz\' and corresponding \'zz\'. ")
                elif len(zz) != len(nz):
                    raise Exception("Please provide \'nz\' and corresponding \'zz\' of the same length. ")

            if self.config["skycut"] == 1: is_conflict_zz(self.config["zz"], self.config["nz"])
            elif self.config["skycut"] > 1:
                self.config["zz"] = np.asarray(self.config["zz"])
                self.config["nz"] = np.asarray(self.config["nz"])
                # print (len(self.config["zz"]), len(self.config["nz"]))
                if len(self.config["zz"]) == self.config["skycut"] and len(self.config["nz"]) == self.config["skycut"]: 
                    for zz, nz in zip(self.config["zz"], self.config["nz"]): is_conflict_zz(zz, nz)
                else:
                    raise Exception("Please provide as many \'nz\' with corresponding \'zz\' (in a list) as the corresponding skycuts. ")
        else:
            self.config["zz"] = None
            self.config["nz"] = None 

        # if self.config["with_quintessence"]: self.config["with_exact_time"] = True


    def setcosmo(self, cosmo_dict, module='class'):
        
        if module is 'class':
            
            from classy import Class

            # Not sure this is useful: does class read z_max_pk?
            if self.config["skycut"] == 1:
                if self.config["with_redshift_bin"]: zmax = max(self.config["zz"])
                else: zmax = self.config["z"]
            elif self.config["skycut"] > 1:
                if self.config["with_redshift_bin"]: 
                    maxbin = np.argmax(self.config["z"])
                    zmax = max(self.config["zz"][maxbin])
                else: zmax = max(self.config["z"])

            cosmo_dict_local = cosmo_dict.copy()
            if self.config["with_bias"]: del cosmo_dict_local["bias"] # Class does not like dictionary with keys other than the ones it reads...

            M = Class()
            M.set(cosmo_dict_local)
            M.set({'output': 'mPk', 'P_k_max_h/Mpc': 1.0, 'z_max_pk': zmax })
            M.compute()

            cosmo = {}

            if self.config["with_bias"]: 
                try: cosmo["bias"] = cosmo_dict["bias"]
                except:
                    print ("Please specify \'bias\'.")
                    raise

            if self.config["skycut"] == 1: zfid = self.config["z"]
            elif self.config["skycut"] > 1: zfid = self.config["z"][self.config["skycut"]//2]

            log10kmax_classy = 0 
            if self.config["with_nnlo_counterterm"]: log10kmax_classy = 1 # slower, but useful for the wiggle-no-wiggle split
            cosmo["k11"] = np.logspace(-5, log10kmax_classy, 200)  # k in h/Mpc
            cosmo["P11"] = np.array([M.pk(k*M.h(), zfid)*M.h()**3 for k in cosmo["k11"]]) # P(k) in (Mpc/h)**3

            if self.config["skycut"] == 1:
                if self.config["multipole"] is not 0: cosmo["f"] = M.scale_independent_growth_factor_f(self.config["z"])
                if self.config["with_nonequal_time"]:
                    cosmo["D"] = M.scale_independent_growth_factor(self.config["z"]) 
                    cosmo["D1"] = M.scale_independent_growth_factor(self.config["z1"]) 
                    cosmo["D2"] = M.scale_independent_growth_factor(self.config["z2"]) 
                    cosmo["f1"] = M.scale_independent_growth_factor_f(self.config["z1"]) 
                    cosmo["f2"] = M.scale_independent_growth_factor_f(self.config["z2"]) 
                if self.config["with_exact_time"] or self.config["with_quintessence"]: 
                    cosmo["z"] = self.config["z"]
                    cosmo["Omega0_m"] = M.Omega0_m()
                    try: cosmo["w0_fld"] = cosmo_dict["w0_fld"]
                    except: pass
                if self.config["with_AP"]:
                    cosmo["DA"] = M.angular_distance(self.config["z"]) * M.Hubble(0.)
                    cosmo["H"] = M.Hubble(self.config["z"]) / M.Hubble(0.)

            elif self.config["skycut"] > 1:
                if self.config["multipole"] is not 0: cosmo["f"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["z"]])
                cosmo["D"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["z"]])
                if self.config["with_AP"]:
                    cosmo["DA"] = np.array([M.angular_distance(z) * M.Hubble(0.) for z in self.config["z"]])
                    cosmo["H"] = np.array([M.Hubble(z) / M.Hubble(0.) for z in self.config["z"]])

            if self.config["with_redshift_bin"]:
                def comoving_distance(z): return M.angular_distance(z) * (1+z) * M.h()
                if self.config["skycut"] == 1:
                    cosmo["D"] = M.scale_independent_growth_factor(self.config["z"])
                    cosmo["Dz"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["zz"]])
                    cosmo["fz"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["zz"]])
                    cosmo["rz"] = np.array([comoving_distance(z) for z in self.config["zz"]])

                elif self.config["skycut"] > 1:
                    cosmo["Dz"] = np.array([ [M.scale_independent_growth_factor(z) for z in zz] for zz in self.config["zz"] ])
                    cosmo["fz"] = np.array([ [M.scale_independent_growth_factor_f(z) for z in zz] for zz in self.config["zz"] ])
                    cosmo["rz"] = np.array([ [comoving_distance(z) for z in zz] for zz in self.config["zz"] ])

            if self.config["with_quintessence"]: 
                # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum. 
                # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
                # This does not work for multi skycuts nor 'with_redshift_bin': True. # eventually to code up
                zm = 5. # z in matter domination
                def scale_factor(z): return 1/(1.+z)
                Omega0_m = cosmo["Omega0_m"]
                w = cosmo["w0_fld"]
                GF = GreenFunction(Omega0_m, w=w, quintessence=True)
                Dq = GF.D(scale_factor(zfid)) / GF.D(scale_factor(zm))
                Dm = M.scale_independent_growth_factor(zfid) / M.scale_independent_growth_factor(zm)
                cosmo["P11"] *= Dq**2 / Dm**2 * ( 1 + (1+w)/(1.-3*w) * (1-Omega0_m)/Omega0_m * (1+zm)**(3*w) )**2 # 1611.07966 eq. (4.15)
                cosmo["f"] = GF.fplus(1/(1.+cosmo["z"]))

            # wiggle-no-wiggle split # algo: 1003.3999; details: 2004.10607
            def get_smooth_wiggle_resc(kk, pk, alpha_rs=1.): # k [h/Mpc], pk [(Mpc/h)**3]
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

            if self.config["with_nnlo_counterterm"]: 
                cosmo["k11"], cosmo["Psmooth"], cosmo["P11"] = get_smooth_wiggle_resc(cosmo["k11"], cosmo["P11"])

            return cosmo

class BiasCorrelator(Correlator): 
    '''
    Class to load pre-computed correlator
    '''
    def __init__(self, config_dict=None, load_engines=False):
        Correlator.__init__(self, config_dict, load_engines=load_engines)

def translate_catalog_to_dict(catalog):
    newdict = dict.fromkeys(catalog)
    for key, option in zip(catalog, catalog.values()):
        newdict[key] = option.value
    return newdict

def typename(onetype):
    if isinstance(onetype, tuple): return [t.__name__ for t in onetype]
    else: return [onetype.__name__]

class Option(object):

    def __init__(self, config_name, config_type, config_list=None, description='', default=None, verbose=False):

        self.verbose = verbose
        self.name = config_name
        self.type = config_type
        self.list = config_list
        self.description = description
        self.default = default
        self.value = None

    def check(self, config_key, config_value):
        is_config = False
        if self.verbose: print("\'%s\': \'%s\'" % (config_key, config_value))
        if isinstance(config_value, self.type):
            if self.list is None: is_config = True
            elif isinstance(config_value, str):
                if any(config_value in o for o in self.list): is_config = True
            elif isinstance(config_value, (int, float, bool)): 
                if any(config_value == o for o in self.list): is_config = True
        if is_config: 
            self.value = config_value
        else: self.error()
        return is_config

    def error(self):
        if self.list is None:
            try: raise Exception("Input error in \'%s\'; input configs: %s. Check Correlator.info() in any doubt." % (self.name, typename(self.type)))
            except Exception as e: print(e)
        else:
            try: raise Exception("Input error in \'%s\'; input configs: %s. Check Correlator.info() in any doubt." % (self.name, self.list))
            except Exception as e: print(e)




