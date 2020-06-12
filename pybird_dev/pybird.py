import os
import numpy as np
from copy import deepcopy

from common import Common, co
from bird import Bird
from nonlinear import NonLinear
from resum import Resum
from projection import Projection
from angular import Angular
from greenfunction import GreenFunction


# import importlib, sys
# importlib.reload(sys.modules['common'])
# importlib.reload(sys.modules['bird'])
# importlib.reload(sys.modules['nonlinear'])
# importlib.reload(sys.modules['resum'])
# importlib.reload(sys.modules['projection'])
# importlib.reload(sys.modules['angular'])
# importlib.reload(sys.modules['greenfunction'])

# from common import Common, co
# from bird import Bird
# from nonlinear import NonLinear
# from resum import Resum
# from projection import Projection
# from angular import Angular
# from greenfunction import GreenFunction

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
                description="Scale independent growth function. To specify if \'skycut\' > 1.", 
                default=None) ,
            "f": Option("f", (float, list, np.ndarray),
                description="Scale independent growth rate (for RSD). Automatically set to 0 for \'output\': \'m__\'.", 
                default=None) ,
            "bias": Option("bias", (dict, list, np.ndarray),
                description="EFT parameters in dict = \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\', \'cr2\', \'ce0\', \'ce1\', \'ce2\' \}.", 
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
            "DAz": Option("DAz", (list, np.ndarray),
                description="Angular distance times H_0 over redshift bin. To specify if \'with_redshift_bin\' and \'with_AP\' are True.", 
                default=None) ,
            "Hz": Option("Hz", (list, np.ndarray),
                description="Hubble parameter by H_0 over redshift bin. To specify if \'with_redshift_bin\' and \'with_AP\' are True.", 
                default=None) ,
            "rz": Option("rz", (list, np.ndarray),
                description="Comoving distance in [Mpc/h] over redshift bin. To specify if \'output\':\'w\'.", 
                default=None) ,
        }
        
        self.config_catalog = {
            "output": Option("output", str, ["mPk", "mCf", "bPk", "bCf", "w"], 
                description="Correlator: matter / biased tracers - power spectrum / correlation function ; \'w\': angular correlation function. ", 
                default="bPk") ,
            "multipole": Option("multipole", int, [0, 2, 3], 
                description="Number of multipoles. 0: real space. 2: monopole + quadrupole. 3: monopole + quadrupole + hexadecapole.",
                default=2) ,
            "wedge": Option("wedge", int, 
                description="Number of wedges. 0: compute multipole instead. 0<: automatically set \'multipole\' to 3.",
                default=0) ,
            "skycut": Option("skycut", int, 
                description="Number of skycuts.",
                default=1) ,
            "with_time": Option("with_time", bool,
                description="Time (in)dependent evaluation (for multi skycuts / redshift bin). For \'with_redshift_bin\': True, \'skycut\' > 1 or \'output\': \'w\', automatically set to False.",
                default=True) ,
            "z": Option("z", (float, list, np.ndarray),
                description="Effective redshift(s). Should match the number of skycuts.",
                default=None) ,
            "km": Option("km", float,
                description="Inverse galaxy spatial extension scale in [h/Mpc].",
                default=1.) ,
            "nd": Option("nd", float,
                description="mean galaxy density",
                default=3e-4) ,
            "with_stoch": Option("with_stoch", bool, 
                description="With stochastic terms.",
                   default=False) ,
            "with_bias": Option("with_bias", bool, 
                description="Bias (in)dependent evalution. Automatically set to False for \'with_time\': False.",
                   default=False) ,
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
            "accboost": Option("accboost", int, [1, 2, 3],
                description="Sampling accuracy boost.",
                default=1) ,
            "optiresum": Option("optiresum", bool,
                description="True: Resumming only with the BAO peak. False: Resummation on the full correlation function.",
                default=False) ,
            "xdata": Option("xdata", (np.ndarray, list),
                description="Array of data points.",
                default=None) ,
            "with_resum": Option("with_resum", bool,
                description="Apply IR-resummation.",
                default=True) ,
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
            "with_nlo_bias": Option("with_nlo_bias", bool,
                description="With next-to-leading bias.",
                default=False) ,
            "with_assembly_bias": Option("with_assembly_bias", bool,
                description="With assembly bias.",
                default=False) ,
            "with_quintessence": Option("with_quintessence", bool,
                description="Clustering quintessence. Automatically set \'with_exact_time\' to True.",
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
        
        # Loading PyBird engines
        if load_engines: self.__load_engines()
        else: self.__load_engines(only_common=True)


    def compute(self, cosmo_dict, module=None):
        
        if module is 'class': cosmo_dict = self.setcosmo(cosmo_dict, module='class')
        
        self.__read_cosmo(cosmo_dict)
        self.__is_cosmo_conflict()

        if self.config["skycut"] == 1:
            self.bird = Bird(self.cosmo, with_bias=self.config["with_bias"], with_stoch=self.config["with_stoch"], with_nlo_bias=self.config["with_nlo_bias"], with_assembly_bias=self.config["with_assembly_bias"], co=self.co)
            self.nonlinear.PsCf(self.bird)
            if self.config["with_bias"]: self.bird.setPsCf(self.bias)
            else: self.bird.setPsCfl()
            if self.config["with_resum"]:
                if self.config["with_cf"]: self.resum.PsCf(self.bird)
                else: self.resum.Ps(self.bird)
            if "w" in self.config["output"]: self.angular.w(self.bird, self.cosmo["Dz"], self.cosmo["fz"], self.cosmo["rz"], self.config["zz"], self.config["nz"])
            else:
                if self.config["with_redshift_bin"]: self.projection.redshift(self.bird, self.cosmo["Dz"], self.cosmo["fz"], self.cosmo["DAz"], self.cosmo["Hz"])
                elif self.config["with_AP"]: self.projection.AP(self.bird)
                if self.config["with_window"]: self.projection.Window(self.bird)
                if self.config["with_fibercol"]: self.projection.fibcolWindow(self.bird)
                if self.config["wedge"] is not 0: self.projection.Wedges(self.bird)
                if self.config["with_binning"]: self.projection.xbinning(self.bird)
                else: self.projection.xdata(self.bird)

        elif self.config["skycut"] > 1:

            self.birds = []

            cosmoi = deepcopy(self.cosmo)

            for i in range(self.config["skycut"]):

                cosmoi["f"] = self.cosmo["f"][i]
                cosmoi["D"] = self.cosmo["D"][i]
                cosmoi["z"] = self.config["z"][i]

                if self.config["with_AP"] and not self.config["with_redshift_bin"]:
                    cosmoi["DA"] = self.cosmo["DA"][i]
                    cosmoi["H"] = self.cosmo["H"][i]

                if i is 0: 
                    self.bird = Bird(cosmoi, with_bias=False, with_stoch=self.config["with_stoch"], with_nlo_bias=self.config["with_nlo_bias"], with_assembly_bias=self.config["with_assembly_bias"], co=self.co)
                    self.nonlinear.PsCf(self.bird)
                    self.bird.setPsCfl()
                    if self.config["with_resum"]: self.resum.Ps(self.bird, makeIR=True, makeQ=True, setPs=False)
                    self.birds.append(self.bird)
                else:
                    birdi = deepcopy(self.bird)
                    birdi.settime(cosmoi)
                    if self.config["with_resum"]:
                        if self.config["with_cf"]: self.resum.PsCf(birdi, makeIR=False, makeQ=True, setPs=False, setCf=True)
                        else: self.resum.Ps(birdi, makeIR=False, makeQ=True, setPs=True)
                    self.birds.append(birdi)
                    
            if self.config["with_resum"]:
                if self.config["with_cf"]: self.resum.PsCf(self.birds[0], makeIR=False, makeQ=False, setPs=False, setCf=True)
                else: self.resum.Ps(self.birds[0], makeIR=False, makeQ=False, setPs=True)

            for i in range(self.config["skycut"]):

                if "w" in self.config["output"]: 
                    self.angular[i].w(self.birds[i], self.cosmo["Dz"][i], self.cosmo["fz"][i], self.cosmo["rz"][i], self.config["zz"][i], self.config["nz"][i])
                else:
                    if self.config["with_redshift_bin"] and self.config["nz"][i] is not None: self.projection[i].redshift(self.birds[i], self.cosmo["Dz"][i], self.cosmo["fz"][i], self.cosmo["DAz"][i], self.cosmo["Hz"][i])
                    elif self.config["with_AP"]: self.projection[i].AP(self.birds[i])
                    if self.config["with_window"]: self.projection[i].Window(self.birds[i])
                    if self.config["with_fibercol"]: self.projection[i].fibcolWindow(self.birds[i])
                    if self.config["wedge"] is not 0: self.projection[i].Wedges(self.birds[i]) 
                    if self.config["with_binning"]: self.projection[i].xbinning(self.birds[i])
                    else: self.projection[i].xdata(self.birds[i])

    def cache(self, as_dict=True):

        if self.config["skycut"] == 1:

            if as_dict:
                correlator_cache = {}
                correlator_cache = {"f": self.cosmo["f"]}
                if "Pk" in self.config["output"]: 
                    correlator_cache["lin"] = self.bird.P11l
                    correlator_cache["loop"] = self.bird.Ploopl
                    correlator_cache["ct"] = self.bird.Pctl
                    if self.config["with_stoch"]: correlator_cache["st"] = self.bird.Pstl
                    if self.config["with_nlo_bias"]: correlator_cache["nlo"] = self.bird.Pnlol
                elif "Cf" in self.config["output"]: 
                    correlator_cache["lin"] = self.bird.C11l
                    correlator_cache["loop"] = self.bird.Cloopl
                    correlator_cache["ct"] = self.bird.Cctl
                    if self.config["with_nlo_bias"]: correlator_cache["nlo"] = self.bird.Cnlol
                elif "w" in self.config["output"]:
                    correlator_cache["lin"] = self.bird.wlin
                    correlator_cache["loop"] = self.bird.wloop
                    correlator_cache["ct"] = self.bird.wct
                    if self.config["with_nlo_bias"]: correlator_cache["nlo"] = self.bird.wnlo
                return correlator_cache

            else: return self.bird

        elif self.config["skycut"] > 1:

            if as_dict:
                correlator_cache = {}
                correlator_cache = {"f": self.cosmo["f"]}
                if "Pk" in self.config["output"]: 
                    correlator_cache["lin"] = [self.birds[i].P11l for i in range(self.config["skycut"])]
                    correlator_cache["loop"] = [self.birds[i].Ploopl for i in range(self.config["skycut"])]
                    correlator_cache["ct"] = [self.birds[i].Pctl for i in range(self.config["skycut"])]
                    if self.config["with_stoch"]: correlator_cache["st"] = [self.birds[i].Pstl for i in range(self.config["skycut"])]
                    if self.config["with_nlo_bias"]: correlator_cache["nlo"] = [self.birds[i].Pnlol for i in range(self.config["skycut"])]
                elif "Cf" in self.config["output"]: 
                    correlator_cache["lin"] = [self.birds[i].C11l for i in range(self.config["skycut"])]
                    correlator_cache["loop"] = [self.birds[i].Cloopl for i in range(self.config["skycut"])]
                    correlator_cache["ct"] = [self.birds[i].Cctl for i in range(self.config["skycut"])]
                    if self.config["with_nlo_bias"]: correlator_cache["nlo"] = [self.birds[i].Cnlol for i in range(self.config["skycut"])]
                elif "w" in self.config["output"]:
                    correlator_cache["lin"] = [self.birds[i].wlin for i in range(self.config["skycut"])]
                    correlator_cache["loop"] = [self.birds[i].wloop for i in range(self.config["skycut"])]
                    correlator_cache["ct"] = [self.birds[i].wct for i in range(self.config["skycut"])]
                    if self.config["with_nlo_bias"]: correlator_cache["nlo"] = [self.birds[i].wnlo for i in range(self.config["skycut"])]
                return correlator_cache

            else: return self.birds

    def setcache(self, correlator_cache, as_dict=True): 

        self.__read_cosmo({})

        if self.config["skycut"] == 1:

            if as_dict:
                self.cosmo["f"] = correlator_cache["f"]
                self.bird = Bird(self.cosmo, with_bias=False, with_stoch=self.config["with_stoch"], with_nlo_bias=self.config["with_nlo_bias"], co=self.co)
                if "Pk" in self.config["output"]: 
                    self.bird.P11l = correlator_cache["lin"] 
                    self.bird.Ploopl = correlator_cache["loop"]
                    self.bird.Pctl = correlator_cache["ct"]
                    if self.config["with_stoch"]: self.bird.Pstl = correlator_cache["st"]
                    if self.config["with_nlo_bias"]: self.bird.Pnlol = correlator_cache["nlo"]
                elif "Cf" in self.config["output"]: 
                    self.bird.C11l = correlator_cache["lin"]
                    self.bird.Cloopl = correlator_cache["loop"]
                    self.bird.Cctl = correlator_cache["ct"]
                    if self.config["with_nlo_bias"]: self.bird.Cnlol = correlator_cache["nlo"]
                elif "w" in self.config["output"]:
                    self.bird.wlin = correlator_cache["lin"]
                    self.bird.wloop = correlator_cache["loop"]
                    self.bird.wct = correlator_cache["ct"]
                    if self.config["with_nlo_bias"]: self.bird.wnlo = correlator_cache["nlo"]

            else: self.bird = correlator_cache

        elif self.config["skycut"] > 1:

            if as_dict:
                self.birds = []
                for i in range(self.config["skycut"]):
                    self.cosmo["f"] = correlator_cache["f"][i]
                    self.bird = Bird(self.cosmo, with_bias=False, with_stoch=self.config["with_stoch"], with_nlo_bias=self.config["with_nlo_bias"], co=self.co)
                    if "Pk" in self.config["output"]: 
                        self.bird.P11l = correlator_cache["lin"][i]
                        self.bird.Ploopl = correlator_cache["loop"][i]
                        self.bird.Pctl = correlator_cache["ct"][i]
                        if self.config["with_stoch"]: self.bird.Pstl = correlator_cache["st"][i]
                        if self.config["with_nlo_bias"]: self.bird.Pnlol = correlator_cache["nlo"][i]
                    elif "Cf" in self.config["output"]: 
                        self.bird.C11l = correlator_cache["lin"][i]
                        self.bird.Cloopl = correlator_cache["loop"][i]
                        self.bird.Cctl = correlator_cache["ct"][i]
                        if self.config["with_nlo_bias"]: self.bird.Cnlol = correlator_cache["nlo"][i]
                    elif "w" in self.config["output"]:
                        self.bird.wlin = correlator_cache["lin"][i]
                        self.bird.wloop = correlator_cache["loop"][i]
                        self.bird.wct = correlator_cache["ct"][i]
                        if self.config["with_nlo_bias"]: self.bird.wnlo = correlator_cache["nlo"][i]
                    self.birds.append(self.bird)

            else: self.birds = correlator_cache


    def get(self, bias=None):

        if self.config["skycut"] == 1:

            if not self.config["with_bias"]: 
                self.__is_bias_conflict(bias)
                if "Pk" in self.config["output"]: self.bird.setreducePslb(self.bias)
                elif "Cf" in self.config["output"]: self.bird.setreduceCflb(self.bias)
                elif "w" in self.config["output"]: self.bird.setw(self.bias)

            if "Pk" in self.config["output"]: return self.bird.fullPs
            elif "Cf" in self.config["output"]: return self.bird.fullCf
            elif "w" in self.config["output"]: return self.bird.w

        elif self.config["skycut"] > 1:

            if not isinstance(bias, (list, np.ndarray)): raise Exception("Please specify bias (in a list of dicts) for each corresponding skycuts. ")
            if len(bias) is not self.config["skycut"]: raise Exception("Please specify bias (in a list of dicts) for each corresponding skycuts. ")
            
            for i in range(self.config["skycut"]):
                self.__is_bias_conflict(bias[i])
                if "w" in self.config["output"]: self.birds[i].setw(self.bias)
                elif "Cf" in self.config["output"]: self.birds[i].setreduceCflb(self.bias)
                elif "Pk" in self.config["output"]: self.birds[i].setreducePslb(self.bias)
            if "Pk" in self.config["output"]: return [self.birds[i].fullPs for i in range(self.config["skycut"])]
            elif "Cf" in self.config["output"]: return [self.birds[i].fullCf for i in range(self.config["skycut"])]
            elif "w" in self.config["output"]: return [self.birds[i].w for i in range(self.config["skycut"])]

    def getmarg(self, b1, model=1):

        def marg(loop, ct, b1, f1, Pst=None, model=model):

            if self.config["with_redshift_bin"]: f = 1.
            else: f = f1

            if loop.ndim is 3:
                loop = np.swapaxes(loop, axis1=0, axis2=1)
                ct = np.swapaxes(ct, axis1=0, axis2=1)
                if Pst is not None: Pst = np.swapaxes(Pst, axis1=0, axis2=1)
            
            if self.config["with_time"]: Pb3 = loop[3] + b1 * loop[7]
            else: Pb3 = f * loop[8] + b1 * loop[16]

            m = np.array([ Pb3.reshape(-1),
                            2 * (f * ct[0+3] + b1 * ct[0]).reshape(-1) / self.config["km"]**2 ])
            
            if "w" not in self.config["output"]:
                if self.config["multipole"] >= 2: m = np.vstack([m, 2 * (f * ct[1+3] + b1 * ct[1]).reshape(-1) / self.config["km"]**2])
                if self.config["multipole"] >= 3: m = np.vstack([m, 2 * (f * ct[2+3] + b1 * ct[2]).reshape(-1) / self.config["km"]**2])
                if self.config["with_stoch"]:
                    # if "Cf" in self.config["output"]:
                    #     if model == 5: m = np.vstack([m, Pst[0].reshape(-1)])
                    # elif "Pk" in self.config["output"]:
                    if model <= 4: m = np.vstack([m, Pst[2].reshape(-1) ])
                    if model == 1: m = np.vstack([m, Pst[0].reshape(-1) ])
                    if model == 3: m = np.vstack([m, Pst[1].reshape(-1) ])
                    if model == 4: m = np.vstack([m, Pst[1].reshape(-1) , Pst[0].reshape(-1) ])

            return m

        if self.config["skycut"] == 1:
            if "Pk" in self.config["output"]: return marg(self.bird.Ploopl, self.bird.Pctl, b1=b1, f1=self.bird.f, Pst=self.bird.Pstl)
            elif "Cf" in self.config["output"]: return marg(self.bird.Cloopl, self.bird.Cctl, b1=b1, f1=self.bird.f, Pst=self.bird.Cstl)
            elif "w" in self.config["output"]: return marg(self.bird.wloop, self.bird.wct, b1=b1, f1=self.bird.f)
        elif self.config["skycut"] > 1:
            if "Pk" in self.config["output"]: return [ marg(self.birds[i].Ploopl, self.birds[i].Pctl, b1=b1[i], f1=self.birds[i].f, Pst=self.birds[i].Pstl) for i in range(self.config["skycut"]) ]
            elif "Cf" in self.config["output"]: return [ marg(self.birds[i].Cloopl, self.birds[i].Cctl, b1=b1[i], f1=self.birds[i].f, Pst=self.birds[i].Cstl) for i in range(self.config["skycut"]) ]
            elif "w" in self.config["output"]: return [ marg(self.birds[i].wloop, self.birds[i].wct, b1=b1[i], f1=self.birds[i].f) for i in range(self.config["skycut"]) ]

    def __load_engines(self, only_common=False):

        self.co = Common(Nl=self.config["multipole"], kmax=self.config["kmax"], km=self.config["km"], nd=self.config["nd"], 
            with_cf=self.config["with_cf"], with_time=self.config["with_time"], optiresum=self.config["optiresum"], exact_time=self.config["with_exact_time"])
        
        if not only_common:
            self.nonlinear = NonLinear(load=True, save=True, co=self.co)
            self.resum = Resum(co=self.co)

            if "w" in self.config["output"]:
                if self.config["skycut"] == 1:
                    self.angular = Angular(self.config["xdata"], co=self.co)
                elif self.config["skycut"] > 1:
                    self.angular = []
                    for i in range(self.config["skycut"]): 
                        if len(self.config["xdata"]) is self.config["skycut"]: xdata = self.config["xdata"][i]
                        else: xdata = self.config["xdata"] 
                        self.angular.append(Angular(xdata, co=self.co))

            else:
                if self.config["skycut"] == 1: 
                    self.projection = Projection(self.config["xdata"], Om_AP=self.config["Omega_m_AP"], z_AP=self.config["z_AP"], 
                        window_fourier_name=self.config["windowPk"], path_to_window='', window_configspace_file=self.config["windowCf"], 
                        binning=self.config["with_binning"], fibcol=self.config["with_fibercol"], Nwedges=self.config["wedge"], 
                        zz=self.config["zz"], nz=self.config["nz"], co=self.co)
                
                elif self.config["skycut"] > 1:
                    
                    self.projection = []

                    for i in range(self.config["skycut"]):

                        if len(self.config["xdata"]) == 1: xdata = self.config["xdata"][i]
                        elif len(self.config["xdata"]) is self.config["skycut"]: xdata = self.config["xdata"][i]
                        else: xdata = self.config["xdata"]

                        if self.config["with_window"]:
                            windowPk = self.config["windowPk"][i]
                            windowCf = self.config["windowCf"][i]
                        else: 
                            windowPk = None
                            windowCf = None

                        if self.config["with_AP"]:
                            if isinstance(self.config["Omega_m_AP"], float): Om_AP = self.config["Omega_m_AP"]
                            elif len(self.config["Omega_m_AP"]) is self.config["skycut"]: Om_AP = self.config["Omega_m_AP"][i]

                            if isinstance(self.config["z_AP"], float): z_AP = self.config["z_AP"]
                            elif len(self.config["z_AP"]) is self.config["skycut"]: z_AP = self.config["z_AP"][i]
                        else:
                            Om_AP = None
                            z_AP = None

                        if self.config["with_redshift_bin"]:
                            zz = self.config["zz"][i]
                            nz = self.config["nz"][i]
                        else:
                            zz = None
                            nz = None

                        self.projection.append( Projection(xdata, Om_AP=Om_AP, z_AP=z_AP, 
                            window_fourier_name=windowPk, path_to_window='', window_configspace_file=windowCf, 
                            binning=self.config["with_binning"], fibcol=self.config["with_fibercol"], Nwedges=self.config["wedge"], 
                            zz=zz, nz=nz,
                            co=self.co) ) 

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
            elif len(self.cosmo["f"]) is not self.config["skycut"]: 
                raise Exception("Please specify (in a list) as many \'f\' as the corresponding skycuts.")

        if self.config["with_bias"]:
            self.__is_bias_conflict()

        if self.config["with_AP"] and not self.config["with_redshift_bin"]:
            if self.cosmo["DA"] is None or self.cosmo["H"] is None:
                raise Exception("You asked to apply the AP effect. Please specify \'DA\' and \'H\'. ")
            
            if self.config["skycut"] == 1:
                if not isinstance(self.cosmo["DA"], float) and not isinstance(self.cosmo["H"], float):
                    raise Exception("Please provide a single pair of \'DA\' and \'H\'.")
            elif len(self.cosmo["DA"]) is not self.config["skycut"] or len(self.cosmo["H"]) is not self.config["skycut"]:
                raise Exception("Please specify (in lists) as many \'DA\' and \'H\' as the corresponding skycuts.")

        if self.config["with_redshift_bin"]:
            if self.cosmo["Dz"] is None or self.cosmo["fz"] is None:
                raise Exception("You asked to account the galaxy counts distribution. Please specify \'Dz\' and \'fz\'. ")

            if self.config["skycut"] == 1:
                if len(self.cosmo["Dz"]) is not len(self.config["zz"]) or len(self.cosmo["fz"]) is not len(self.config["zz"]):
                    raise Exception("Please specify \'Dz\' and \'fz\' with same length as \'zz\'. ")
            elif len(self.cosmo["Dz"]) is not self.config["skycut"] or len(self.cosmo["fz"]) is not self.config["skycut"]:
                raise Exception("Please specify (in lists) as many \'Dz\' and \'fz\' as the corresponding skycuts.")

            if self.config["with_AP"]:
                if self.cosmo["DAz"] is None or self.cosmo["Hz"] is None:
                    raise Exception("You asked to account the galaxy counts distribution and apply the AP effect. Please specify \'DAz\' and \'Hz\'. ")

                if self.config["skycut"] == 1:
                    if len(self.cosmo["DAz"]) is not len(self.config["zz"]) or len(self.cosmo["Hz"]) is not len(self.config["zz"]):
                        raise Exception("Please specify \'DAz\' and \'Hz\' with same length as \'zz\'. ")
                elif len(self.cosmo["DAz"]) is not self.config["skycut"] or len(self.cosmo["Hz"]) is not self.config["skycut"]:
                    raise Exception("Please specify (in lists) as many \'DAz\' and \'Hz\' as the corresponding skycuts.")

            if "w" in self.config["output"]:
                if self.cosmo["rz"] is None:
                    raise Exception("You asked angular statistics. Please specify \'rz\'. ")

                if self.config["skycut"] == 1:
                    if len(self.cosmo["rz"]) is not len(self.config["zz"]):
                        raise Exception("Please specify \'rz\' with same length as \'zz\'. ")
                elif len(self.cosmo["rz"]) is not self.config["skycut"]:
                    raise Exception("Please specify (in a list) as many \'rz\'as the corresponding skycuts.")

    def __is_bias_conflict(self, bias=None):

        ###raise Exception("Input error in \'%s\'; input configs: %s. Check Correlator.info() in any doubt." % ())

        if bias is not None: self.cosmo["bias"] = bias

        if self.cosmo["bias"] is None: raise Exception("Please specify \'bias\'. ")
        if isinstance(self.cosmo["bias"], (list, np.ndarray)): self.cosmo["bias"] = self.cosmo["bias"][0]
        if not isinstance(self.cosmo["bias"], dict): raise Exception("Please specify bias in a dict. ")

        if "m" in self.config["output"]:
            if self.config["multipole"] == 0:
                if len(self.cosmo["bias"]) is not 1: raise Exception("Please specify a dict of 1 bias: \{ \'cct\' \}. ")
                else: self.bias = { "b1": 1., "b2": 1., "b3": 1., "b4": 0., "cct": self.cosmo["bias"]["cct"], "cr1": 0., "cr2": 0., "ce0": 0., "ce1": 0., "ce2": 0. }
            elif self.config["multipole"] == 2:
                if len(self.cosmo["bias"]) is not 2: raise Exception("Please specify a dict of 2 biases: \{ \'cct\', \'cr1\' \}. ")
                else: self.bias = { "b1": 1., "b2": 1., "b3": 1., "b4": 0., "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": 0., "ce0": 0., "ce1": 0., "ce2": 0. }
            elif self.config["multipole"] == 3:
                if len(self.cosmo["bias"]) is not 3: raise Exception("Please specify a dict of 3 biases: \{ \'cct\', \'cr1\', \'cr2\' \}. ")
                else: self.bias = { "b1": 1., "b2": 1., "b3": 1., "b4": 0., "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": self.cosmo["bias"]["cr2"], "ce0": 0., "ce1": 0., "ce2": 0. }
        else:

            Nextra = 0
            if self.config["with_nlo_bias"]: Nextra += 1
            if self.config["with_assembly_bias"]: Nextra += 1

            if not self.config["with_stoch"]:
                if self.config["multipole"] == 0 or "w" in self.config["output"]:
                    if len(self.cosmo["bias"]) is not 5+Nextra: raise Exception("Please specify a dict of 5 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\' \}. ")
                    else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": 0., "cr2": 0., "ce0": 0., "ce1": 0., "ce2": 0. }
                elif self.config["multipole"] == 2:
                    if len(self.cosmo["bias"]) is not 6+Nextra: raise Exception("Please specify a dict of 6 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\' \}. ")
                    else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": 0., "ce0": 0., "ce1": 0., "ce2": 0. }
                elif self.config["multipole"] == 3:
                    if len(self.cosmo["bias"]) is not 7+Nextra: raise Exception("Please specify a dict of 7 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\', \'cr2\' \}. ")
                    else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": self.cosmo["bias"]["cr2"], "ce0": 0., "ce1": 0., "ce2": 0. }
            else:
                if self.config["multipole"] == 0:
                    if len(self.cosmo["bias"]) is not 6+Nextra: raise Exception("Please specify a dict of 6 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'ce0\' \}. ")
                    else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": 0., "cr2": 0., "ce0": self.cosmo["bias"]["ce0"], "ce1": 0., "ce2": 0. }
                elif self.config["multipole"] == 2:
                    if len(self.cosmo["bias"]) is not 9+Nextra: raise Exception("Please specify a dict of 9 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\', \'ce0\', \'ce1\', \'ce2\'  \}. ")
                    else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": 0., "ce0": self.cosmo["bias"]["ce0"], "ce1": self.cosmo["bias"]["ce1"], "ce2": self.cosmo["bias"]["ce2"] }
                elif self.config["multipole"] == 3:
                    if len(self.cosmo["bias"]) is not 10+Nextra: raise Exception("Please specify a dict of 10 biases: \{ \'b1\', \'b2\', \'b3\', \'b4\', \'cct\', \'cr1\', \'cr2\', \'ce0\', \'ce1\', \'ce2\' \}. ")
                    else: self.bias = { "b1": self.cosmo["bias"]["b1"], "b2": self.cosmo["bias"]["b2"], "b3": self.cosmo["bias"]["b3"], "b4": self.cosmo["bias"]["b4"], "cct": self.cosmo["bias"]["cct"], "cr1": self.cosmo["bias"]["cr1"], "cr2": self.cosmo["bias"]["cr2"], "ce0": self.cosmo["bias"]["ce0"], "ce1": self.cosmo["bias"]["ce1"], "ce2": self.cosmo["bias"]["ce2"] }

            if self.config["with_nlo_bias"]: 
                try: self.bias["bnlo"] = self.cosmo["bias"]["bnlo"]
                except: raise Exception ("Please specify the next-to-leading bias \'bnlo\'.  ")

            if self.config["with_assembly_bias"]: 
                try: self.bias["bq"] = self.cosmo["bias"]["bq"]
                except: raise Exception ("Please specify the assembly bias \'bq\'.  ")

    def __read_config(self, config_dict):

        # Checking if the inputs are consistent with the options
        for (name, config) in zip(self.config_catalog, self.config_catalog.values()):
                for config_key in config_dict:
                    if config_key == name:
                        config.check(config_key, config_dict[config_key])
            
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
        elif "w" in self.config["output"]:
            self.config["multipole"] = 3
            self.config["wedge"] = 0
            self.config["with_cf"] = True
            self.config["with_bias"] = False
            self.config["with_time"] = True
            self.config["with_redshift_bin"] = True
            self.config["with_AP"] = False
            self.config["with_window"] = False
        else:
            self.config["with_cf"] = False

        # if self.config["with_cf"]: self.config["with_stoch"] = False

        if self.config["wedge"] is not 0: self.config["multipole"] = 3 # enforced
            
        if self.config["skycut"] > 1:
            self.config["with_time"] = False
            self.config["with_bias"] = False
            if len(self.config["z"]) is not self.config["skycut"]:
                raise Exception("Please specify as many redshifts \'z\' as the number of skycuts.")
            self.config["z"] = np.asarray(self.config["z"])
        
        if self.config["xdata"] is None:
            raise Exception("Please specify a data point array \'xdata\'.")
        if len(self.config["xdata"]) == 1 and isinstance(self.config["xdata"][0], (list, np.ndarray)):
            self.config["xdata"] = self.config["xdata"][0]
        # else:
        #     self.config["xdata"] = np.asarray(self.config["xdata"])

        #     def is_conflict_xdata(xdata):
        #         if "w" in self.config["output"]:
        #             pass
        #         elif "Cf" in self.config["output"]:
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
        
        if self.config["with_AP"]:
            if self.config["z_AP"] is None or self.config["Omega_m_AP"] is None:
                raise Exception("You asked to apply the AP effect. Please specify \'z_AP\' and \'Omega_m_AP\'.")
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

            def is_conflict_zz(zz, nz):
                if zz is None or nz is None:
                    pass#raise Exception("You asked to account for the galaxy counts distribution over a redshift bins. Please provide a distribution \'nz\' and corresponding \'zz\'. ")
                elif len(zz) != len(nz):
                    raise Exception("Please provide \'nz\' and corresponding \'zz\' of the same length. ")

            if self.config["skycut"] == 1: is_conflict_zz(self.config["zz"], self.config["nz"])
            elif self.config["skycut"] > 1:
                self.config["zz"] = np.asarray(self.config["zz"])
                self.config["nz"] = np.asarray(self.config["nz"])
                if len(self.config["zz"]) is self.config["skycut"] and len(self.config["nz"]) is self.config["skycut"]: 
                    for zz, nz in zip(self.config["zz"], self.config["nz"]): is_conflict_zz(zz, nz)
                else:
                    raise Exception("Please provide as many \'nz\' with corresponding \'zz\' (in a list) as the corresponding skycuts. ")
        else:
            self.config["zz"] = None
            self.config["nz"] = None 

        if self.config["with_quintessence"]: self.config["with_exact_time"] = True


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

            M = Class()
            M.set(cosmo_dict)
            M.set({'output': 'mPk', 'P_k_max_1/Mpc': 1.0, 'z_max_pk': zmax })
            M.compute()

            cosmo = {}

            if self.config["with_bias"]: 
                try: cosmo["bias"] = cosmo_dict["bias"]
                except:
                    print ("Please specify \'bias\'.")
                    raise

            if self.config["skycut"] == 1: zfid = self.config["z"]
            elif self.config["skycut"] > 1: zfid = self.config["z"][0]

            cosmo["k11"] = np.logspace(-5, 0, 200) # k in h/Mpc
            cosmo["P11"] = [M.pk(k*M.h(), zfid)*M.h()**3 for k in cosmo["k11"]] # P(k) in (Mpc/h)**3

            if self.config["skycut"] == 1:
                if self.config["multipole"] is not 0: cosmo["f"] = M.scale_independent_growth_factor_f(self.config["z"])
                if self.config["with_exact_time"]: 
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
                if self.config["with_AP"] and not self.config["with_redshift_bin"]:
                    cosmo["DA"] = np.array([M.angular_distance(z) * M.Hubble(0.) for z in self.config["z"]])
                    cosmo["H"] = np.array([M.Hubble(z) / M.Hubble(0.) for z in self.config["z"]])

            if self.config["with_redshift_bin"]:
                if self.config["skycut"] == 1:
                    cosmo["D"] = M.scale_independent_growth_factor(self.config["z"])
                    
                    cosmo["Dz"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["zz"]])
                    cosmo["fz"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["zz"]])

                    cosmo["DAz"] = np.array([M.angular_distance(z) * M.Hubble(0.) for z in self.config["zz"]])
                    cosmo["Hz"] = np.array([M.Hubble(z) / M.Hubble(0.) for z in self.config["zz"]])

                elif self.config["skycut"]   > 1:
                    cosmo["Dz"] = np.array([ [M.scale_independent_growth_factor(z) for z in zz] for zz in self.config["zz"] ])
                    cosmo["fz"] = np.array([ [M.scale_independent_growth_factor_f(z) for z in zz] for zz in self.config["zz"] ])

                    cosmo["DAz"] = np.array([ [M.angular_distance(z) * M.Hubble(0.) for z in zz] for zz in self.config["zz"] ])
                    cosmo["Hz"] = np.array([ [M.Hubble(z) / M.Hubble(0.)  for z in zz] for zz in self.config["zz"] ])

            if "w" in self.config["output"]:
                def comoving_distance(z): return M.angular_distance(z)*(1+z)*M.h()
                if self.config["skycut"] == 1: cosmo["rz"] = np.array([comoving_distance(z) for z in self.config["zz"]])
                elif self.config["skycut"] > 1: cosmo["rz"] = np.array([ [comoving_distance(z) for z in zz] for zz in self.config["zz"] ])

            return cosmo


class BiasCorrelator(Correlator):

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

