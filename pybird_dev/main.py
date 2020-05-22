import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from scipy.integrate import quad, dblquad, simps
from common import Common, common
from nonlinear import NonLinear
from resum import Resum
from projection import Projection

import sys
del sys.modules['common.Common'] 
del sys.modules['common.common'] 
del sys.modules['nonlinear.Nonlinear']
del sys.modules['resum.Resum'] 
del sys.modules['projection.Projection'] 
from common import Common, common
from nonlinear import NonLinear
from resum import Resum
from projection import Projection

def typename(onetype):
    if isinstance(onetype, tuple): return [t.__name__ for t in onetype]
    else: return [onetype.__name__]

class Option(object):
    def __init__(self, option_name, option_type, option_list=None, description='', default=None, verbose=False):
        self.verbose = verbose
        self.name = option_name
        self.type = option_type
        self.list = option_list
        self.description = description
        self.default = default
        self.value = None

    def check(self, option_key, option_value):
        is_option = False
        if self.verbose: print("\'%s\': \'%s\'" % (option_key, option_value))
        if isinstance(option_value, self.type):
            if self.list is None: is_option = True
            elif isinstance(option_value, str):
                if any(option_value in o for o in self.list): is_option = True
            elif isinstance(option_value, (int, float, bool)): 
                if any(option_value == o for o in self.list): is_option = True
        if is_option: 
            self.value = option_value
        else: self.error()
        return is_option

    def error(self):
        if self.list is None:
            try: raise Exception("Input error in \'%s\'; input options: %s. Check Correlator.info() in any doubt." % (self.name, typename(self.type)))
            except Exception as e: print(e)
        else:
            try: raise Exception("Input error in \'%s\'; input options: %s. Check Correlator.info() in any doubt." % (self.name, self.list))
            except Exception as e: print(e)

class Correlator(object):
    
    def __init__(self, option_dict=None):
        
        self.option_catalog = {
            "output": Option("output", str, ["mPk", "mCf", "bPk", "bCf"], 
                description="Correlator: matter / biased tracers - power spectrum / correlation function", 
                default="mPk") ,
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
                description="Time (in)dependent evaluation (for multi skycuts / redshift bin). For \'output\': \'w\' or \'skycut\' > 1, automatically set to False.",
                default=True) ,
            "z": Option("z", (float, list, np.ndarray),
                description="Effective redshift(s). Should match the number of skycuts.",
                default=None) ,
            "with_bias": Option("with_bias", bool, 
                description="Bias (in)dependent evalution. Automatically set to False for \'with_redshift\': False.",
                   default=True) ,
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
                default=False)
            "xdata": Option("xdata", (list, np.ndarray),
                description="Array of data points.",
                default=None) ,
            "with_AP": Option("AP", bool,
                description="Apply Alcock Paczynski effect.",
                default=False) ,
            "z_AP": Option("z_AP", (float, list, np.ndarray),
                description="Fiducial redshift used to convert coordinates to distances. A list can be provided for multi skycuts. ",
                default=None) ,
            "Omega_m_AP": Option("Omega_m_AP", (float, list, np.ndarray),
                description="Fiducial matter abundance used to convert coordinates to distances. A list can be provided for multi skycuts. If only one value is passed, use it for all skycuts.",
                default=None) ,
            "with_window": Option("window", bool,
                description="Apply mask.",
                default=False) ,
            "windowPk": Option("windowPk", (str, list),
                description="Path to Fourier convolution window file for \'output\': \'_Pk\'. If not provided, read \'windowCf\' and precompute it.",
                default=None) ,
            "windowCf": Option("windowCf", (str, list),
                description="Path to configuration space window file with columns: s [Mpc/h], Q0, Q2, Q4. A list can be provided for multi skycuts. Put \'None\' for each skycut without window.",
                default=None) ,
            "with_binning": Option("binning", bool,
                description="Apply binning for linear-spaced data bins.",
                default=False) ,
            "with_fibercol": Option("fibercol", bool,
                description="Apply fiber collision effective window corrections.",
                default=False) ,
        }
        
        if option_dict is not None: self.set(option_dict)
            
    def info(self, description=True):
        print ("Configuration \'set\' commands")
        print ("------------------------------")
        for (name, option) in zip(self.option_catalog, self.option_catalog.values()):
            if option.list is None: print("\'%s\': %s" % (name, typename(option.type)))
            else: print("\'%s\': %s ; options: %s" % (name, typename(option.type), option.list))
            if description:
                print ('    - %s' % option.description)
                print ('    * default: %s' % option.default)
    
    def set(self, option_dict):
        
        # Reading options provided by user
        for (name, option) in zip(self.option_catalog, self.option_catalog.values()):
            for option_key in option_dict:
                if option_key is name:
                    option.check(option_key, option_dict[option_key])
        
        # Setting unspecified options to default value 
        for (name, option) in zip(self.option_catalog, self.option_catalog.values()):
            if option.value is None: option.value = option.default
        
        # Setting optional configuration
        for (name, option) in zip(self.option_catalog, self.option_catalog.values()):
            if name is "output": ### At some point add multiple choice for output
                if "Cf" in option.value: self.cf = True
                else: self.cf = False
                if "m" in option.value: self.matter = True ### add a conflict flag
                else: self.matter = False
            if name is "multipole": self.Nl = option.value
            if name is "wedge": self.Nw = option.value
            if name is "skycut": self.Nsc = option.value
            if name is "with_time": self.with_time = option.value
            if name is "z": self.z = option.value
            if name is "with_bias": self.with_bias = option.value
            if name is "kmax": self.kmax = option.value
            #if name is "smin": self.smin = option.value
            if name is "accboost": self.accboost = float(option.value)
            if name is "optiresum": self.optiresum = option.value
            if name is "xdata": self.xdata = option.value
            if name is "with_AP": self.with_AP = option.value
            if name is "z_AP": self.z_AP = option.value
            if name is "Omega_m_AP": self.Omega_m_AP = option.value        
            if name is "with_window": self.with_window = option.value
            if name is "windowPk": self.windowPk = option.value
            if name is "windowCf": self.windowCf = option.value
            if name is "with_binning": self.with_binning = option.value
            if name is "with_fibercol": self.with_fibcol = option.value
        
        # Setting no-optional configuration
        self.smin = 1.
        self.smax = 1000.
        self.kmin = 0.001
        
        # Resolving conflict
        if self.xdata is not None:
            if self.cf:
                if self.xdata[0] < self.smin or self.xdata[-1] > self.smax:
                    raise Exception("Please specify a data point array \'xdata\' in: (%s, %s)." % (self.smin, self.smax))
            else:
                if self.xdata[0] < self.kmin or self.xdata[-1] > self.kmax:
                    raise Exception("Please specify a data point array \'xdata\' in: (%s, %s) or increase the kmax." % (self.kmin, self.kmax))
        
        if self.Nw is not 0: self.Nl = 3 # enforced
            
        if self.Nsc > 1: 
            self.with_time = False
            self.with_bias = False
            if len(self.z) is not self.Nsc:
                raise Exception("Please specify as many redshifts \'z\' as the number of skycuts.")
        
        if self.with_time: self.z = None
        
        if self.with_AP:
            if self.z_AP is None or self.Omega_m_AP is None:
                raise Exception("You asked to apply the AP effect. Please specify \'z_AP\' and \'Omega_m_AP\'.")
            if self.Nsc > 1:
                if len(self.z_AP) is not self.Nsc:
                    raise Exception("You asked to apply the AP effect. Please specify (in a list) as many \'z_AP\' as the number of skycuts.")
                
        if self.with_window:
            
            def conflict_window(windowCf):
                try: 
                    test = np.loadtxt(windowCf)
                    if self.cf: self.windowPk = None
                except:
                    print("You asked to apply a mask. Please specify a correct path to the configuration space window file.")
                    raise
            
            if self.Nsc is 1:
                conflict_window(self.windowCf)
            if self.Nsc > 1:
                for windowCf in self.windowCf:
                    if windowCf is not None:
                        conflict_window(windowCf)
        else:
            self.windowPk = None
            self.windowCf = None
        
        self.co = Common(Nl=self.Nl, kmax=self.kmax, smin=self.smin, optiresum=self.optiresum)
        self.nonlinear = NonLinear(load=True, save=True, co=self.co)
        self.resum = Resum(co=self.co)
        self.projection = Projection(self.k, self.Om_AP, self.zAP, cf=self.cf, 
            window_fourier_name=self.windowPk, path_to_window='./', window_configspace_file=self.windowCf,
            binning=self.with_binning, fibcol=self.with_fibcol, Nwedges=self.Nw, co=self.co)
        
    def setcosmo(cosmo):