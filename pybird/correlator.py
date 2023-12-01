from pybird.module import *

from pybird.common import Common, co
from pybird.bird import Bird
from pybird.nonlinear import NonLinear
from pybird.nnlo import NNLO_higher_derivative, NNLO_counterterm
from pybird.resum import Resum
from pybird.projection import Projection
from pybird.greenfunction import GreenFunction
from pybird.fourier import FourierTransform
from pybird.matching import Matching

# ### dev mode ###
import importlib, pybird
importlib.reload(pybird.common)
from pybird.common import Common, co
importlib.reload(pybird.bird)
from pybird.bird import Bird
importlib.reload(pybird.nonlinear)
from pybird.nonlinear import NonLinear
importlib.reload(pybird.resum)
from pybird.resum import Resum
importlib.reload(pybird.matching)
from pybird.matching import Matching
importlib.reload(pybird.module)
from pybird.module import *

# import pdb; pdb.set_trace()
################

class Correlator(object):

    def __init__(self, config_dict=None, load_engines=True):

        self.cosmo_catalog = {
            "pk_lin": Option("pk_lin", (list, ndarray),
                description="Linear matter power spectrum in [Mpc/h]^3",
                default=None) ,
            "kk": Option("kk", (list, ndarray),
                description="k-array in [h/Mpc] on which pk_lin is evaluated",
                default=None) ,
            "f": Option("f", float,
                description="Scale independent growth rate (for RSD). Automatically set to 0 for \'output\': \'m__\'.",
                default=None) ,
            "bias": Option("bias", dict,
                description="EFT parameters in dictionary to specify as \
                    (\'eft_basis\': \'eftoflss\') \{ \'b1\'(a), \'b2\'(a), \'b3\'(a), \'b4\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'westcoast\') \{ \'b1\'(a), \'c2\'(a), \'c4\'(a), \'b3\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'eastcoast\') \{ \'b1\'(a), \'b2\'(a), \'bG2\'(a), \'bgamma3\'(a), \'c0\', \'c2\'(b), \'c4\'(c), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    if (a): \'b\' in \'output\'; (b): \'multipole\'>=2; (d): \'with_stoch\' is True ",
                default=None) ,
            "H": Option("H", float,
                description="Hubble parameter by H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "DA": Option("DA", float,
                description="Angular distance times H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "z": Option("z", float,
                description="Effective redshift(s). To specify if \'with_time\' is False or \'with_exact_time\' is True.",
                default=None) ,
            "D": Option("D", float,
                description="Scale independent growth function. To specify if \'with_time\' is False, e.g., \'with_nonequal_time\' or \'with_redshift_bin\' is True.",
                default=None) ,
            "A": Option("A", float,
                description="Amplitude rescaling, i.e, A = A_s / A_s^\{fid\}. Default: A=1. If \'with_time\' is False, can in some ways be used as a fast parameter.",
                default=None) ,
            "Omega0_m": Option("Omega0_m", float,
                description="Fractional matter abundance at present time. To specify if \'with_exact_time\' is True.",
                default=None) ,
            "w0_fld": Option("w0_fld", float,
                description="Dark energy equation of state parameter. To specify in presence of dark energy if \'with_exact_time\' is True (otherwise w0 = -1).",
                default=None) ,
            "Dz": Option("Dz", (list, ndarray),
                description="Scale independent growth function over redshift bin. To specify if \'with_redshift_bin\' is True.",
                default=None) ,
            "fz": Option("fz", (list, ndarray),
                description="Scale independent growth rate over redshift bin. To specify if \'with_redshift_bin\' is True.",
                default=None) ,
            "rz": Option("rz", (list, ndarray),
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
            "Psmooth": Option("Psmooth", (list, ndarray),
                description="Smooth power spectrum. To specify if \'with_nnlo_counterterm\' is True.",
                default=None) ,
            "pk_lin_2": Option("pk_lin_2", (list, ndarray),
                description="Alternative linear matter power spectrum in [Mpc/h]^3 replacing \'pk_lin\' in the internal loop integrals (and resummation)",
                default=None) ,
        }

        self.c_catalog = {
            "output": Option("output", str, ["bPk", "bCf", "mPk", "mCf", "bmPk", "bmCf"],
                description="Correlator: biased tracers / matter / biased tracers-matter -- power spectrum / correlation function.",
                default="bPk") ,
            "multipole": Option("multipole", int, [0, 2, 3],
                description="Number of multipoles. 0: real space. 2: monopole + quadrupole. 3: monopole + quadrupole + hexadecapole.",
                default=2) ,
            "z": Option("z", float,
                description="Effective redshift.",
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
            "kmin": Option("kmin", float,
                description="kmin in [h/Mpc] for \'output\': \'_Pk\', to be chosen between [1e-4, 1e-3]. ",
                default=0.001) ,
            "kmax": Option("kmax", float,
                description="kmax in [h/Mpc] for \'output\': \'_Pk\'",
                default=0.25) ,
            "with_bias": Option("with_bias", bool,
                description="Bias (in)dependent evalution. Automatically set to False for \'with_time\': False.",
                   default=False) ,
            "eft_basis": Option("eft_basis", str,
                description="Basis of EFT parameters: \'eftoflss\' (default), \'westcoast\', or \'eastcoast\'. See cosmology command \'bias\' for more details.",
                default="eftoflss") ,
            "with_stoch": Option("with_stoch", bool,
                description="With stochastic terms.",
                   default=False) ,
            "with_nnlo_counterterm": Option("with_nnlo_counterterm", bool,
                description="With next-to-next-to-leading counterterm k^4 pk_lin.",
                default=False) ,
            "with_tidal_alignments": Option("with_tidal_alignments", bool,
                description="With tidal alignements: bq * (\mu^2 - 1/3) \delta_m ",
                default=False) ,
            "with_time": Option("with_time", bool,
                description="Time (in)dependent evaluation. For \'with_redshift_bin\': True, automatically set to False.",
                default=True) ,
            "with_exact_time": Option("with_exact_time", bool,
                description="Exact time dependence or EdS approximation.",
                default=False) ,
            "with_quintessence": Option("with_quintessence", bool,
                description="Clustering quintessence.",
                default=False) ,
            "with_nonequal_time": Option("with_nonequal_time", bool,
                description="Non equal time correlator. Automatically set \'with_time\' to False ",
                default=False) ,
            "z1": Option("z1", float,
                description="Redshift z_1 for non equal time correlator.",
                default=None) ,
            "z2": Option("z2", float,
                description="Redshift z_2 for non equal time correlator.",
                default=None) ,
            "with_resum": Option("with_resum", bool,
                description="Apply IR-resummation.",
                default=True) ,
            "optiresum": Option("optiresum", bool,
                description="[depreciated: keep on default False] True: Resumming only with the BAO peak. False: Resummation on the full correlation function.",
                default=False) ,
            "xdata": Option("xdata", (list, ndarray),
                description="Array of k [h/Mpc] (or s [Mpc/h]) on which to output the correlator. If \'with_binning\' is True, please provide the central k (or s). If not, it can be bin-weighted k (or s). If no \'xdata\' provided, output is on internal default array. ",
                default=None) ,
            "with_binning": Option("with_binning", bool,
                description="Apply binning for linear-spaced bins.",
                default=False) ,
            "binsize": Option("binsize", float,
                description="size of the bin.",
                default=None) ,
            "with_ap": Option("wity_AP", bool,
                description="Apply Alcock Paczynski effect. ",
                default=False) ,
            "H_fid": Option("H_fid", float,
                description="Hubble parameter by H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "D_fid": Option("D_fid", float,
                description="Angular distance times H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "with_survey_mask": Option("with_survey_mask", bool,
                description="Apply survey mask. Automatically set to False for \'output\': \'_Cf\'.",
                default=False) ,
            "survey_mask_arr_p": Option("survey_mask_arr_p", (list, ndarray),
                description="Mask convolution array for \'output\': \'_Pk\'.",
                default=None) ,
            "survey_mask_mat_kp": Option("survey_mask_mat_kp", (list, ndarray),
                description="Mask convolution matrix for \'output\': \'_Pk\'.",
                default=None) ,
            "with_fibercol": Option("with_fibercol", bool,
                description="Apply fiber collision effective window corrections.",
                default=False) ,
            "with_wedge": Option("with_wedge", bool,
                description="Rotate multipoles to wedges",
                default=False) ,
            "wedge_mat_wl": Option("wedge_mat_wl", (list, ndarray),
                description="multipole-to-wedge rotation matrix",
                default=None) ,
            "with_redshift_bin": Option("with_redshift_bin", bool,
                description="Account for the galaxy count distribution over a redshift bin.",
                default=False) ,
            "redshift_bin_zz": Option("redshift_bin_zz", (list, ndarray),
                description="Array of redshift points inside a redshift bin.",
                default=None) ,
            "redshift_bin_nz": Option("redshift_bin_nz", (list, ndarray),
                description="Galaxy counts distribution over a redshift bin.",
                default=None) ,
            "accboost": Option("accboost", int, [1, 2, 3],
                description="Sampling accuracy boost factor. Default k sampling: dk ~ 0.005 (k<0.3), dk ~ 0.01 (k>0.3). ",
                default=1) ,
            "fftaccboost": Option("fftaccboost", int, [1, 2, 3],
                description="FFTLog accuracy boost factor. Default FFTLog sampling : NFFT ~ 256. ",
                default=1) ,
            "fftbias": Option("fftbias", float,
                description="real power bias for fftlog decomposition of pk_lin (usually to keep to default value)",
                default=-1.6) ,
            "with_uvmatch_2": Option("with_uvmatch_2", bool,
                description="In case two linear power spectra \`pk_lin\` and \`pk_lin_2\` are provided (see description in cosmo_catalog), match the UV as in the case if only \`pk_lin\` would be provided. Implemented only for output=\`Pk\`. ",
                default=False) ,
            "with_irmatch_2": Option("with_uvmatch_2", bool,
                description="In case two linear power spectra \`pk_lin\` and \`pk_lin_2\` are provided (see description in cosmo_catalog), match the IR as in the case if only \`pk_lin\` would be provided. Implemented only for output=\`Pk\`. In practice, mostly useless since the IR pieces anyway cancel once adding 13 and 22, and for fftbias < -1.5, are set to 0 by dim. reg. ",
                default=False) ,
            "keep_loop_pieces_independent": Option("keep_loop_pieces_independent", bool,
                description="keep the loop pieces 13 and 22 independent (mainly for debugging)",
                default=False) ,
        }

        if config_dict is not None: self.set(config_dict, load_engines=load_engines)


    def info(self, description=True):

        for on in ['config', 'cosmo']:

            print ("\n")
            if on == 'config':
                print ("Configuration commands [.set(config_dict)]")
                print ("----------------------")
                catalog = self.c_catalog
            elif on == 'cosmo':
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
        self.c["smin"] = 1.
        self.c["smax"] = 1000.
        # self.c["kmin"] = 0.001

        # Checking for config conflict
        self.__is_config_conflict()

        # Setting list of EFT parameters required by the user to provide later
        self.__set_eft_parameters_list()

        # Loading PyBird engines
        self.__load_engines(load_engines=load_engines)

    def compute(self, cosmo_dict=None, cosmo_module=None, cosmo_engine=None, correlator_engine=None, do_core=True, do_survey_specific=True):

        if cosmo_dict: cosmo_dict_local = cosmo_dict.copy()
        elif cosmo_module and cosmo_engine: cosmo_dict_local = {}
        else: raise Exception('provide cosmo_dict or CLASSy engine with cosmo_module=\'class\' ')
        if cosmo_module: # works only with classy now
            cosmo_dict_class = self.set_cosmo(cosmo_dict, module=cosmo_module, engine=cosmo_engine)
            cosmo_dict_local.update(cosmo_dict_class)

        self.__read_cosmo(cosmo_dict_local)
        self.__is_cosmo_conflict()

        if do_core:
            self.bird = Bird(self.cosmo, with_bias=self.c["with_bias"], eft_basis=self.c["eft_basis"], with_stoch=self.c["with_stoch"], with_nnlo_counterterm=self.c["with_nnlo_counterterm"], co=self.co)
            if self.c["with_nnlo_counterterm"]: # we use smooth power spectrum since we don't want spurious BAO signals
                ilogPsmooth = interp1d(log(self.bird.kin), log(self.cosmo["Psmooth"]), fill_value='extrapolate')
                if self.c["with_cf"]: self.nnlo_counterterm.Cf(self.bird, ilogPsmooth)
                else: self.nnlo_counterterm.Ps(self.bird, ilogPsmooth)
            if not correlator_engine: self.nonlinear.PsCf(self.bird)
            elif correlator_engine: correlator_engine.nonlinear.PsCf(self.bird, c_alpha) # emu
            if self.c["with_uvmatch_2"]: self.matching.UVPsCf(self.bird) 
            if self.c["with_irmatch_2"]: self.matching.IRPsCf(self.bird) 
            if self.c["with_bias"]: self.bird.setPsCf(self.bias)
            else: self.bird.setPsCfl()
            if self.c["with_resum"]:
                if not correlator_engine: self.resum.PsCf(self.bird, makeIR=True, makeQ=False, setIR=False, setPs=False, setCf=False) # compute IR-correction pieces
                elif correlator_engine: correlator_engine.resum.PsCf(self.bird, c_alpha) # emu

        if do_survey_specific:
            if not self.c["with_time"]: self.bird.settime(self.cosmo, co=self.co)
            if self.c["with_resum"]: self.resum.PsCf(self.bird, makeIR=False, makeQ=True, setIR=True, setPs=True, setCf=self.c["with_cf"])
            if self.c["with_redshift_bin"]: self.projection.redshift(self.bird, self.cosmo["rz"], self.cosmo["Dz"], self.cosmo["fz"], pk=self.c["output"])
            if self.c["with_ap"]: self.projection.AP(self.bird)
            if self.c["with_fibercol"]: self.projection.fibcolWindow(self.bird)
            if self.c["with_survey_mask"]: self.projection.Window(self.bird)
            elif self.c["with_binning"]: self.projection.xbinning(self.bird) # no binning if 'with_survey_mask' since the mask should account for it.
            elif self.c["xdata"] is not None: self.projection.xdata(self.bird)
            if self.c["with_wedge"]: self.projection.Wedges(self.bird)

    def get(self, bias=None, what="full"):

        if not self.c["with_bias"]:
            self.__is_bias_conflict(bias)
            if "Pk" in self.c["output"]: self.bird.setreducePslb(self.bias, what=what)
            elif "Cf" in self.c["output"]: self.bird.setreduceCflb(self.bias, what=what)
        if "Pk" in self.c["output"]: return self.bird.fullPs
        elif "Cf" in self.c["output"]: return self.bird.fullCf

    def getmarg(self, bias, marg_gauss_eft_parameters_list):

        for p in marg_gauss_eft_parameters_list:
            if p not in self.gauss_eft_parameters_list:
                raise Exception("The parameter %s specified in getmarg() is not an available Gaussian EFT parameter to marginalize. Check your options. " % p)

        def marg(loopl, ctl, b1, f, stl=None, nnlol=None, bq=0):

            # concatenating multipoles: loopl.shape = (Nl, Nloop, Nk) -> loop.shape = (Nloop, Nl * Nk)
            loop = swapaxes(loopl, axis1=0, axis2=1).reshape(loopl.shape[1],-1)
            ct = swapaxes(ctl, axis1=0, axis2=1).reshape(ctl.shape[1],-1)
            if stl is not None: st = swapaxes(stl, axis1=0, axis2=1).reshape(stl.shape[1],-1)
            if nnlol is not None: nnlo = swapaxes(nnlol, axis1=0, axis2=1).reshape(nnlol.shape[1],-1)

            pg = empty(shape=(len(marg_gauss_eft_parameters_list), loop.shape[1]))
            for i, p in enumerate(marg_gauss_eft_parameters_list):
                if p in ['b3', 'bGamma3']:
                    if self.co.Nloop == 12: pg[i] = loop[3] + b1 * loop[7]                          # config["with_time"] = True
                    elif self.co.Nloop == 18: pg[i] = loop[3] + b1 * loop[7] + bq * loop[16]        # config["with_time"] = True, config["with_tidal_alignments"] = True
                    elif self.co.Nloop == 22: pg[i] = f * loop[8] + b1 * loop[16]                   # config["with_time"] = False, config["with_exact_time"] = False
                    elif self.co.Nloop == 35: pg[i] = f * loop[18] + b1 * loop[29]                  # config["with_time"] = False, config["with_exact_time"] = True
                    if p == 'bGamma3': pg[i] *= 6. # b3 = b1 + 15. * bG2 + 6. * bGamma3 : config["eft_basis"] = 'eastcoast'
                # counterterm : config["eft_basis"] = 'eftoflss' or 'westcoast'
                elif p == 'cct': pg[i] = 2 * (f * ct[0+3] + b1 * ct[0]) / self.c["km"]**2 # ~ 2 (b1 + f * mu^2) k^2/km^2 pk_lin
                elif p == 'cr1': pg[i] = 2 * (f * ct[1+3] + b1 * ct[1]) / self.c["kr"]**2 # ~ 2 (b1 mu^2 + f * mu^4) k^2/kr^2 pk_lin
                elif p == 'cr2': pg[i] = 2 * (f * ct[2+3] + b1 * ct[2]) / self.c["kr"]**2 # ~ 2 (b1 mu^4 + f * mu^6) k^2/kr^2 pk_lin
                # counterterm : config["eft_basis"] = 'eastcoast'                       # (2.15) and (2.23) of 2004.10607
                elif p in ['c0', 'c2', 'c4']:
                    ct0, ct2, ct4 = - 2 * ct[0], - 2 * f * ct[1], - 2 * f**2 * ct[2]    # - 2 ct0 k^2 pk_lin , - 2 ct2 f mu^2 k^2 pk_lin , - 2 ct4 f^2 mu^4 k^2 pk_lin
                    if p == 'c0':   pg[i] = ct0
                    elif p == 'c2': pg[i] = - f/3. * ct0 + ct2
                    elif p == 'c4': pg[i] = 3/35. * f**2 * ct0 - 6/7. * f * ct2 + ct4
                # stochastic term
                elif p == 'ce0': pg[i] = st[0] / self.c["nd"] # k^0 / nd mono
                elif p == 'ce1': pg[i] = st[1] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd mono
                elif p == 'ce2': pg[i] = st[2] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd quad
                # nnlo term: config["eft_basis"] = 'eftoflss' or 'westcoast'
                elif p == 'cr4': pg[i] = 0.25 * b1**2 * nnlo[0] / self.c["kr"]**4 # ~ 1/4 b1^2 k^4/kr^4 mu^4 pk_lin
                elif p == 'cr6': pg[i] = 0.25 * b1 * nnlo[1] / self.c["kr"]**4    # ~ 1/4 b1 k^4/kr^4 mu^6 pk_lin
                # nnlo term: config["eft_basis"] = 'eastcoast'
                elif p == 'ct': pg[i] = - f**4 * (b1**2 * nnlo[0] + 2. * b1 * f * nnlo[1] + f**2 * nnlo[2]) # ~ k^4 mu^4 pk_lin

            return pg

        def marg_from_bird(bird, bias_local):
            self.__is_bias_conflict(bias_local)
            if self.c["with_tidal_alignments"]: bq = self.bias["bq"]
            else: bq = 0.
            if "Pk" in self.c["output"]: return marg(bird.Ploopl, bird.Pctl, self.bias["b1"], bird.f, stl=bird.Pstl, nnlol=bird.Pnnlol, bq=bq)
            elif "Cf" in self.c["output"]: return marg(bird.Cloopl, bird.Cctl, self.bias["b1"], bird.f, stl=bird.Cstl, nnlol=bird.Cnnlol, bq=bq)

        return marg_from_bird(self.bird, bias)

    def __load_engines(self, load_engines=True):

        self.co = Common(Nl=self.c["multipole"], kmin=self.c["kmin"], kmax=self.c["kmax"], km=self.c["km"], kr=self.c["kr"], nd=self.c["nd"], eft_basis=self.c["eft_basis"],
            halohalo=self.c["halohalo"], with_cf=self.c["with_cf"], with_time=self.c["with_time"], accboost=self.c["accboost"], optiresum=self.c["optiresum"],
            exact_time=self.c["with_exact_time"], quintessence=self.c["with_quintessence"], with_uvmatch=self.c["with_uvmatch_2"], with_irmatch=self.c["with_irmatch_2"], 
            with_tidal_alignments=self.c["with_tidal_alignments"], nonequaltime=self.c["with_common_nonequal_time"], keep_loop_pieces_independent=self.c["keep_loop_pieces_independent"])
        if load_engines:
            self.nonlinear = NonLinear(load_matrix=True, save_matrix=True, NFFT=64, 
                                       # NFFT=256*self.c["fftaccboost"], 
                                       fftbias=self.c["fftbias"], co=self.co)
            if self.c["with_uvmatch_2"] or self.c["with_irmatch_2"]: self.matching = Matching(self.nonlinear, co=self.co)
            self.resum = Resum(co=self.co)
            self.projection = Projection(self.c["xdata"],
                with_ap=self.c["with_ap"], H_fid=self.c["H_fid"], D_fid=self.c["D_fid"],
                with_survey_mask=self.c["with_survey_mask"], survey_mask_arr_p=self.c["survey_mask_arr_p"], survey_mask_mat_kp=self.c["survey_mask_mat_kp"],
                with_binning=self.c["with_binning"], binsize=self.c["binsize"],
                fibcol=self.c["with_fibercol"],
                with_wedge=self.c["with_wedge"], wedge_mat_wl=self.c["wedge_mat_wl"],
                with_redshift_bin=self.c["with_redshift_bin"], redshift_bin_zz=self.c["redshift_bin_zz"], redshift_bin_nz=self.c["redshift_bin_nz"],
                co=self.co)
            if self.c["with_nnlo_counterterm"]: self.nnlo_counterterm = NNLO_counterterm(co=self.co)

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

        if self.c["with_bias"]: self.__is_bias_conflict()

        if self.cosmo["kk"] is None or self.cosmo["pk_lin"] is None:
            raise Exception("Please provide a linear matter power spectrum \'pk_lin\' and the corresponding \'kk\'. ")

        if len(self.cosmo["kk"]) != len(self.cosmo["pk_lin"]):
            raise Exception("Please provide a linear matter power spectrum \'pk_lin\' and the corresponding \'kk\' of same length.")

        if self.cosmo["kk"][0] > 1e-4 or self.cosmo["kk"][-1] < 1.:
            raise Exception("Please provide a linear matter spectrum \'pk_lin\' and the corresponding \'kk\' with min(kk) < 1e-4 and max(kk) > 1.")

        if self.c["multipole"] == 0: self.cosmo["f"] = 0.
        elif not self.c["with_redshift_bin"] and self.cosmo["f"] is None:
            raise Exception("Please specify the growth rate \'f\'.")
        elif self.c["with_redshift_bin"] and (self.cosmo["Dz"] is None or self.cosmo["fz"] is None):
            raise Exception("You asked to account the galaxy counts distribution. Please specify \'Dz\' and \'fz\'. ")

        if not self.c["with_time"] and self.cosmo["D"] is None:
            raise Exception("Please specify the growth factor \'D\'.")

        if self.c["with_nonequal_time"] and (self.cosmo["D1"] is None or self.cosmo["D2"] is None or self.cosmo["f1"] is None or self.cosmo["f2"] is None):
            raise Exception("You asked nonequal time correlator. Pleas specify: \'D1\', \'D2\', \'f1\', \'f2\'.  ")

        if self.c["with_ap"] and (self.cosmo["H"] is None or self.cosmo["DA"] is None):
            raise Exception("You asked to apply the AP effect. Please specify \'H\' and \'DA\'. ")

        if not self.c["with_time"] and self.cosmo["A"]: self.cosmo["D"] *= self.cosmo["A"]**.5

    def __is_bias_conflict(self, bias=None):
        if bias is not None: self.cosmo["bias"] = bias
        if self.cosmo["bias"] is None: raise Exception("Please specify \'bias\'. ")
        if isinstance(self.cosmo["bias"], (list, ndarray)): self.cosmo["bias"] = self.cosmo["bias"][0]
        if not isinstance(self.cosmo["bias"], dict): raise Exception("Please specify bias in a dict. ")

        for p in self.eft_parameters_list:
            if p not in self.cosmo["bias"]:
                raise Exception ("%s not found, please provide (given command \'eft_basis\': \'%s\') %s" % (p, self.c["eft_basis"], self.eft_parameters_list))

        # PZ: here I should auto-fill the EFT parameters for all output options!!!

        self.bias = self.cosmo["bias"]

        if "b" in self.c["output"]:
            if "westcoast" in self.c["eft_basis"]:
                self.bias["b2"] = 2.**-.5 * (self.bias["c2"] + self.bias["c4"])
                self.bias["b4"] = 2.**-.5 * (self.bias["c2"] - self.bias["c4"])
            elif "eastcoast" in self.c["eft_basis"]:
                self.bias["b2"] = self.bias["b1"] + 7/2. * self.bias["bG2"]
                self.bias["b3"] = self.bias["b1"] + 15. * self.bias["bG2"] + 6. * self.bias["bGamma3"]
                self.bias["b4"] = 1/2. * self.bias["bt2"] - 7/2. * self.bias["bG2"]
        elif "m" in self.c["output"]: self.bias.update({"b1": 1., "b2": 1., "b3": 1., "b4": 0.})
        if self.c["multipole"] == 0: self.bias.update({"cr1": 0., "cr2": 0.})

    def __set_eft_parameters_list(self):

        if self.c["eft_basis"] in ["eftoflss", "westcoast"]:
            self.gauss_eft_parameters_list = ['cct']
            if self.c["multipole"] >= 2: self.gauss_eft_parameters_list.extend(['cr1', 'cr2'])
        elif self.c["eft_basis"] == "eastcoast":
            self.gauss_eft_parameters_list = ['c0']
            if self.c["multipole"] >= 2: self.gauss_eft_parameters_list.extend(['c2', 'c4'])
        if self.c["with_stoch"]: self.gauss_eft_parameters_list.extend(['ce0', 'ce1', 'ce2'])
        if self.c["with_nnlo_counterterm"]:
            if self.c["eft_basis"] in ["eftoflss", "westcoast"]: self.gauss_eft_parameters_list.extend(['cr4', 'cr6'])
            elif self.c["eft_basis"] == "eastcoast": self.gauss_eft_parameters_list.append('ct')
        self.eft_parameters_list = deepcopy(self.gauss_eft_parameters_list)
        if "b" in self.c["output"]:
            if self.c["eft_basis"] in ["eftoflss", "westcoast"]: self.gauss_eft_parameters_list.append('b3')
            elif self.c["eft_basis"] == "eastcoast": self.gauss_eft_parameters_list.append('bGamma3')
            if self.c["eft_basis"] == "eftoflss": self.eft_parameters_list.extend(['b1', 'b2', 'b3', 'b4'])
            elif self.c["eft_basis"] == "westcoast": self.eft_parameters_list.extend(['b1', 'c2', 'b3', 'c4'])
            elif self.c["eft_basis"] == "eastcoast": self.eft_parameters_list.extend(['b1', 'bt2', 'bG2', 'bGamma3'])
        if self.c["with_tidal_alignments"]: self.eft_parameters_list.append('bq')

    def __read_config(self, config_dict):

        # Checking if the inputs are consistent with the options
        for config_key in config_dict:
            is_config = False
            for (name, config) in zip(self.c_catalog, self.c_catalog.values()):
                if config_key == name:
                    config.check(config_key, config_dict[config_key])
                    is_config = True
            ### Keep this warning for typos in options that are then unread...
            if not is_config:
                raise Exception("%s is not an available configuration option. Please check correlator.info() for help. " % config_key)

        # Setting unspecified configs to default value
        for (name, config) in zip(self.c_catalog, self.c_catalog.values()):
            if config.value is None: config.value = config.default

        # Translating the catalog to a dict
        self.c = translate_catalog_to_dict(self.c_catalog)

        self.c["accboost"] = float(self.c["accboost"])

    def __is_config_conflict(self):

        if "Cf" in self.c["output"]: self.c.update({"with_cf": True, "with_survey_mask": False, "with_stoch": False})
        else: self.c["with_cf"] = False

        if "bm" in self.c["output"]: self.c["halohalo"] = False
        else: self.c["halohalo"] = True

        if self.c["with_quintessence"]: self.c["with_exact_time"] = True

        self.c["with_common_nonequal_time"] = False # this is to pass for the common Class to setup the numbers of loops (22 and 13 gathered by default)
        if self.c["with_nonequal_time"]:
            self.c.update({"with_bias": False, "with_time": False, "with_common_nonequal_time": True}) # with_common_nonequal_time is to pass for the common Class to setup the numbers of loops (22 and 13 seperated since they have different time dependence)
            if self.c["z1"] is None or self.c["z2"] is None: print("Please specify \'z1\' and \'z2\' for nonequaltime correlator. ")

        if self.c["with_ap"] and (self.c["H_fid"] is None or self.c["D_fid"] is None):
                raise Exception("You asked to apply the AP effect. Please specify \'H_fid\' and \'D_fid\'. ")

        if self.c["with_survey_mask"] and (self.c["survey_mask_arr_p"] is None or self.c["survey_mask_mat_kp"] is None): raise Exception("Survey mask: on. Please specify \'survey_mask_arr_p\' and \'survey_mask_mat_kp\'. ")
        if self.c["with_binning"] and self.c["binsize"] is None: raise Exception("Binning: on. Please provide \'binsize\'.")
        if self.c["with_redshift_bin"]:
            self.c.update({"with_bias": False, "with_time": False, "with_cf": True}) # even for the Pk, we first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            if self.c["redshift_bin_zz"] is None or self.c["redshift_bin_nz"] is None: raise Exception("You asked to account for the galaxy counts distribution over a redshift bins. Please provide a distribution \'redshift_bin_nz\' and corresponding \'redshift_bin_zz\'. ")
        if self.c["with_wedge"] and self.c["wedge_mat_wl"] is None: raise Exception("Please specify \'wedge_mat_wl\'.")

    def set_cosmo(self, cosmo_dict, module='class', engine=None):

        cosmo = {}

        if module == 'class':

            log10kmax = 0
            if self.c["with_nnlo_counterterm"]: log10kmax = 1 # slower, but required for the wiggle-no-wiggle split scheme

            if not engine:
                from classy import Class
                cosmo_dict_local = cosmo_dict.copy()
                if self.c["with_bias"] and "bias" in cosmo_dict: del cosmo_dict_local["bias"] # remove to not pass it to classy that otherwise complains
                if not self.c["with_time"] and "A" in cosmo_dict: del cosmo_dict_local["A"] # same as above
                if self.c["with_redshift_bin"]: zmax = max(self.c["redshift_bin_zz"])
                else: zmax = self.c["z"]
                M = Class()
                M.set(cosmo_dict_local)
                M.set({'output': 'mPk', 'P_k_max_h/Mpc': 10.**log10kmax, 'z_max_pk': zmax })
                M.compute()
            else: M = engine

            cosmo["kk"] = logspace(-5, log10kmax, 200)  # k in h/Mpc
            cosmo["pk_lin"] = array([M.pk_lin(k*M.h(), self.c["z"])*M.h()**3 for k in cosmo["kk"]]) # P(k) in (Mpc/h)**3

            if self.c["multipole"] > 0: cosmo["f"] = M.scale_independent_growth_factor_f(self.c["z"])
            if not self.c["with_time"]: cosmo["D"] = M.scale_independent_growth_factor(self.c["z"])
            if self.c["with_nonequal_time"]:
                cosmo["D1"] = M.scale_independent_growth_factor(self.c["z1"])
                cosmo["D2"] = M.scale_independent_growth_factor(self.c["z2"])
                cosmo["f1"] = M.scale_independent_growth_factor_f(self.c["z1"])
                cosmo["f2"] = M.scale_independent_growth_factor_f(self.c["z2"])
            if self.c["with_exact_time"] or self.c["with_quintessence"]:
                cosmo["z"] = self.c["z"]
                cosmo["Omega0_m"] = M.Omega0_m()
                if "w0_fld" in cosmo_dict: cosmo["w0_fld"] = cosmo_dict["w0_fld"]
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

            if self.c["with_nnlo_counterterm"]: cosmo["kk"], cosmo["Psmooth"], cosmo["pk_lin"] = get_smooth_wiggle_resc(cosmo["kk"], cosmo["pk_lin"])

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
