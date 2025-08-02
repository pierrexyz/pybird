from pybird.module import *
from pybird.common import Common, co
from pybird.bird import Bird
from pybird.nonlinear import NonLinear
# from pybird.nnlo import NNLO_higher_derivative, NNLO_counterterm
from pybird.resum import Resum
from pybird.projection import Projection
from pybird.greenfunction import GreenFunction
from pybird.fourier import FourierTransform
from pybird.matching import Matching
from pybird.cosmo import Cosmo
from pybird.utils import get_data_path


class Correlator(object):
    """A class for computing cosmological correlation functions.
    
    The Correlator class provides functionality to compute various cosmological 
    correlation functions, including power spectra and correlation functions for
    biased tracers and matter. It implements Effective Field Theory (EFT) of 
    Large Scale Structure calculations with various options for bias models, 
    IR resummation, AP effect, and other corrections.
    
    Attributes:
        cosmo_catalog (dict): Catalog of cosmological parameters options.
        c_catalog (dict): Catalog of configuration options.
        cosmo (dict): Current cosmological parameters.
        c (dict): Current configuration parameters.
        bias (dict): Current bias parameters.
        bird (Bird): Main computational object containing results.
        co (Common): Common parameters and utilities.
        nonlinear (NonLinear): Engine for nonlinear calculations.
        resum (Resum): Engine for IR resummation.
        projection (Projection): Engine for projections and coordinate transformations.
        matching (Matching): Engine for IR/UV matching if needed.
        emulator (Emulator): Emulator for faster calculations if enabled.
    
    Methods:
        info(): Display information about available configuration and cosmology parameters.
        set(): Set configuration parameters and initialize engines.
        compute(): Compute cosmological correlations based on provided parameters.
        get(): Get the computed power spectrum or correlation function.
        getmarg(): Get marginalized EFT parameters.
        load_engines(): Load and initialize computational engines.
    
    Private Methods:
        __read_cosmo(): Read and validate cosmological parameters.
        __is_cosmo_conflict(): Check for conflicts in cosmological parameters.
        __is_bias_conflict(): Check for conflicts in bias parameters.
        __set_eft_parameters_list(): Set lists of required EFT parameters.
        __read_config(): Read and validate configuration parameters.
        __is_config_conflict(): Check for conflicts in configuration parameters.
    """

    def __init__(self, config_dict=None, load_engines=True):

        self.cosmo_catalog = {
            "pk_lin": Option("pk_lin", (list, ndarray),
                description="Linear matter power spectrum in [Mpc/h]^3",
                default=None) ,
            "kk": Option("kk", (list, ndarray),
                description="k-array in [h/Mpc] on which pk_lin is evaluated",
                default=None) ,
            "f": Option("f", (float, list, ndarray),
                description="Scale independent growth rate (for RSD). Automatically set to 0 for \'output\': \'m__\'.",
                default=None) ,
            "bias": Option("bias", dict,
                description="EFT parameters in dictionary to specify as \
                    (\'eft_basis\': \'eftoflss\') \{ \'b1\'(a), \'b2\'(a), \'b3\'(a), \'b4\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'westcoast\') \{ \'b1\'(a), \'c2\'(a), \'c4\'(a), \'b3\'(a), \'cct\', \'cr1\'(b), \'cr2\'(b), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    (\'eft_basis\': \'eastcoast\') \{ \'b1\'(a), \'b2\'(a), \'bG2\'(a), \'bgamma3\'(a), \'c0\', \'c2\'(b), \'c4\'(c), \'ce0\'(d), \'ce1\'(d), \'ce2\'(d)] \} \
                    if (a): \'b\' in \'output\'; (b): \'multipole\'>=2; (d): \'with_stoch\' is True ",
                default=None) ,
            "H": Option("H", (float, list, ndarray),
                description="Hubble parameter by H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "DA": Option("DA", (float, list, ndarray),
                description="Angular distance times H_0. To specify if \'with_ap\' is True.",
                default=None) ,
            "z": Option("z", (float, list, ndarray),
                description="Effective redshift(s). To specify if \'with_time\' is False or \'with_exact_time\' is True.",
                default=None) ,
            "D": Option("D", (float, list, ndarray),
                description="Scale independent growth function. To specify if \'with_time\' is False, e.g., \'with_nonequal_time\' or \'with_redshift_bin\' is True.",
                default=None) ,
            "A": Option("A", float,
                description="Amplitude rescaling, i.e, A = A_s / A_s^\{fid\}. Default: A=1. If \'with_time\' is False, can in some ways be used as a fast parameter.",
                default=None) ,
            "Omega0_m": Option("Omega0_m", (float, list, ndarray),
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
                default=True) ,
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
            "survey_mask_arr_p": Option("survey_mask_arr_p", (list, ndarray, np.ndarray),
                description="Mask convolution array for \'output\': \'_Pk\'.",
                default=None) ,
            "survey_mask_mat_kp": Option("survey_mask_mat_kp", (list, ndarray, np.ndarray),
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
                default=2) ,
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
            "with_emu": Option("with_emu", bool,
                description="Use emulator",
                default=False) ,
            "emu_path": Option("emu_path", str,
                description="Path to emulators",
                default=str((get_data_path()).resolve())),
            "knots_path": Option("knots_path", str,
                description="Path to emulator knots",
                default=str((get_data_path() / "knots.npy").resolve())),
        }

        if config_dict is not None: self.set(config_dict, load_engines=load_engines)


    def info(self, description=True):
        """Display information about available configuration and cosmology parameters.
        
        Parameters
        ----------
        description : bool, optional
            Whether to include parameter descriptions and defaults, by default True
            
        Notes
        -----
        This method prints two catalogs:
        - Configuration commands: parameters for .set(config_dict)
        - Cosmology commands: parameters for .compute(cosmo_dict)
        """

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
        """Set configuration parameters and initialize engines.
        
        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration parameters
        load_engines : bool, optional
            Whether to load and initialize computational engines, by default True
            
        Notes
        -----
        This method:
        1. Reads and validates the provided configuration
        2. Sets default values for optional parameters
        3. Checks for configuration conflicts
        4. Loads PyBird computational engines if requested
        """

        # Reading config provided by user
        self.__read_config(config_dict)

        # Setting no-optional config
        self.c["smin"] = 1.
        self.c["smax"] = 1000.

        # Checking for config conflict
        self.__is_config_conflict() 

        # Setting list of EFT parameters required by the user to provide later
        self.__set_eft_parameters_list()

        # Loading PyBird engines
        self.load_engines(load_engines=load_engines)
        
    def compute(self, cosmo_dict=None, cosmo_module=None, cosmo_engine=None, do_core=True, do_survey_specific=True):
        """Compute cosmological correlations based on provided parameters.
        
        Parameters
        ----------
        cosmo_dict : dict, optional
            Dictionary of cosmological parameters for PyBird inputs
        cosmo_module : str, optional
            Name of Boltzmann solver to call internally (e.g., 'classy', 'camb')
        cosmo_engine : object, optional
            External Boltzmann solver engine object
        do_core : bool, optional
            Whether to perform core EFT calculations, by default True
        do_survey_specific : bool, optional
            Whether to apply survey-specific effects, by default True
            
        Notes
        -----
        You must provide either:
        - cosmo_dict with PyBird parameters, or
        - cosmo_dict with cosmological parameters + cosmo_module, or
        - cosmo_module + cosmo_engine for external Boltzmann solver
        
        The method handles cosmological parameter processing, validates inputs,
        and performs the main EFT calculations for power spectra or correlation functions.
        """

        if cosmo_dict:
            if not cosmo_module: cosmo_dict_local = cosmo_dict.copy()
            else: # cosmo parameters to be passed to cosmo_module called internally 
                cosmo_class = Cosmo(config = self.c)
                cosmo_dict_local = cosmo_class.set_cosmo(cosmo_dict, module=cosmo_module, engine=cosmo_engine)

        elif cosmo_module and cosmo_engine: 
            cosmo_dict_local = {}
            cosmo_class = Cosmo(config = self.c)
            cosmo_dict_class = cosmo_class.set_cosmo(cosmo_dict, module=cosmo_module, engine=cosmo_engine)
            cosmo_dict_local.update(cosmo_dict_class)

        else: raise Exception('provide \'cosmo_dict\' of PyBird inputs or \'cosmo_dict\' of cosmological parameters to be passed either to a \'cosmo_module\' (name of the Boltzmann solver) to be called internally, or to an external \'cosmo_engine\' (Boltzmann solver)')
        self.__read_cosmo(cosmo_dict_local)
        self.__is_cosmo_conflict()

        if self.c["with_bias"]: self.__is_bias_conflict()

        if do_core:
            self.bird = Bird(self.cosmo, with_bias=self.c["with_bias"], eft_basis=self.c["eft_basis"], with_stoch=self.c["with_stoch"], with_nnlo_counterterm=self.c["with_nnlo_counterterm"], co=self.co)
            if self.c["with_nnlo_counterterm"]: # we used to use a smooth power spectrum since we don't want spurious BAO signals
                # ilogPsmooth = interp1d(log(self.bird.kin), log(self.cosmo["Psmooth"]), fill_value='extrapolate')
                # if self.c["with_cf"]: self.nnlo_counterterm.Cf(self.bird, ilogPsmooth)
                # else: self.nnlo_counterterm.Ps(self.bird, ilogPsmooth)
                self.bird.Pnnlo = self.co.k**4 * self.bird.P11 # good approximation in the end
            if not self.c["with_emu"]:
                self.nonlinear.PsCf(self.bird)

            if self.c["with_bias"]:
                self.bird.setPsCf(self.bias)
            else:
                if not self.c["with_emu"]:
                    self.bird.setPsCfl()
                else:
                    self.bird.setPsCfl(with_loop_and_cf=False)
                    if not self.c["with_resum"]: # if we are doing resum we capture this part anyway 
                        self.emulator.setPsCfl(bird=self.bird, kk=self.cosmo["kk"], pk=self.cosmo["pk_lin"], time=False, make_params=True)

                if self.c["with_uvmatch_2"]: self.matching.UVPsCf(self.bird) 
                if self.c["with_irmatch_2"]: self.matching.IRPsCf(self.bird) 

            if self.c["with_resum"]:
                if not self.c["with_emu"]: self.resum.PsCf(self.bird, setCf=self.c["with_cf"]) # PZ: setCf should be only for Cf no?
                else: self.emulator.PsCf_resum(self.bird, self.cosmo["kk"], self.cosmo["pk_lin"], f=self.cosmo["f"], time=False, make_params=True) # emu resum

        if do_survey_specific:
            if self.c["with_redshift_bin"]: self.projection.redshift(self.bird, self.cosmo["rz"], self.cosmo["Dz"], self.cosmo["fz"], pk=self.c["output"])
            if self.c["with_ap"]: self.projection.AP(self.bird)
            if self.c["with_fibercol"]: self.projection.fibcolWindow(self.bird)
            if self.c["with_survey_mask"]:
                self.projection.Window(self.bird)
            elif self.c["with_binning"]: self.projection.xbinning(self.bird) # no binning if 'with_survey_mask' since the mask should account for it.
            elif self.c["xdata"] is not None: self.projection.xdata(self.bird)
            if self.c["with_wedge"]: self.projection.Wedges(self.bird)

    def get(self, bias=None, what="full"):
        """Get the computed power spectrum or correlation function.
        
        Parameters
        ----------
        bias : dict, optional
            Bias parameters dictionary. If None, uses the bias set during compute()
        what : str, optional
            Which components to return ('full', 'linear', 'loop', etc.), by default "full"
            
        Returns
        -------
        ndarray
            Computed power spectrum (if output='Pk') or correlation function (if output='Cf')
            Shape depends on configuration (multipoles, redshift bins, etc.)
        """

        if not self.c["with_bias"]:
            self.__is_bias_conflict(bias)
            if "Pk" in self.c["output"]: self.bird.setreducePslb(self.bias, what=what)
            elif "Cf" in self.c["output"]: self.bird.setreduceCflb(self.bias, what=what)
        if "Pk" in self.c["output"]: return self.bird.fullPs
        elif "Cf" in self.c["output"]: return self.bird.fullCf

    def getmarg(self, bias, marg_gauss_eft_parameters_list):
        """Get marginalized EFT parameters.
        
        Parameters
        ----------
        bias : dict
            Bias parameters dictionary
        marg_gauss_eft_parameters_list : list
            List of EFT parameter names to marginalize over using Gaussian priors
            
        Notes
        -----
        This method implements analytical marginalization over nuisance EFT parameters
        assuming Gaussian priors. Common parameters to marginalize include stochastic
        terms like 'ce0', 'ce1', 'ce2' and higher-order bias terms.
        """

        for p in marg_gauss_eft_parameters_list:
            if p not in self.gauss_eft_parameters_list:
                raise Exception("The parameter %s specified in getmarg() is not an available Gaussian EFT parameter to marginalize. Check your options. " % p)

        def marg(loopl, ctl, b1, f, stl=None, nnlol=None, bq=0):
            if is_jax: # jax version of this function
                loop = swapaxes(loopl, 0, 1).reshape(loopl.shape[1], -1)
                ct = swapaxes(ctl, 0, 1).reshape(ctl.shape[1], -1)
                st = swapaxes(stl, 0, 1).reshape(stl.shape[1], -1) if stl is not None else None
                nnlo = swapaxes(nnlol, 0, 1).reshape(nnlol.shape[1], -1) if nnlol is not None else None

                pg = empty((len(marg_gauss_eft_parameters_list), loop.shape[1]))

                for i, p in enumerate(marg_gauss_eft_parameters_list):
                    if p in ['b3', 'bGamma3']:
                        # Implementing the conditional logic with jnp.where for different 'Nloop' values
                        pg_val = where(self.co.Nloop == 12, loop[3] + b1 * loop[7],
                                where(self.co.Nloop == 18, loop[3] + b1* loop[7] + bq * loop[16],
                                where(self.co.Nloop == 22, f * loop[8] + b1 * loop[16],
                                where(self.co.Nloop == 35, f * loop[18] + b1* loop[29], 0))))
                        if p == 'bGamma3':
                            pg_val *= 6.0
                        pg = pg.at[i].set(pg_val)
                    elif p == 'cct':
                        pg_val = 2 * (f * ct[3] + b1 * ct[0]) / self.c["km"]**2
                        pg = pg.at[i].set(pg_val)
                    elif p == 'cr1':
                        pg_val = 2 * (f * ct[4] + b1 * ct[1]) / self.c["kr"]**2
                        pg = pg.at[i].set(pg_val)
                    elif p == 'cr2':
                        pg_val = 2 * (f * ct[5] + b1 * ct[2]) / self.c["kr"]**2
                        pg = pg.at[i].set(pg_val)
                    elif p in ['c0', 'c2', 'c4']:
                        ct0, ct2, ct4 = -2 * ct[0], -2 * f * ct[1], -2 * f**2 * ct[2]
                        pg_val = where(p == 'c0', ct0,
                                where(p == 'c2', -f/3. * ct0 + ct2,
                                where(p == 'c4', 3/35. * f**2 * ct0 - 6/7. * f * ct2 + ct4, 0)))
                        pg = pg.at[i].set(pg_val)
                    elif p == 'ce0':
                        pg_val = st[0] / self.c["nd"]
                        pg = pg.at[i].set(pg_val)
                    elif p == 'ce1':
                        pg_val = st[1] / self.c["km"]**2 / self.c["nd"]
                        pg = pg.at[i].set(pg_val)
                    elif p == 'ce2':
                        if self.co.eft_basis == 'eftoflss':
                            pg_val = f * st[2] / self.c["km"]**2 / self.c["nd"] # f mu^2 k^2 / km^2 / nd
                        elif self.co.eft_basis in ['eastcoast', 'westcoast']:
                            pg_val = st[2] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd quad | mu^2 k^2 / km^2 / nd
                        pg = pg.at[i].set(pg_val)
                    elif p == 'cr4':
                        pg_val = 0.25 * b1**2 * nnlo[0] / self.c["kr"]**4 # ~ 1/4 b1^2 k^4/kr^4 mu^4 pk_lin
                        pg = pg.at[i].set(pg_val)
                    elif p == 'cr6':
                        pg_val = 0.25 * b1 * nnlo[1] / self.c["kr"]**4  # ~ 1/4 b1 k^4/kr^4 mu^6 pk_lin
                        pg = pg.at[i].set(pg_val)
                    elif p == 'ct':
                        pg_val = -f**4 * (b1**2 * nnlo[0] + 2. * b1 * f * nnlo[1] + f**2 * nnlo[2])
                        pg = pg.at[i].set(pg_val)
                return pg

            else:
                # concatenating multipoles: loopl.shape = (Nl, Nloop, Nk) -> loop.shape = (Nloop, Nl * Nk)
                loop =swapaxes(loopl, axis1=0, axis2=1).reshape(loopl.shape[1],-1)
                ct = swapaxes(ctl, axis1=0, axis2=1).reshape(ctl.shape[1],-1)
                if stl is not None: st = swapaxes(stl, axis1=0, axis2=1).reshape(stl.shape[1],-1)
                if nnlol is not None: nnlo = swapaxes(nnlol, axis1=0, axis2=1).reshape(nnlol.shape[1],-1)

                pg = empty(shape=(len(marg_gauss_eft_parameters_list), loop.shape[1]))
                for i, p in enumerate(marg_gauss_eft_parameters_list):
                    if p in ['b3', 'bGamma3']:
                        if self.co.Nloop == 12: pg[i] = loop[3] + b1 * loop[7]                          # config["with_time"] = True
                        elif self.co.Nloop == 18: pg[i] = loop[3] + b1 * loop[7] + bq * loop[16]        # config["with_time"] = True, config["with_tidal_alignments"] = True
                        elif self.co.Nloop == 22: pg[i] = f * loop[8] + b1 * loop[16]                   # config["with_time"] = False, config["with_exact_time"] = False
                        elif self.co.Nloop == 35: 
                            pg[i] = f * loop[18] + b1 * loop[29]                  # config["with_time"] = False, config["with_exact_time"] = True
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
                    elif p == 'ce0': pg[i] = st[0] / self.c["nd"] # k^0 / nd 
                    elif p == 'ce1': pg[i] = st[1] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd 
                    elif p == 'ce2': 
                        if self.co.eft_basis == 'eftoflss':
                            pg[i] = f * st[2] / self.c["km"]**2 / self.c["nd"] # f mu^2 k^2 / kr^2 / nd
                        elif self.co.eft_basis in ['eastcoast', 'westcoast']:
                            pg[i] = st[2] / self.c["km"]**2 / self.c["nd"] # k^2 / km^2 / nd quad | mu^2 k^2 / km^2 / nd
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

    def load_engines(self, load_engines=True):
        """Load and initialize computational engines.
        
        Parameters
        ----------
        load_engines : bool, optional
            Whether to actually load the engines, by default True
            
        Notes
        -----
        This method initializes all the computational engines needed for PyBird:
        - Common: shared parameters and utilities
        - NonLinear: one-loop calculations
        - Resum: IR resummation
        - Projection: multipole projections and survey effects
        - Matching: UV/IR matching (if needed)
        - Emulator: neural network emulator (if enabled)
        """

        self.co = Common(Nl=self.c["multipole"], kmin=self.c["kmin"], kmax=self.c["kmax"], km=self.c["km"], kr=self.c["kr"], nd=self.c["nd"], eft_basis=self.c["eft_basis"],
            halohalo=self.c["halohalo"], with_cf=self.c["with_cf"], with_time=self.c["with_time"], accboost=self.c["accboost"], optiresum=self.c["optiresum"],
            exact_time=self.c["with_exact_time"], quintessence=self.c["with_quintessence"], with_uvmatch=self.c["with_uvmatch_2"], with_irmatch=self.c["with_irmatch_2"], 
            with_emu=self.c["with_emu"],
            with_tidal_alignments=self.c["with_tidal_alignments"], nonequaltime=self.c["with_common_nonequal_time"], keep_loop_pieces_independent=self.c["keep_loop_pieces_independent"])
        if load_engines:
            if self.c["with_emu"]:
                from pybird.emulator import Emulator
                self.emulator = Emulator(self.c["emu_path"], self.c["knots_path"], co=self.co)
            else: 
                self.nonlinear = NonLinear(load_matrix=True, save_matrix=True, NFFT=256*self.c["fftaccboost"], fftbias=self.c["fftbias"], co=self.co) 
                self.resum = Resum(co=self.co)
            if self.c["with_uvmatch_2"] or self.c["with_irmatch_2"]: 
                self.matching = Matching(self.nonlinear, co=self.co)
            self.projection = Projection(self.c["xdata"],
                with_ap=self.c["with_ap"], H_fid=self.c["H_fid"], D_fid=self.c["D_fid"],
                with_survey_mask=self.c["with_survey_mask"], survey_mask_arr_p=self.c["survey_mask_arr_p"], survey_mask_mat_kp=self.c["survey_mask_mat_kp"],
                with_binning=self.c["with_binning"], binsize=self.c["binsize"],
                fibcol=self.c["with_fibercol"],
                with_wedge=self.c["with_wedge"], wedge_mat_wl=self.c["wedge_mat_wl"],
                with_redshift_bin=self.c["with_redshift_bin"], redshift_bin_zz=self.c["redshift_bin_zz"], redshift_bin_nz=self.c["redshift_bin_nz"],
                co=self.co)
            # if self.c["with_nnlo_counterterm"]: self.nnlo_counterterm = NNLO_counterterm(co=self.co)

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
        if is_jax:
            error_messages = []
            
            # Check for required kk and pk_lin
            if self.cosmo["kk"] is None or self.cosmo["pk_lin"] is None:
                error_messages.append("Please provide a linear matter power spectrum 'pk_lin' and the corresponding 'kk'.")
            
            # Handle f based on multipole setting
            if self.c["multipole"] == 0:
                # We can still do direct assignment here
                self.cosmo["f"] = 0.
            elif not self.c["with_redshift_bin"] and self.cosmo["f"] is None:
                error_messages.append("Please specify the growth rate 'f'.")
            
            # Check redshift bin requirements
            if self.c["with_redshift_bin"] and (self.cosmo["Dz"] is None or self.cosmo["fz"] is None):
                error_messages.append("Please specify 'Dz' and 'fz' for galaxy counts distribution.")
            
            # Check for growth factor if with_time is False
            if not self.c["with_time"] and self.cosmo["D"] is None:
                error_messages.append("Please specify the growth factor 'D'.")
            
            # Check nonequal time requirements
            if self.c["with_nonequal_time"] and (self.cosmo["D1"] is None or self.cosmo["D2"] is None or 
                                                self.cosmo["f1"] is None or self.cosmo["f2"] is None):
                error_messages.append("You asked for nonequal time correlator. Please specify: 'D1', 'D2', 'f1', 'f2'.")
            
            # Check AP effect requirements
            if self.c["with_ap"] and (self.cosmo["H"] is None or self.cosmo["DA"] is None):
                error_messages.append("You asked to apply the AP effect. Please specify 'H' and 'DA'.")
            
            # Handle amplitude scaling (can be moved outside JAX compatibility section if needed)
            if not self.c["with_time"] and self.cosmo["A"] is not None:
                # This will need special handling in a fully JAX context to avoid mutation
                # One approach is to return this as a separate output to be applied by the caller
                scaling_factor = self.cosmo["A"]**0.5
                self.cosmo["D"] = self.cosmo["D"] * scaling_factor
            
            # Return joined error messages (empty string if no errors)
            if len(error_messages) > 0:
                raise Exception("\n".join(error_messages))
        
        else:
            if self.c["with_bias"]: 
                self.__is_bias_conflict()

            if self.cosmo["kk"] is None or self.cosmo["pk_lin"] is None:
                raise Exception("Please provide a linear matter power spectrum 'pk_lin' and the corresponding 'kk'.")

            if self.cosmo["kk"][0] > 1e-4 or self.cosmo["kk"][-1] < 0.5:
                raise Exception("Please provide a linear matter spectrum 'pk_lin' and the corresponding 'kk' with min(kk) < 1e-4 and max(kk) > 0.5")

            if self.c["multipole"] == 0: self.cosmo["f"] = 0.
            elif not self.c["with_redshift_bin"] and self.cosmo["f"] is None:
                raise Exception("Please specify the growth rate 'f'.")
            elif self.c["with_redshift_bin"] and (self.cosmo["Dz"] is None or self.cosmo["fz"] is None):
                raise Exception("You asked to account the galaxy counts distribution. Please specify 'Dz' and 'fz'.")

            if not self.c["with_time"] and self.cosmo["D"] is None:
                raise Exception("Please specify the growth factor 'D'.")

            if self.c["with_nonequal_time"] and (self.cosmo["D1"] is None or self.cosmo["D2"] is None or 
                                                self.cosmo["f1"] is None or self.cosmo["f2"] is None):
                raise Exception("You asked nonequal time correlator. Please specify: 'D1', 'D2', 'f1', 'f2'.")

            if self.c["with_ap"] and (self.cosmo["H"] is None or self.cosmo["DA"] is None):
                raise Exception("You asked to apply the AP effect. Please specify 'H' and 'DA'.")

            if not self.c["with_time"] and self.cosmo["A"]: self.cosmo["D"] *= self.cosmo["A"]**.5

    def __is_bias_conflict(self, bias=None):
        if bias is not None: self.cosmo["bias"] = bias
        if self.cosmo["bias"] is None: raise Exception("Please specify 'bias'.")
        if isinstance(self.cosmo["bias"], (list, ndarray)): self.cosmo["bias"] = self.cosmo["bias"][0]
        if not isinstance(self.cosmo["bias"], dict): raise Exception("Please specify bias in a dict.")

        for p in self.eft_parameters_list:
            if p not in self.cosmo["bias"]:
                raise Exception ("%s not found, please provide (given command 'eft_basis': '%s') %s" % (p, self.c["eft_basis"], self.eft_parameters_list))

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

        if 'Pk' in self.c['output']: 
            if self.c['xdata'] is not None: 
                if self.c['kmax'] < self.c['xdata'][-1]: self.c['kmax'] = self.c['xdata'][-1] + 0.05

        if "bm" in self.c["output"]: self.c["halohalo"] = False
        else: self.c["halohalo"] = True

        if self.c["with_quintessence"]: self.c["with_exact_time"] = True

        self.c["with_common_nonequal_time"] = False # this is to pass for the common Class to setup the numbers of loops (22 and 13 gathered by default)
        if self.c["with_nonequal_time"]:
            self.c.update({"with_bias": False, "with_time": False, "with_common_nonequal_time": True}) # with_common_nonequal_time is to pass for the common Class to setup the numbers of loops (22 and 13 seperated since they have different time dependence)
            if self.c["z1"] is None or self.c["z2"] is None: print("Please specify 'z1' and 'z2' for nonequaltime correlator. ")

        if self.c["with_ap"] and (self.c["H_fid"] is None or self.c["D_fid"] is None):
                raise Exception("You asked to apply the AP effect. Please specify 'H_fid' and 'D_fid'.")

        if self.c["with_survey_mask"] and (self.c["survey_mask_arr_p"] is None or self.c["survey_mask_mat_kp"] is None): raise Exception("Survey mask: on. Please specify 'survey_mask_arr_p' and 'survey_mask_mat_kp'.")
        if self.c["with_binning"] and self.c["binsize"] is None: raise Exception("Binning: on. Please provide 'binsize'.")
        if self.c["with_redshift_bin"]:
            self.c.update({"with_bias": False, "with_time": False, "with_cf": True}) # even for the Pk, we first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            if self.c["redshift_bin_zz"] is None or self.c["redshift_bin_nz"] is None: raise Exception("You asked to account for the galaxy counts distribution over a redshift bins. Please provide a distribution 'redshift_bin_nz' and corresponding 'redshift_bin_zz'.")
        if self.c["with_wedge"] and self.c["wedge_mat_wl"] is None: raise Exception("Please specify 'wedge_mat_wl'.")

        if self.c['with_emu']: 
            self.c['with_time'] = True
            self.c['with_exact_time'] = True

class BiasCorrelator(Correlator):
    """A class for loading pre-computed correlations.
    
    BiasCorrelator extends the Correlator class to work with pre-computed
    correlation functions. It initializes with load_engines=False by default
    since computational engines are typically not needed for pre-computed results.
    
    Attributes:
        Inherits all attributes from Correlator.
    
    Methods:
        Inherits all methods from Correlator.
    """
    
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
    """A class for handling configuration options with validation.
    
    The Option class represents a single configuration parameter with its
    allowed types, values, description, and default value. It provides
    methods for validating user-provided values.
    
    Attributes:
        name (str): Name of the configuration option.
        type (type or tuple): Allowed type(s) for this option.
        list (list): List of allowed values if restricted, None otherwise.
        description (str): Description of the option.
        default (any): Default value for the option.
        value (any): Current value of the option.
        verbose (bool): Whether to print verbose information.
    
    Methods:
        check(): Check if a provided value is valid for this option.
        error(): Raise an appropriate error for an invalid configuration value.
    """

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
        else: 
            self.error(config_value)
        return is_config
    def error(self, config_value):
        if self.list is None:
            try: raise Exception("Input error in '%s'; expecting: %s (but provided: %s). Check Correlator.info() in any doubt." % (self.name, typename(self.type), type(config_value)))
            except Exception as e: print(e)
        else:
            try: raise Exception("Input error in '%s'; expecting: %s (but provided: %s). Check Correlator.info() in any doubt." % (self.name, self.list, type(config_value)))
            except Exception as e: print(e)
