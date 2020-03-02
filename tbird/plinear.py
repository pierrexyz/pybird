import os
import subprocess

keys_cbird = ["PathToOutput", "PathToLinearPowerSpectrum",
              "knl", "km", "nbar", "Sum_mnu",
              "ComputePowerSpectrum", "ResumPowerSpectrum", "ComputeBispectrum",
              "PathToTriangles", "aperp", "apar",
              'cbird_folder', 'cbird_exe', 'UseCosmoRef', 'ImportResummationMatrix', 'ExportResummationMatrix',
              'EpsRel_IntegrBispectrumAP',
              'EpsAbs_NoCosmoRef', 'EpsRel_NoCosmoRef', 'EpsAbs_YesCosmoRef', 'EpsRel_YesCosmoRef',
              'outpath', 'basename', 'pid', 'zbEFT_configf',
              'zbEFTw_configf',
              'logfile',
              'DM', 'invh'
              'kren']

keys_class = ["H0", "h", "100*theta_s",
              "T_cmb", "Omega_g", "omega_g",
              "Omega_cdm", "omega_cdm",
              "Omega_b", "omega_b",
              "N_ur", "Omega_ur", "omega_ur",
              "N_ncdm", "m_ncdm", "Omega_ncdm"
              "T_ncdm", "ksi_ncdm", "deg_ncdm",
              "k_pivot", "A_s", "ln10^{10}A_s", "sigma8",
              "n_s", "alpha_s",
              "Omega_k",
              "Omega_Lambda", "Omega_fld", "fluid_equation_of_state",
              "w0_fld", "wa_fld", "cs2_fld",
              "Omega_EDE",
              "YHe", "recombination",
              "reio_parametrization", "z_reio", "tau_reio",
              "output",
              "gauge",
              "P_k_max_h/Mpc",
              "z_pk", "z_max_pk",
              "root", "headers",
              "format",
              "write background", "write thermodynamics", "write primordial", "write parameters"]


class LinearPower(object):
    """
    An object to compute the linear power spectrum and related quantities,
    using a transfer function from the CLASS code or the analytic
    Eisenstein & Hu approximation.
    Parameters
    ----------
    cosmo : dictionary
        the dictionary of parameters, to be read by Class
    klist : array-like
        the k's in unit of h/Mpc where the Plin is evaluated
    Attributes
    ----------
    pk : class:`Cosmology`
        the object giving the cosmological parameters
    sigma8 : float
        the z=0 amplitude of matter fluctuations
    redshift : float
        the redshift to compute the power at
    transfer : str
        the type of transfer function used
    """
    def __init__(self, cosmo, klist):
        self.cosmo = {k: v for (k, v) in cosmo.items() if k not in keys_cbird}
        if set(("A_s", "ln10^{10}A_s")).issubset(self.cosmo.keys()):
            del self.cosmo["ln10^{10}A_s"]
        self.class_exe = os.path.abspath(os.path.join(cosmo["class_folder"],
                                         cosmo["class_exe"]))
        self.outdir = cosmo["PathToOutput"]
        self.redshift = float(self.cosmo['z_pk'])
        self.Omega_m = (float(self.cosmo['omega_cdm']) + float(self.cosmo['omega_b'])) / float(self.cosmo['h'])**2
        self.klist = klist

    def create_parfile(self):
        """
        Creates the output directory and parameter file
        """
        try:
            if not os.path.isdir(self.outdir):
                os.makedirs(self.outdir)
        except IOError:
            print("Cannot create directory: %s" % self.outdir)
        parfile = os.path.join(self.outdir, 'classpar.ini')
        with open(parfile, 'w') as f:
            for k, v in self.cosmo.items():
                f.write("%s = %s \n" % (k, v))
            f.write("root = %s \n" % os.path.join(self.outdir, 'class_'))
        return parfile

    def compute(self):
        """
        Computes the linear PS using Class
        """
        parfile = self.create_parfile()
        self._command = [self.class_exe, parfile]
        process = subprocess.Popen(self._command)
        try:
            # process.wait(timeout=300)
            process.wait()
        # except (KeyboardInterrupt, subprocess.TimeoutExpired) as e:  # TimeoutExpired only in Python >= 3.3
        except Exception as e:
            process.kill()
            raise e
        return

    @property
    def sigma8(self):
        """
        The present day value of ``sigma_r(r=8 Mpc/h)``, used to normalize
        the power spectrum, which is proportional to the square of this value.
        The power spectrum can re-normalized by setting a different
        value for this parameter
        """
        return self._sigma8
