import numpy as np
import os
import subprocess


keys_cbird = ["PathToOutput", "PathToLinearPowerSpectrum",
              "knl", "km", "nbar",
              "ComputePowerSpectrum", "ResumPowerSpectrum", "ComputeBispectrum",
              "z_pk", "ln10^{10}A_s", "n_s", "h", "omega_b", "omega_cdm","N_ncdm","Sum_mnu",
              "PathToTriangles", "aperp", "apar"]


class NonLinearPower(object):
    """
    An object that calls CBIRD and computes the non-linear power spectrum.
    Parameters
    ----------
    paramdict : dictionary
        dictionary containing, among others, the parameters read by CBIRD
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
    Plin : np.array
        The linear power spectrum at z_pk
    """
    def __init__(self, paramdict, kmin=None, kmax=None):
        self.cbird_exe = os.path.abspath(os.path.join(paramdict["cbird_folder"],
                                         paramdict["cbird_exe"]))
        if not os.path.isfile(self.cbird_exe):
            print("You want your CBIRD code in %s. You must compile it!" % self.cbird_exe)
            raise IOError
        self.paramdict = {}
        for k, v in paramdict.items():
            if ('path' in k.lower()) or ('folder' in k.lower()):
                self.paramdict[k] = v  # os.path.abspath(v)
            else:
                self.paramdict[k] = v
        if "A_s" in paramdict.keys():
            paramdict["ln10^{10}A_s"] = np.log(float(paramdict["A_s"]) * 1e10)
        self.outdir = self.paramdict["PathToOutput"]
        self.redshift = self.paramdict["z_pk"]
        self.kmin = kmin
        self.kmax = kmax

    def create_parfile(self):
        """
        Creates the output directory and parameter file
        """
        try:
            if not os.path.isdir(self.outdir):
                os.makedirs(self.outdir)
        except IOError:
            print("Cannot create directory: %s" % self.outdir)
        parfile = os.path.join(self.outdir, 'cbird.ini')
        with open(parfile, 'w') as f:
            for k, v in self.paramdict.items():
                f.write("%s = %s \n" % (k, v))
        return parfile

    def compute(self):
        """
        Creates and saves the parameter file and calls CBIRD with it
        """
        parfile = self.create_parfile()
        self._command = [self.cbird_exe, parfile]
        print(self._command)
        process = subprocess.Popen(self._command)
        try:
            # process.wait(timeout=300)
            process.wait()
        # except (KeyboardInterrupt, subprocess.TimeoutExpired) as e:  # TimeoutExpired only in Python >= 3.3
        except Exception as e:
            process.kill()
            raise e
        return

    def get_Plin_resum(self):
        """
        Gets the linear resummed power spectrum
        """
        Plin_file = os.path.join(self.outdir, "PowerSpectra.dat")
        try:
            Plin = np.loadtxt(Plin_file)
        except IOError:
            print("You have to compute the power spectra first!")
            return
        if (self.kmin is not None) and (self.kmax is not None):
            kin = Plin[:, 0]
            kmask = np.where((kin>=self.kmin)&(kin<=self.kmax))[0]
            Plin = Plin[kmask, :4]
        return Plin

    def get_Ploop_resum(self):
        """
        Gets the loop resummed power spectrum
        """
        Ploop_file = os.path.join(self.outdir, "PowerSpectra.dat")
        try:
            Ploop = np.loadtxt(Ploop_file)
        except IOError:
            print("You have to compute the power spectra first!")
            return
        if (self.kmin is not None) and (self.kmax is not None):
            kin = Ploop[:, 0]
            kmask = np.where((kin>=self.kmin)&(kin<=self.kmax))[0]
            Ploop = Ploop[kmask]
        return np.concatenate([Ploop[:, :1], Ploop[:, 4:]], axis=1)
