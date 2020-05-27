from cosmosis.datablock import names, option_section
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import numpy as np
import os
import pybird as pb


class PowerSpectrum(object):
    # I use this class as in project_2d.py
    # More useful to keep everything in a class, can extend to other calculations
    def __init__(self, options):
        # General options
        kmin = options.get_double(option_section, "kmin")
        kmax = options.get_double(option_section, "kmax")
        # print(kmin, kmax)
        self.Nl = options.get_int(option_section, "Nl")
        Om_AP = options.get_double(option_section, "Om_AP")
        z_AP = options.get_double(option_section, "z_AP")

        self.use_window = options.get_bool(option_section, "use_window")
        if self.use_window:
            path_to_window = options.get_string(option_section, "path_to_window")
            window_fourier_name = options.get_string(option_section, "window_fourier_name")
            window_configspace_file = os.path.join(path_to_window, options.get_string(option_section, "window_configspace_file"))
        else:
            path_to_window = None
            window_fourier_name = None
            window_configspace_file = None

        self.binning = options.get(option_section, "binning")
        self.fibcol_window = options.get(option_section, "fibcol_window")

        kdata, PSdata = self.__load_data(options)

        self.k = kdata.reshape(3, -1)[0]
        self.Nk = len(self.k)
        kmask0 = np.argwhere((self.k <= kmax) & (self.k >= kmin))[:, 0]
        self.kmask = kmask0
        for i in range(self.Nl - 1):
            kmaski = np.argwhere((self.k <= kmax) & (self.k >= kmin))[:, 0] + (i + 1) * self.Nk
            self.kmask = np.concatenate((self.kmask, kmaski))
        print("Nk = ", self.Nk)
        self.zp = options.get_double(option_section, "z")
        self.common = pb.Common(Nl=self.Nl, kmax=kmax + 0.05, optiresum=False)  # , orderresum=8)
        self.nonlinear = pb.NonLinear(load=True, save=True, co=self.common)
        self.resum = pb.Resum(co=self.common)
        self.projection = pb.Projection(self.k, Om_AP, z_AP, cf=False,
                                        window_fourier_name=window_fourier_name, path_to_window=path_to_window, window_configspace_file=window_configspace_file,
                                        binning=self.binning, fibcol=self.fibcol_window, Nwedges=0, co=self.common)
        self.input_section = options.get_string(option_section, "input_section", names.matter_power_lin)
        self.output_section = options.get_string(option_section, "output_section", names.matter_power_nl)

    def __load_data(self, options):
        """
        Helper function to read in the full data vector.
        """
        # print("Load data?")
        data_directory = options.get_string(option_section, "data_dir")
        data_file = options.get_string(option_section, "ps_file")
        fname = os.path.join(data_directory, data_file)
        try:
            kPS, PSdata, _ = np.loadtxt(fname, unpack=True)
        except:
            kPS, PSdata = np.loadtxt(fname, unpack=True)
        return kPS, PSdata

    def load_distance_splines(self, block):
        # Extract some useful distance splines
        # have to copy these to get into C ordering (because we reverse them)
        # Distances are in 1/Mpc units
        z_distance = block[names.distances, 'z']
        Hz = block[names.distances, 'h']
        DA = block[names.distances, 'd_a']
        if z_distance[1] < z_distance[0]:
            z_distance = z_distance[::-1].copy()
            Hz = Hz[::-1].copy()
            DA = DA[::-1].copy()
        H0Mpc = block[names.cosmological_parameters, "hubble"] / 299792.458
        self.DA = InterpolatedUnivariateSpline(z_distance, DA * H0Mpc)
        self.Hz = InterpolatedUnivariateSpline(z_distance, Hz / H0Mpc)

    def load_powergrowth(self, block):
        # Get linear PS
        zsamples, self.kin, pgrid = block.get_grid(self.input_section, "z", "k_h", "p_k")
        ip2d = RectBivariateSpline(zsamples, self.kin, pgrid)
        self.plin = ip2d.ev(self.zp, self.kin)
        # Get growth factors, and interpolate f on the PS redshift
        self.zg = block.get_double_array_1d(names.growth_parameters, "z")
        self.Dz = block.get_double_array_1d(names.growth_parameters, "d_z")
        self.fz = block.get_double_array_1d(names.growth_parameters, "f_z")
        self.ff = InterpolatedUnivariateSpline(self.zg, self.fz)(self.zp)
        # Let's write in the block the f since it's useful for biasing
        block.put_double(names.growth_parameters, "f_PS", self.ff)

    def compute_spectra(self, block):
        # Here kin is in h/Mpc and Plin is in (Mpc/h)^3
        self.bird = pb.Bird(self.kin, self.plin, f=self.ff, DA=self.DA(self.zp),
                            H=self.Hz(self.zp), z=self.zp, which='all', co=self.common)
        self.bird.setPsCfl()
        # print(self.bird.P11l)
        self.nonlinear.PsCf(self.bird)
        # print(self.bird.P22)
        # print(self.bird.P13)
        self.bird.setPsCfl()
        # print(self.bird.P11l)
        self.resum.Ps(self.bird)
        # print(self.bird.P11l)
        self.projection.AP(self.bird)
        # print(self.bird.P11l)

        # print(self.bird.P11l.shape, self.bird.Ploopl.shape, self.bird.Pctl.shape)
        if self.use_window:
            self.projection.Window(self.bird)
        if self.fibcol_window:
            self.projection.fibcolWindow(self.bird)
        if self.binning:
            self.projection.kbinning(self.bird)
        else:
            self.projection.kdata(self.bird)
        # print(self.bird.P11l.shape, self.bird.Ploopl.shape, self.bird.Pctl.shape)
        block[names.matter_power_nl, 'P11l'] = self.bird.P11l
        block[names.matter_power_nl, 'Ploopl'] = self.bird.Ploopl
        block[names.matter_power_nl, 'Pctl'] = self.bird.Pctl
        # block.put(names.matter_power_nl, 'Bird', self.bird)

    def execute(self, block):
        self.load_distance_splines(block)
        self.load_powergrowth(block)
        self.compute_spectra(block)
        return 0


def setup(options):
    return PowerSpectrum(options)


def execute(block, config):
    return config.execute(block)


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
