from cosmosis.datablock import names, option_section
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import numpy as np
import os
from scipy.interpolate import interp1d
from astropy.io import fits
import pybird as pb


class Wtheta(object):
    # I use this class as in project_2d.py
    # More useful to keep everything in a class, can extend to other calculations
    def __init__(self, options):
        # General options
        self.config = {}
        self.Nbin = options.get_int(option_section, "skycut")
        self.config["skycut"] = self.Nbin
        self.zeff = options.get_double_array_1d(option_section, "zeff")
        self.config["z"] = self.zeff
        # self.config["xmin"] = options.get_int_array_1d(option_section, "xmin")
        self.config["xdata"] = self.__load_data(options)
        self.config["model"] = options.get_int(option_section, "model")
        self.config["output"] = options.get_string(option_section, "output")
        self.config["multipole"] = 3
        self.config["with_AP"] = False
        self.config["with_redshift_bin"] = True
        self.config["with_stoch"] = False
        self.config["with_exact_time"] = False

        # setting pybird correlator configuration, and save it to the datablock
        self.correlator = pb.Correlator()

        self.correlator.set(self.config)
        # print("CONFIG OUTPUT", self.config['output'])
        # print("CORRELATOR OUTPUT", self.correlator.config['output'])
        self.input_section = options.get_string(option_section, "input_section", names.matter_power_lin)

    def __load_data(self, options):
        """
        Helper function to read in the full data vector.
        """
        # print("Load data?")
        data_file = options.get_string(option_section, "data")

        with fits.open(data_file) as des:
            tam = np.empty(shape=(20))
            for i, line in enumerate(des['wtheta'].data[:20]):
                bin1, bin2, angbin, val, ang, npairs = line
                tam[i] = ang
            t = tam * np.pi / (60. * 180.)
            N = des['nz_lens'].data.shape[0]
            zdes = np.empty(shape=(N))
            ndes = np.empty(shape=(self.Nbin, N))
            for i, line in enumerate(des['nz_lens'].data):
                zlow, zmid, zhigh, bin1, bin2, bin3, bin4, bin5 = line
                zdes[i] = zmid
                for j in range(self.Nbin):
                    ndes[j, i] = line[3 + j] / (zhigh - zlow)
            for j in range(self.Nbin):
                ndes[j] /= np.trapz(ndes[j], x=zdes)
            Nz = 200
            zz = np.empty(shape=(self.Nbin, Nz))
            nz = np.empty(shape=(self.Nbin, Nz))
            for i in range(self.Nbin):
                zz[i] = np.linspace(self.zeff[i] - 0.15, self.zeff[i] + 0.15, Nz)
                nz[i] = interp1d(zdes, ndes[i], kind='cubic')(zz[i])
            self.config["zz"] = zz
            self.config["nz"] = nz
        return t

    def load_distance_splines(self, block):
        # Extract some useful distance splines
        # have to copy these to get into C ordering (because we reverse them)
        # Distances are in 1/Mpc units
        z_distance = block[names.distances, 'z']
        Hz = block[names.distances, 'h']
        DA = block[names.distances, 'd_a']
        DM = block[names.distances, 'd_m']
        if z_distance[1] < z_distance[0]:
            z_distance = z_distance[::-1].copy()
            Hz = Hz[::-1].copy()
            DA = DA[::-1].copy()
            DM = DM[::-1].copy()
        H0Mpc = block[names.cosmological_parameters, "hubble"] / 299792.458
        h = block[names.cosmological_parameters, "h0"]
        self.iDAz = InterpolatedUnivariateSpline(z_distance, DA * H0Mpc)
        self.iHz = InterpolatedUnivariateSpline(z_distance, Hz / H0Mpc)
        self.irz = InterpolatedUnivariateSpline(z_distance, DM * h)

    def load_powergrowth(self, block):
        # Get growth factors, and interpolate them
        self.zg = block.get_double_array_1d(names.growth_parameters, "z")
        self.Dz = block.get_double_array_1d(names.growth_parameters, "d_z")
        self.fz = block.get_double_array_1d(names.growth_parameters, "f_z")
        self.ifz = InterpolatedUnivariateSpline(self.zg, self.fz)
        self.iDz = InterpolatedUnivariateSpline(self.zg, self.Dz)
        # Let's write in the block the f since it's useful for biasing
        # block.put_double(names.growth_parameters, "f_PS", self.ff)

    def __set_cosmo(self, block):
        cosmo = {}
        # Copy the cosmo parameter section into a new config section, which we use for the self.config dictionary
        # block._copy_section(names.cosmological_parameters, "bird_config")
        for k, v in self.config.items():
            block["bird_config", k] = v

        # Get linear PS and put it into cosmo
        zsamples, self.kin, pgrid = block.get_grid(self.input_section, "z", "k_h", "p_k")
        ip2d = RectBivariateSpline(zsamples, self.kin, pgrid)
        zfid = self.config["z"][0]
        cosmo["k11"] = self.kin  # k in h/Mpc
        cosmo["P11"] = ip2d.ev(zfid, self.kin)  # P(k) in (Mpc/h)**3
        if self.config["skycut"] == 1:
            # if self.config["multipole"] is not 0:
            cosmo["f"] = self.ifz(self.zp)
            if self.config["with_exact_time"]:
                cosmo["z"] = self.config["z"][0]
                cosmo["Omega0_m"] = block[names.cosmological_parameters, "omega_m"]
                try:
                    cosmo["w0_fld"] = block[names.cosmological_parameters, "w"]
                except:
                    pass

            if self.config["with_AP"]:
                cosmo["DA"] = self.iDAz(self.zp)
                cosmo["H"] = self.iHz(self.zp)

        elif self.config["skycut"] > 1:
            # if self.config["multipole"] is not 0:
            cosmo["f"] = np.array([self.ifz(z) for z in self.config["z"]])
            cosmo["D"] = np.array([self.iDz(z) for z in self.config["z"]])

            if self.config["with_AP"] and not self.config["with_redshift_bin"]:
                cosmo["DA"] = np.array([self.iDAz(z) for z in self.config["z"]])
                cosmo["H"] = np.array([self.iHz(z) for z in self.config["z"]])

        if self.config["with_redshift_bin"]:
            if self.config["skycut"] == 1:
                cosmo["D"] = self.iD(zfid)

                cosmo["Dz"] = np.array([self.iDz(z) for z in self.config["zz"]])
                cosmo["fz"] = np.array([self.ifz(z) for z in self.config["zz"]])

                if self.config["with_AP"]:
                    cosmo["DAz"] = np.array([self.iDAz(z) for z in self.config["zz"]])
                    cosmo["Hz"] = np.array([self.iHz(z) for z in self.config["zz"]])

            elif self.config["skycut"] > 1:
                cosmo["Dz"] = np.array([[self.iDz(z) for z in zz] for zz in self.config["zz"]])
                cosmo["fz"] = np.array([[self.ifz(z) for z in zz] for zz in self.config["zz"]])

                if self.config["with_AP"]:
                    cosmo["DAz"] = np.array([[self.iDAz(z) for z in zz] for zz in self.config["zz"]])
                    cosmo["Hz"] = np.array([[self.iHz(z) for z in zz] for zz in self.config["zz"]])

        if "w" in self.config["output"]:
            if self.config["skycut"] is 1:
                cosmo["rz"] = np.array([self.irz(z) for z in self.config["zz"]])
            elif self.config["skycut"] > 1:
                cosmo["rz"] = np.array([[self.irz(z) for z in zz] for zz in self.config["zz"]])

        return cosmo

    def compute_spectra(self, block):
        # Here kin is in h/Mpc and Plin is in (Mpc/h)^3
        cosmo = self.__set_cosmo(block)
        self.correlator.compute(cosmo)
        cache_dict = self.correlator.cache(as_dict=True)
        # np.savetxt("lin.txt", np.array(cache_dict["lin"]).flatten())
        # np.save("loopguido.npy", np.array(cache_dict["loop"]))
        for k, v in cache_dict.items():
            if isinstance(v, list):
                v = np.array(v)
            block[names.matter_power_nl, k] = v

    def execute(self, block):
        self.load_distance_splines(block)
        self.load_powergrowth(block)
        self.compute_spectra(block)
        return 0


def setup(options):
    return Wtheta(options)


def execute(block, config):
    return config.execute(block)


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
