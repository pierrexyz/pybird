from cosmosis.datablock import names, option_section, SectionOptions
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from astropy.io import fits
import pybird as pb


class BirdLikelihood(object):
    # I take as example TwoPointLikelihood
    # They subclass the Gaussian one, but we can't
    like_name = "bird_like"

    def __init__(self, options):
        # Load data and covariance
        self.__load_data(options)

        # Check for BBNprior
        self.use_BBNprior = False
        try:
            self.omega_b_BBNsigma = options.get_double("omega_b_BBNsigma")
            self.omega_b_BBNcenter = options.get_double("omega_b_BBNcenter")
            self.use_BBNprior = True
            print ('BBN prior on omega_b: on')
        except:
            print ('BBN prior on omega_b: none')

    def __load_data(self, options):
        """
        Helper function to read in the full data vector.
        """
        # print("Load data?")
        data_file = options.get_string("data")
        # The following automaticallly closes the file
        with fits.open(data_file) as des:
            Nbin = 5
            tam = np.empty(shape=(20))
            wdes = np.empty(shape=(Nbin * 20))
            for i, line in enumerate(des['wtheta'].data):
                bin1, bin2, angbin, val, ang, npairs = line
                if i < 20:
                    tam[i] = ang
                wdes[i] = val
            wdes = wdes.reshape(Nbin, 20)
            cov = des['COVMAT'].data[-100:, -100:]
            t = tam * np.pi / (60. * 180.)  # Convert arcmins to radians
            N = des['nz_lens'].data.shape[0]
            zdes = np.empty(shape=(N))
            ndes = np.empty(shape=(Nbin, N))
            for i, line in enumerate(des['nz_lens'].data):
                zlow, zmid, zhigh, bin1, bin2, bin3, bin4, bin5 = line
                zdes[i] = zmid
                for j in range(Nbin):
                    ndes[j, i] = line[3 + j] / (zhigh - zlow)
            for j in range(Nbin):
                ndes[j] /= np.trapz(ndes[j], x=zdes)
            Nz = 200
            zeff = np.array([0.24, 0.38, 0.525, 0.68, 0.83])

            zz = np.empty(shape=(Nbin, Nz))
            nz = np.empty(shape=(Nbin, Nz))

            for i in range(Nbin):
                zz[i] = np.linspace(zeff[i] - 0.15, zeff[i] + 0.15, Nz)
                nz[i] = interp1d(zdes, ndes[i], kind='cubic')(zz[i])
            tamin = options.get_double_array_1d("xmin")
            # print(tamin)
            tmask0 = np.argwhere((tam >= min(tamin)))[:, 0]
            self.tmask = np.concatenate([np.argwhere((tam >= tamin[i]))[:, 0] + i * 20 for i in range(Nbin)])
            # print(self.tmask)
            covred = cov[self.tmask.reshape((len(self.tmask), 1)), self.tmask]
            self.invcov = np.linalg.inv(covred)
            ydata = wdes.reshape(-1)[self.tmask]
            xdata = tam[tmask0]
            # print("DATA: ", ydata.shape, ydata)
            self.chi2data = np.dot(ydata, np.dot(self.invcov, ydata))
            self.invcovdata = np.dot(ydata, self.invcov)
        return

    def __set_prior(self, multipole, model=5):
        if model == 0:
            priors = np.array([2., 2.])
            b3, cct = priors
            print ('EFT priors: b3: %s, cct: %s (default)' % (b3, cct))

        if multipole is 2:
            if model == 1:
                priors = np.array([2., 2., 8., 2., 2.])
                b3, cct, cr1, ce2, sn = priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, ce2, sn))
            elif model == 2:
                priors = np.array([2., 2., 8., 2.])
                b3, cct, cr1, ce2 = priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s (default)' % (b3, cct, cr1, ce2))
            elif model == 3:
                priors = np.array([2., 2., 8., 2., 2.])  # np.array([ 10., 4., 8., 4., 2. ])
                b3, cct, cr1, ce2, ce1 = priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, ce1: %s (default)' % (b3, cct, cr1, ce2, ce1))
            elif model == 4:
                priors = np.array([2., 2., 8., 2., 2., 2.])
                b3, cct, cr1, ce2, ce1, sn = priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, ce1: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, ce2, ce1, sn))
            elif model == 5:
                priors = np.array([2., 2., 8.])
                b3, cct, cr1 = priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s (default)' % (b3, cct, cr1))

        if multipole is 3:
            if model == 1:
                priors = np.array([2., 2., 4., 4., 2., 2.])
                b3, cct, cr1, cr2, ce2, sn = priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, cr2, ce2, sn))
            elif model == 2:
                priors = np.array([2., 2., 4., 4., 2.])
                b3, cct, cr1, cr2, ce2 = priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s (default)' % (b3, cct, cr1, cr2, ce2))
            elif model == 3:
                priors = np.array([2., 2., 4., 4., 2., 2.])  # np.array([ 10., 4., 8., 4., 2. ])
                b3, cct, cr1, cr2, ce2, ce1 = priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s, ce1: %s (default)' %
                       (b3, cct, cr1, cr2, ce2, ce1))
            elif model == 4:
                priors = np.array([2., 2., 4., 4., 2., 2., 2.])
                b3, cct, cr1, cr2, ce2, ce1, sn = priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s, ce1: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, cr2, ce2, ce1, sn))
            elif model == 5:
                priors = np.array([2., 2., 4., 4.])
                b3, cct, cr1, cr2 = priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s (default)' % (b3, cct, cr1, cr2))

        priormat = np.diagflat(1. / priors**2)
        return priormat

    def bias_array_to_dict(self, bs):
        if self.config["with_stoch"]:
            if self.config["multipole"] is 2:
                bdict = {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3],
                         "cct": bs[4], "cr1": bs[5],
                         "ce0": bs[7], "ce1": bs[8], "ce2": bs[9]}
            elif self.config["multipole"] is 3:
                bdict = {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3],
                         "cct": bs[4], "cr1": bs[5], "cr2": bs[6],
                         "ce0": bs[7], "ce1": bs[8], "ce2": bs[9]}
        else:
            bdict = {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3],
                     "cct": bs[4], "cr1": bs[5], "cr2": bs[6]}
        return bdict

    def bias_custom_to_all(self, bs):
        return [bs[0], bs[1] / np.sqrt(2.), 0., bs[1] / np.sqrt(2.), 0., 0., 0., 0., 0., 0.]

    def biasing(self, block):
        bval = np.array([block["bias", n]
                         for n in ("b11", "c21", "b12", "c22", "b13", "c23", "b14", "c24", "b15", "c25")])
        bval = bval.reshape(self.config["skycut"], -1)
        bdict = np.array([self.bias_array_to_dict(self.bias_custom_to_all(bs)) for bs in bval])
        b1 = np.array([bval[i, 0] for i in range(self.config["skycut"])])
        correlator = self.biascorrelator.get(bdict)
        marg_correlator = self.biascorrelator.getmarg(b1, model=self.config["model"])
        return correlator, marg_correlator

    def __get_Pi_for_marg(self, marg_correlator, xmask):
        Pi = marg_correlator

        if self.config["with_bao"]:  # BAO
            newPi = np.zeros(shape=(Pi.shape[0], Pi.shape[1] + 2))
            newPi[:Pi.shape[0], :Pi.shape[1]] = Pi
            Pi = 1. * newPi

        Pi = Pi[:, xmask]

        return Pi

    def __get_chi2(self, modelX, Pi, invcov, invcovdata, chi2data, priormat):
        Covbi = np.dot(Pi, np.dot(invcov, Pi.T)) + priormat
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = np.dot(modelX, np.dot(invcov, Pi.T)) - np.dot(invcovdata, Pi.T)
        chi2nomar = np.dot(modelX, np.dot(invcov, modelX)) - 2. * np.dot(invcovdata, modelX) + chi2data
        chi2mar = - np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.abs(np.linalg.det(Covbi)))
        chi2tot = chi2mar + chi2nomar - priormat.shape[0] * np.log(2. * np.pi)
        #print (np.dot(modelX, np.dot(invcov, modelX)), -2. * np.dot(invcovdata, modelX), chi2data)
        #print (- np.dot(vectorbi, np.dot(Cinvbi, vectorbi)), np.log(np.abs(np.linalg.det(Covbi))))
        #print (chi2nomar, chi2mar )
        return chi2tot

    def get_cache(self, block):
        # Function to get self.config and cache_dict from the datablock
        self.config = {}
        self.cache_dict = {}
        for sec, k in block.keys():
            if sec == "bird_config":
                self.config[k] = block[sec, k]
            if sec == names.matter_power_nl:
                self.cache_dict[k] = block[sec, k]
        return

    def do_likelihood(self, block):
        # Get the config dictionary from the datablock
        self.biascorrelator = pb.BiasCorrelator()
        self.get_cache(block)
        self.biascorrelator.set(self.config)
        # Reconstruct the cache_dict from the datablock
        self.biascorrelator.setcache(self.cache_dict, as_dict=True)
        # Do the biasing
        correlator, marg_correlator = self.biasing(block)

        # Assign priors to the bias parameters to marginalize
        # shape: (Nbin * Nmarg, Nbin * Nmarg)
        priormatdiag = []
        for i in range(self.config["skycut"]):
            priormatdiag.append(np.diag(self.__set_prior(self.config["multipole"], model=self.config["model"])))
        priormatdiag = np.array(priormatdiag).reshape(-1)
        self.priormat = np.diagflat(priormatdiag)

        chi2 = 0.

        if "w" in self.config["output"]:
            # print(type(correlator), len(correlator))
            # print(np.asarray(correlator).reshape(-1))
            # print(len(np.asarray(correlator).reshape(-1)))
            modelX = np.asarray(correlator).reshape(-1)[self.tmask]
            np.save("correlatorguido.npy", correlator)
            np.save("modelXguido.npy", modelX)
            # print(modelX.shape, modelX)
            Pi = block_diag(*marg_correlator)[:, self.tmask]
            chi2 += self.__get_chi2(modelX, Pi, self.invcov, self.invcovdata, self.chi2data, self.priormat)
        else:
            for i in range(self.config["skycut"]):
                if self.config["skycut"] is 1:
                    modelX = correlator.reshape(-1)
                elif self.config["skycut"] > 1:
                    modelX = correlator[i].reshape(-1)

            modelX = modelX[self.xmask[i]]

            if self.config["skycut"] is 1:
                Pi = self.__get_Pi_for_marg(marg_correlator, self.xmask[i])
            elif self.config["skycut"] > 1:
                Pi = self.__get_Pi_for_marg(marg_correlator[i], self.xmask[i])

            chi2 += self.__get_chi2(modelX, Pi, self.invcov[i],
                                    self.invcovdata[i], self.chi2data[i], self.priormat[i])

        prior = 0.
        # BBN prior?
        if self.use_BBNprior:
            omb = block[names.cosmological_parameters, "ombh2"]
            prior += -0.5 * ((omb - self.omega_b_BBNcenter) / self.omega_b_BBNsigma)**2

        lkl = - 0.5 * chi2 + prior
        # Now save the resulting likelihood
        block[names.likelihoods, self.like_name + "_LIKE"] = lkl
        return lkl

    def cleanup(self):
        """
        You can override the cleanup method if you do something unusual to get your data,
        like open a database or something.
        It is run just once, at the end of the pipeline.
        """
        pass

    @classmethod
    def build_module(cls):

        def setup(options):
            options = SectionOptions(options)
            likelihoodCalculator = cls(options)
            return likelihoodCalculator

        def execute(block, config):
            likelihoodCalculator = config
            likelihoodCalculator.do_likelihood(block)
            return 0

        def cleanup(config):
            likelihoodCalculator = config
            likelihoodCalculator.cleanup()

        return setup, execute, cleanup


setup, execute, cleanup = BirdLikelihood.build_module()
