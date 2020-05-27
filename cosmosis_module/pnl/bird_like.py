from cosmosis.datablock import names, option_section, SectionOptions
import numpy as np
import os


class BirdLikelihood(object):
    # I take as example TwoPointLikelihood
    # They subclass the Gaussian one, but we can't
    like_name = "bird_like"

    def __init__(self, options):
        # General options
        self.options = options
        kmin = options.get_double("kmin")
        kmax = options.get_double("kmax")
        self.Nl = options.get_int("Nl")
        self.model = options.get_int("model")
        self.data_directory = options.get_string("dir")
        cov_file = options.get_string("cov_file")

        # Load data PS and mask the relevant k
        kdata, PSdata = self.__load_data()

        self.k = kdata.reshape(3, -1)[0]
        self.Nk = len(self.k)
        kmask0 = np.argwhere((self.k <= kmax) & (self.k >= kmin))[:, 0]
        self.kmask = kmask0
        # print(self.kmask)
        for i in range(self.Nl - 1):
            kmaski = np.argwhere((self.k <= kmax) & (self.k >= kmin))[:, 0] + (i + 1) * self.Nk
            self.kmask = np.concatenate((self.kmask, kmaski))
        # print(self.kmask)
        self.ydata = PSdata[self.kmask]

        # Load data covariance, mask and invert it
        cov = np.loadtxt(os.path.join(self.data_directory, cov_file))
        # print(cov.shape)
        covred = cov[self.kmask.reshape((len(self.kmask), 1)), self.kmask]
        # print(covred.shape)
        self.invcov = np.linalg.inv(covred)
        self.chi2data = np.dot(self.ydata, np.dot(self.invcov, self.ydata))
        self.invcovdata = np.dot(self.ydata, self.invcov)

        # Assign priors to the bias parameters to marginalize
        self.assign_priors()

        # Check for BBNprior
        self.use_BBNprior = False
        try:
            self.omega_b_BBNsigma = options.get_double("omega_b_BBNsigma")
            self.omega_b_BBNcenter = options.get_double("omega_b_BBNcenter")
            self.use_BBNprior = True
            print ('BBN prior on omega_b: on')
        except:
            print ('BBN prior on omega_b: none')

    def __load_data(self):
        """
        Helper function to read in the full data vector.
        """
        # print("Load data?")
        data_file = self.options.get_string("ps_file")
        fname = os.path.join(self.data_directory, data_file)
        try:
            kPS, PSdata, _ = np.loadtxt(fname, unpack=True)
        except:
            kPS, PSdata = np.loadtxt(fname, unpack=True)
        return kPS, PSdata

    def assign_priors(self):
        # Assigns priors to marginalized bias parameters
        if self.Nl is 2:
            self.use_prior = True
            if self.model == 1:
                self.priors = np.array([2., 2., 8., 2., 2.])
                b3, cct, cr1, ce2, sn = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, ce2, sn))
            elif self.model == 2:
                self.priors = np.array([2., 2., 8., 2.])
                b3, cct, cr1, ce2 = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s (default)' % (b3, cct, cr1, ce2))
            elif self.model == 3:
                self.priors = np.array([2., 2., 8., 2., 2.])  # np.array([ 10., 4., 8., 4., 2. ])
                b3, cct, cr1, ce2, ce1 = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, ce1: %s (default)' % (b3, cct, cr1, ce2, ce1))
            elif self.model == 4:
                self.priors = np.array([2., 2., 8., 2., 2., 2.])
                b3, cct, cr1, ce2, ce1, sn = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, ce1: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, ce2, ce1, sn))
            elif self.model == 5:
                self.priors = np.array([2., 2., 8.])
                b3, cct, cr1 = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s (default)' % (b3, cct, cr1))
        elif self.Nl is 3:
            self.use_prior = True
            if self.model == 1:
                self.priors = np.array([2., 2., 4., 4., 2., 2.])
                b3, cct, cr1, cr2, ce2, sn = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, cr2, ce2, sn))
            elif self.model == 2:
                self.priors = np.array([2., 2., 4., 4., 2.])
                b3, cct, cr1, ce2 = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s (default)' % (b3, cct, cr1, cr2, ce2))
            elif self.model == 3:
                self.priors = np.array([2., 2., 4., 4., 2., 2.])  # np.array([ 10., 4., 8., 4., 2. ])
                b3, cct, cr1, cr2, ce2, ce1 = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s, ce1: %s (default)' %
                       (b3, cct, cr1, cr2, ce2, ce1))
            elif self.model == 4:
                self.priors = np.array([2., 2., 4., 4., 2., 2., 2.])
                b3, cct, cr1, cr2, ce2, ce1, sn = self.priors
                print ('EFT priors: b3: %s, cct: %s, cr1: %s, cr2: %s, ce2: %s, ce1: %s, shotnoise: %s (default)' %
                       (b3, cct, cr1, cr2, ce2, ce1, sn))
        self.priormat = np.diagflat(1. / self.priors**2)

    def biasing(self, block):
        self.knl = self.options.get_double("knl")
        self.km = self.options.get_double("km")
        self.nd = self.options.get_double("nd")

        bval = [block["bias", n]
                for n in ("b1", "c2", "b3", "c4", "b5",
                          "b6", "b7", "b8", "b9", "b10")]
        b1 = bval[0]
        self.b1 = b1
        b2 = (bval[1] + bval[3]) / np.sqrt(2.)
        b3 = bval[2]
        b4 = (bval[1] - bval[3]) / np.sqrt(2.)
        b5 = bval[4] / self.knl**2
        b6 = bval[5] / self.km**2
        b7 = 0.

        # The PS are correctly read in the right shape
        self.P11l = block[names.matter_power_nl, 'P11l']
        self.Ploopl = block[names.matter_power_nl, 'Ploopl']
        self.Pctl = block[names.matter_power_nl, 'Pctl']
        self.f = block[names.growth_parameters, "f_PS"]

        b11 = np.array([b1**2, 2. * b1 * self.f, self.f**2])
        bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7,
                        2. * self.f * b5, 2. * self.f * b6, 2. * self.f * b7])
        bloop = np.array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])

        Ps0 = np.einsum('b,lbx->lx', b11, self.P11l)
        Ps1 = np.einsum('b,lbx->lx', bloop, self.Ploopl) + np.einsum('b,lbx->lx', bct, self.Pctl)
        self.fullPs = Ps0 + Ps1
        # print(self.fullPs.shape)
        self.Pb3 = self.Ploopl[:, 3] + b1 * self.Ploopl[:, 7]

        if self.use_prior:
            self.prior = - 0.5 * (
                (bval[1] / 10.)**2                                    # c2
                + (bval[3] / 2.)**2                                     # c4
                + (bval[2] / self.priors[0])**2                         # b3
                + (bval[4] / self.knl**2 / self.priors[1])**2           # cct
                + (bval[5] / self.km**2 / self.priors[2])**2            # cr1(+cr2)
            )
            if self.model <= 4:
                self.prior += 0.5 * (bval[9] / self.nd / self.km**2 / self.priors[3])**2  # ce,l2
            if self.model == 1:
                self.prior += -0.5 * (bval[7] / self.nd / self.priors[4])**2               # ce0
            if self.model == 3:
                self.prior += -0.5 * (bval[8] / self.nd / self.km**2 / self.priors[4])**2  # ce,l0

    def __get_Pi_for_marg(self, Pct, Pb3, b1, f, model=2):
        if self.Nl is 2:
            Pi = np.array([
                Pb3.reshape(-1),                                          # *b3
                (2 * f * Pct[:, 0 + 3] + 2 * b1 * Pct[:, 0]).reshape(-1) / self.knl**2,  # *cct
                (2 * f * Pct[:, 1 + 3] + 2 * b1 * Pct[:, 1]).reshape(-1) / self.km**2  # *cr1
            ])

        elif self.Nl is 3:
            Pi = np.array([
                Pb3.reshape(-1),                                          # *b3
                (2 * f * Pct[:, 0 + 3] + 2 * b1 * Pct[:, 0]).reshape(-1) / self.knl**2,  # *cct
                (2 * f * Pct[:, 1 + 3] + 2 * b1 * Pct[:, 1]).reshape(-1) / self.km**2,  # *cr1
                (2 * f * Pct[:, 2 + 3] + 2 * b1 * Pct[:, 2]).reshape(-1) / self.km**2  # *cr2
            ])

        if model <= 4:
            kp2l2 = np.zeros(shape=(self.Nl, self.Nk))
            kp2l2[1] = self.k**2 / self.nd / self.km**2  # k^2 quad
            Pi = np.vstack([Pi, kp2l2.reshape(-1)])

        if model == 1:
            Onel0 = np.zeros(shape=(self.Nl, self.Nk))
            Onel0[0] = np.ones(self.Nk) / self.nd  # shot-noise mono
            Pi = np.vstack([Pi, Onel0.reshape(-1)])
        elif model == 3:
            kp2l0 = np.zeros(shape=(self.Nl, self.Nk))
            kp2l0[0] = self.k**2 / self.nd / self.km**2  # k^2 mono
            Pi = np.vstack([Pi, kp2l0.reshape(-1)])
        elif model == 4:
            kp2l0 = np.zeros(shape=(self.Nl, self.Nk))
            kp2l0[0] = self.k**2 / self.nd / self.km**2  # k^2 mono
            Onel0 = np.zeros(shape=(self.Nl, self.Nk))
            Onel0[0] = np.ones(self.Nk) / self.nd  # shot-noise mono
            Pi = np.vstack([Pi, kp2l0.reshape(-1), Onel0.reshape(-1)])
        # print(self.kmask.shape, Pi.shape)
        Pi = Pi[:, self.kmask]
        # print(Pi.shape)
        return Pi

    def do_likelihood(self, block):
        self.biasing(block)
        modelX = self.fullPs.reshape(-1)
        modelX = modelX[self.kmask]

        Pi = self.__get_Pi_for_marg(self.Pctl, self.Pb3, self.b1, self.f, model=self.model)
        # print(Pi.shape, self.invcov.shape)
        Covbi = np.dot(Pi, np.dot(self.invcov, Pi.T)) + self.priormat
        # print(Covbi.shape)
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = np.dot(modelX, np.dot(self.invcov, Pi.T)) - np.dot(self.invcovdata, Pi.T)
        chi2nomar = (np.dot(modelX, np.dot(self.invcov, modelX))
                     - 2. * np.dot(self.invcovdata, modelX)
                     + self.chi2data)
        chi2mar = -np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.abs(np.linalg.det(Covbi)))
        chi2 = chi2mar + chi2nomar - self.priors.shape[0] * np.log(2. * np.pi)

        if self.use_BBNprior:
            omb = block[names.cosmological_parameters, "ombh2"]
            self.prior += -0.5 * ((omb - self.omega_b_BBNcenter) / self.omega_b_BBNsigma)**2

        lkl = - 0.5 * chi2 + self.prior

        # Now save the resulting likelihood
        block[names.likelihoods, self.like_name + "_LIKE"] = lkl

    def cleanup(self):
        """
        You can override the cleanup method if you do something 
        unusual to get your data, like open a database or something.
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
