import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from scipy.special import gamma
from .fftlog import FFTLog, MPC, CoefWindow
from .common import co


class NonLinear(object):
    """
    given a Bird() object, computes the one-loop power spectrum and one-loop correlation function.
    The correlation function is useful to perform the IR-resummation of the power spectrum.
    The loop and spherical Bessel transform matrices are either loaded either precomputed and stored at the instanciation of the class.

    Attributes
    ----------
    co : class
        An object of type Common() used to share data
    fftsettings: dict
    fft : class
        An object of type FFTLog() to perform the FFTLog
    M22 : ndarray
        22-loop power spectrum matrices
    M13 : ndarray
        13-loop power spectrum matrices
    Mcf11 : ndarray
        Spherical Bessel transform matrices of the linear power spectrum to correlation function
    Ml : ndarray
        Spherical Bessel transform matrices of the loop power spectrum to correlation function multipole. Auxiliary matrices used for the loop correlation function matrices.
    Mcf22 : ndarray
        22-loop correlation function multipole matrices
    Mcf13 : ndarray
        13-loop correlation function multipole matrices
    kPow : ndarray
        k's to the powers on which to perform the FFTLog to evaluate the loop power spectrum
    sPow : ndarray
        s's to the powers on which to perform the FFTLog to evaluate the loop correlation function
    optipathP22 : NumPy einsum_path
        Optimization settings for NumPy einsum when performing matrix multiplications to compute the 22-loop power spectrum. For speedup purpose in repetitive evaluations.
    optipathC13l : NumPy einsum_path
        Optimization settings for NumPy einsum when performing matrix multiplications to compute the 13-loop correlation function. For speedup purpose in repetitive evaluations.
    optipathC22l : NumPy einsum_path
        Optimization settings for NumPy einsum when performing matrix multiplications to compute the 22-loop correlation function. For speedup purpose in repetitive evaluations.
    """

    def __init__(self, load=True, save=True, path="./", NFFT=256, fftbias = -1.6, co=co):

        self.co = co

        if self.co.with_cf:
            self.fftsettings = dict(Nmax=NFFT, xmin=1.0e-4, xmax=100.0, bias=fftbias)
        else:
            self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1000.0, bias=fftbias)
        
        # self.fftsettings = dict(Nmax=NFFT, xmin=1.e-4, xmax=100., bias=fftbias) # notice that if one wants to resolve the Cf up to s ~ 1000 (which is clearly way beyond what we can analyze) use here xmin=1e-5 instead 
        
        
        # self.fftsettings = dict(Nmax=NFFT, xmin=1.0e-5, xmax=1000.0, bias=-1.6)

        self.fft = FFTLog(**self.fftsettings)

        if self.co.halohalo:
            if self.co.with_cf:
                self.pyegg = os.path.join(path, "pyegg%s_cf_nl%s.npz") % (NFFT, self.co.Nl)
                if self.co.with_tidal_alignments:
                    self.pyegg = os.path.join(path, "pyegg%s_cf_nl%s_tidal_alignments.npz") % (NFFT, self.co.Nl)
            elif self.co.exact_time:
                self.pyegg = os.path.join(path, "pyegg%s_nl%s_exact_time.npz") % (NFFT, self.co.Nl)
            elif self.co.with_tidal_alignments:
                self.pyegg = os.path.join(path, "pyegg%s_nl%s_tidal_alignments.npz") % (NFFT, self.co.Nl)
            else:
                self.pyegg = os.path.join(path, "pyegg%s_nl%s.npz") % (NFFT, self.co.Nl)
        else:
            self.pyegg = os.path.join(path, "pyegg%s_gm_nl%s.npz") % (NFFT, self.co.Nl)

        if load is True:
            try:
                L = np.load(self.pyegg)
                if (self.fft.Pow - L["Pow"]).any():
                    print(
                        "Loaded loop matrices do not correspond to asked FFTLog configuration. \n Computing new matrices."
                    )
                    load = False
                else:
                    self.M22, self.M13, self.Mcf11, self.Mcf22, self.Mcf13, self.Mcfct = (
                        L["M22"],
                        L["M13"],
                        L["Mcf11"],
                        L["Mcf22"],
                        L["Mcf13"],
                        L["Mcfct"],
                    )
                    save = False
            except:
                print("Can't load loop matrices at %s. \n Computing new matrices." % path)
                load = False

        if load is False:
            self.setM22()
            self.setM13()
            self.setMl()
            self.setMcf11()
            self.setMcf22()
            self.setMcf13()
            self.setMcfct()

        if save is True:
            try:
                np.savez(
                    self.pyegg,
                    Pow=self.fft.Pow,
                    M22=self.M22,
                    M13=self.M13,
                    Mcf11=self.Mcf11,
                    Mcf22=self.Mcf22,
                    Mcf13=self.Mcf13,
                    Mcfct=self.Mcfct,
                )
            except:
                print("Can't save loop matrices at %s." % path)

        self.setkPow()
        self.setsPow()

        # To speed-up matrix multiplication:
        self.optipathP22 = np.einsum_path("nk,mk,bnm->bk", self.kPow, self.kPow, self.M22, optimize="optimal")[0]
        self.optipathC13l = np.einsum_path("ns,ms,blnm->bls", self.sPow, self.sPow, self.Mcf22, optimize="optimal")[0]
        self.optipathC22l = np.einsum_path("ns,ms,blnm->bls", self.sPow, self.sPow, self.Mcf13, optimize="optimal")[0]

    def setM22(self):
        """ Compute the 22-loop power spectrum matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M22 = np.empty(shape=(self.co.N22, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype="complex")
        # common piece of M22
        Ma = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype="complex")
        for u, n1 in enumerate(-0.5 * self.fft.Pow):
            for v, n2 in enumerate(-0.5 * self.fft.Pow):
                Ma[u, v] = M22a(n1, n2)
        for i in range(self.co.N22):
            # singular piece of M22
            Mb = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype="complex")
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    if self.co.halohalo:
                        if self.co.exact_time:
                            Mb[u, v] = M22e[i](n1, n2)
                        elif self.co.with_tidal_alignments:
                            Mb[u, v] = M22ta[i](n1, n2)
                        else:
                            Mb[u, v] = M22b[i](n1, n2)
                    else:
                        Mb[u, v] = M22gm[i](n1, n2)
            self.M22[i] = Ma * Mb

    def setM13(self):
        """ Compute the 13-loop power spectrum matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M13 = np.empty(shape=(self.co.N13, self.fft.Pow.shape[0]), dtype="complex")
        Ma = M13a(-0.5 * self.fft.Pow)
        for i in range(self.co.N13):
            if self.co.halohalo:
                if self.co.exact_time:
                    self.M13[i] = Ma * M13e[i](-0.5 * self.fft.Pow)
                elif self.co.with_tidal_alignments:
                    self.M13[i] = Ma * M13ta[i](-0.5 * self.fft.Pow)
                else:
                    self.M13[i] = Ma * M13b[i](-0.5 * self.fft.Pow)
            else:
                self.M13[i] = Ma * M13gm[i](-0.5 * self.fft.Pow)

    def setMcf11(self):
        """ Compute the 11-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf11 = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype="complex")
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                self.Mcf11[l, u] = 1j ** (2 * l) * MPC(2 * l, n1)

    def setMl(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Ml = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype="complex")
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    self.Ml[l, u, v] = 1j ** (2 * l) * MPC(2 * l, n1 + n2 - 1.5)

    def setMcf22(self):
        """ Compute the 22-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf22 = np.einsum("lnm,bnm->blnm", self.Ml, self.M22)

    def setMcf13(self):
        """ Compute the 13-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf13 = np.einsum("lnm,bn->blnm", self.Ml, self.M13)

    def setMcfct(self):
        """ Compute the counterterm correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcfct = np.empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype="complex")
        for l in range(self.co.Nl):
            for u, n1 in enumerate(-0.5 * self.fft.Pow - 1.0):
                self.Mcfct[l, u] = 1j ** (2 * l) * MPC(2 * l, n1)

    def setkPow(self):
        """ Compute the k's to the powers of the FFTLog to evaluate the loop power spectrum. Called at the instantiation of the class. """
        self.kPow = exp(np.einsum("n,k->nk", self.fft.Pow, log(self.co.k)))

    def setsPow(self):
        """ Compute the s's to the powers of the FFTLog to evaluate the loop correlation function. Called at the instantiation of the class. """
        self.sPow = exp(np.einsum("n,s->ns", -self.fft.Pow - 3.0, log(self.co.s)))

    def CoefkPow(self, Coef):
        """ Multiply the coefficients with the k's to the powers of the FFTLog to evaluate the loop power spectrum. """
        return np.einsum("n,nk->nk", Coef, self.kPow)

    def CoefsPow(self, Coef):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the correlation function. """
        return np.einsum("n,ns->ns", Coef, self.sPow)

    def makeP22(self, CoefkPow, bird):
        """ Perform the 22-loop power spectrum matrix multiplications """
        bird.P22 = self.co.k ** 3 * np.real(
            np.einsum("nk,mk,bnm->bk", CoefkPow, CoefkPow, self.M22, optimize=self.optipathP22)
        )

    def makeP13(self, CoefkPow, bird):
        """ Perform the 13-loop power spectrum matrix multiplications """
        bird.P13 = self.co.k ** 3 * bird.P11 * np.real(np.einsum("nk,bn->bk", CoefkPow, self.M13))

    def makeC11(self, CoefsPow, bird):
        """ Perform the linear correlation function matrix multiplications """
        bird.C11 = np.real(np.einsum("ns,ln->ls", CoefsPow, self.Mcf11))

    def makeCct(self, CoefsPow, bird):
        """ Perform the counterterm correlation function matrix multiplications """
        bird.Cct = self.co.s ** -2 * np.real(np.einsum("ns,ln->ls", CoefsPow, self.Mcfct))

    def makeC22l(self, CoefsPow, bird):
        """ Perform the 22-loop correlation function matrix multiplications """
        bird.C22l = np.real(np.einsum("ns,ms,blnm->lbs", CoefsPow, CoefsPow, self.Mcf22, optimize=self.optipathC22l))

    def makeC13l(self, CoefsPow, bird):
        """ Perform the 13-loop correlation function matrix multiplications """
        bird.C13l = np.real(np.einsum("ns,ms,blnm->lbs", CoefsPow, CoefsPow, self.Mcf13, optimize=self.optipathC13l))

    def Coef(self, bird, window=None):
        """Perform the FFTLog (i.e. calculate the coefficients of the FFTLog) of the input linear power spectrum in the given a Bird().

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        return self.fft.Coef(bird.kin, bird.Pin, window=window)

    def Ps(self, bird, window=None):
        """Compute the loop power spectrum given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird, window=0.2)
        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)

    def Cf(self, bird, window=None):
        """Compute the loop correlation function given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird, window=0.2)
        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22l(coefsPow, bird)
        self.makeC13l(coefsPow, bird)

    def PsCf(self, bird, window=None):
        """Compute the loop power spectrum and correlation function given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird, window=0.2)

        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)

        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22l(coefsPow, bird)
        self.makeC13l(coefsPow, bird)


def M13a(n1):
    """ Common part of the 13-loop matrices """
    return np.tan(n1 * pi) / (14.0 * (-3 + n1) * (-2 + n1) * (-1 + n1) * n1 * pi)


def M22a(n1, n2):
    """ Common part of the 22-loop matrices """
    return (gamma(1.5 - n1) * gamma(1.5 - n2) * gamma(-1.5 + n1 + n2)) / (
        8.0 * pi ** 1.5 * gamma(n1) * gamma(3 - n1 - n2) * gamma(n2)
    )


# specific part of the 13-loop matrices
M13b = {
    0: lambda n1: 1.125,
    1: lambda n1: -(1 / (1 + n1)),
    2: lambda n1: 2.25,
    3: lambda n1: (3 * (-1 + 3 * n1)) / (4.0 * (1 + n1)),
    4: lambda n1: -(1 / (1 + n1)),
    5: lambda n1: -9 / (4 + 4 * n1),
    6: lambda n1: (9 + 18 * n1) / (4 + 4 * n1),
    7: lambda n1: (3 * (-5 + 3 * n1)) / (8.0 * (1 + n1)),
    8: lambda n1: -9 / (4 + 4 * n1),
    9: lambda n1: (9 * n1) / (4.0 + 4 * n1),
}

# specific part of the 22-loop matrices
M22b = {
    0: lambda n1, n2: (
        6
        + n1 ** 4 * (4 - 24 * n2)
        - 7 * n2
        + 8 * n1 ** 5 * n2
        - 13 * n2 ** 2
        + 4 * n2 ** 3
        + 4 * n2 ** 4
        + n1 ** 2 * (-13 + 38 * n2 + 12 * n2 ** 2 - 8 * n2 ** 3)
        + 2 * n1 ** 3 * (2 - 5 * n2 - 4 * n2 ** 2 + 8 * n2 ** 3)
        + n1 * (-7 - 6 * n2 + 38 * n2 ** 2 - 10 * n2 ** 3 - 24 * n2 ** 4 + 8 * n2 ** 5)
    )
    / (4.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (
        -18
        + n1 ** 2 * (1 - 11 * n2)
        - 12 * n2
        + n2 ** 2
        + 10 * n2 ** 3
        + 2 * n1 ** 3 * (5 + 7 * n2)
        + n1 * (-12 - 38 * n2 - 11 * n2 ** 2 + 14 * n2 ** 3)
    )
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: (-3 * n1 + 2 * n1 ** 2 + n2 * (-3 + 2 * n2)) / (n1 * n2),
    3: lambda n1, n2: (
        -4 * (-24 + n2 + 10 * n2 ** 2)
        + 2 * n1 * (-2 + 51 * n2 + 21 * n2 ** 2)
        + n1 ** 2 * (-40 + 42 * n2 + 98 * n2 ** 2)
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    4: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7.0 * n1 * n2),
    5: lambda n1, n2: 2,
    6: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            -2
            + 3 * n2
            + 4 * n1 ** 4 * n2
            + 3 * n2 ** 2
            - 2 * n2 ** 3
            + n1 ** 3 * (-2 - 2 * n2 + 4 * n2 ** 2)
            + n1 ** 2 * (3 - 10 * n2 - 4 * n2 ** 2 + 4 * n2 ** 3)
            + n1 * (3 + 2 * n2 - 10 * n2 ** 2 - 2 * n2 ** 3 + 4 * n2 ** 4)
        )
    )
    / (2.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    7: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2) * (2 + 4 * n2 + 5 * n2 ** 2 + n1 ** 2 * (5 + 7 * n2) + n1 * (4 + 10 * n2 + 7 * n2 ** 2))
    )
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    8: lambda n1, n2: ((n1 + n2) * (-3 + 2 * n1 + 2 * n2)) / (n1 * n2),
    9: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            10
            - 23 * n2
            + 28 * n1 ** 4 * n2
            + 5 * n2 ** 2
            + 2 * n2 ** 3
            + n1 ** 3 * (2 - 46 * n2 + 28 * n2 ** 2)
            + n1 ** 2 * (5 - 38 * n2 - 28 * n2 ** 2 + 28 * n2 ** 3)
            + n1 * (-23 + 94 * n2 - 38 * n2 ** 2 - 46 * n2 ** 3 + 28 * n2 ** 4)
        )
    )
    / (14.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    10: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-58 + 4 * n2 + 35 * n2 ** 2 + 7 * n1 ** 2 * (5 + 7 * n2) + n1 * (4 + 14 * n2 + 49 * n2 ** 2))
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    11: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-8 + 7 * n1 + 7 * n2)) / (7.0 * n1 * n2),
    12: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + 2 * n1 ** 3 - n2 - n2 ** 2 + 2 * n2 ** 3 - n1 ** 2 * (1 + 2 * n2) - n1 * (1 + 2 * n2 + 2 * n2 ** 2))
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    13: lambda n1, n2: ((1 + n1 + n2) * (2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    14: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-6 - n1 + 2 * n1 ** 2 - n2 + 2 * n2 ** 2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    15: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            38
            + 41 * n2
            + 112 * n1 ** 3 * n2
            - 66 * n2 ** 2
            + 2 * n1 ** 2 * (-33 - 18 * n2 + 56 * n2 ** 2)
            + n1 * (41 - 232 * n2 - 36 * n2 ** 2 + 112 * n2 ** 3)
        )
    )
    / (56.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    16: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2))
    / (14.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    17: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (5 + 5 * n1 + 5 * n2 + 7 * n1 * n2))
    / (14.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    18: lambda n1, n2: (3 - 2 * n1 - 2 * n2) / (2.0 * n1 * n2),
    19: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (2.0 * n1 * n2),
    20: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            50
            - 9 * n2
            + 98 * n1 ** 3 * n2
            - 35 * n2 ** 2
            + 7 * n1 ** 2 * (-5 - 18 * n2 + 28 * n2 ** 2)
            + n1 * (-9 - 66 * n2 - 126 * n2 ** 2 + 98 * n2 ** 3)
        )
    )
    / (196.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    21: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + n1 + 4 * n1 ** 3 + n2 - 8 * n1 * n2 - 8 * n1 ** 2 * n2 - 8 * n1 * n2 ** 2 + 4 * n2 ** 3)
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    22: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    23: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (-2 + 7 * n1 + 7 * n2))
    / (56.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    24: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            26
            + 9 * n2
            + 56 * n1 ** 3 * n2
            - 38 * n2 ** 2
            + 2 * n1 ** 2 * (-19 - 18 * n2 + 56 * n2 ** 2)
            + n1 * (9 - 84 * n2 - 36 * n2 ** 2 + 56 * n2 ** 3)
        )
    )
    / (56.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    25: lambda n1, n2: (3 * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    26: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (1 + 2 * n1 + 2 * n2)
        * (1 + 2 * n1 ** 2 - 8 * n1 * n2 + 2 * n2 ** 2)
    )
    / (16.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    27: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (3 + 2 * n1 + 2 * n2))
    / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
}

# with tidal alignments
M13ta = {
    0: lambda n1: (5 + 9 * n1) / (4 + 4 * n1),
    1: lambda n1: -(1 / (1 + n1)),
    2: lambda n1: -(1 + 9 * n1) / (12.0 * (1 + n1)),
    3: lambda n1: (1 + 9 * n1) / (8 + 8 * n1),
    4: lambda n1: 1.125,
    5: lambda n1: -(1 / (1 + n1)),
    6: lambda n1: -(5 + 9 * n1) / (12.0 * (1 + n1)),
    7: lambda n1: 1 / (3 + 3 * n1),
    8: lambda n1: (1 + 9 * n1) / (72 + 72 * n1),
    9: lambda n1: 2.25,
    10: lambda n1: -0.75,
    11: lambda n1: 2.25,
    12: lambda n1: (3 * (-1 + 3 * n1)) / (4.0 * (1 + n1)),
    13: lambda n1: -(1 / (1 + n1)),
    14: lambda n1: (7 - 9 * n1) / (12 + 12 * n1),
    15: lambda n1: (-7 + 9 * n1) / (4.0 * (1 + n1)),
    16: lambda n1: -9 / (4 + 4 * n1),
    17: lambda n1: (9 + 18 * n1) / (4 + 4 * n1),
    18: lambda n1: 3 / (4 + 4 * n1),
    19: lambda n1: (-3 * (3 + n1)) / (4.0 * (1 + n1)),
    20: lambda n1: (9 * n1) / (4 + 4 * n1),
    21: lambda n1: (3 * (-5 + 3 * n1)) / (8.0 * (1 + n1)),
    22: lambda n1: -9 / (4 + 4 * n1),
    23: lambda n1: (9 * n1) / (4 + 4 * n1),
}

M22ta = {
    0: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            -2
            + n1 * (-1 + (13 - 6 * n1) * n1)
            - n2
            + 2 * n1 * (-3 + 2 * n1) * (-9 + n1 * (3 + 7 * n1)) * n2
            + (13 + 2 * n1 * (-27 + 14 * (-1 + n1) * n1)) * n2 ** 2
            + 2 * (-3 + n1 * (-15 + 14 * n1)) * n2 ** 3
            + 28 * n1 * n2 ** 4
        )
    )
    / (14.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-22 + 7 * n1 ** 2 * (5 + 7 * n2) + n2 * (16 + 35 * n2) + n1 * (16 + 7 * n2 * (6 + 7 * n2)))
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-4 + 7 * n1 + 7 * n2)) / (7.0 * n1 * n2),
    3: lambda n1, n2: -(
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            58
            + 98 * n1 ** 3 * n2
            + (3 - 91 * n2) * n2
            + 7 * n1 ** 2 * (-13 - 2 * n2 + 28 * n2 ** 2)
            + n1 * (3 + 2 * n2 * (-73 + 7 * n2 * (-1 + 7 * n2)))
        )
    )
    / (294.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    4: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            58
            + 98 * n1 ** 3 * n2
            + (3 - 91 * n2) * n2
            + 7 * n1 ** 2 * (-13 - 2 * n2 + 28 * n2 ** 2)
            + n1 * (3 + 2 * n2 * (-73 + 7 * n2 * (-1 + 7 * n2)))
        )
    )
    / (196.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    5: lambda n1, n2: (
        6
        + n1 * (1 + 2 * n1) * (-7 + n1 + 2 * n1 ** 2)
        - 7 * n2
        + 2 * n1 * (-3 + n1 * (19 + n1 * (-5 + 4 * (-3 + n1) * n1))) * n2
        + (-13 + 2 * n1 * (19 + 6 * n1 - 4 * n1 ** 2)) * n2 ** 2
        + 2 * (2 + n1 * (-5 - 4 * n1 + 8 * n1 ** 2)) * n2 ** 3
        + 4 * (1 - 6 * n1) * n2 ** 4
        + 8 * n1 * n2 ** 5
    )
    / (4.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    6: lambda n1, n2: (
        n1 ** 2 * (1 - 11 * n2)
        + 2 * n1 ** 3 * (5 + 7 * n2)
        + (-3 + 2 * n2) * (6 + n2 * (8 + 5 * n2))
        + n1 * (-12 + n2 * (-38 + n2 * (-11 + 14 * n2)))
    )
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    7: lambda n1, n2: (-3 + 2 * n1) / n2 + (-3 + 2 * n2) / n1,
    8: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            2
            + n1
            + n2
            - 28 * n1 ** 4 * n2
            + n2 ** 2 * (-13 + 6 * n2)
            + n1 ** 3 * (6 + 30 * n2 - 28 * n2 ** 2)
            - 2 * n1 * n2 * (-3 + 2 * n2) * (-9 + n2 * (3 + 7 * n2))
            + n1 ** 2 * (-13 + 2 * n2 * (27 - 14 * (-1 + n2) * n2))
        )
    )
    / (42.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    9: lambda n1, n2: (
        2
        * (48 - 2 * n1 * (1 + 10 * n1) - 2 * n2 + 3 * n1 * (17 + 7 * n1) * n2 + (-20 + 7 * n1 * (3 + 7 * n1)) * n2 ** 2)
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    10: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7.0 * n1 * n2),
    11: lambda n1, n2: -(
        (-3 + 2 * n1 + 2 * n2)
        * (-22 + 7 * n1 ** 2 * (5 + 7 * n2) + n2 * (16 + 35 * n2) + n1 * (16 + 7 * n2 * (6 + 7 * n2)))
    )
    / (147.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    12: lambda n1, n2: 2.0,
    13: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-4 + 7 * n1 + 7 * n2)) / (21.0 * n1 * n2),
    14: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            58
            + 98 * n1 ** 3 * n2
            + (3 - 91 * n2) * n2
            + 7 * n1 ** 2 * (-13 - 2 * n2 + 28 * n2 ** 2)
            + n1 * (3 + 2 * n2 * (-73 + 7 * n2 * (-1 + 7 * n2)))
        )
    )
    / (1764.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    15: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            -2
            + 3 * n2
            + 4 * n1 ** 4 * n2
            + (3 - 2 * n2) * n2 ** 2
            + 2 * n1 ** 3 * (-1 + n2) * (1 + 2 * n2)
            + n1 * (1 + 2 * n2) * (3 + 2 * (-2 + n2) * n2 * (1 + n2))
            + n1 ** 2 * (3 + 2 * n2 * (-5 + 2 * (-1 + n2) * n2))
        )
    )
    / (2.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    16: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2) * (2 + n2 * (4 + 5 * n2) + n1 ** 2 * (5 + 7 * n2) + n1 * (4 + n2 * (10 + 7 * n2)))
    )
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    17: lambda n1, n2: ((n1 + n2) * (-3 + 2 * n1 + 2 * n2)) / (n1 * n2),
    18: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            -10
            + n1
            + n2
            - 14 * n1 ** 3 * n2
            + 17 * n2 ** 2
            - 2 * n1 * n2 * (-11 + n2 * (3 + 7 * n2))
            + n1 ** 2 * (17 - 2 * n2 * (3 + 14 * n2))
        )
    )
    / (42.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    19: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            10
            + 14 * n1 ** 3 * n2
            - n2 * (1 + 17 * n2)
            + n1 ** 2 * (-17 + 6 * n2 + 28 * n2 ** 2)
            + n1 * (-1 + 2 * n2 * (-11 + n2 * (3 + 7 * n2)))
        )
    )
    / (14.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    20: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            28 * n1 ** 4 * n2
            + (-2 + n2) * (5 + n2) * (-1 + 2 * n2)
            + n1 ** 3 * (2 - 46 * n2 + 28 * n2 ** 2)
            + n1 ** 2 * (5 + 2 * n2 * (-19 + 14 * (-1 + n2) * n2))
            + n1 * (-23 + 2 * n2 * (47 + n2 * (-19 + n2 * (-23 + 14 * n2))))
        )
    )
    / (14.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    21: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-58 + 7 * n1 ** 2 * (5 + 7 * n2) + n2 * (4 + 35 * n2) + n1 * (4 + 7 * n2 * (2 + 7 * n2)))
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    22: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-8 + 7 * n1 + 7 * n2)) / (7.0 * n1 * n2),
    23: lambda n1, n2: -(
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            46
            + 98 * n1 ** 3 * n2
            + (13 - 63 * n2) * n2
            + 7 * n1 ** 2 * (-9 + 2 * n2 * (-5 + 14 * n2))
            + n1 * (13 + 2 * n2 * (-69 + 7 * n2 * (-5 + 7 * n2)))
        )
    )
    / (294.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    24: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            46
            + 98 * n1 ** 3 * n2
            + (13 - 63 * n2) * n2
            + 7 * n1 ** 2 * (-9 + 2 * n2 * (-5 + 14 * n2))
            + n1 * (13 + 2 * n2 * (-69 + 7 * n2 * (-5 + 7 * n2)))
        )
    )
    / (98.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    25: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + (-1 + n1) * n1 * (1 + 2 * n1) - n2 - 2 * n1 * (1 + n1) * n2 - (1 + 2 * n1) * n2 ** 2 + 2 * n2 ** 3)
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    26: lambda n1, n2: ((1 + n1 + n2) * (2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    27: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (6 + n1 - 2 * n1 ** 2 + n2 - 2 * n2 ** 2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    28: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            38
            + 112 * n1 ** 3 * n2
            + (41 - 66 * n2) * n2
            + 2 * n1 ** 2 * (-33 + 2 * n2 * (-9 + 28 * n2))
            + n1 * (41 + 4 * n2 * (-58 + n2 * (-9 + 28 * n2)))
        )
    )
    / (56.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    29: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2))
    / (14.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    30: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (5 + 5 * n1 + 5 * n2 + 7 * n1 * n2))
    / (14.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    31: lambda n1, n2: (3 - 2 * n1 - 2 * n2) / (2.0 * n1 * n2),
    32: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (2.0 * n1 * n2),
    33: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (6 + 7 * n1 + 7 * n2))
    / (168.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    34: lambda n1, n2: -(
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            26
            + 28 * n1 ** 3 * n2
            - n2 * (7 + 48 * n2)
            + 8 * n1 ** 2 * (-6 + n2 * (5 + 7 * n2))
            + n1 * (-7 + 4 * n2 * (-12 + n2 * (10 + 7 * n2)))
        )
    )
    / (84.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    35: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            34
            + n1
            + n2
            + 56 * n1 ** 3 * n2
            - 54 * n2 ** 2
            + 2 * n1 ** 2 * (-27 - 2 * n2 + 56 * n2 ** 2)
            + 4 * n1 * n2 * (-21 + n2 * (-1 + 14 * n2))
        )
    )
    / (56.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    36: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            50
            + 98 * n1 ** 3 * n2
            - n2 * (9 + 35 * n2)
            + 7 * n1 ** 2 * (-5 + 2 * n2 * (-9 + 14 * n2))
            + n1 * (-9 + 2 * n2 * (-33 + 7 * n2 * (-9 + 7 * n2)))
        )
    )
    / (196.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    37: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + n1 + 4 * n1 ** 3 + n2 - 8 * n1 ** 2 * n2 + 4 * n2 ** 3 - 8 * n1 * n2 * (1 + n2))
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    38: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    39: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (-2 + 7 * n1 + 7 * n2))
    / (56.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    40: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            26
            + 56 * n1 ** 3 * n2
            + (9 - 38 * n2) * n2
            + 2 * n1 ** 2 * (-19 + 2 * n2 * (-9 + 28 * n2))
            + n1 * (9 + 4 * n2 * (-21 + n2 * (-9 + 14 * n2)))
        )
    )
    / (56.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    41: lambda n1, n2: (3 * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    42: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (1 + 2 * n1 + 2 * n2)
        * (1 + 2 * (n1 ** 2 - 4 * n1 * n2 + n2 ** 2))
    )
    / (16.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    43: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (3 + 2 * n1 + 2 * n2))
    / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
}


# Exact time dependence loops
M13e = {
    0: lambda n1: 1.125,
    1: lambda n1: -(1 / (1 + n1)),
    2: lambda n1: 2.25,
    3: lambda n1: -(1 / (1 + n1)),
    4: lambda n1: 1.125,
    5: lambda n1: 5.25,
    6: lambda n1: 10.5,
    7: lambda n1: 5.25,
    8: lambda n1: 5.25,
    9: lambda n1: -21 / (4 + 4 * n1),
    10: lambda n1: (21 + 42 * n1) / (4 + 4 * n1),
    11: lambda n1: -21 / (4 + 4 * n1),
    12: lambda n1: (21 * n1) / (4 + 4 * n1),
    13: lambda n1: -21 / (1 + n1),
    14: lambda n1: -21 / (1 + n1),
}

M22e = {
    0: lambda n1, n2: (
        6
        + n1 * (1 + 2 * n1) * (-7 + n1 + 2 * n1 ** 2)
        - 7 * n2
        + 2 * n1 * (-3 + n1 * (19 + n1 * (-5 + 4 * (-3 + n1) * n1))) * n2
        + (-13 + 2 * n1 * (19 + 6 * n1 - 4 * n1 ** 2)) * n2 ** 2
        + 2 * (2 + n1 * (-5 - 4 * n1 + 8 * n1 ** 2)) * n2 ** 3
        + 4 * (1 - 6 * n1) * n2 ** 4
        + 8 * n1 * n2 ** 5
    )
    / (4.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (
        n1 ** 2 * (1 - 11 * n2)
        + 2 * n1 ** 3 * (5 + 7 * n2)
        + (-3 + 2 * n2) * (6 + n2 * (8 + 5 * n2))
        + n1 * (-12 + n2 * (-38 + n2 * (-11 + 14 * n2)))
    )
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: (-3 + 2 * n1) / n2 + (-3 + 2 * n2) / n1,
    3: lambda n1, n2: (
        2
        * (48 - 2 * n1 * (1 + 10 * n1) - 2 * n2 + 3 * n1 * (17 + 7 * n1) * n2 + (-20 + 7 * n1 * (3 + 7 * n1)) * n2 ** 2)
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    4: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7.0 * n1 * n2),
    5: lambda n1, n2: 2,
    6: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            -2
            + 3 * n2
            + 4 * n1 ** 4 * n2
            + (3 - 2 * n2) * n2 ** 2
            + 2 * n1 ** 3 * (-1 + n2) * (1 + 2 * n2)
            + n1 * (1 + 2 * n2) * (3 + 2 * (-2 + n2) * n2 * (1 + n2))
            + n1 ** 2 * (3 + 2 * n2 * (-5 + 2 * (-1 + n2) * n2))
        )
    )
    / (2.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    7: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2) * (2 + n2 * (4 + 5 * n2) + n1 ** 2 * (5 + 7 * n2) + n1 * (4 + n2 * (10 + 7 * n2)))
    )
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    8: lambda n1, n2: ((n1 + n2) * (-3 + 2 * n1 + 2 * n2)) / (n1 * n2),
    9: lambda n1, n2: (
        (-2 + n1 + n2)
        * (-3 + 2 * n1 + 2 * n2)
        * (
            -2
            + n1 * (3 + 2 * n1)
            + 3 * n2
            + 2 * n1 * (-4 + n1 * (-1 + 2 * n1)) * n2
            - 2 * (-1 + n1) * n2 ** 2
            + 4 * n1 * n2 ** 3
        )
    )
    / (2.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    10: lambda n1, n2: ((-2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (8 + 5 * n2 + n1 * (5 + 7 * n2)))
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    11: lambda n1, n2: ((-2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2)) / (n1 * n2),
    12: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + (-1 + n1) * n1 * (1 + 2 * n1) - n2 - 2 * n1 * (1 + n1) * n2 - (1 + 2 * n1) * n2 ** 2 + 2 * n2 ** 3)
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    13: lambda n1, n2: ((1 + n1 + n2) * (2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    14: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (6 + n1 - 2 * n1 ** 2 + n2 - 2 * n2 ** 2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    15: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            2
            + (11 - 6 * n1) * n1
            + 11 * n2
            + 4 * (-2 + n1) * n1 * (5 + 4 * n1) * n2
            + 2 * (-3 - 6 * n1 + 8 * n1 ** 2) * n2 ** 2
            + 16 * n1 * n2 ** 3
        )
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    16: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2))
    / (14.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    17: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (5 + 5 * n1 + 5 * n2 + 7 * n1 * n2))
    / (14.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    18: lambda n1, n2: (3 - 2 * n1 - 2 * n2) / (2.0 * n1 * n2),
    19: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (2.0 * n1 * n2),
    20: lambda n1, n2: (
        (-2 + n1 + n2) * (-1 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 * n2)
    )
    / (4.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    21: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + n1 + 4 * n1 ** 3 + n2 - 8 * n1 ** 2 * n2 + 4 * n2 ** 3 - 8 * n1 * n2 * (1 + n2))
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    22: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    23: lambda n1, n2: -((-2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (8.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    24: lambda n1, n2: (
        (-2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (-1 + 4 * n1 * n2)
    )
    / (8.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    25: lambda n1, n2: (3 * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    26: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (1 + 2 * n1 + 2 * n2)
        * (1 + 2 * (n1 ** 2 - 4 * n1 * n2 + n2 ** 2))
    )
    / (16.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    27: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (3 + 2 * n1 + 2 * n2))
    / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    28: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (n1 * (-1 + 2 * n1) + (-2 + n2) * (3 + 2 * n2)))
    / (2.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    29: lambda n1, n2: (2 * (-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2))
    / (7.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    30: lambda n1, n2: (-6 + 4 * n1 + 4 * n2) / (n1 * n2),
    31: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (2.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    32: lambda n1, n2: ((-2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (2.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    33: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (2.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    34: lambda n1, n2: ((1 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2))
    / (2.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    35: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (n1 * (1 + n1) * n2 * (1 + n2)),
}

# galaxy matter
M13gm = {
    0: lambda n1: (5 + 9 * n1) / (8 + 8 * n1),
    1: lambda n1: -(1 / (2 + 2 * n1)),
    2: lambda n1: (3 * (5 + 9 * n1)) / (8.0 * (1 + n1)),
    3: lambda n1: -(1 / (2 + 2 * n1)),
    4: lambda n1: (-7 + 9 * n1) / (8.0 * (1 + n1)),
    5: lambda n1: -9 / (8 + 8 * n1),
    6: lambda n1: (9 + 18 * n1) / (8 + 8 * n1),
    7: lambda n1: -9 / (8 + 8 * n1),
    8: lambda n1: (3 * (-2 + 9 * n1)) / (8.0 * (1 + n1)),
    9: lambda n1: -9 / (4 + 4 * n1),
    10: lambda n1: (9 * n1) / (4 + 4 * n1),
}

M22gm = {
    0: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            -2
            + n1 * (-1 + (13 - 6 * n1) * n1)
            - n2
            + 2 * n1 * (-3 + 2 * n1) * (-9 + n1 * (3 + 7 * n1)) * n2
            + (13 + 2 * n1 * (-27 + 14 * (-1 + n1) * n1)) * n2 ** 2
            + 2 * (-3 + n1 * (-15 + 14 * n1)) * n2 ** 3
            + 28 * n1 * n2 ** 4
        )
    )
    / (28.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-22 + 7 * n1 ** 2 * (5 + 7 * n2) + n2 * (16 + 35 * n2) + n1 * (16 + 7 * n2 * (6 + 7 * n2)))
    )
    / (98.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-4 + 7 * n1 + 7 * n2)) / (14.0 * n1 * n2),
    3: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            -14
            + n1 * (19 + (41 - 46 * n1) * n1)
            + 19 * n2
            + 2 * n1 * (63 + n1 * (-96 + n1 * (-31 + 42 * n1))) * n2
            + (41 + 4 * n1 * (-48 + 5 * n1 * (-3 + 7 * n1))) * n2 ** 2
            + 2 * (-23 + n1 * (-31 + 70 * n1)) * n2 ** 3
            + 84 * n1 * n2 ** 4
        )
    )
    / (28.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    4: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-22 + 7 * n1 ** 2 * (5 + 7 * n2) + n2 * (16 + 35 * n2) + n1 * (16 + 7 * n2 * (6 + 7 * n2)))
    )
    / (49.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    5: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-4 + 7 * n1 + 7 * n2)) / (7.0 * n1 * n2),
    6: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            46
            + 98 * n1 ** 3 * n2
            + (13 - 63 * n2) * n2
            + 7 * n1 ** 2 * (-9 + 2 * n2 * (-5 + 14 * n2))
            + n1 * (13 + 2 * n2 * (-69 + 7 * n2 * (-5 + 7 * n2)))
        )
    )
    / (196.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    7: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (
            2
            + n1 * (1 + 2 * n1) * (-1 + 4 * (-1 + n1) * n1)
            - n2
            - 8 * n1 * (-2 + n1 ** 2) * n2
            - 2 * (3 + 8 * n1 ** 2) * n2 ** 2
            - 4 * (1 + 2 * n1) * n2 ** 3
            + 8 * n2 ** 4
        )
    )
    / (16.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    8: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            66
            + 28 * n1 ** 3 * (-1 + 6 * n2)
            + 8 * n1 ** 2 * (-17 + 6 * n2 + 28 * n2 ** 2)
            + n2 * (27 - 4 * n2 * (34 + 7 * n2))
            + n1 * (27 + 4 * n2 * (-65 + 6 * n2 * (2 + 7 * n2)))
        )
    )
    / (112.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    9: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2))
    / (28.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    10: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (5 + 5 * n1 + 5 * n2 + 7 * n1 * n2))
    / (28.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    11: lambda n1, n2: (3 - 2 * n1 - 2 * n2) / (4.0 * n1 * n2),
    12: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (4.0 * n1 * n2),
    13: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (6 + 7 * n1 + 7 * n2))
    / (112.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    14: lambda n1, n2: (
        3
        * (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            202
            + 392 * n1 ** 3 * n2
            + (37 - 294 * n2) * n2
            + 98 * n1 ** 2 * (1 + 2 * n2) * (-3 + 4 * n2)
            + n1 * (37 + 4 * n2 * (-141 + 49 * n2 * (-1 + 2 * n2)))
        )
    )
    / (784.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    15: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (2 + n1 + 4 * n1 ** 3 + n2 - 8 * n1 ** 2 * n2 + 4 * n2 ** 3 - 8 * n1 * n2 * (1 + n2))
    )
    / (16.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    16: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2))
    / (16.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    17: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            18
            + 28 * n1 ** 3
            + 28 * n1 ** 2 * (1 - 4 * n2)
            + n1 * (-15 + 16 * (1 - 7 * n2) * n2)
            + n2 * (-15 + 28 * n2 * (1 + n2))
        )
    )
    / (112.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    18: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (
            66
            + 28 * n1 ** 3 * (-1 + 6 * n2)
            + 4 * n1 ** 2 * (-33 - 4 * n2 + 84 * n2 ** 2)
            + n2 * (25 - 4 * n2 * (33 + 7 * n2))
            + n1 * (25 + 8 * n2 * (-28 + n2 * (-2 + 21 * n2)))
        )
    )
    / (112.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    19: lambda n1, n2: (3 * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
    20: lambda n1, n2: (
        (-3 + 2 * n1 + 2 * n2)
        * (-1 + 2 * n1 + 2 * n2)
        * (1 + 2 * n1 + 2 * n2)
        * (1 + 2 * (n1 ** 2 - 4 * n1 * n2 + n2 ** 2))
    )
    / (16.0 * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    21: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (3 + 2 * n1 + 2 * n2))
    / (32.0 * n1 * (1 + n1) * n2 * (1 + n2)),
}
