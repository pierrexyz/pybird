from pybird.module import *
from fftlog.fftlog import FFTLog
from fftlog.sbt import MPC
from fftlog.utils import CoefWindow
from pybird.common import co
import numpy as np #this module is numpy-based as it does not get called during the liklelihood at runtime 

class NonLinear(object):
    """
    Computes the one-loop power spectrum and correlation function using FFTLog.
    
    The NonLinear class calculates the one-loop power spectrum (P22, P13) and one-loop
    correlation function multipoles using the FFTLog method, which allows for efficient
    spherical Bessel transforms between Fourier and configuration space. The correlation
    function is particularly useful for performing the IR-resummation of the power spectrum.
    
    Loop integrals and spherical Bessel transform matrices are either loaded from disk
    or computed at instantiation and optionally saved for future use.
    
    Attributes:
        co (Common): Common parameters shared across calculations.
        fftsettings (dict): Settings for FFTLog (Nmax, xmin, xmax, bias, window).
        fft (FFTLog): FFTLog engine for the transforms.
        pyegg (str): Path to the cached matrices file.
        
        M22 (ndarray): 22-loop power spectrum matrices.
        M13 (ndarray): 13-loop power spectrum matrices.
        Mcf11 (ndarray): Linear correlation function multipole matrices.
        Ml (ndarray): Auxiliary matrices for loop correlation function.
        Mcf22 (ndarray): 22-loop correlation function multipole matrices.
        Mcf13 (ndarray): 13-loop correlation function multipole matrices.
        Mcfct (ndarray): Counterterm correlation function matrices.
        
        kPow (ndarray): k^n powers for the loop power spectrum evaluation.
        sPow (ndarray): s^n powers for the loop correlation function evaluation.
        
        optipathP22 (einsum_path): Optimized einsum path for 22-loop power spectrum.
        optipathC13l (einsum_path): Optimized einsum path for 13-loop correlation function.
        optipathC22l (einsum_path): Optimized einsum path for 22-loop correlation function.
    
    Methods:
        setM22(): Compute the 22-loop power spectrum matrices.
        setM13(): Compute the 13-loop power spectrum matrices.
        setMcf11(): Compute the linear correlation function matrices.
        setMl(): Compute the auxiliary matrices for loop correlation function.
        setMcf22(): Compute the 22-loop correlation function matrices.
        setMcf13(): Compute the 13-loop correlation function matrices.
        setMcfct(): Compute the counterterm correlation function matrices.
        
        setkPow(): Compute the k^n powers for the loop power spectrum.
        setsPow(): Compute the s^n powers for the loop correlation function.
        
        CoefkPow(Coef): Multiply coefficients with k^n powers.
        CoefsPow(Coef): Multiply coefficients with s^n powers.
        
        makeP22(CoefkPow, bird): Compute the 22-loop power spectrum.
        makeP13(CoefkPow, bird): Compute the 13-loop power spectrum.
        makeC11(CoefsPow, bird): Compute the linear correlation function.
        makeCct(CoefsPow, bird): Compute the counterterm correlation function.
        makeC22l(CoefsPow, bird): Compute the 22-loop correlation function.
        makeC13l(CoefsPow, bird): Compute the 13-loop correlation function.
        
        Coef(bird): Compute the FFTLog coefficients for the input linear power spectrum.
        Ps(bird): Compute the loop power spectrum.
        Cf(bird): Compute the loop correlation function.
        PsCf(bird): Compute both loop power spectrum and correlation function.
        clean_lowk(bird): Clean up low-k values of loop power spectra.
    """

    def __init__(self, load_matrix=True, save_matrix=True, path=None, NFFT=256, fftbias=-1.6, co=co):
        """Initialize the NonLinear engine for one-loop calculations.
        
        Parameters
        ----------
        load_matrix : bool, optional
            Whether to load pre-computed matrices from disk, by default True
        save_matrix : bool, optional
            Whether to save computed matrices to disk for future use, by default True
        path : str, optional
            Directory path for saving/loading matrix files. If None, uses pybird/data/tmp, by default None
        NFFT : int, optional
            Number of FFT points for FFTLog transforms, by default 256
        fftbias : float, optional
            Real power bias parameter for FFTLog decomposition, by default -1.6
        co : Common, optional
            Common parameters object, by default co
            
        Notes
        -----
        The initialization computes or loads the loop matrices (M22, M13) and correlation
        function transform matrices. Matrix files are named based on NFFT, bias, and
        multipole settings to ensure correct caching.
        """

        self.co = co

        self.fftsettings = dict(Nmax=NFFT, xmin=1.e-4, xmax=100., bias=fftbias, window=0.2) # notice that if one wants to resolve the Cf up to s ~ 1000 (which is clearly way beyond what we can analyze) use here xmin=1e-5 instead 
        self.fft = FFTLog(**self.fftsettings)

        # Set default path to platform-appropriate cache directory if not specified
        if path is None:
            import os
            
            # Use platformdirs for proper cross-platform cache directory
            try:
                import platformdirs
                cache_dir = platformdirs.user_cache_dir("pybird", "pybird")
            except ImportError:
                # Fallback if platformdirs somehow not available
                import tempfile
                cache_dir = os.path.join(tempfile.gettempdir(), "pybird_cache")
            
            path = os.path.join(cache_dir, 'loop_matrices')
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)

        if self.co.halohalo:
            if self.co.exact_time: self.pyegg = os.path.join(path, 'pyegg%s_nl%s_exact_time.npz') % (NFFT, self.co.Nl)
            elif self.co.with_tidal_alignments: self.pyegg = os.path.join(path, 'pyegg%s_nl%s_tidal_alignments.npz') % (NFFT, self.co.Nl)
            else: self.pyegg = os.path.join(path, 'pyegg%s_fftbias_%s_nl%s.npz') % (NFFT, fftbias, self.co.Nl)
        else:
            self.pyegg = os.path.join(path, 'pyegg%s_gm_nl%s.npz') % (NFFT, self.co.Nl)

        if load_matrix:
            try:
                L = load( self.pyegg )
                if (self.fft.Pow - L['Pow']).any():
                    print ('Loaded loop matrices do not correspond to asked FFTLog configuration. \n Computing new matrices.')
                    load_matrix = False
                else:
                    self.M22, self.M13, self.Mcf11, self.Mcf22, self.Mcf13, self.Mcfct = L['M22'], L['M13'], L['Mcf11'], L['Mcf22'], L['Mcf13'], L['Mcfct']
                    save_matrix = False
            except:
                print ('Can\'t load loop matrices at %s.' % path)
                load_matrix = False

        if not load_matrix:
            if is_jax: print ('WARNING: Loop matrix computation is extremely slow because jax is enabled! \n ADVICE: stop this run and start a new one with jax disabled. \n The loop matrices, fastly computed, will be saved. \n Then, restart your run in jax that will be able to load the pre-computed loop matrices.  ')
            print ('Computing loop matrices...')
            self.setM22()
            self.setM13()
            self.setMl()
            self.setMcf11()
            self.setMcf22()
            self.setMcf13()
            self.setMcfct()
            print ('Loop matrices computed!')

        if save_matrix:
            try: savez(self.pyegg, Pow=self.fft.Pow, M22=self.M22, M13=self.M13, Mcf11=self.Mcf11, Mcf22=self.Mcf22, Mcf13=self.Mcf13, Mcfct=self.Mcfct)
            except: print ('Can\'t save loop matrices at %s.' % path)

        self.setkPow()
        self.setsPow()

        # To speed-up matrix multiplication:
        self.optipathP13 = einsum_path('nk,bn->bk', self.kPow, self.M13, optimize='optimal')[0]
        self.optipathP22 = einsum_path('nk,mk,bnm->bk', self.kPow, self.kPow, self.M22, optimize='optimal')[0]
        self.optipathP13 = einsum_path('nk,bn->bk', self.kPow, self.M13, optimize='optimal')[0]
        self.optipathC13l = einsum_path('ns,ms,blnm->bls', self.sPow, self.sPow, self.Mcf22, optimize='optimal')[0]
        self.optipathC22l = einsum_path('ns,ms,blnm->bls', self.sPow, self.sPow, self.Mcf13, optimize='optimal')[0]

    def setM22(self):
        """ Compute the 22-loop power spectrum matrices. Called at the instantiation of the class if the matrices are not loaded. """
        # self.M22 = empty(shape=(self.co.N22, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        # # common piece of M22
        # Ma = empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        # for u, n1 in enumerate(-0.5 * self.fft.Pow):
        #     for v, n2 in enumerate(-0.5 * self.fft.Pow):
        #         Ma[u, v] = M22a(n1, n2)
        # for i in range(self.co.N22):
        #     # singular piece of M22
        #     Mb = empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        #     for u, n1 in enumerate(-0.5 * self.fft.Pow):
        #         for v, n2 in enumerate(-0.5 * self.fft.Pow):
        #             if self.co.halohalo:
        #                 if self.co.exact_time: Mb[u, v] = M22e[i](n1, n2)
        #                 elif self.co.with_tidal_alignments: Mb[u, v] = M22ta[i](n1, n2)
        #                 else: Mb[u, v] = M22b[i](n1, n2)
        #             else: Mb[u, v] = M22gm[i](n1, n2)
        #     self.M22[i] = Ma * Mb

        def _M22(i, n1, n2):
            if self.co.halohalo:
                if self.co.exact_time: return M22e[i](n1, n2)
                elif self.co.with_tidal_alignments: return M22ta[i](n1, n2)
                else: return M22b[i](n1, n2)
            else: return M22gm[i](n1, n2)

        Ma = array([[M22a(n1, n2) for n2 in -0.5 * self.fft.Pow] for n1 in -0.5 * self.fft.Pow]) # common piece of M22
        self.M22 = self.co.N22 * [None]
        for i in range(self.co.N22):
            Mb = array([[_M22(i, n1, n2) for n2 in -0.5 * self.fft.Pow] for n1 in -0.5 * self.fft.Pow]) # singular piece of M22
            self.M22[i] = Ma * Mb
        self.M22 = array(self.M22)

    def setM13(self):
        """ Compute the 13-loop power spectrum matrices. Called at the instantiation of the class if the matrices are not loaded. """
        # self.M13 = empty(shape=(self.co.N13, self.fft.Pow.shape[0]), dtype='complex')
        self.M13 = self.co.N13 * [None]
        Ma = M13a(-0.5 * self.fft.Pow)
        for i in range(self.co.N13):
            if self.co.halohalo:
                if self.co.exact_time: self.M13[i] = Ma * M13e[i](-0.5 * self.fft.Pow)
                elif self.co.with_tidal_alignments: self.M13[i] = Ma * M13ta[i](-0.5 * self.fft.Pow)
                else: self.M13[i] = Ma * M13b[i](-0.5 * self.fft.Pow)
            else: self.M13[i] = Ma * M13gm[i](-0.5 * self.fft.Pow)
        self.M13 = array(self.M13)

    def setMcf11(self):
        """ Compute the 11-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        # self.Mcf11 = empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        # for l in range(self.co.Nl):
        #     for u, n1 in enumerate(-0.5 * self.fft.Pow):
        #         self.Mcf11[l, u] = 1j**(2*l) * MPC(2 * l, n1)
        self.Mcf11 = array([[1j**(2*l) * MPC(2 * l, n1) for n1 in -0.5 * self.fft.Pow] for l in range(self.co.Nl)])

    def setMl(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        # self.Ml = empty(shape=(self.co.Nl, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        # for l in range(self.co.Nl):
        #     for u, n1 in enumerate(-0.5 * self.fft.Pow):
        #         for v, n2 in enumerate(-0.5 * self.fft.Pow):
        #             self.Ml[l, u, v] = 1j**(2*l) * MPC(2 * l, n1 + n2 - 1.5)
        self.Ml = array([[[1j**(2*l) * MPC(2 * l, n1 + n2 - 1.5) for n1 in -0.5 * self.fft.Pow] for n2 in -0.5 * self.fft.Pow] for l in range(self.co.Nl)])

    def setMcf22(self):
        """ Compute the 22-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf22 = array(np.einsum('lnm,bnm->blnm', self.Ml, self.M22))

    def setMcf13(self):
        """ Compute the 13-loop correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mcf13 = array(np.einsum('lnm,bn->blnm', self.Ml, self.M13))

    def setMcfct(self):
        """ Compute the counterterm correlation function matrices. Called at the instantiation of the class if the matrices are not loaded. """
        # self.Mcfct = empty(shape=(self.co.Nl, self.fft.Pow.shape[0]), dtype='complex')
        # for l in range(self.co.Nl):
        #     for u, n1 in enumerate(-0.5 * self.fft.Pow - 1.):
        #         self.Mcfct[l, u] = 1j**(2*l) * MPC(2 * l, n1)
        self.Mcfct = array([[1j**(2*l) * MPC(2 * l, n1) for n1 in -0.5 * self.fft.Pow - 1.] for l in range(self.co.Nl)])

    def setkPow(self):
        """ Compute the k's to the powers of the FFTLog to evaluate the loop power spectrum. Called at the instantiation of the class. """
        self.kPow = array(np.exp(np.einsum('n,k->nk', self.fft.Pow, log(self.co.k))))

    def setsPow(self):
        """ Compute the s's to the powers of the FFTLog to evaluate the loop correlation function. Called at the instantiation of the class. """
        self.sPow = array(np.exp(np.einsum('n,s->ns', -self.fft.Pow - 3., log(self.co.s))))

    def CoefkPow(self, Coef):
        """ Multiply the coefficients with the k's to the powers of the FFTLog to evaluate the loop power spectrum. """
        return array(einsum('n,nk->nk', Coef, self.kPow))

    def CoefsPow(self, Coef):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the correlation function. """
        return array(einsum('n,ns->ns', Coef, self.sPow))

    def makeP22(self, CoefkPow, bird):
        """ Perform the 22-loop power spectrum matrix multiplications """
        bird.P22 = self.co.k**3 * real(einsum('nk,mk,bnm->bk', CoefkPow, CoefkPow, self.M22, optimize=self.optipathP22))

    def makeP13(self, CoefkPow, bird):
        """ Perform the 13-loop power spectrum matrix multiplications """
        bird.P13 = self.co.k**3 * bird.P11 * real(einsum('nk,bn->bk', CoefkPow, self.M13, optimize=self.optipathP13))

    def makeC11(self, CoefsPow, bird):
        """ Perform the linear correlation function matrix multiplications """
        bird.C11 = real(einsum('ns,ln->ls', CoefsPow, self.Mcf11))

    def makeCct(self, CoefsPow, bird):
        """ Perform the counterterm correlation function matrix multiplications """
        bird.Cct = self.co.s**-2 * real(einsum('ns,ln->ls', CoefsPow, self.Mcfct))

    def makeC22l(self, CoefsPow, bird):
        """ Perform the 22-loop correlation function matrix multiplications """
        bird.C22l = real(einsum('ns,ms,blnm->lbs', CoefsPow, CoefsPow, self.Mcf22, optimize=self.optipathC22l))

    def makeC13l(self, CoefsPow, bird):
        """ Perform the 13-loop correlation function matrix multiplications """
        bird.C13l = real(einsum('ns,ms,blnm->lbs', CoefsPow, CoefsPow, self.Mcf13, optimize=self.optipathC13l))

    def Coef(self, bird):
        """ Perform the FFTLog (i.e. calculate the coefficients of the FFTLog) of the input linear power spectrum in the given a Bird().

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        return self.fft.Coef(bird.kin, bird.Pin_2)

    def Ps(self, bird):
        """ Compute the loop power spectrum given a Bird(). Perform the FFTLog and the matrix multiplications.

        Parameters
        ----------
        bird : class
            an object of type Bird()
        """
        coef = self.Coef(bird)
        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)
        self.clean_lowk(bird)

    def Cf(self, bird):
        """Compute the loop correlation function given a Bird().
        
        Performs the FFTLog spherical Bessel transform and matrix multiplications
        to compute the one-loop correlation function multipoles.

        Parameters
        ----------
        bird : Bird
            Bird object containing cosmological parameters and linear power spectrum
            
        Notes
        -----
        This method computes C11, C22l, and C13l terms, which are the correlation
        function multipoles corresponding to linear, 22-loop, and 13-loop contributions.
        The results are stored in the bird object.
        """
        coef = self.Coef(bird)
        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22l(coefsPow, bird)
        self.makeC13l(coefsPow, bird)

    def PsCf(self, bird):
        """Compute both power spectrum and correlation function loop terms.
        
        This is the main method that computes both the one-loop power spectrum
        and correlation function multipoles for a given Bird object.

        Parameters
        ----------
        bird : Bird
            Bird object containing cosmological parameters and linear power spectrum
            
        Notes
        -----
        This method combines Ps() and Cf() calculations, computing all one-loop
        terms (P22, P13, C11, C22l, C13l) and counterterms efficiently.
        """
        coef = self.Coef(bird)

        coefkPow = self.CoefkPow(coef)
        self.makeP22(coefkPow, bird)
        self.makeP13(coefkPow, bird)
        self.clean_lowk(bird)

        coefsPow = self.CoefsPow(coef)
        self.makeC11(coefsPow, bird)
        self.makeCct(coefsPow, bird)
        self.makeC22l(coefsPow, bird)
        self.makeC13l(coefsPow, bird)

    def clean_lowk(self, bird):
        if self.co.id_kstable > 0: # = 1 if kmin < 0.001,
            bird.P22[:, 0] = 1. * bird.P22[:, 1] # replace junky P22(kmin) = P22(0.001)
            bird.P13[:, 0] = 1. * bird.P13[:, 1] # replace junky P13(kmin) = P22(0.001)

def M13a(n1):
    """ Common part of the 13-loop matrices """
    return np.tan(n1 * pi) / (14. * (-3 + n1) * (-2 + n1) * (-1 + n1) * n1 * pi)

def M22a(n1, n2):
    """ Common part of the 22-loop matrices """
    return (gamma(1.5 - n1) * gamma(1.5 - n2) * gamma(-1.5 + n1 + n2)) / (8. * pi**1.5 * gamma(n1) * gamma(3 - n1 - n2) * gamma(n2))

# specific part of the 13-loop matrices
M13b = {
    0: lambda n1: 1.125,
    1: lambda n1: -(1 / (1 + n1)),
    2: lambda n1: 2.25,
    3: lambda n1: (3 * (-1 + 3 * n1)) / (4. * (1 + n1)),
    4: lambda n1: -(1 / (1 + n1)),
    5: lambda n1: -9 / (4 + 4 * n1),
    6: lambda n1: (9 + 18 * n1) / (4 + 4 * n1),
    7: lambda n1: (3 * (-5 + 3 * n1)) / (8. * (1 + n1)),
    8: lambda n1: -9 / (4 + 4 * n1),
    9: lambda n1: (9 * n1) / (4. + 4 * n1),
}

# specific part of the 22-loop matrices
M22b = {
    0: lambda n1, n2: (6 + n1**4 * (4 - 24 * n2) - 7 * n2 + 8 * n1**5 * n2 - 13 * n2**2 + 4 * n2**3 + 4 * n2**4 + n1**2 * (-13 + 38 * n2 + 12 * n2**2 - 8 * n2**3) + 2 * n1**3 * (2 - 5 * n2 - 4 * n2**2 + 8 * n2**3) + n1 * (-7 - 6 * n2 + 38 * n2**2 - 10 * n2**3 - 24 * n2**4 + 8 * n2**5)) / (4. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (-18 + n1**2 * (1 - 11 * n2) - 12 * n2 + n2**2 + 10 * n2**3 + 2 * n1**3 * (5 + 7 * n2) + n1 * (-12 - 38 * n2 - 11 * n2**2 + 14 * n2**3)) / (7. * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: (-3 * n1 + 2 * n1**2 + n2 * (-3 + 2 * n2)) / (n1 * n2),
    3: lambda n1, n2: (-4 * (-24 + n2 + 10 * n2**2) + 2 * n1 * (-2 + 51 * n2 + 21 * n2**2) + n1**2 * (-40 + 42 * n2 + 98 * n2**2)) / (49. * n1 * (1 + n1) * n2 * (1 + n2)),
    4: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7. * n1 * n2),
    5: lambda n1, n2: 2,
    6: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-2 + 3 * n2 + 4 * n1**4 * n2 + 3 * n2**2 - 2 * n2**3 + n1**3 * (-2 - 2 * n2 + 4 * n2**2) + n1**2 * (3 - 10 * n2 - 4 * n2**2 + 4 * n2**3) + n1 * (3 + 2 * n2 - 10 * n2**2 - 2 * n2**3 + 4 * n2**4))) / (2. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    7: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (2 + 4 * n2 + 5 * n2**2 + n1**2 * (5 + 7 * n2) + n1 * (4 + 10 * n2 + 7 * n2**2))) / (7. * n1 * (1 + n1) * n2 * (1 + n2)),
    8: lambda n1, n2: ((n1 + n2) * (-3 + 2 * n1 + 2 * n2)) / (n1 * n2),
    9: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (10 - 23 * n2 + 28 * n1**4 * n2 + 5 * n2**2 + 2 * n2**3 + n1**3 * (2 - 46 * n2 + 28 * n2**2) + n1**2 * (5 - 38 * n2 - 28 * n2**2 + 28 * n2**3) + n1 * (-23 + 94 * n2 - 38 * n2**2 - 46 * n2**3 + 28 * n2**4))) / (14. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    10: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-58 + 4 * n2 + 35 * n2**2 + 7 * n1**2 * (5 + 7 * n2) + n1 * (4 + 14 * n2 + 49 * n2**2))) / (49. * n1 * (1 + n1) * n2 * (1 + n2)),
    11: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-8 + 7 * n1 + 7 * n2)) / (7. * n1 * n2),
    12: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (2 + 2 * n1**3 - n2 - n2**2 + 2 * n2**3 - n1**2 * (1 + 2 * n2) - n1 * (1 + 2 * n2 + 2 * n2**2))) / (8. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    13: lambda n1, n2: ((1 + n1 + n2) * (2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (8. * n1 * (1 + n1) * n2 * (1 + n2)),
    14: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-6 - n1 + 2 * n1**2 - n2 + 2 * n2**2)) / (8. * n1 * (1 + n1) * n2 * (1 + n2)),
    15: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (38 + 41 * n2 + 112 * n1**3 * n2 - 66 * n2**2 + 2 * n1**2 * (-33 - 18 * n2 + 56 * n2**2) + n1 * (41 - 232 * n2 - 36 * n2**2 + 112 * n2**3))) / (56. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    16: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (9 + 3 * n1 + 3 * n2 + 7 * n1 * n2)) / (14. * n1 * (1 + n1) * n2 * (1 + n2)),
    17: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (5 + 5 * n1 + 5 * n2 + 7 * n1 * n2)) / (14. * n1 * (1 + n1) * n2 * (1 + n2)),
    18: lambda n1, n2: (3 - 2 * n1 - 2 * n2) / (2. * n1 * n2),
    19: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (2. * n1 * n2),
    20: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (50 - 9 * n2 + 98 * n1**3 * n2 - 35 * n2**2 + 7 * n1**2 * (-5 - 18 * n2 + 28 * n2**2) + n1 * (-9 - 66 * n2 - 126 * n2**2 + 98 * n2**3))) / (196. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    21: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (2 + n1 + 4 * n1**3 + n2 - 8 * n1 * n2 - 8 * n1**2 * n2 - 8 * n1 * n2**2 + 4 * n2**3)) / (8. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    22: lambda n1, n2: ((2 + n1 + n2) * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2)) / (8. * n1 * (1 + n1) * n2 * (1 + n2)),
    23: lambda n1, n2: -((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (-2 + 7 * n1 + 7 * n2)) / (56. * n1 * (1 + n1) * n2 * (1 + n2)),
    24: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (26 + 9 * n2 + 56 * n1**3 * n2 - 38 * n2**2 + 2 * n1**2 * (-19 - 18 * n2 + 56 * n2**2) + n1 * (9 - 84 * n2 - 36 * n2**2 + 56 * n2**3))) / (56. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    25: lambda n1, n2: (3 * (-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2)) / (32. * n1 * (1 + n1) * n2 * (1 + n2)),
    26: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (1 + 2 * n1**2 - 8 * n1 * n2 + 2 * n2**2)) / (16. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    27: lambda n1, n2: ((-3 + 2 * n1 + 2 * n2) * (-1 + 2 * n1 + 2 * n2) * (1 + 2 * n1 + 2 * n2) * (3 + 2 * n1 + 2 * n2)) / (32. * n1 * (1 + n1) * n2 * (1 + n2)),
}

# with tidal alignments
M13ta = {
    0: lambda n1: (5 + 9*n1)/(4 + 4*n1),
    1: lambda n1: -(1/(1 + n1)),
    2: lambda n1: -(1 + 9*n1)/(12.*(1 + n1)),
    3: lambda n1: (1 + 9*n1)/(8 + 8*n1),
    4: lambda n1: 1.125,
    5: lambda n1: -(1/(1 + n1)),
    6: lambda n1: -(5 + 9*n1)/(12.*(1 + n1)),
    7: lambda n1: 1/(3 + 3*n1),
    8: lambda n1: (1 + 9*n1)/(72 + 72*n1),
    9: lambda n1: 2.25,
    10: lambda n1: -0.75,
    11: lambda n1: 2.25,
    12: lambda n1: (3*(-1 + 3*n1))/(4.*(1 + n1)),
    13: lambda n1: -(1/(1 + n1)),
    14: lambda n1: (7 - 9*n1)/(12 + 12*n1),
    15: lambda n1: (-7 + 9*n1)/(4.*(1 + n1)),
    16: lambda n1: -9/(4 + 4*n1),
    17: lambda n1: (9 + 18*n1)/(4 + 4*n1),
    18: lambda n1: 3/(4 + 4*n1),
    19: lambda n1: (-3*(3 + n1))/(4.*(1 + n1)),
    20: lambda n1: (9*n1)/(4 + 4*n1),
    21: lambda n1: (3*(-5 + 3*n1))/(8.*(1 + n1)),
    22: lambda n1: -9/(4 + 4*n1),
    23: lambda n1: (9*n1)/(4 + 4*n1),
}

M22ta = {
    0: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + n1*(-1 + (13 - 6*n1)*n1) - n2 + 2*n1*(-3 + 2*n1)*(-9 + n1*(3 + 7*n1))*n2 + (13 + 2*n1*(-27 + 14*(-1 + n1)*n1))*n2**2 + 2*(-3 + n1*(-15 + 14*n1))*n2**3 + 28*n1*n2**4))/(14.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    1: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-22 + 7*n1**2*(5 + 7*n2) + n2*(16 + 35*n2) + n1*(16 + 7*n2*(6 + 7*n2))))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    2: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-4 + 7*n1 + 7*n2))/(7.*n1*n2),
    3: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(58 + 98*n1**3*n2 + (3 - 91*n2)*n2 + 7*n1**2*(-13 - 2*n2 + 28*n2**2) + n1*(3 + 2*n2*(-73 + 7*n2*(-1 + 7*n2)))))/(294.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    4: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(58 + 98*n1**3*n2 + (3 - 91*n2)*n2 + 7*n1**2*(-13 - 2*n2 + 28*n2**2) + n1*(3 + 2*n2*(-73 + 7*n2*(-1 + 7*n2)))))/(196.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    5: lambda n1, n2: (6 + n1*(1 + 2*n1)*(-7 + n1 + 2*n1**2) - 7*n2 + 2*n1*(-3 + n1*(19 + n1*(-5 + 4*(-3 + n1)*n1)))*n2 + (-13 + 2*n1*(19 + 6*n1 - 4*n1**2))*n2**2 + 2*(2 + n1*(-5 - 4*n1 + 8*n1**2))*n2**3 + 4*(1 - 6*n1)*n2**4 + 8*n1*n2**5)/(4.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    6: lambda n1, n2: (n1**2*(1 - 11*n2) + 2*n1**3*(5 + 7*n2) + (-3 + 2*n2)*(6 + n2*(8 + 5*n2)) + n1*(-12 + n2*(-38 + n2*(-11 + 14*n2))))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    7: lambda n1, n2: (-3 + 2*n1)/n2 + (-3 + 2*n2)/n1,
    8: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(2 + n1 + n2 - 28*n1**4*n2 + n2**2*(-13 + 6*n2) + n1**3*(6 + 30*n2 - 28*n2**2) - 2*n1*n2*(-3 + 2*n2)*(-9 + n2*(3 + 7*n2)) + n1**2*(-13 + 2*n2*(27 - 14*(-1 + n2)*n2))))/(42.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    9: lambda n1, n2: (2*(48 - 2*n1*(1 + 10*n1) - 2*n2 + 3*n1*(17 + 7*n1)*n2 + (-20 + 7*n1*(3 + 7*n1))*n2**2))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    10: lambda n1, n2: (4*(3 - 2*n2 + n1*(-2 + 7*n2)))/(7.*n1*n2),
    11: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-22 + 7*n1**2*(5 + 7*n2) + n2*(16 + 35*n2) + n1*(16 + 7*n2*(6 + 7*n2))))/(147.*n1*(1 + n1)*n2*(1 + n2)),
    12: lambda n1, n2: 2.,
    13: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-4 + 7*n1 + 7*n2))/(21.*n1*n2),
    14: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(58 + 98*n1**3*n2 + (3 - 91*n2)*n2 + 7*n1**2*(-13 - 2*n2 + 28*n2**2) + n1*(3 + 2*n2*(-73 + 7*n2*(-1 + 7*n2)))))/(1764.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    15: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + 3*n2 + 4*n1**4*n2 + (3 - 2*n2)*n2**2 + 2*n1**3*(-1 + n2)*(1 + 2*n2) + n1*(1 + 2*n2)*(3 + 2*(-2 + n2)*n2*(1 + n2)) + n1**2*(3 + 2*n2*(-5 + 2*(-1 + n2)*n2))))/(2.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    16: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(2 + n2*(4 + 5*n2) + n1**2*(5 + 7*n2) + n1*(4 + n2*(10 + 7*n2))))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    17: lambda n1, n2: ((n1 + n2)*(-3 + 2*n1 + 2*n2))/(n1*n2),
    18: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-10 + n1 + n2 - 14*n1**3*n2 + 17*n2**2 - 2*n1*n2*(-11 + n2*(3 + 7*n2)) + n1**2*(17 - 2*n2*(3 + 14*n2))))/(42.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    19: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(10 + 14*n1**3*n2 - n2*(1 + 17*n2) + n1**2*(-17 + 6*n2 + 28*n2**2) + n1*(-1 + 2*n2*(-11 + n2*(3 + 7*n2)))))/(14.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    20: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(28*n1**4*n2 + (-2 + n2)*(5 + n2)*(-1 + 2*n2) + n1**3*(2 - 46*n2 + 28*n2**2) + n1**2*(5 + 2*n2*(-19 + 14*(-1 + n2)*n2)) + n1*(-23 + 2*n2*(47 + n2*(-19 + n2*(-23 + 14*n2))))))/(14.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    21: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-58 + 7*n1**2*(5 + 7*n2) + n2*(4 + 35*n2) + n1*(4 + 7*n2*(2 + 7*n2))))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    22: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-8 + 7*n1 + 7*n2))/(7.*n1*n2),
    23: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(46 + 98*n1**3*n2 + (13 - 63*n2)*n2 + 7*n1**2*(-9 + 2*n2*(-5 + 14*n2)) + n1*(13 + 2*n2*(-69 + 7*n2*(-5 + 7*n2)))))/(294.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    24: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(46 + 98*n1**3*n2 + (13 - 63*n2)*n2 + 7*n1**2*(-9 + 2*n2*(-5 + 14*n2)) + n1*(13 + 2*n2*(-69 + 7*n2*(-5 + 7*n2)))))/(98.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    25: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + (-1 + n1)*n1*(1 + 2*n1) - n2 - 2*n1*(1 + n1)*n2 - (1 + 2*n1)*n2**2 + 2*n2**3))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    26: lambda n1, n2: ((1 + n1 + n2)*(2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    27: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(6 + n1 - 2*n1**2 + n2 - 2*n2**2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    28: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(38 + 112*n1**3*n2 + (41 - 66*n2)*n2 + 2*n1**2*(-33 + 2*n2*(-9 + 28*n2)) + n1*(41 + 4*n2*(-58 + n2*(-9 + 28*n2)))))/(56.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    29: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(9 + 3*n1 + 3*n2 + 7*n1*n2))/(14.*n1*(1 + n1)*n2*(1 + n2)),
    30: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(5 + 5*n1 + 5*n2 + 7*n1*n2))/(14.*n1*(1 + n1)*n2*(1 + n2)),
    31: lambda n1, n2: (3 - 2*n1 - 2*n2)/(2.*n1*n2),
    32: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*n2),
    33: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(6 + 7*n1 + 7*n2))/(168.*n1*(1 + n1)*n2*(1 + n2)),
    34: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(26 + 28*n1**3*n2 - n2*(7 + 48*n2) + 8*n1**2*(-6 + n2*(5 + 7*n2)) + n1*(-7 + 4*n2*(-12 + n2*(10 + 7*n2)))))/(84.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    35: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(34 + n1 + n2 + 56*n1**3*n2 - 54*n2**2 + 2*n1**2*(-27 - 2*n2 + 56*n2**2) + 4*n1*n2*(-21 + n2*(-1 + 14*n2))))/(56.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    36: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(50 + 98*n1**3*n2 - n2*(9 + 35*n2) + 7*n1**2*(-5 + 2*n2*(-9 + 14*n2)) + n1*(-9 + 2*n2*(-33 + 7*n2*(-9 + 7*n2)))))/(196.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    37: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + n1 + 4*n1**3 + n2 - 8*n1**2*n2 + 4*n2**3 - 8*n1*n2*(1 + n2)))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    38: lambda n1, n2: ((2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    39: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(-2 + 7*n1 + 7*n2))/(56.*n1*(1 + n1)*n2*(1 + n2)),
    40: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(26 + 56*n1**3*n2 + (9 - 38*n2)*n2 + 2*n1**2*(-19 + 2*n2*(-9 + 28*n2)) + n1*(9 + 4*n2*(-21 + n2*(-9 + 14*n2)))))/(56.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    41: lambda n1, n2: (3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
    42: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(1 + 2*(n1**2 - 4*n1*n2 + n2**2)))/(16.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    43: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(3 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
}




# Exact time dependence loops
M13e = {
    0: lambda n1: 1.125,
    1: lambda n1: -(1/(1 + n1)),
    2: lambda n1: 2.25,
    3: lambda n1: -(1/(1 + n1)),
    4: lambda n1: 1.125,
    5: lambda n1: 5.25,
    6: lambda n1: 10.5,
    7: lambda n1: 5.25,
    8: lambda n1: 5.25,
    9: lambda n1: -21/(4 + 4*n1),
    10: lambda n1: (21 + 42*n1)/(4 + 4*n1),
    11: lambda n1: -21/(4 + 4*n1),
    12: lambda n1: (21*n1)/(4 + 4*n1),
    13: lambda n1: -21/(1 + n1),
    14: lambda n1: -21/(1 + n1),
}

M22e = {
    0: lambda n1, n2: (6 + n1*(1 + 2*n1)*(-7 + n1 + 2*n1**2) - 7*n2 + 2*n1*(-3 + n1*(19 + n1*(-5 + 4*(-3 + n1)*n1)))*n2 + (-13 + 2*n1*(19 + 6*n1 - 4*n1**2))*n2**2 + 2*(2 + n1*(-5 - 4*n1 + 8*n1**2))*n2**3 + 4*(1 - 6*n1)*n2**4 + 8*n1*n2**5)/(4.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    1: lambda n1, n2: (n1**2*(1 - 11*n2) + 2*n1**3*(5 + 7*n2) + (-3 + 2*n2)*(6 + n2*(8 + 5*n2)) + n1*(-12 + n2*(-38 + n2*(-11 + 14*n2))))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    2: lambda n1, n2: (-3 + 2*n1)/n2 + (-3 + 2*n2)/n1,
    3: lambda n1, n2: (2*(48 - 2*n1*(1 + 10*n1) - 2*n2 + 3*n1*(17 + 7*n1)*n2 + (-20 + 7*n1*(3 + 7*n1))*n2**2))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    4: lambda n1, n2: (4*(3 - 2*n2 + n1*(-2 + 7*n2)))/(7.*n1*n2),
    5: lambda n1, n2: 2,
    6: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + 3*n2 + 4*n1**4*n2 + (3 - 2*n2)*n2**2 + 2*n1**3*(-1 + n2)*(1 + 2*n2) + n1*(1 + 2*n2)*(3 + 2*(-2 + n2)*n2*(1 + n2)) + n1**2*(3 + 2*n2*(-5 + 2*(-1 + n2)*n2))))/(2.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    7: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(2 + n2*(4 + 5*n2) + n1**2*(5 + 7*n2) + n1*(4 + n2*(10 + 7*n2))))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    8: lambda n1, n2: ((n1 + n2)*(-3 + 2*n1 + 2*n2))/(n1*n2),
    9: lambda n1, n2: ((-2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-2 + n1*(3 + 2*n1) + 3*n2 + 2*n1*(-4 + n1*(-1 + 2*n1))*n2 - 2*(-1 + n1)*n2**2 + 4*n1*n2**3))/(2.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    10: lambda n1, n2: ((-2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(8 + 5*n2 + n1*(5 + 7*n2)))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    11: lambda n1, n2: ((-2 + n1 + n2)*(-3 + 2*n1 + 2*n2))/(n1*n2),
    12: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + (-1 + n1)*n1*(1 + 2*n1) - n2 - 2*n1*(1 + n1)*n2 - (1 + 2*n1)*n2**2 + 2*n2**3))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    13: lambda n1, n2: ((1 + n1 + n2)*(2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    14: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(6 + n1 - 2*n1**2 + n2 - 2*n2**2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    15: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + (11 - 6*n1)*n1 + 11*n2 + 4*(-2 + n1)*n1*(5 + 4*n1)*n2 + 2*(-3 - 6*n1 + 8*n1**2)*n2**2 + 16*n1*n2**3))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    16: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(9 + 3*n1 + 3*n2 + 7*n1*n2))/(14.*n1*(1 + n1)*n2*(1 + n2)),
    17: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(5 + 5*n1 + 5*n2 + 7*n1*n2))/(14.*n1*(1 + n1)*n2*(1 + n2)),
    18: lambda n1, n2: (3 - 2*n1 - 2*n2)/(2.*n1*n2),
    19: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*n2),
    20: lambda n1, n2: ((-2 + n1 + n2)*(-1 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1*n2))/(4.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    21: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + n1 + 4*n1**3 + n2 - 8*n1**2*n2 + 4*n2**3 - 8*n1*n2*(1 + n2)))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    22: lambda n1, n2: ((2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    23: lambda n1, n2: -((-2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(8.*n1*(1 + n1)*n2*(1 + n2)),
    24: lambda n1, n2: ((-2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(-1 + 4*n1*n2))/(8.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    25: lambda n1, n2: (3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
    26: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(1 + 2*(n1**2 - 4*n1*n2 + n2**2)))/(16.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    27: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(3 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
    28: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(n1*(-1 + 2*n1) + (-2 + n2)*(3 + 2*n2)))/(2.*n1*(1 + n1)*n2*(1 + n2)),
    29: lambda n1, n2: (2*(-3 + 2*n1 + 2*n2)*(9 + 3*n1 + 3*n2 + 7*n1*n2))/(7.*n1*(1 + n1)*n2*(1 + n2)),
    30: lambda n1, n2: (-6 + 4*n1 + 4*n2)/(n1*n2),
    31: lambda n1, n2: ((2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*(1 + n1)*n2*(1 + n2)),
    32: lambda n1, n2: ((-2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*(1 + n1)*n2*(1 + n2)),
    33: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*(1 + n1)*n2*(1 + n2)),
    34: lambda n1, n2: ((1 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(2.*n1*(1 + n1)*n2*(1 + n2)),
    35: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(n1*(1 + n1)*n2*(1 + n2)),
}

# galaxy matter
M13gm = {
    0: lambda n1: (5 + 9*n1)/(8 + 8*n1),
    1: lambda n1: -(1/(2 + 2*n1)),
    2: lambda n1: (3*(5 + 9*n1))/(8.*(1 + n1)),
    3: lambda n1: -(1/(2 + 2*n1)),
    4: lambda n1: (-7 + 9*n1)/(8.*(1 + n1)),
    5: lambda n1: -9/(8 + 8*n1),
    6: lambda n1: (9 + 18*n1)/(8 + 8*n1),
    7: lambda n1: -9/(8 + 8*n1),
    8: lambda n1: (3*(-2 + 9*n1))/(8.*(1 + n1)),
    9: lambda n1: -9/(4 + 4*n1),
    10: lambda n1: (9*n1)/(4 + 4*n1),
}

M22gm = {
    0: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + n1*(-1 + (13 - 6*n1)*n1) - n2 + 2*n1*(-3 + 2*n1)*(-9 + n1*(3 + 7*n1))*n2 + (13 + 2*n1*(-27 + 14*(-1 + n1)*n1))*n2**2 + 2*(-3 + n1*(-15 + 14*n1))*n2**3 + 28*n1*n2**4))/(28.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    1: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-22 + 7*n1**2*(5 + 7*n2) + n2*(16 + 35*n2) + n1*(16 + 7*n2*(6 + 7*n2))))/(98.*n1*(1 + n1)*n2*(1 + n2)),
    2: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-4 + 7*n1 + 7*n2))/(14.*n1*n2),
    3: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-14 + n1*(19 + (41 - 46*n1)*n1) + 19*n2 + 2*n1*(63 + n1*(-96 + n1*(-31 + 42*n1)))*n2 + (41 + 4*n1*(-48 + 5*n1*(-3 + 7*n1)))*n2**2 + 2*(-23 + n1*(-31 + 70*n1))*n2**3 + 84*n1*n2**4))/(28.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    4: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-22 + 7*n1**2*(5 + 7*n2) + n2*(16 + 35*n2) + n1*(16 + 7*n2*(6 + 7*n2))))/(49.*n1*(1 + n1)*n2*(1 + n2)),
    5: lambda n1, n2: ((-3 + 2*n1 + 2*  n2)*(-4 + 7*n1 + 7*n2))/(7.*n1*n2),
    6: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(46 + 98*n1**3*n2 + (13 - 63*n2)*n2 + 7*n1**2*(-9 + 2*n2*(-5 + 14*n2)) + n1*(13 + 2*n2*(-69 + 7*n2*(-5 + 7*n2)))))/(196.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    7: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(2 + n1*(1 + 2*n1)*(-1 + 4*(-1 + n1)*n1) - n2 - 8*n1*(-2 + n1**2)*n2 - 2*(3 + 8*n1**2)*n2**2 - 4*(1 + 2*n1)*n2**3 + 8*n2**4))/(16.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    8: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(66 + 28*n1**3*(-1 + 6*n2) + 8*n1**2*(-17 + 6*n2 + 28*n2**2) + n2*(27 - 4*n2*(34 + 7*n2)) + n1*(27 + 4*n2*(-65 + 6*n2*(2 + 7*n2)))))/(112.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    9: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(9 + 3*n1 + 3*n2 + 7*n1*n2))/(28.*n1*(1 + n1)*n2*(1 + n2)),
    10: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(5 + 5*n1 + 5*n2 + 7*n1*n2))/(28.*n1*(1 + n1)*n2*(1 + n2)),
    11: lambda n1, n2: (3 - 2*n1 - 2*n2)/(4.*n1*n2),
    12: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(4.*n1*n2),
    13: lambda n1, n2: -((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(6 + 7*n1 + 7*n2))/(112.*n1*(1 + n1)*n2*(1 + n2)),
    14: lambda n1, n2: (3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(202 + 392*n1**3*n2 + (37 - 294*n2)*n2 + 98*n1**2*(1 + 2*n2)*(-3 + 4*n2) + n1*(37 + 4*n2*(-141 + 49*n2*(-1 + 2*n2)))))/(784.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    15: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(2 + n1 + 4*n1**3 + n2 - 8*n1**2*n2 + 4*n2**3 - 8*n1*n2*(1 + n2)))/(16.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    16: lambda n1, n2: ((2 + n1 + n2)*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2))/(16.*n1*(1 + n1)*n2*(1 + n2)),
    17: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(18 + 28*n1**3 + 28*n1**2*(1 - 4*n2) + n1*(-15 + 16*(1 - 7*n2)*n2) + n2*(-15 + 28*n2*(1 + n2))))/(112.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    18: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(66 + 28*n1**3*(-1 + 6*n2) + 4*n1**2*(-33 - 4*n2 + 84*n2**2) + n2*(25 - 4*n2*(33 + 7*n2)) + n1*(25 + 8*n2*(-28 + n2*(-2 + 21*n2)))))/(112.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    19: lambda n1, n2: (3*(-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
    20: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(1 + 2*(n1**2 - 4*n1*n2 + n2**2)))/(16.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    21: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(1 + 2*n1 + 2*n2)*(3 + 2*n1 + 2*n2))/(32.*n1*(1 + n1)*n2*(1 + n2)),
}