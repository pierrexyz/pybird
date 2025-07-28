from pybird.module import *
# from pybird.fftlog import FFTLog, MPC, CoefWindow
from fftlog.fftlog import FFTLog
from fftlog.utils import CoefWindow
from fftlog.sbt import MPC
from pybird.common import co
from pybird.resumfactor import Qa, Qawithhex, Qawithhex20

class Resum(object):
    """A class for IR-resummation of the power spectrum and correlation function.
    
    The Resum class performs the IR-resummation of the power spectrum by computing
    the bulk flow effects on the BAO feature. Two resummation schemes are available:
    
    1. fullresum (default): FFTLog transforms are performed on the full integrands
       from s = 0.1 to s = 10000 Mpc/h. This is the more complete treatment.
    
    2. optiresum: FFTLog transforms are performed only on the extracted BAO peak
       (after removing the broadband). The peak is padded with zeros and transforms
       run from s = 0.1 to s = 1000 Mpc/h. This is computationally faster.
    
    Attributes:
        co (Common): Common parameters shared across calculations.
        LambdaIR (float): Integral cutoff for IR-filters X and Y.
            Default is 0.2 for fullresum, but 1.0 works for either scheme.
        
        # Correlation function arrays
        sr (ndarray): Separation array for resummation.
        sbao (ndarray): Separation array for BAO peak (optiresum).
        snobao (ndarray): Separation array for broadband (optiresum).
        
        # Power spectrum arrays
        kr (ndarray): Wavenumber array for resummation.
        k2p (ndarray): Powers of k^2 for IR corrections.
        
        # FFTLog settings and objects
        fftsettings (dict): Settings for IR correction FFTLog.
        fft (FFTLog): FFTLog engine for IR corrections.
        M (ndarray): Spherical Bessel transform matrices for IR corrections.
        kPow (ndarray): k-dependent power terms for IR corrections.
        
        Xfftsettings (dict): Settings for IR-filter FFTLog.
        Xfft (FFTLog): FFTLog engine for IR-filters.
        XM (ndarray): Spherical Bessel transform matrices for IR-filters.
        XsPow (ndarray): s-dependent power terms for IR-filters.
        
        Cfftsettings (dict): Settings for correlation function FFTLog.
        Cfft (FFTLog): FFTLog engine for correlation functions.
        Ml (ndarray): Spherical Bessel transform matrices for correlation functions.
        sPow (ndarray): s-dependent power terms for correlation functions.
        
        # Damping windows
        dampPs (ndarray): Damping windows for power spectrum.
        dampCf (ndarray): Damping window for correlation function.
        
        # Optimization paths
        optipath_cf (list): Optimized einsum path for correlation function.
        optipath_XpYpC (list): Optimized path for XpYp * correlation function.
        optipath_IRPs (list): Optimized path for IR power spectrum.
        optipath_k2IRPs (list): Optimized path for k^2 * IR power spectrum.
    
    Methods:
        setXsPow(): Compute s-dependent power terms for IR-filters.
        setXM(): Compute transform matrices for IR-filters.
        IRFilters(): Compute the IR-filters X and Y.
        setkPow(): Compute k-dependent power terms for IR corrections.
        setM(): Compute transform matrices for IR corrections.
        
        IRn(): Compute spherical Bessel transform for IR correction order n.
        extractBAO(): Extract BAO feature from correlation function.
        setXpYp(): Compute powers of IR-filters X and Y.
        makeQ(): Compute bulk flow coefficients Q.
        
        setMl(): Compute transform matrices for correlation functions.
        setsPow(): Compute s-dependent power terms for correlation functions.
        Ps2Cf(): Transform power spectrum to correlation function.
        
        IRCf(): Compute IR corrections in configuration space.
        IRPs(): Compute IR corrections in Fourier space.
        PsCf(): Compute both power spectrum and correlation function with IR-resummation.
        Ps(): Compute power spectrum with IR-resummation.
    """

    def __init__(self, LambdaIR=.1, NFFT=192, fft=False, co=co):
        """Initialize the Resum engine for IR-resummation calculations.
        
        Parameters
        ----------
        LambdaIR : float, optional
            Integral cutoff for IR-filters X and Y in Mpc/h, by default 0.1
        NFFT : int, optional
            Number of FFT points for transforms, by default 192
        fft : bool, optional
            Whether to use O(NFFT logNFFT) FFT instead of O(NFFT x Nk) sum, by default False
        co : Common, optional
            Common parameters object, by default co
            
        Notes
        -----
        The choice of LambdaIR depends on the resummation scheme:
        - fullresum: LambdaIR = 0.2 (default for complete treatment)
        - optiresum: LambdaIR = 1.0 (faster, works for either scheme)
        """

        self.co = co
        self.LambdaIR = LambdaIR

        self.is_fft = fft # spherical bessel transform using O(NFFT logNFFT)-FFT instead of O(NFFT x Nk)-sum; not neccessarily faster because of the large-dimension arrays involved here

        if self.co.optiresum:
            self.sLow = 70.
            self.sHigh = 190.
            self.idlow = where(self.co.s > self.sLow)[0][0]
            self.idhigh = where(self.co.s > self.sHigh)[0][0]
            self.sbao = self.co.s[self.idlow:self.idhigh]
            self.snobao = concatenate([self.co.s[:self.idlow], self.co.s[self.idhigh:]])
            self.sr = self.sbao
        else:
            self.sr = self.co.s

        self.klow = 0.02
        self.kr = self.co.k[self.klow <= self.co.k]
        self.Nkr = self.kr.shape[0]
        self.Nlow = where(self.klow <= self.co.k)[0][0]
        k2pi = array([self.kr**(2*(p+1)) for p in range(self.co.NIR)])
        self.k2p = concatenate((k2pi, k2pi))

        self.fftsettings = dict(Nmax=NFFT, xmin=.1, xmax=10000., bias=-0.6, window=0.2)
        self.fft = FFTLog(**self.fftsettings)
        self.setM()
        self.setkPow()

        self.Xfftsettings = dict(Nmax=32, xmin=1.5e-5, xmax=10., bias=-2.6, window=0.2)
        self.Xfft = FFTLog(**self.Xfftsettings)
        self.setXM()
        self.setXsPow()

        self.Cfftsettings = dict(Nmax=256, xmin=1.e-3, xmax=10., bias=-0.6, window=0.2)
        self.Cfft = FFTLog(**self.Cfftsettings)
        self.setMl()
        self.setsPow()

        #self.damping = CoefWindow(self.co.Nk-1, window=.2, left=False, right=True)
        self.kl2 = self.co.k[self.co.k < 0.5]
        Nkl2 = len(self.kl2)

        self.kl4 = self.co.k[self.co.k < 0.4]
        Nkl4 = len(self.kl4)

        self.dampPs = array([
            CoefWindow(self.co.Nk-1, window=.25, left=False, right=True),
            pad(CoefWindow(Nkl2-1, window=.25, left=False, right=True), (0,self.co.Nk-Nkl2), mode='constant'),
            pad(CoefWindow(Nkl4-1, window=.25, left=False, right=True), (0,self.co.Nk-Nkl4), mode='constant')
            ])

        self.scut = self.co.s[self.co.s < 70.]
        self.dampCf = pad(CoefWindow(self.co.Ns-len(self.scut)-1, window=.25, left=True, right=True), (len(self.scut),0), mode='constant')

        self.sign_ellp = array([real((-1j)**(2*l)) for l in range(self.co.Nl)]) # for eq. (8) of 2003.07956

        ### To speed-up matrix multiplication:
        def create_einsum_path(N):
            fake_cf = empty(shape=(N, self.co.Nl, self.sr.shape[0]))
            fake_XpYp = empty(shape=(2*self.co.NIR, self.sr.shape[0]))
            optipath_cf = einsum_path('l,jls->jls', self.sign_ellp, fake_cf, optimize='optimal')[0]
            optipath_XpYpC = einsum_path('ns,jls->jlns', fake_XpYp, fake_cf, optimize='optimal')[0]
            fake_XpYpC = einsum('ns,jls->jlns', fake_XpYp, fake_cf, optimize=optipath_XpYpC)
            if self.is_fft:
                u_m = self.fft.Coef(self.sr, fake_XpYpC, to_sum=False, extrap='padding')
                optipath_IRPs = einsum_path('...m,lm->...lm', u_m, self.M, optimize='optimal')[0]
                fake_IRPs = real(self.fft.sbt(self.sr, fake_XpYpC, y=self.kr, kernel=self.M, sum_ell=False, einsum_optimize=optipath_IRPs, extrap='padding')) 
            else: 
                Coef = self.fft.Coef(self.sr, fake_XpYpC, extrap='padding')
                optipath_IRPs = einsum_path('jlnm,mk,am->jlnak', Coef, self.kPow, self.M, optimize='optimal')[0]
                fake_IRPs = einsum('jlnm,mk,am->jlnak', Coef, self.kPow, self.M, optimize=optipath_IRPs)
            optipath_k2IRPs = einsum_path('nk,jlnak->jlnak', self.k2p, fake_IRPs, optimize='optimal')[0]
            return optipath_cf, optipath_XpYpC, optipath_IRPs, optipath_k2IRPs

        self.optipath_cf, self.optipath_XpYpC, self.optipath_IRPs, self._optipath_k2IRPs = create_einsum_path(2) # with_bias = True
        self._optipath_cf, self._optipath_XpYpC, self._optipath_IRPs, self._optipath_k2IRPs = create_einsum_path(14) # with_bias = False

    def setXsPow(self):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the IR-filters X and Y. """
        self.XsPow = exp(einsum('n,s->ns', -self.Xfft.Pow - 3., log(self.sr)))

    def setXM(self):
        """ Compute the matrices to evaluate the IR-filters X and Y. Called at instantiation. """
        # self.XM = empty(shape=(2, self.Xfft.Pow.shape[0]), dtype='complex')
        # for l in range(2): self.XM[l] = MPC(2 * l, -0.5 * self.Xfft.Pow)
        self.XM = array([MPC(2 * l, -0.5 * self.Xfft.Pow) for l in range(2)])

    def IRFilters(self, bird, soffset=1., LambdaIR=None, RescaleIR=1.):
        """ Compute the IR-filters X and Y. """
        if LambdaIR is None: LambdaIR = self.LambdaIR
        if self.co.exact_time and self.co.quintessence: Pin = bird.G1**2 * bird.Pin
        else: Pin = bird.Pin
        Coef = self.Xfft.Coef(bird.kin, Pin * exp(-bird.kin**2 / LambdaIR**2) / bird.kin**2, extrap='padding')
        CoefsPow = einsum('n,ns->ns', Coef, self.XsPow)
        X02 = real(einsum('ns,ln->ls', CoefsPow, self.XM))
        X0offset = real(einsum('n,n->', einsum('n,n->n', Coef, soffset**(-self.Xfft.Pow - 3.)), self.XM[0]))
        if is_jax: X02 = X02.at[0].set(X0offset - X02[0])
        else: X02[0] = X0offset - X02[0]
        # if self.co.nonequaltime:
        #     X = RescaleIR * 2/3. * bird.D1*bird.D2/bird.D**2 * (X02[0] - X02[1]) + 1/3. * (bird.D1-bird.D2)**2/bird.D**2 * X0offset
        #     Y = 2. * bird.D1*bird.D2/bird.D**2 * X02[1]
        # else:
        X = RescaleIR * 2. / 3. * (X02[0] - X02[1])
        Y = 2. * X02[1]
        return X, Y

    def setkPow(self):
        """ Multiply the coefficients with the k's to the powers of the FFTLog to evaluate the IR-corrections. """
        self.kPow = exp(einsum('n,s->ns', -self.fft.Pow - 3., log(self.kr)))

    def setM(self, Nl=3):
        """ Compute the matrices to evaluate the IR-corrections. Called at instantiation. """
        # self.M = empty(shape=(Nl, self.fft.Pow.shape[0]), dtype='complex')
        # for l in range(Nl): self.M[l] = 8.*pi**3 * MPC(2 * l, -0.5 * self.fft.Pow)
        self.M = array([8.*pi**3 * MPC(2 * l, -0.5 * self.fft.Pow) for l in range(Nl)])
        

    def IRn(self, XpYpC):
        """ Compute the spherical Bessel transform in the IR correction of order n given [XY]^n """
        Coef = self.fft.Coef(self.sr, XpYpC, extrap='padding')
        CoefkPow = einsum('n,nk->nk', Coef, self.kPow)
        return real(einsum('nk,ln->lk', CoefkPow, self.M[:self.co.Na]))

    def extractBAO(self, cf):
        """ Given a correlation function cf,
            - if fullresum, return cf
            - if optiresum, extract the BAO peak """
        if self.co.optiresum:
            cfnobao = concatenate([cf[..., :self.idlow], cf[..., self.idhigh:]], axis=-1)
            nobao = interp1d(self.snobao, self.snobao**2 * cfnobao, kind='linear', axis=-1)(self.sbao) * self.sbao**-2
            bao = cf[..., self.idlow:self.idhigh] - nobao
            return bao
        else:
            return cf

    def setXpYp(self, bird):
        X, Y = self.IRFilters(bird)
        Xp = array([X**(p+1) for p in range(self.co.NIR)])
        XpY = array([Y * X**p for p in range(self.co.NIR)])
        XpYp = concatenate((Xp, XpY))
        #return array([item for pair in zip(Xp, XpY + [0]) for item in pair])
        return XpYp

    def makeQ(self, f):
        """ Compute the bulk coefficients Q^{ll'}_{||N-j}(n, \alpha, f) """
        # Q = empty(shape=(2, self.co.Nl, self.co.Nl, self.co.Nn))
        # for a in range(2):
        #     for l in range(self.co.Nl):
        #         for lpr in range(self.co.Nl):
        #             for u in range(self.co.Nn):
        #                 if self.co.NIR == 8: Q[a][l][lpr][u] = Qa[1 - a][2 * l][2 * lpr][u](f)
        #                 elif self.co.NIR == 16: Q[a][l][lpr][u] = Qawithhex[1 - a][2 * l][2 * lpr][u](f)
        #                 elif self.co.NIR == 20: Q[a][l][lpr][u] = Qawithhex20[1 - a][2 * l][2 * lpr][u](f)
        # return Q
        if self.co.NIR == 8: Q_ = Qa
        elif self.co.NIR == 16: Q_ = Qawithhex
        elif self.co.NIR == 20: Q_ = Qawithhex20
        Q = array([[[[Q_[1 - a][2 * l][2 * lpr][u](f) for u in range(self.co.Nn)] for lpr in range(self.co.Nl)] for l in range(self.co.Nl)] for a in range(2)])
        return Q

    def setMl(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation. """
        # self.Ml = empty(shape=(self.co.Nl, self.Cfft.Pow.shape[0]), dtype='complex')
        # for l in range(self.co.Nl):
        #     self.Ml[l] = 1j**(2*l) * MPC(2 * l, -0.5 * self.Cfft.Pow)
        self.Ml = array([(-1j)**(2*l) * MPC(2 * l, -0.5 * self.Cfft.Pow) for l in range(self.co.Nl)])

    def setsPow(self):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the IR corrections in configuration space. """
        self.sPow = exp(einsum('n,s->ns', -self.Cfft.Pow - 3., log(self.co.s)))

    def Ps2Cf(self, P, l=0):
        Coef = self.Cfft.Coef(self.co.k, P * self.dampPs[l], extrap='padding')
        CoefsPow = einsum('n,ns->ns', Coef, self.sPow)
        return real(einsum('ns,n->s', CoefsPow, self.Ml[l])) * self.dampCf

    def IRCf(self, bird):
        """ Compute the IR corrections in configuration space by spherical Bessel transforming the IR corrections in Fourier space.  """

        if bird.with_bias:
            for a, IRa in enumerate(bird.fullIRPs): # this can be speedup x2 by doing FFTLog[lin+loop] instead of separately
                for l, IRal in enumerate(IRa):
                    bird.fullIRCf[a,l] = self.Ps2Cf(IRal, l=l)
        else:
            for l, IRl in enumerate(bird.fullIRPs11):
                for j, IRlj in enumerate(IRl):
                    bird.fullIRCf11[l,j] = self.Ps2Cf(IRlj, l=l)
            for l, IRl in enumerate(bird.fullIRPsct):
                for j, IRlj in enumerate(IRl):
                    bird.fullIRCfct[l,j] = self.Ps2Cf(IRlj, l=l)
            for l, IRl in enumerate(bird.fullIRPsloop):
                for j, IRlj in enumerate(IRl):
                    bird.fullIRCfloop[l,j] = self.Ps2Cf(IRlj, l=l)

    def PsCf(self, bird, makeIR=True, makeQ=True, setIR=True, setPs=True, setCf=True):
        """Perform IR-resummation for both power spectrum and correlation function.
        
        This is the main method that applies IR-resummation to both the power spectrum
        and correlation function, handling the bulk flow effects on the BAO feature.

        Parameters
        ----------
        bird : Bird
            Bird object containing the power spectrum and correlation function
        makeIR : bool, optional
            Whether to compute IR-resummed power spectrum, by default True
        makeQ : bool, optional
            Whether to compute the Q factors, by default True
        setIR : bool, optional
            Whether to set IR-resummed components, by default True
        setPs : bool, optional
            Whether to set power spectrum components, by default True
        setCf : bool, optional
            Whether to set correlation function components, by default True
            
        Notes
        -----
        This method orchestrates the complete IR-resummation procedure, including
        computing Q factors, IR filters, and applying resummation to both Fourier
        and configuration space quantities.
        """

        self.Ps(bird, makeIR=makeIR, makeQ=makeQ, setIR=setIR, setPs=setPs)
        if setCf:
            self.IRCf(bird)
            bird.setresumCf()

    def Ps(self, bird, makeIR=True, makeQ=True, setIR=True, setPs=True):
        """Apply IR-resummation to the power spectrum.
        
        Computes the IR-resummed power spectrum by applying bulk flow corrections
        to the BAO feature using the IR-resummation formalism.

        Parameters
        ----------
        bird : Bird
            Bird object containing the power spectrum to be resummed
        makeIR : bool, optional
            Whether to compute IR-resummed components, by default True
        makeQ : bool, optional
            Whether to compute Q factors for resummation, by default True
        setIR : bool, optional
            Whether to set IR components in the bird object, by default True
        setPs : bool, optional
            Whether to set final power spectrum components, by default True
            
        Notes
        -----
        The IR-resummation process involves:
        1. Computing Q factors from the growth rate
        2. Computing IR-resummed power spectrum components
        3. Setting the resummed components in the bird object
        """

        if makeIR: self.IRPs(bird)
        if makeQ: bird.Q = self.makeQ(bird.f) #can edit Q here 
        if setIR: bird.setIRPs()
        if setPs: bird.setresumPs()

    def getIRPs(self, cf, XpYp, optipath_cf=False, optipath_XpYpC=False, optipath_IRPs=False, optipath_k2IRPs=False):
        # for notation, see eq. (8) of 2003.07956
        cf = einsum('l,jls->jls', self.sign_ellp, self.extractBAO(cf), optimize=optipath_cf)
        XpYpC = einsum('ns,jls->jlns', XpYp, cf, optimize=optipath_XpYpC) #j: (0: lin, 1: loop); l': multipoles; n: IR-exp; s: distance seperation
        if self.is_fft:
            IRPs = real(self.fft.sbt(self.sr, XpYpC, y=self.kr, kernel=self.M, sum_ell=False, einsum_optimize=optipath_IRPs, extrap='padding')) 
        else:
            Coef = self.fft.Coef(self.sr, XpYpC, extrap='padding')
            IRPs = real(einsum('jlnm,mk,am->jlnak', Coef, self.kPow, self.M, optimize=optipath_IRPs)) # a: order of spherical-bessel j_a
        IRPs = einsum('nk,jlnak->jlnak', self.k2p, IRPs, optimize=optipath_k2IRPs)
        ### IRPs = real(einsum('nk,jlnm,mk,am->jlnak', self.k2p, Coef, self.kPow, self.M, optimize=optipath_IRPs)) 
        IRPs = IRPs.reshape(IRPs.shape[0], IRPs.shape[1], IRPs.shape[2] * IRPs.shape[3], IRPs.shape[4])
        IRPs = pad(IRPs, ((0,0), (0,0), (0,0), (self.Nlow,0)), mode='constant', constant_values=0)
        return IRPs


    def IRPs(self, bird):
        """ This is the main method of the class. Compute the IR corrections in Fourier space. """

        XpYp = self.setXpYp(bird)

        if bird.with_bias:
            bird.IRPs = self.getIRPs(bird.Cf[:2], XpYp, optipath_cf=self.optipath_cf, optipath_XpYpC=self.optipath_XpYpC, optipath_IRPs=self.optipath_IRPs, optipath_k2IRPs=self._optipath_k2IRPs)
        else:
            IRPs = self.getIRPs(concatenate((array([bird.C11]), array([bird.Cct]), swapaxes(bird.Cloopl, axis1=0, axis2=1)), axis=0), XpYp, 
                optipath_cf=self._optipath_cf, optipath_XpYpC=self._optipath_XpYpC, optipath_IRPs=self._optipath_IRPs, optipath_k2IRPs=self._optipath_k2IRPs)
            bird.IRPs11, bird.IRPsct, bird.IRPsloop = IRPs[0], IRPs[1], swapaxes(IRPs[2:], axis1=0, axis2=1)
