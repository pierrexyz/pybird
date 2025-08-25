from pybird.module import *

from pybird.common import co, mu
from pybird.greenfunction import GreenFunction

class Bird(object):
    """Main class for computing power spectra and correlation functions for biased tracers.
    
    The Bird (Biased tracers in redshift space) class computes the power spectrum and 
    correlation function for biased tracers, given a cosmology and a set of EFT parameters.
    It implements perturbation theory calculations decomposed by multipoles and bias 
    parameters, allowing for flexible assembly based on the chosen EFT basis.
    
    Attributes:
        co (Common): Common parameters shared across calculations.
        with_bias (bool): Whether to include bias parameters in the calculation.
        eft_basis (str): Basis for the EFT parameters ('eftoflss', 'westcoast', 'eastcoast').
        with_stoch (bool): Whether to include stochastic terms.
        with_nnlo_counterterm (bool): Whether to include NNLO counterterms.
        with_tidal_alignments (bool): Whether to include tidal alignment terms.
        
        kin (ndarray): k-array on which the input linear power spectrum is evaluated.
        Pin (ndarray): Input linear power spectrum.
        Plin (callable): Interpolated function of the linear power spectrum.
        P11 (ndarray): Linear power spectrum evaluated on internal k-array.
        
        P22 (ndarray): Power spectrum 22-loop terms.
        P13 (ndarray): Power spectrum 13-loop terms.
        Ps (ndarray): Power spectrum multipoles (linear, 1-loop, NNLO).
        
        C11 (ndarray): Correlation function multipole linear terms.
        C22l (ndarray): Correlation function multipole 22-loop terms.
        C13l (ndarray): Correlation function multipole 13-loop terms.
        Cct (ndarray): Correlation function multipole counter terms.
        Cf (ndarray): Correlation function multipoles (linear, 1-loop, NNLO).
        
        fullPs (ndarray): Full power spectrum multipoles (linear + loop).
        fullCf (ndarray): Full correlation function multipoles (linear + loop).
        
        b11 (ndarray): EFT parameters for linear terms per multipole.
        b13 (ndarray): EFT parameters for 13-loop terms per multipole.
        b22 (ndarray): EFT parameters for 22-loop terms per multipole.
        bct (ndarray): EFT parameters for counter terms per multipole.
        
        f (float): Growth rate (for redshift space distortion).
        DA (float): Angular distance (for AP effect).
        H (float): Hubble parameter (for AP effect).
        z (float): Redshift.
    
    Methods:
        setcosmo(): Set cosmological parameters and compute the linear power spectrum.
        setBias(): Set EFT parameters for different terms and multipoles.
        setPs(): Set power spectrum multipoles with provided bias parameters.
        setCf(): Set correlation function multipoles with provided bias parameters.
        setPsCf(): Set both power spectrum and correlation function multipoles.
        setfullPs(): Combine linear and loop parts for the full power spectrum.
        setfullCf(): Combine linear and loop parts for the full correlation function.
        setPsCfl(): Create multipoles for each term weighted by bias parameters.
        reducePsCfl(): Regroup terms that share the same EFT parameters.
    """

    def __init__(self, cosmology=None, with_bias=True, eft_basis='eftoflss', with_stoch=False, with_nnlo_counterterm=False, co=co, which='full'):

        self.co = co
        self.with_bias = with_bias
        self.eft_basis = eft_basis
        self.with_stoch = with_stoch
        self.with_nnlo_counterterm = with_nnlo_counterterm
        self.with_tidal_alignments = self.co.with_tidal_alignments

        if cosmology is not None: self.setcosmo(cosmology)


        self.P22 = empty(shape=(self.co.N22, self.co.Nk))
        self.P13 = empty(shape=(self.co.N13, self.co.Nk))
        self.Ps = zeros(shape=(3, self.co.Nl, self.co.Nk)) # 3: linear, 1-loop, NNLO

        self.C11 = empty(shape=(self.co.Nl, self.co.Ns))
        self.C22l = empty(shape=(self.co.Nl, self.co.N22, self.co.Ns))
        self.C13l = empty(shape=(self.co.Nl, self.co.N13, self.co.Ns))
        self.Cct = empty(shape=(self.co.Nl, self.co.Ns))
        self.Cf = zeros(shape=(3, self.co.Nl, self.co.Ns)) # 3: linear, 1-loop, NNLO

        if not with_bias:
            self.Ploopl = empty(shape=(self.co.Nl, self.co.Nloop, self.co.Nk))
            self.P11l = empty(shape=(self.co.Nl, self.co.N11, self.co.Nk))
            self.Pctl = empty(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
            self.P22l = empty(shape=(self.co.Nl, self.co.N22, self.co.Nk))
            self.P13l = empty(shape=(self.co.Nl, self.co.N13, self.co.Nk))

            self.Cloopl = empty(shape=(self.co.Nl, self.co.Nloop, self.co.Ns))
            self.C11l = empty(shape=(self.co.Nl, self.co.N11, self.co.Ns))
            self.Cctl = empty(shape=(self.co.Nl, self.co.Nct, self.co.Ns))

            ###
            self.IRPs11 = zeros(shape=(self.co.Nl, self.co.Nn, self.co.Nk))
            self.IRPsct = zeros(shape=(self.co.Nl, self.co.Nn, self.co.Nk))
            self.IRPsloop = zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nn, self.co.Nk))

            self.IRCf11 = zeros(shape=(self.co.Nl, self.co.N11, self.co.Nn, self.co.Ns))
            self.IRCfct = zeros(shape=(self.co.Nl, self.co.Nct, self.co.Nn, self.co.Ns))
            self.IRCfloop = zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nn, self.co.Ns))

            self.fullIRPs11 = zeros(shape=(self.co.Nl, self.co.N11, self.co.Nk))
            self.fullIRPsct = zeros(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
            self.fullIRPsloop = zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nk))

            self.fullIRCf11 = zeros(shape=(self.co.Nl, self.co.N11, self.co.Ns))
            self.fullIRCfct = zeros(shape=(self.co.Nl, self.co.Nct, self.co.Ns))
            self.fullIRCfloop = zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Ns))

        self.IRPs = zeros(shape=(2, self.co.Nl, self.co.Nn, self.co.Nk))
        self.IRCf = zeros(shape=(2, self.co.Nl, self.co.Nn, self.co.Ns))

        self.fullIRPs = empty(shape=(2, self.co.Nl, self.co.Nk))
        self.fullIRCf = empty(shape=(2, self.co.Nl, self.co.Ns))

        if self.with_bias:
            self.b11 = empty(shape=(self.co.Nl))
            self.b13 = empty(shape=(self.co.Nl, self.co.N13))
            self.b22 = empty(shape=(self.co.Nl, self.co.N22))
            self.bct = empty(shape=(self.co.Nl))
        else:
            self.b11 = empty(shape=(self.co.N11))
            self.bct = empty(shape=(self.co.Nct))
            self.bloop = empty(shape=(self.co.Nloop))

        if self.with_stoch:
            # if self.co.with_cf: # no stochastic term for cf in general ; below is the stochastic terms from a Pade expansion of the Fourier-space stochastic terms
            #     self.bst = zeros(shape=(self.co.Nst))
            #     self.Cstl = zeros(shape=(self.co.Nl, self.co.Nst, self.co.Ns))
            #     self.Cstl[0,0] = exp(-self.co.km * self.co.s) * self.co.km**2 / (4.*pi*self.co.s) / self.co.nd
            #     self.Cstl[0,1] = -self.co.km**2*exp(-self.co.km * self.co.s) / (4.*pi*self.co.s**2) / self.co.nd
            #     self.Cstl[0,2] = exp(-self.co.km * self.co.s) * (3.+3.*self.co.km*self.co.s+self.co.km**2*self.co.s**2) / (4.*pi*self.co.s**3) / self.co.nd
            # else:
            self.bst = zeros(shape=(self.co.Nst))
            self.Pstl = zeros(shape=(self.co.Nl, self.co.Nst, self.co.Nk))
            if is_jax:
                self.Pstl = self.Pstl.at[0,0].set(self.co.k**0)
                if self.eft_basis == "westcoast":
                    self.Pstl = self.Pstl.at[0,1].set(self.co.k**2) 
                    self.Pstl = self.Pstl.at[1,2].set(self.co.k**2) 
                elif self.eft_basis in ["eftoflss", 'eastcoast']:
                    for i in range(self.co.Nl):
                        self.Pstl = self.Pstl.at[i,1].set(mu[0][2*i] * self.co.k**2)
                        self.Pstl = self.Pstl.at[i,2].set(mu[2][2*i] * self.co.k**2) 
            else: 
                self.Pstl[0,0] = self.co.k**0 # / self.co.nd
                if self.eft_basis == "westcoast":
                    self.Pstl[0,1] = self.co.k**2 # / self.co.km**2 / self.co.nd
                    self.Pstl[1,2] = self.co.k**2 # / self.co.km**2 / self.co.nd
                elif self.eft_basis in ["eftoflss", 'eastcoast']:
                    for i in range(self.co.Nl):
                        self.Pstl[i,1] = mu[0][2*i] * self.co.k**2 # / self.co.km**2 / self.co.nd
                        self.Pstl[i,2] = mu[2][2*i] * self.co.k**2 # / self.co.km**2 / self.co.nd

        else:
            if self.co.with_cf: self.Cstl = None
            else: self.Pstl = None

        if self.with_nnlo_counterterm:
            self.cnnlo = zeros(shape=(self.co.Nl))
            self.Cnnlo = empty(shape=(self.co.Nl, self.co.Ns))
            self.Cnnlol = empty(shape=(self.co.Nl, self.co.Nnnlo, self.co.Ns))
            self.Pnnlo = empty(shape=(self.co.Nk))
            self.Pnnlol = empty(shape=(self.co.Nl, self.co.Nnnlo, self.co.Nk))
        else: # this was clashing with redshift_bin: True, because for output: 'bpk', it is the correlation that is first computed, so co.with_cf = True at the instatiation of the bird... need to change that # PZ
            self.Pnnlol = None
            self.Cnnlol = None
            # if self.co.with_cf: self.Cnnlol = None
            # else: self.Pnnlol = None
    

    def setcosmo(self, cosmo):

        self.kin = cosmo["kk"]
        self.Pin = cosmo["pk_lin"]

        if self.Pin is not None:


            self.Plin = interp1d(self.kin, self.Pin, kind='cubic', axis=-1)
            self.P11 = self.Plin(self.co.k)

        else:
            self.Plin = None
            self.P11 = None

        if cosmo["pk_lin_2"] is not None: self.Pin_2 = cosmo["pk_lin_2"]
        else: self.Pin_2 = self.Pin

        if not self.co.with_time: self.D = cosmo["D"]
        self.f = cosmo["f"]
        self.DA = cosmo["DA"]
        self.H = cosmo["H"]
        if self.co.exact_time:
            if "w0_fld" in cosmo: self.w0 = cosmo["w0_fld"]
            else: self.w0 = None
            self.Omega0_m = cosmo["Omega0_m"]
            self.z = cosmo["z"]
            self.Y1 = 0.
            self.G1t = 3/7.
            self.V12t = 1/7.
            self.G1 = 1.
        if self.co.nonequaltime:
            self.D = cosmo["D"]
            self.D1 = cosmo["D1"]
            self.D2 = cosmo["D2"]
            self.f1 = cosmo["f1"]
            self.f2 = cosmo["f2"]

    def setBias(self, bias):
        """ Given an array of EFT parameters, set them among linear, loops and counter terms, and among multipoles

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """

        f = self.f
        # print ()

        if self.co.exact_time:
            ## EdS: Y1 = 0., G1t = 3/7., V12t = 1/7.
            G1 = self.G1
            Y1 = self.Y1
            G1t = self.G1t
            V12t = self.V12t

        b1 = bias["b1"]
        b2 = bias["b2"]
        b3 = bias["b3"]
        b4 = bias["b4"]
        if self.eft_basis in ["eftoflss", "westcoast"]:
            b5 = bias["cct"] / self.co.km**2
            b6 = bias["cr1"] / self.co.kr**2
            b7 = bias["cr2"] / self.co.kr**2
        elif self.eft_basis == 'eastcoast': # inversion of (2.23) of 2004.10607
            ct0 = bias["c0"] - f/3. * bias["c2"] + 3/35. * f**2 * bias["c4"]
            ct2 = bias["c2"] - 6/7. * f * bias["c4"]
            ct4 = bias["c4"]

        if self.with_stoch:
            if self.eft_basis in ["eastcoast", "westcoast"]:
                self.bst = array([bias["ce0"], bias["ce1"] / self.co.km**2, bias["ce2"] / self.co.km**2]) / self.co.nd
            elif self.eft_basis == "eftoflss":
                self.bst = array([bias["ce0"], bias["ce1"] / self.co.km**2, f*bias["ce2"] / self.co.km**2]) / self.co.nd
        
        

        if self.co.halohalo:

            if self.with_tidal_alignments: bq = bias["bq"]

            if self.with_bias: # evaluation with biases specified
                # cnnlo
                if self.with_nnlo_counterterm: 
                    if self.eft_basis in ["eftoflss", "westcoast"]: self.cnnlo = array([0.25 * (b1**2 * bias["cr4"] * mu[4][2*i] + b1 * bias["cr6"] * mu[6][2*i]) / self.co.kr**4 for i in range(self.co.Nl)])
                    elif self.eft_basis == "eastcoast": self.cnnlo = array([- bias["ct"] * f**4 * (b1**2 * mu[4][2*i] + 2. * b1 * f * mu[6][2*i] + f**2 * mu[8][2*i]) for i in range(self.co.Nl)])
                # b11
                if self.with_tidal_alignments: self.b11 = array([(b1-bq/3.)**2 * mu[0][2*i] + 2. * (b1-bq/3.) * (f+bq) * mu[2][2*i] + (f+bq)**2 * mu[4][2*i] for i in range(self.co.Nl)])
                else: self.b11 = array([b1**2 * mu[0][2*i] + 2. * b1 * f * mu[2][2*i] + f**2 * mu[4][2*i] for i in range(self.co.Nl)])
                # bct
                if self.eft_basis in ["eftoflss", "westcoast"]: self.bct = array([2. * b1 * (b5 * mu[0][2*i] + b6 * mu[2][2*i] + b7 * mu[4][2*i]) + 2. * f * (b5 * mu[2][2*i] + b6 * mu[4][2*i] + b7 * mu[6][2*i]) for i in range(self.co.Nl)])
                elif self.eft_basis == "eastcoast": self.bct = array([- 2. * (ct0 * mu[0][2*i] + ct2 * f * mu[2][2*i] + ct4 * f**2 * mu[4][2*i]) for i in range(self.co.Nl)])
                # loop, exact_time
                if self.co.exact_time:
                    self.b22 = array([array([b1**2*G1**2*mu[0][2*i], b1*b2*G1*mu[0][2*i], b1*b4*G1*mu[0][2*i], b2**2*mu[0][2*i], b2*b4*mu[0][2*i], b4**2*mu[0][2*i], b1**2*f*G1*mu[2][2*i], b1*b2*f*mu[2][2*i], b1*b4*f*mu[2][2*i], b1*f*G1**2*mu[2][2*i], b2*f*G1*mu[2][2*i], b4*f*G1*mu[2][2*i], b1**2*f**2*mu[2][2*i], b1**2*f**2*mu[4][2*i], b1*f**2*G1*mu[2][2*i], b1*f**2*G1*mu[4][2*i], b2*f**2*mu[2][2*i], b2*f**2*mu[4][2*i], b4*f**2*mu[2][2*i], b4*f**2*mu[4][2*i], f**2*G1**2*mu[4][2*i], b1*f**3*mu[4][2*i], b1*f**3*mu[6][2*i], f**3*G1*mu[4][2*i], f**3*G1*mu[6][2*i], f**4*mu[4][2*i], f**4*mu[6][2*i], f**4*mu[8][2*i], b1*f*G1*G1t*mu[2][2*i], b2*f*G1t*mu[2][2*i], b4*f*G1t*mu[2][2*i], b1*f**2*G1t*mu[4][2*i], f**2*G1*G1t*mu[4][2*i], f**3*G1t*mu[4][2*i], f**3*G1t*mu[6][2*i], f**2*G1t**2*mu[4][2*i]]) for i in range(self.co.Nl)])
                    if self.co.with_uvmatch: self.b13 = array([array([b1**2*G1**2*mu[0][2*i], b1*b3*mu[0][2*i], b1*f*G1**2*mu[2][2*i], b3*f*mu[2][2*i], f**2*G1**2*mu[4][2*i], b1**2*Y1*mu[0][2*i], b1*f*mu[2][2*i]*Y1, f**2*mu[4][2*i]*Y1, b1**2*f*G1t*mu[2][2*i], b1*f**2*G1t*mu[2][2*i], b1*f**2*G1t*mu[4][2*i], f**3*G1t*mu[4][2*i], f**3*G1t*mu[6][2*i], b1*f*mu[2][2*i]*V12t, f**2*mu[4][2*i]*V12t, b1**2 * f * mu[2][2*i], b1**2 * f**2 * mu[2][2*i], b1 * f**2 * mu[4][2*i], b1 * f**3 * mu[4][2*i], f**3 * mu[6][2*i], f**4 * mu[6][2*i]]) for i in range(self.co.Nl)])
                    else: self.b13 = array([array([b1**2*G1**2*mu[0][2*i], b1*b3*mu[0][2*i], b1*f*G1**2*mu[2][2*i], b3*f*mu[2][2*i], f**2*G1**2*mu[4][2*i], b1**2*Y1*mu[0][2*i], b1*f*mu[2][2*i]*Y1, f**2*mu[4][2*i]*Y1, b1**2*f*G1t*mu[2][2*i], b1*f**2*G1t*mu[2][2*i], b1*f**2*G1t*mu[4][2*i], f**3*G1t*mu[4][2*i], f**3*G1t*mu[6][2*i], b1*f*mu[2][2*i]*V12t, f**2*mu[4][2*i]*V12t]) for i in range(self.co.Nl)])
                # loop, tidal_alignments (when not exact_time)
                elif self.with_tidal_alignments:
                    self.b22 = array([array([b1*bq*mu[2][2*i], b2*bq*mu[2][2*i], b4*bq*mu[2][2*i], bq**2*mu[2][2*i], bq**2*mu[4][2*i], b1**2*mu[0][2*i], b1*b2*mu[0][2*i], b1*b4*mu[0][2*i], b1*bq*mu[0][2*i], b2**2*mu[0][2*i], b2*b4*mu[0][2*i], b2*bq*mu[0][2*i], b4**2*mu[0][2*i], b4*bq*mu[0][2*i], bq**2*mu[0][2*i], b1**2*f*mu[2][2*i], b1*b2*f*mu[2][2*i], b1*b4*f*mu[2][2*i], b1*bq*f*mu[2][2*i], b1*bq*f*mu[4][2*i], b1*f*mu[2][2*i], b2*f*mu[2][2*i], b4*f*mu[2][2*i], bq*f*mu[2][2*i], bq*f*mu[4][2*i], b1**2*f**2*mu[2][2*i], b1**2*f**2*mu[4][2*i], b1*f**2*mu[2][2*i], b1*f**2*mu[4][2*i], b2*f**2*mu[2][2*i], b2*f**2*mu[4][2*i], b4*f**2*mu[2][2*i], b4*f**2*mu[4][2*i], bq*f**2*mu[2][2*i], bq*f**2*mu[4][2*i], bq*f**2*mu[6][2*i], f**2*mu[4][2*i], b1*f**3*mu[4][2*i], b1*f**3*mu[6][2*i], f**3*mu[4][2*i], f**3*mu[6][2*i], f**4*mu[4][2*i], f**4*mu[6][2*i], f**4*mu[8][2*i]]) for i in range(self.co.Nl)])
                    self.b13 = array([array([b1*bq*mu[2][2*i], b3*bq*mu[2][2*i], bq**2*mu[2][2*i], bq**2*mu[4][2*i], b1**2*mu[0][2*i], b1*b3*mu[0][2*i], b1*bq*mu[0][2*i], b3*bq*mu[0][2*i], bq**2*mu[0][2*i], b1**2*f*mu[2][2*i], b1*bq*f*mu[2][2*i], b1*bq*f*mu[4][2*i], b1*f*mu[2][2*i], b3*f*mu[2][2*i], bq*f*mu[2][2*i], bq*f*mu[4][2*i], b1*f**2*mu[2][2*i], b1*f**2*mu[4][2*i], bq*f**2*mu[2][2*i], bq*f**2*mu[4][2*i], bq*f**2*mu[6][2*i], f**2*mu[4][2*i], f**3*mu[4][2*i], f**3*mu[6][2*i]]) for i in range(self.co.Nl)])
                # loop, EdS (no tidal_alignments, not exact_time)
                else:
                    self.b22 = array([array([b1**2 * mu[0][2*i], b1 * b2 * mu[0][2*i], b1 * b4 * mu[0][2*i], b2**2 * mu[0][2*i], b2 * b4 * mu[0][2*i], b4**2 * mu[0][2*i], b1**2 * f * mu[2][2*i], b1 * b2 * f * mu[2][2*i], b1 * b4 * f * mu[2][2*i], b1 * f * mu[2][2*i], b2 * f * mu[2][2*i], b4 * f * mu[2][2*i], b1**2 * f**2 * mu[2][2*i], b1**2 * f**2 * mu[4][2*i], b1 * f**2 * mu[2][2*i], b1 * f**2 * mu[4][2*i], b2 * f**2 * mu[2][2*i], b2 * f**2 * mu[4][2*i], b4 * f**2 * mu[2][2*i], b4 * f**2 * mu[4][2*i], f**2 * mu[4][2*i], b1 * f**3 * mu[4][2*i], b1 * f**3 * mu[6][2*i], f**3 * mu[4][2*i], f**3 * mu[6][2*i], f**4 * mu[4][2*i], f**4 * mu[6][2*i], f**4 * mu[8][2*i]]) for i in range(self.co.Nl)])
                    if self.co.with_uvmatch: self.b13 = array([array([b1**2 * mu[0][2*i], b1 * b3 * mu[0][2*i], b1**2 * f * mu[2][2*i], b1 * f * mu[2][2*i], b3 * f * mu[2][2*i], b1 * f**2 * mu[2][2*i], b1 * f**2 * mu[4][2*i], f**2 * mu[4][2*i], f**3 * mu[4][2*i], f**3 * mu[6][2*i], b1**2 * f**2 * mu[2][2*i], b1 * f**3 * mu[4][2*i], f**4 * mu[6][2*i]]) for i in range(self.co.Nl)])
                    else: self.b13 = array([array([b1**2 * mu[0][2*i], b1 * b3 * mu[0][2*i], b1**2 * f * mu[2][2*i], b1 * f * mu[2][2*i], b3 * f * mu[2][2*i], b1 * f**2 * mu[2][2*i], b1 * f**2 * mu[4][2*i], f**2 * mu[4][2*i], f**3 * mu[4][2*i], f**3 * mu[6][2*i]]) for i in range(self.co.Nl)])
            else: # evaluation with biases unspecified
                if self.with_nnlo_counterterm:
                    if self.eft_basis in ["eftoflss", "westcoast"]: self.cnnlo = 0.25 * array([b1**2 * bias["cr4"], b1 * bias["cr6"]]) / self.co.kr**4
                    elif self.eft_basis == "eastcoast": self.cnnlo = - bias["ct"] * f**4 * array([b1**2, 2. * b1 * f, f**2])   # these are not divided by kr^4 according to eastcoast definition; the prior is adjusted accordingly
                if self.with_tidal_alignments: self.b11 = array([(b1-bq/3.)**2, 2. * (b1-bq/3.) * (f+bq), (f+bq)**2])
                else: self.b11 = array([b1**2, 2. * b1 * f, f**2])
                if self.eft_basis in ["eftoflss", "westcoast"]: self.bct = array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
                elif self.eft_basis == "eastcoast": self.bct = - 2. * array([ct0, f * ct2, f**2 * ct4]) # these are not divided by km^2 or kr^2 according to eastcoast definition; the prior is adjusted accordingly
                if self.co.Nloop == 12: self.bloop = array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
                elif self.co.Nloop == 22: self.bloop = array([f**2, f**3, f**4, b1*f, b1*f**2, b1*f**3, b2*f, b2*f**2, b3*f, b4*f, b4*f**2, b1**2, b1**2*f, b1**2*f**2, b1*b2, b1*b2*f, b1*b3, b1*b4, b1*b4*f, b2**2, b2*b4, b4**2])
                elif self.co.Nloop == 35: self.bloop = array([f**2, f**2*G1t, f**2*G1t**2, f**2*Y1, f**2*V12t, f**3, f**3*G1t, f**4, b1*f, b1*f*G1t, b1*f*Y1, b1*f*V12t, b1*f**2, b1*f**2*G1t, b1*f**3, b2*f, b2*f*G1t, b2*f**2, b3*f, b4*f, b4*f*G1t, b4*f**2, b1**2, b1**2*Y1, b1**2*f, b1**2*f*G1t, b1**2*f**2, b1*b2, b1*b2*f, b1*b3, b1*b4, b1*b4*f, b2**2, b2*b4, b4**2])
                elif self.co.Nloop == self.co.N22+self.co.N13: self.bloop = array([
                    b1**2, b1 * b2, b1 * b4, b2**2, b2 * b4, b4**2,
                    b1**2 * f, b1 * b2 * f, b1 * b4 * f, b1 * f, b2 * f, b4 * f,
                    b1**2 * f**2, b1**2 *f**2, b1 * f**2, b2 * f**2, b2 * f**2, b2 * f**2, b4 * f**2, b4 * f**2, f**2,
                    b1 * f**3, b1 * f**3, f**3, f**3, f**4, f**4, f**4,
                    b1**2, b1 * b3, b1**2 * f, b1 * f, b3 * f, b1 * f**2, b1 * f**2, f**2, f**3, f**3])
                elif self.co.Nloop == 18: self.bloop = array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4, bq, bq * bq, bq * b1, bq * b2, bq * b3, bq * b4]) # with_tidal_alignements

        else: # halo-matter

            d5 = bias["dct"] / self.co.km**2 # matter counterterm
            d6 = bias["dr1"] / self.co.km**2 # matter redshift counterterm 1
            d7 = bias["dr2"] / self.co.km**2 # matter redshift counterterm 2

            if self.with_bias:
                for i in range(self.co.Nl):
                    l = 2 * i
                    self.b11[i] = b1 * mu[0][l] + b1 * f * mu[2][l] + f * mu[2][l] + f**2 * mu[4][l]
                    self.bct[i] = b1 * (d5 * mu[0][l] + d6 * mu[2][l] + d7 * mu[4][l]) + f * (d5 * mu[2][l] + d6 * mu[4][l] + d7 * mu[6][l]) + b5 * (mu[0][l] + f * mu[2][l]) + b6 * (mu[2][l] + f * mu[4][l]) + b7 * (mu[4][l] + f * mu[6][l])
                    self.b22[i] = array([ b1*mu[0][l], b2*mu[0][l], b4*mu[0][l], b1*f*mu[2][l], b2*f*mu[2][l], b4*f*mu[2][l], f*mu[2][l], b1*f**2*mu[2][l], b1*f**2*mu[4][l], b2*f**2*mu[2][l], b2*f**2*mu[4][l], b4*f**2*mu[2][l], b4*f**2*mu[4][l], f**2*mu[2][l], f**2*mu[4][l], b1*f**3*mu[4][l], b1*f**3*mu[6][l], f**3*mu[4][l], f**3*mu[6][l], f**4*mu[4][l], f**4*mu[6][l], f**4*mu[8][l] ])
                    self.b13[i] = array([ b1*mu[0][l], b3*mu[0][l], b1*f*mu[2][l], b3*f*mu[2][l], f*mu[2][l], b1*f**2*mu[2][l], b1*f**2*mu[4][l], f**2*mu[2][l], f**2*mu[4][l], f**3*mu[4][l], f**3*mu[6][l] ])
            else:
                self.b11 = array([b1, b1 * f, f, f**2])
                self.bct = array([b1 * dct, b1 * dr1, b1 * dr2, f * dct, f * dr1, f * dr2, cct, cr1, cr2, f * cct, f * cr1, f * cr2])
                if self.co.Nloop == 5: self.bloop = array([1., b1, b2, b3, b4])

    def setPs(self, bs=None, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiplies them accordingly to the power spectrum multipole terms and adds the resulting terms together per loop order

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        if bs is not None: self.setBias(bs)
        self.Ps = [None] * 2
        self.Ps[0] = einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = einsum('lb,bx->lx', self.b22, self.P22) + einsum('lb,bx->lx', self.b13, self.P13) + einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11) 
        Ps1_k_at_0 = einsum('lb,b->l', self.b22, self.P22[:, self.co.id_kstable]) + einsum('lb,b->l', self.b13, self.P13[:, self.co.id_kstable]) + self.bct * self.co.k[self.co.id_kstable]**2 * self.P11[self.co.id_kstable] # self.co.id_kstable = 0 if kmin = 0.001 (default), = 1 if kmin < 0.001 (option)
        Ps1_k_at_0 = tile(Ps1_k_at_0, (self.co.Nk,1)).T
        self.Ps[1] -= Ps1_k_at_0 # setting loop = 0 at k ~ 0 
        if self.with_stoch: self.Ps[1] += einsum('b,lbx->lx', self.bst, self.Pstl)
        if self.with_nnlo_counterterm: self.Ps[1] += einsum('l,x->lx', self.cnnlo, self.Pnnlo)
        self.Ps = array(self.Ps)
        if setfull: self.setfullPs()

    def setCf(self, bs=None, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        if bs is not None: self.setBias(bs)
        self.Cf = [None] * 2
        self.Cf[0] = einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = einsum('lb,lbx->lx', self.b22, self.C22l) + einsum('lb,lbx->lx', self.b13, self.C13l) + einsum('l,lx->lx', self.bct, self.Cct)
        # if self.with_stoch: self.Cf[1] += einsum('b,lbx->lx', self.bst, self.Cstl) # no stochastic term for Cf
        if self.with_nnlo_counterterm: self.Cf[1] += einsum('l,lx->lx', self.cnnlo, self.Cnnlo)
        self.Cf = array(self.Cf)
        if setfull: self.setfullCf()

    def setPsCf(self, bs, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the power spectrum and correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.setPs(setfull=setfull)
        self.setCf(setfull=setfull)

    def setfullPs(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full power spectrum multipoles """
        self.fullPs = sum(self.Ps, axis=0)

    def setfullCf(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full correlation function multipoles """
        self.fullCf = sum(self.Cf, axis=0)

    def setPsCfl(self, with_loop_and_cf=True):
        """ For option: which='all'. Creates multipoles for each term weighted accordingly """
        self.P11l = einsum('x,ln->lnx', self.P11, self.co.l11)
        self.Pctl = einsum('x,x,ln->lnx', self.co.k**2, self.P11, self.co.lct)

        if self.with_nnlo_counterterm:
            self.Pnnlol = einsum('x,ln->lnx', self.Pnnlo, self.co.lnnlo)
            self.Cnnlol = einsum('lx,ln->lnx', self.Cnnlo, self.co.lnnlo)

        if with_loop_and_cf:
            self.P22l = einsum('nx,ln->lnx', self.P22, self.co.l22)
            self.P13l = einsum('nx,ln->lnx', self.P13, self.co.l13)
            self.Cctl = einsum('lx,ln->lnx', self.Cct, self.co.lct)
            self.C11l = einsum('lx,ln->lnx', self.C11, self.co.l11)
            self.C22l = einsum('lnx,ln->lnx', self.C22l, self.co.l22)
            self.C13l = einsum('lnx,ln->lnx', self.C13l, self.co.l13)

            self.reducePsCfl()
        return

    def reducePsCfl(self):
        """ For option: which='all'. Regroups terms that share the same EFT parameter(s) (more generally, the same time functions) """
        if is_jax: ### jax 
            if self.co.halohalo:

                if self.co.exact_time:      # config["with_exact_time"] == True
                    if self.co.Nloop == 12: # config["with_time"] == True
                        f1 = self.f

                        ## EdS: Y1 = 0., G1t = 3/7., V12t = 1/7.
                        G1 = self.G1
                        Y1 = self.Y1
                        G1t = self.G1t
                        V12t = self.V12t

                        self.Ploopl = self.Ploopl.at[:, 0].set(G1**2 * f1**2 * self.P22l[:, 20] + G1 * f1**3 * self.P22l[:, 23] + G1 * f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + \
                        G1 * G1t * f1**2 * self.P22l[:, 32] + G1t * f1**3 * self.P22l[:, 33] + G1t * f1**3 * self.P22l[:, 34] + G1t**2 * f1**2 * self.P22l[:, 35] + \
                        G1**2 * f1**2 * self.P13l[:, 4] + Y1 * f1**2 * self.P13l[:, 7] + G1t * f1**3 * self.P13l[:, 11] + G1t * f1**3 * self.P13l[:, 12] + V12t * f1**2 * self.P13l[:, 14]) # *1
                        self.Ploopl = self.Ploopl.at[:, 1].set(G1**2 * f1 * self.P22l[:, 9] + G1 * f1**2 * self.P22l[:, 14] + G1 * f1**2 * self.P22l[:, 15] + f1**3 * self.P22l[:, 21] + f1**3 * self.P22l[:, 22] + G1 * G1t * f1 * self.P22l[:, 28] + G1t * f1**2 * self.P22l[:, 31] + \
                            G1**2 * f1 * self.P13l[:, 2] + Y1 * f1 * self.P13l[:, 6] + G1t * f1**2 * self.P13l[:, 9] + G1t * f1**2 * self.P13l[:, 10] + V12t * f1 * self.P13l[:, 13]) # *b1
                        self.Ploopl = self.Ploopl.at[:, 2].set(G1 * f1 * self.P22l[:, 10] + f1**2 * self.P22l[:, 16] + f1**2 * self.P22l[:, 17] + G1t * f1 * self.P22l[:, 29]) # *b2
                        self.Ploopl = self.Ploopl.at[:, 3].set(f1 * self.P13l[:, 3]) # *b3
                        self.Ploopl = self.Ploopl.at[:, 4].set(G1 * f1 * self.P22l[:, 11] + f1**2 * self.P22l[:, 18] + f1**2 * self.P22l[:, 19] + G1t * f1 * self.P22l[:, 30]) # *b4
                        self.Ploopl = self.Ploopl.at[:, 5].set(G1**2 * self.P22l[:, 0] + G1 * f1 * self.P22l[:, 6] + f1**2 * self.P22l[:, 12] + f1**2 * self.P22l[:, 13] + G1**2 * self.P13l[:, 0] + Y1 * self.P13l[:, 5] + G1t * f1 * self.P13l[:, 8])  # *b1*b1
                        self.Ploopl = self.Ploopl.at[:, 6].set(G1 * self.P22l[:, 1] + f1 * self.P22l[:, 7])  # *b1*b2
                        self.Ploopl = self.Ploopl.at[:, 7].set(self.P13l[:, 1])  # *b1*b3
                        self.Ploopl = self.Ploopl.at[:, 8].set(G1 * self.P22l[:, 2] + f1 * self.P22l[:, 8])  # *b1*b4
                        self.Ploopl = self.Ploopl.at[:, 9].set(self.P22l[:, 3])  # *b2*b2
                        self.Ploopl = self.Ploopl.at[:, 10].set(self.P22l[:, 4])  # *b2*b4
                        self.Ploopl = self.Ploopl.at[:, 11].set(self.P22l[:, 5])  # *b4*b4

                        self.Cloopl = self.Cloopl.at[:, 0].set(G1**2 * f1**2 * self.C22l[:, 20] + G1 * f1**3 * self.C22l[:, 23] + G1 * f1**3 * self.C22l[:, 24] + f1**4 * self.C22l[:, 25] + f1**4 * self.C22l[:, 26] + f1**4 * self.C22l[:, 27] + \
                            G1 * G1t * f1**2 * self.C22l[:, 32] + G1t * f1**3 * self.C22l[:, 33] + G1t * f1**3 * self.C22l[:, 34] + G1t**2 * f1**2 * self.C22l[:, 35] + \
                            G1**2 * f1**2 * self.C13l[:, 4] + Y1 * f1**2 * self.C13l[:, 7] + G1t * f1**3 * self.C13l[:, 11] + G1t * f1**3 * self.C13l[:, 12] + V12t * f1**2 * self.C13l[:, 14]) # *1
                        self.Cloopl = self.Cloopl.at[:, 1].set(G1**2 * f1 * self.C22l[:, 9] + G1 * f1**2 * self.C22l[:, 14] + G1 * f1**2 * self.C22l[:, 15] + f1**3 * self.C22l[:, 21] + f1**3 * self.C22l[:, 22] + G1 * G1t * f1 * self.C22l[:, 28] + G1t * f1**2 * self.C22l[:, 31] + \
                            G1**2 * f1 * self.C13l[:, 2] + Y1 * f1 * self.C13l[:, 6] + G1t * f1**2 * self.C13l[:, 9] + G1t * f1**2 * self.C13l[:, 10] + V12t * f1 * self.C13l[:, 13]) # *b1
                        self.Cloopl = self.Cloopl.at[:, 2].set(G1 * f1 * self.C22l[:, 10] + f1**2 * self.C22l[:, 16] + f1**2 * self.C22l[:, 17] + G1t * f1 * self.C22l[:, 29]) # *b2
                        self.Cloopl = self.Cloopl.at[:, 3].set(f1 * self.C13l[:, 3]) # *b3
                        self.Cloopl = self.Cloopl.at[:, 4].set(G1 * f1 * self.C22l[:, 11] + f1**2 * self.C22l[:, 18] + f1**2 * self.C22l[:, 19] + G1t * f1 * self.C22l[:, 30]) # *b4
                        self.Cloopl = self.Cloopl.at[:, 5].set(G1**2 * self.C22l[:, 0] + G1 * f1 * self.C22l[:, 6] + f1**2 * self.C22l[:, 12] + f1**2 * self.C22l[:, 13] + G1**2 * self.C13l[:, 0] + Y1 * self.C13l[:, 5] + G1t * f1 * self.C13l[:, 8])  # *b1*b1
                        self.Cloopl = self.Cloopl.at[:, 6].set(G1 * self.C22l[:, 1] + f1 * self.C22l[:, 7])  # *b1*b2
                        self.Cloopl = self.Cloopl.at[:, 7].set(self.C13l[:, 1])  # *b1*b3
                        self.Cloopl = self.Cloopl.at[:, 8].set(G1 * self.C22l[:, 2] + f1 * self.C22l[:, 8])  # *b1*b4
                        self.Cloopl = self.Cloopl.at[:, 9].set(self.C22l[:, 3])  # *b2*b2
                        self.Cloopl = self.Cloopl.at[:, 10].set(self.C22l[:, 4])  # *b2*b4
                        self.Cloopl = self.Cloopl.at[:, 11].set(self.C22l[:, 5])  # *b4*b4

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl = self.Ploopl.at[:, 0].add(f1**4 * self.P13l[:, -1] + f1**3 * self.P13l[:, -2]) # *1
                            self.Ploopl = self.Ploopl.at[:, 1].add(f1**3 * self.P13l[:, -3] + f1**2 * self.P13l[:, -4]) # *b1
                            self.Ploopl = self.Ploopl.at[:, 5].add(f1**2 * self.P13l[:, -5] + f1 * self.P13l[:, -6]) # *b1*b1

                            self.Cloopl = self.Cloopl.at[:, 0].add(f1**4 * self.C13l[:, -1] + f1**3 * self.C13l[:, -2]) # *1
                            self.Cloopl = self.Cloopl.at[:, 1].add(f1**3 * self.C13l[:, -3] + f1**2 * self.C13l[:, -4]) # *b1
                            self.Cloopl = self.Cloopl.at[:, 5].add(f1**2 * self.C13l[:, -5] + f1 * self.C13l[:, -6]) # *b1*b1

                    elif self.co.Nloop == 35: # config["with_time"] == False
                        self.Ploopl = self.Ploopl.at[:, 0].set(self.P22l[:, 20] + self.P13l[:, 4])   # *f^2
                        self.Ploopl = self.Ploopl.at[:, 1].set(self.P22l[:, 32])    # *f^2*G1t
                        self.Ploopl = self.Ploopl.at[:, 2].set(self.P22l[:, 35])    # *f^2*G1t**2
                        self.Ploopl = self.Ploopl.at[:, 3].set(self.P13l[:, 7])     # *f^2*Y1
                        self.Ploopl = self.Ploopl.at[:, 4].set(self.P13l[:, 14])    # *f^2*V12t
                        self.Ploopl = self.Ploopl.at[:, 5].set(self.P22l[:, 23] + self.P22l[:, 24])   # *f^3
                        self.Ploopl = self.Ploopl.at[:, 6].set(self.P22l[:, 33] + self.P22l[:, 34] + self.P13l[:, 11] + self.P13l[:, 12]) # *f^3*G1t
                        self.Ploopl = self.Ploopl.at[:, 7].set(self.P22l[:, 25] + self.P22l[:, 26] + self.P22l[:, 27])   # *f^4
                        self.Ploopl = self.Ploopl.at[:, 8].set(self.P22l[:, 9] + self.P13l[:, 2])  # *b1*f
                        self.Ploopl = self.Ploopl.at[:, 9].set(self.P22l[:, 28])    # *b1*f*G1t
                        self.Ploopl = self.Ploopl.at[:, 10].set(self.P13l[:, 6])     # *b1*f*Y1
                        self.Ploopl = self.Ploopl.at[:, 11].set(self.P13l[:, 13])    # *b1*f*V12t
                        self.Ploopl = self.Ploopl.at[:, 12].set(self.P22l[:, 14] + self.P22l[:, 15])     # *b1*f^2
                        self.Ploopl = self.Ploopl.at[:, 13].set(self.P22l[:, 31] + self.P13l[:, 9] + self.P13l[:, 10])   # *b1*f^2*G1t
                        self.Ploopl = self.Ploopl.at[:, 14].set(self.P22l[:, 21] + self.P22l[:, 22])     # *b1*f^3
                        self.Ploopl = self.Ploopl.at[:, 15].set(self.P22l[:, 10])    # *b2*f
                        self.Ploopl = self.Ploopl.at[:, 16].set(self.P22l[:, 29])    # *b2*f*G1t
                        self.Ploopl = self.Ploopl.at[:, 17].set(self.P22l[:, 16] + self.P22l[:, 17])     # *b2*f^2
                        self.Ploopl = self.Ploopl.at[:, 18].set(self.P13l[:, 3])     # *b3*f
                        self.Ploopl = self.Ploopl.at[:, 19].set(self.P22l[:, 11])    # *b4*f
                        self.Ploopl = self.Ploopl.at[:, 20].set(self.P22l[:, 30])    # *b4*f*G1t
                        self.Ploopl = self.Ploopl.at[:, 21].set(self.P22l[:, 18] + self.P22l[:, 19])     # *b4*f^2
                        self.Ploopl = self.Ploopl.at[:, 22].set(self.P22l[:, 0] + self.P13l[:, 0])   # *b1^2
                        self.Ploopl = self.Ploopl.at[:, 23].set(self.P13l[:, 5])     # *b1^2*Y1
                        self.Ploopl = self.Ploopl.at[:, 24].set(self.P22l[:, 6])     # *b1^2*f
                        self.Ploopl = self.Ploopl.at[:, 25].set(self.P13l[:, 8])     # *b1^2*f*G1t
                        self.Ploopl = self.Ploopl.at[:, 26].set(self.P22l[:, 12] + self.P22l[:, 13])     # *b1^2*f^2
                        self.Ploopl = self.Ploopl.at[:, 27].set(self.P22l[:, 1])     # *b1*b2
                        self.Ploopl = self.Ploopl.at[:, 28].set(self.P22l[:, 7])     # *b1*b2*f
                        self.Ploopl = self.Ploopl.at[:, 29].set(self.P13l[:, 1])     # *b1*b3
                        self.Ploopl = self.Ploopl.at[:, 30].set(self.P22l[:, 2])     # *b1*b4
                        self.Ploopl = self.Ploopl.at[:, 31].set(self.P22l[:, 8])     # *b1*b4*f
                        self.Ploopl = self.Ploopl.at[:, 32].set(self.P22l[:, 3])     # *b2^2
                        self.Ploopl = self.Ploopl.at[:, 33].set(self.P22l[:, 4])     # *b2*b4
                        self.Ploopl = self.Ploopl.at[:, 34].set(self.P22l[:, 5])     # *b4^2

                        self.Cloopl = self.Cloopl.at[:, 0].set(self.C22l[:, 20] + self.C13l[:, 4])   # *f^2
                        self.Cloopl = self.Cloopl.at[:, 1].set(self.C22l[:, 32])    # *f^2*G1t
                        self.Cloopl = self.Cloopl.at[:, 2].set(self.C22l[:, 35])    # *f^2*G1t**2
                        self.Cloopl = self.Cloopl.at[:, 3].set(self.C13l[:, 7])     # *f^2*Y1
                        self.Cloopl = self.Cloopl.at[:, 4].set(self.C13l[:, 14])    # *f^2*V12t
                        self.Cloopl = self.Cloopl.at[:, 5].set(self.C22l[:, 23] + self.C22l[:, 24])   # *f^3
                        self.Cloopl = self.Cloopl.at[:, 6].set(self.C22l[:, 33] + self.C22l[:, 34] + self.C13l[:, 11] + self.C13l[:, 12]) # *f^3*G1t
                        self.Cloopl = self.Cloopl.at[:, 7].set(self.C22l[:, 25] + self.C22l[:, 26] + self.C22l[:, 27])   # *f^4
                        self.Cloopl = self.Cloopl.at[:, 8].set(self.C22l[:, 9] + self.C13l[:, 2])  # *b1*f
                        self.Cloopl = self.Cloopl.at[:, 9].set(self.C22l[:, 28])    # *b1*f*G1t
                        self.Cloopl = self.Cloopl.at[:, 10].set(self.C13l[:, 6])     # *b1*f*Y1
                        self.Cloopl = self.Cloopl.at[:, 11].set(self.C13l[:, 13])    # *b1*f*V12t
                        self.Cloopl = self.Cloopl.at[:, 12].set(self.C22l[:, 14] + self.C22l[:, 15])     # *b1*f^2
                        self.Cloopl = self.Cloopl.at[:, 13].set(self.C22l[:, 31] + self.C13l[:, 9] + self.C13l[:, 10])   # *b1*f^2*G1t
                        self.Cloopl = self.Cloopl.at[:, 14].set(self.C22l[:, 21] + self.C22l[:, 22])     # *b1*f^3
                        self.Cloopl = self.Cloopl.at[:, 15].set(self.C22l[:, 10])    # *b2*f
                        self.Cloopl = self.Cloopl.at[:, 16].set(self.C22l[:, 29])    # *b2*f*G1t
                        self.Cloopl = self.Cloopl.at[:, 17].set(self.C22l[:, 16] + self.C22l[:, 17])     # *b2*f^2
                        self.Cloopl = self.Cloopl.at[:, 18].set(self.C13l[:, 3])     # *b3*f
                        self.Cloopl = self.Cloopl.at[:, 19].set(self.C22l[:, 11])    # *b4*f
                        self.Cloopl = self.Cloopl.at[:, 20].set(self.C22l[:, 30])    # *b4*f*G1t
                        self.Cloopl = self.Cloopl.at[:, 21].set(self.C22l[:, 18] + self.C22l[:, 19])     # *b4*f^2
                        self.Cloopl = self.Cloopl.at[:, 22].set(self.C22l[:, 0] + self.C13l[:, 0])   # *b1^2
                        self.Cloopl = self.Cloopl.at[:, 23].set(self.C13l[:, 5])     # *b1^2*Y1
                        self.Cloopl = self.Cloopl.at[:, 24].set(self.C22l[:, 6])     # *b1^2*f
                        self.Cloopl = self.Cloopl.at[:, 25].set(self.C13l[:, 8])     # *b1^2*f*G1t
                        self.Cloopl = self.Cloopl.at[:, 26].set(self.C22l[:, 12] + self.C22l[:, 13])     # *b1^2*f^2
                        self.Cloopl = self.Cloopl.at[:, 27].set(self.C22l[:, 1])     # *b1*b2
                        self.Cloopl = self.Cloopl.at[:, 28].set(self.C22l[:, 7])     # *b1*b2*f
                        self.Cloopl = self.Cloopl.at[:, 29].set(self.C13l[:, 1])     # *b1*b3
                        self.Cloopl = self.Cloopl.at[:, 30].set(self.C22l[:, 2])     # *b1*b4
                        self.Cloopl = self.Cloopl.at[:, 31].set(self.C22l[:, 8])     # *b1*b4*f
                        self.Cloopl = self.Cloopl.at[:, 32].set(self.C22l[:, 3])     # *b2^2
                        self.Cloopl = self.Cloopl.at[:, 33].set(self.C22l[:, 4])     # *b2*b4
                        self.Cloopl = self.Cloopl.at[:, 34].set(self.C22l[:, 5])     # *b4^2

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl = self.Ploopl.at[:, 7].add(self.P13l[:, -1]) # *f^4
                            self.Ploopl = self.Ploopl.at[:, 5].add(self.P13l[:, -2]) # *f^3
                            self.Ploopl = self.Ploopl.at[:, 14].add(self.P13l[:, -3]) # *b1*f^3
                            self.Ploopl = self.Ploopl.at[:, 12].add(self.P13l[:, -4]) # *b1*f^2
                            self.Ploopl = self.Ploopl.at[:, 26].add(self.P13l[:, -5]) # *b1*b1*f^2
                            self.Ploopl = self.Ploopl.at[:, 24].add(self.P13l[:, -6]) # *b1*b1*f

                            self.Cloopl = self.Cloopl.at[:, 7].add(self.C13l[:, -1]) # *f^4
                            self.Cloopl = self.Ploopl.at[:, 5].add(self.C13l[:, -2]) # *f^3
                            self.Cloopl = self.Cloopl.at[:, 14].add(self.C13l[:, -3]) # *b1*f^3
                            self.Cloopl = self.Cloopl.at[:, 12].add(self.C13l[:, -4]) # *b1*f^2
                            self.Cloopl = self.Cloopl.at[:, 26].add(self.C13l[:, -5]) # *b1*b1*f^2
                            self.Cloopl = self.Cloopl.at[:, 24].add(self.C13l[:, -6]) # *b1*b1*f

                else:                       # config["with_exact_time"] == False
                    if self.co.Nloop == 12: # config["with_time"] == True
                        f1 = self.f

                        self.Ploopl = self.Ploopl.at[:, 0].set(f1**2 * self.P22l[:, 20] + f1**3 * self.P22l[:, 23] + f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + f1**2 * self.P13l[:, 7] + f1**3 * self.P13l[:, 8] + f1**3 * self.P13l[:, 9])  # *1
                        self.Ploopl = self.Ploopl.at[:, 1].set(f1 * self.P22l[:, 9] + f1**2 * self.P22l[:, 14] + f1**2 * self.P22l[:, 15] + f1**3 * self.P22l[:, 21] + f1**3 * self.P22l[:, 22] + f1 * self.P13l[:, 3] + f1**2 * self.P13l[:, 5] + f1**2 * self.P13l[:, 6])  # *b1
                        self.Ploopl = self.Ploopl.at[:, 2].set(f1 * self.P22l[:, 10] + f1**2 * self.P22l[:, 16] + f1**2 * self.P22l[:, 17])  # *b2
                        self.Ploopl = self.Ploopl.at[:, 3].set(f1 * self.P13l[:, 4])  # *b3
                        self.Ploopl = self.Ploopl.at[:, 4].set(f1 * self.P22l[:, 11] + f1**2 * self.P22l[:, 18] + f1**2 * self.P22l[:, 19])  # *b4
                        self.Ploopl = self.Ploopl.at[:, 5].set(self.P22l[:, 0] + f1 * self.P22l[:, 6] + f1**2 * self.P22l[:, 12] + f1**2 * self.P22l[:, 13] + self.P13l[:, 0] + f1 * self.P13l[:, 2])  # *b1*b1
                        self.Ploopl = self.Ploopl.at[:, 6].set(self.P22l[:, 1] + f1 * self.P22l[:, 7])  # *b1*b2
                        self.Ploopl = self.Ploopl.at[:, 7].set(self.P13l[:, 1])  # *b1*b3
                        self.Ploopl = self.Ploopl.at[:, 8].set(self.P22l[:, 2] + f1 * self.P22l[:, 8])  # *b1*b4
                        self.Ploopl = self.Ploopl.at[:, 9].set(self.P22l[:, 3])  # *b2*b2
                        self.Ploopl = self.Ploopl.at[:, 10].set(self.P22l[:, 4])  # *b2*b4
                        self.Ploopl = self.Ploopl.at[:, 11].set(self.P22l[:, 5])  # *b4*b4

                        self.Cloopl = self.Cloopl.at[:, 0].set(f1**2 * self.C22l[:, 20] + f1**3 * self.C22l[:, 23] + f1**3 * self.C22l[:, 24] + f1**4 * self.C22l[:, 25] + f1**4 * self.C22l[:, 26] + f1**4 * self.C22l[:, 27] + f1**2 * self.C13l[:, 7] + f1**3 * self.C13l[:, 8] + f1**3 * self.C13l[:, 9])  # *1
                        self.Cloopl = self.Cloopl.at[:, 1].set(f1 * self.C22l[:, 9] + f1**2 * self.C22l[:, 14] + f1**2 * self.C22l[:, 15] + f1**3 * self.C22l[:, 21] + f1**3 * self.C22l[:, 22] + f1 * self.C13l[:, 3] + f1**2 * self.C13l[:, 5] + f1**2 * self.C13l[:, 6])  # *b1
                        self.Cloopl = self.Cloopl.at[:, 2].set(f1 * self.C22l[:, 10] + f1**2 * self.C22l[:, 16] + f1**2 * self.C22l[:, 17])  # *b2
                        self.Cloopl = self.Cloopl.at[:, 3].set(f1 * self.C13l[:, 4])  # *b3
                        self.Cloopl = self.Cloopl.at[:, 4].set(f1 * self.C22l[:, 11] + f1**2 * self.C22l[:, 18] + f1**2 * self.C22l[:, 19])  # *b4
                        self.Cloopl = self.Cloopl.at[:, 5].set(self.C22l[:, 0] + f1 * self.C22l[:, 6] + f1**2 * self.C22l[:, 12] + f1**2 * self.C22l[:, 13] + self.C13l[:, 0] + f1 * self.C13l[:, 2])  # *b1*b1
                        self.Cloopl = self.Cloopl.at[:, 6].set(self.C22l[:, 1] + f1 * self.C22l[:, 7])  # *b1*b2
                        self.Cloopl = self.Cloopl.at[:, 7].set(self.C13l[:, 1])  # *b1*b3
                        self.Cloopl = self.Cloopl.at[:, 8].set(self.C22l[:, 2] + f1 * self.C22l[:, 8])  # *b1*b4
                        self.Cloopl = self.Cloopl.at[:, 9].set(self.C22l[:, 3])  # *b2*b2
                        self.Cloopl = self.Cloopl.at[:, 10].set(self.C22l[:, 4])  # *b2*b4
                        self.Cloopl = self.Cloopl.at[:, 11].set(self.C22l[:, 5])  # *b4*b4

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl = self.Ploopl.at[:, 0].add(f1**4 * self.P13l[:, -1]) # *1
                            self.Ploopl = self.Ploopl.at[:, 1].add(f1**3 * self.P13l[:, -2]) # *b1
                            self.Ploopl = self.Ploopl.at[:, 5].add(f1**2 * self.P13l[:, -3]) # *b1*b1

                            self.Cloopl = self.Cloopl.at[:, 0].add(f1**4 * self.C13l[:, -1]) # *1
                            self.Cloopl = self.Cloopl.at[:, 1].add(f1**3 * self.C13l[:, -2]) # *b1
                            self.Cloopl = self.Cloopl.at[:, 5].add(f1**2 * self.C13l[:, -3]) # *b1*b1

                    elif self.co.Nloop == 22: # config["with_time"] == False
                        self.Ploopl = self.Ploopl.at[:, 0].set(self.P22l[:, 20] + self.P13l[:, 7])   # *f^2
                        self.Ploopl = self.Ploopl.at[:, 1].set(self.P22l[:, 23] + self.P22l[:, 24] + self.P13l[:, 8] + self.P13l[:, 9])   # *f^3
                        self.Ploopl = self.Ploopl.at[:, 2].set(self.P22l[:, 25] + self.P22l[:, 26] + self.P22l[:, 27])   # *f^4
                        self.Ploopl = self.Ploopl.at[:, 3].set(self.P22l[:, 9] + self.P13l[:, 3])  # *b1*f
                        self.Ploopl = self.Ploopl.at[:, 4].set(self.P22l[:, 14] + self.P22l[:, 15] + self.P13l[:, 5] + self.P13l[:, 6])   # *b1*f^2
                        self.Ploopl = self.Ploopl.at[:, 5].set(self.P22l[:, 21] + self.P22l[:, 22])   # *b1*f^3
                        self.Ploopl = self.Ploopl.at[:, 6].set(self.P22l[:, 10])   # *b2*f
                        self.Ploopl = self.Ploopl.at[:, 7].set(self.P22l[:, 16] + self.P22l[:, 17])  # *b2*f^2
                        self.Ploopl = self.Ploopl.at[:, 8].set(self.P13l[:, 4])   # *b3*f
                        self.Ploopl = self.Ploopl.at[:, 9].set(self.P22l[:, 11])   # *b4*f
                        self.Ploopl = self.Ploopl.at[:, 10].set(self.P22l[:, 18] + self.P22l[:, 19])   # *b4*f^2
                        self.Ploopl = self.Ploopl.at[:, 11].set(self.P22l[:, 0] + self.P13l[:, 0])   # *b1*b1
                        self.Ploopl = self.Ploopl.at[:, 12].set(self.P22l[:, 6] + self.P13l[:, 2])   # *b1*b1*f
                        self.Ploopl = self.Ploopl.at[:, 13].set(self.P22l[:, 12] + self.P22l[:, 13])   # *b1*b1*f^2
                        self.Ploopl = self.Ploopl.at[:, 14].set(self.P22l[:, 1])  # *b1*b2
                        self.Ploopl = self.Ploopl.at[:, 15].set(self.P22l[:, 7])  # *b1*b2*f
                        self.Ploopl = self.Ploopl.at[:, 16].set(self.P13l[:, 1])  # *b1*b3
                        self.Ploopl = self.Ploopl.at[:, 17].set(self.P22l[:, 2])  # *b1*b4
                        self.Ploopl = self.Ploopl.at[:, 18].set(self.P22l[:, 8])  # *b1*b4*f
                        self.Ploopl = self.Ploopl.at[:, 19].set(self.P22l[:, 3])  # *b2*b2
                        self.Ploopl = self.Ploopl.at[:, 20].set(self.P22l[:, 4])  # *b2*b4
                        self.Ploopl = self.Ploopl.at[:, 21].set(self.P22l[:, 5])  # *b4*b4

                        self.Cloopl = self.Cloopl.at[:, 0].set(self.C22l[:, 20] + self.C13l[:, 7])   # *f^2
                        self.Cloopl = self.Cloopl.at[:, 1].set(self.C22l[:, 23] + self.C22l[:, 24] + self.C13l[:, 8] + self.C13l[:, 9])   # *f^3
                        self.Cloopl = self.Cloopl.at[:, 2].set(self.C22l[:, 25] + self.C22l[:, 26] + self.C22l[:, 27])   # *f^4
                        self.Cloopl = self.Cloopl.at[:, 3].set(self.C22l[:, 9] + self.C13l[:, 3])  # *b1*f
                        self.Cloopl = self.Cloopl.at[:, 4].set(self.C22l[:, 14] + self.C22l[:, 15] + self.C13l[:, 5] + self.C13l[:, 6])   # *b1*f^2
                        self.Cloopl = self.Cloopl.at[:, 5].set(self.C22l[:, 21] + self.C22l[:, 22])   # *b1*f^3
                        self.Cloopl = self.Cloopl.at[:, 6].set(self.C22l[:, 10])   # *b2*f
                        self.Cloopl = self.Cloopl.at[:, 7].set(self.C22l[:, 16] + self.C22l[:, 17])  # *b2*f^2
                        self.Cloopl = self.Cloopl.at[:, 8].set(self.C13l[:, 4])   # *b3*f
                        self.Cloopl = self.Cloopl.at[:, 9].set(self.C22l[:, 11])   # *b4*f
                        self.Cloopl = self.Cloopl.at[:, 10].set(self.C22l[:, 18] + self.C22l[:, 19])   # *b4*f^2
                        self.Cloopl = self.Cloopl.at[:, 11].set(self.C22l[:, 0] + self.C13l[:, 0])   # *b1*b1
                        self.Cloopl = self.Cloopl.at[:, 12].set(self.C22l[:, 6] + self.C13l[:, 2])   # *b1*b1*f
                        self.Cloopl = self.Cloopl.at[:, 13].set(self.C22l[:, 12] + self.C22l[:, 13])   # *b1*b1*f^2
                        self.Cloopl = self.Cloopl.at[:, 14].set(self.C22l[:, 1])   # *b1*b2
                        self.Cloopl = self.Cloopl.at[:, 15].set(self.C22l[:, 7])   # *b1*b2*f
                        self.Cloopl = self.Cloopl.at[:, 16].set(self.C13l[:, 1])  # *b1*b3
                        self.Cloopl = self.Cloopl.at[:, 17].set(self.C22l[:, 2])   # *b1*b4
                        self.Cloopl = self.Cloopl.at[:, 18].set(self.C22l[:, 8])   # *b1*b4*f
                        self.Cloopl = self.Cloopl.at[:, 19].set(self.C22l[:, 3])  # *b2*b2
                        self.Cloopl = self.Cloopl.at[:, 20].set(self.C22l[:, 4])  # *b2*b4
                        self.Cloopl = self.Cloopl.at[:, 21].set(self.C22l[:, 5])  # *b4*b4

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl = self.Ploopl.at[:, 2].add(self.P13l[:, -1]) # *f^4
                            self.Ploopl = self.Ploopl.at[:, 5].add(self.P13l[:, -2]) # *b1*f^3
                            self.Ploopl = self.Ploopl.at[:, 13].add(self.P13l[:, -3]) # *b1*b1*f^2

                            self.Cloopl = self.Cloopl.at[:, 2].add(self.C13l[:, -1]) # *f^4
                            self.Cloopl = self.Cloopl.at[:, 5].add(self.C13l[:, -2]) # *b1*f^3
                            self.Cloopl = self.Cloopl.at[:, 13].add(self.C13l[:, -3]) # *b1*b1*f^2

                    elif self.co.Nloop == 18: # config["with_tidal_alignements"] = True
                        f1 = self.f

                        self.Ploopl = self.Ploopl.at[:, 0].set(f1**2 * self.P22l[:, 36] + f1**3 * self.P22l[:, 39] + f1**3 * self.P22l[:, 40] + f1**4 * self.P22l[:, 41] + f1**4 * self.P22l[:, 42] + f1**4 * self.P22l[:, 43] + f1**2 * self.P13l[:, 21] + f1**3 * self.P13l[:, 22] + f1**3 * self.P13l[:, 23])  # *1
                        self.Ploopl = self.Ploopl.at[:, 1].set(f1 * self.P22l[:, 20] + f1**2 * self.P22l[:, 27] + f1**2 * self.P22l[:, 28] + f1**3 * self.P22l[:, 37] + f1**3 * self.P22l[:, 38] + f1 * self.P13l[:, 12] + f1**2 * self.P13l[:, 16] + f1**2 * self.P13l[:, 17])  # *b1
                        self.Ploopl = self.Ploopl.at[:, 2].set(f1 * self.P22l[:, 21] + f1**2 * self.P22l[:, 29] + f1**2 * self.P22l[:, 30])  # *b2
                        self.Ploopl = self.Ploopl.at[:, 3].set(f1 * self.P13l[:, 13])  # *b3
                        self.Ploopl = self.Ploopl.at[:, 4].set(f1 * self.P22l[:, 22] + f1**2 * self.P22l[:, 31] + f1**2 * self.P22l[:, 32])  # *b4
                        self.Ploopl = self.Ploopl.at[:, 5].set(self.P22l[:, 5] + f1 * self.P22l[:, 15] + f1**2 * self.P22l[:, 25] + f1**2 * self.P22l[:, 26] + self.P13l[:, 4] + f1 * self.P13l[:, 9])  # *b1*b1
                        self.Ploopl = self.Ploopl.at[:, 6].set(self.P22l[:, 6] + f1 * self.P22l[:, 16])  # *b1*b2
                        self.Ploopl = self.Ploopl.at[:, 7].set(self.P13l[:, 5])  # *b1*b3
                        self.Ploopl = self.Ploopl.at[:, 8].set(self.P22l[:, 7] + f1 * self.P22l[:, 17])  # *b1*b4
                        self.Ploopl = self.Ploopl.at[:, 9].set(self.P22l[:, 9])  # *b2*b2
                        self.Ploopl = self.Ploopl.at[:, 10].set(self.P22l[:, 10])  # *b2*b4
                        self.Ploopl = self.Ploopl.at[:, 11].set(self.P22l[:, 12])  # *b4*b4

                        self.Ploopl = self.Ploopl.at[:, 12].set(f1 * self.P22l[:, 23] + f1 * self.P22l[:, 24] + f1**2 * self.P22l[:, 33] + f1**2 * self.P22l[:, 34] + f1**2 * self.P22l[:, 35] + f1 * self.P13l[:, 14] + f1 * self.P13l[:, 15] + f1**2 * self.P13l[:, 18] + f1**2 * self.P13l[:, 19] + f1**2 * self.P13l[:, 20]) # *bq
                        self.Ploopl = self.Ploopl.at[:, 13].set(self.P22l[:, 3] + self.P22l[:, 4] + self.P22l[:, 14] + self.P13l[:, 2] + self.P13l[:, 3] + self.P13l[:, 8]) # *bq*bq
                        self.Ploopl = self.Ploopl.at[:, 14].set(self.P22l[:, 0] + self.P22l[:, 8] + f1 * self.P22l[:, 18] + f1 * self.P22l[:, 19] + self.P13l[:, 0] + self.P13l[:, 6] + f1 * self.P13l[:, 10] + f1 * self.P13l[:, 11]) # *bq*b1
                        self.Ploopl = self.Ploopl.at[:, 15].set(self.P22l[:, 1] + self.P22l[:, 11]) # *bq*b2
                        self.Ploopl = self.Ploopl.at[:, 16].set(self.P13l[:, 1] + self.P13l[:, 7]) # *bq*b3
                        self.Ploopl = self.Ploopl.at[:, 17].set(self.P22l[:, 2] + self.P22l[:, 13]) # *bq*b4

                        self.Cloopl = self.Cloopl.at[:, 0].set(f1**2 * self.C22l[:, 36] + f1**3 * self.C22l[:, 39] + f1**3 * self.C22l[:, 40] + f1**4 * self.C22l[:, 41] + f1**4 * self.C22l[:, 42] + f1**4 * self.C22l[:, 43] + f1**2 * self.C13l[:, 21] + f1**3 * self.C13l[:, 22] + f1**3 * self.C13l[:, 23])  # *1
                        self.Cloopl = self.Cloopl.at[:, 1].set(f1 * self.C22l[:, 20] + f1**2 * self.C22l[:, 27] + f1**2 * self.C22l[:, 28] + f1**3 * self.C22l[:, 37] + f1**3 * self.C22l[:, 38] + f1 * self.C13l[:, 12] + f1**2 * self.C13l[:, 16] + f1**2 * self.C13l[:, 17])  # *b1
                        self.Cloopl = self.Cloopl.at[:, 2].set(f1 * self.C22l[:, 21] + f1**2 * self.C22l[:, 29] + f1**2 * self.C22l[:, 30])  # *b2
                        self.Cloopl = self.Cloopl.at[:, 3].set(f1 * self.C13l[:, 13])  # *b3
                        self.Cloopl = self.Cloopl.at[:, 4].set(f1 * self.C22l[:, 22] + f1**2 * self.C22l[:, 31] + f1**2 * self.C22l[:, 32])  # *b4
                        self.Cloopl = self.Cloopl.at[:, 5].set(self.C22l[:, 5] + f1 * self.C22l[:, 15] + f1**2 * self.C22l[:, 25] + f1**2 * self.C22l[:, 26] + self.C13l[:, 4] + f1 * self.C13l[:, 9])  # *b1*b1
                        self.Cloopl = self.Cloopl.at[:, 6].set(self.C22l[:, 6] + f1 * self.C22l[:, 16])  # *b1*b2
                        self.Cloopl = self.Cloopl.at[:, 7].set(self.C13l[:, 5])  # *b1*b3
                        self.Cloopl = self.Cloopl.at[:, 8].set(self.C22l[:, 7] + f1 * self.C22l[:, 17])  # *b1*b4
                        self.Cloopl = self.Cloopl.at[:, 9].set(self.C22l[:, 9])  # *b2*b2
                        self.Cloopl = self.Cloopl.at[:, 10].set(self.C22l[:, 10])  # *b2*b4
                        self.Cloopl = self.Cloopl.at[:, 11].set(self.C22l[:, 12])  # *b4*b4

                        self.Cloopl = self.Cloopl.at[:, 12].set(f1 * self.C22l[:, 23] + f1 * self.C22l[:, 24] + f1**2 * self.C22l[:, 33] + f1**2 * self.C22l[:, 34] + f1**2 * self.C22l[:, 35] + f1 * self.C13l[:, 14] + f1 * self.C13l[:, 15] + f1**2 * self.C13l[:, 18] + f1**2 * self.C13l[:, 19] + f1**2 * self.C13l[:, 20]) # *bq
                        self.Cloopl = self.Cloopl.at[:, 13].set(self.C22l[:, 3] + self.C22l[:, 4] + self.C22l[:, 14] + self.C13l[:, 2] + self.C13l[:, 3] + self.C13l[:, 8]) # *bq*bq
                        self.Cloopl = self.Cloopl.at[:, 14].set(self.C22l[:, 0] + self.C22l[:, 8] + f1 * self.C22l[:, 18] + f1 * self.C22l[:, 19] + self.C13l[:, 0] + self.C13l[:, 6] + f1 * self.C13l[:, 10] + f1 * self.C13l[:, 11]) # *bq*b1
                        self.Cloopl = self.Cloopl.at[:, 15].set(self.C22l[:, 1] + self.C22l[:, 11]) # *bq*b2
                        self.Cloopl = self.Cloopl.at[:, 16].set(self.C13l[:, 1] + self.C13l[:, 7]) # *bq*b3
                        self.Cloopl = self.Cloopl.at[:, 17].set(self.C22l[:, 2] + self.C22l[:, 13]) # *bq*b4

                    if self.co.Nloop == self.co.N22 + self.co.N13:
                        self.Ploopl[:, :self.co.N22] = self.P22l
                        self.Ploopl[:, self.co.N22:] = self.P13l
                        self.Cloopl[:, :self.co.N22] = self.C22l
                        self.Cloopl[:, self.co.N22:] = self.C13l

            else: # halo-matter
                if self.co.Nloop == 5:
                    f1 = self.f

                    self.Ploopl = self.Ploopl.at[:, 0].set(f1**2 * self.P22l[:, 6] + f1**2 * self.P22l[:, 13] + f1**2 * self.P22l[:, 14] + f1**3 * self.P22l[:, 17] + f1**3 * self.P22l[:, 18] + f1**4 * self.P22l[:, 19] + f1**4 * self.P22l[:, 20] + f1**4 * self.P22l[:, 21] + \
                        f1 * self.P13l[:, 4] + f1**2 * self.P13l[:, 7] + f1**2 * self.P13l[:, 8] + f1**3 * self.P13l[:, 9] + f1**3 * self.P13l[:, 10]) # *1
                    self.Ploopl = self.Ploopl.at[:, 1].set(self.P22l[:, 0] + f1**2 * self.P22l[:, 7] + f1**2 * self.P22l[:, 8]  + f1**3 * self.P22l[:, 15] + f1**3 * self.P22l[:, 16] + f1 * self.P13l[:, 2] + f1**2 * self.P13l[:, 5] + f1**2 * self.P13l[:, 6]) # *b1
                    self.Ploopl = self.Ploopl.at[:, 2].set(self.P22l[:, 1] + f1 * self.P22l[:, 4] + f1**2 * self.P22l[:, 9] + f1**2 * self.P22l[:, 10]) # *b2
                    self.Ploopl = self.Ploopl.at[:, 3].set(self.P13l[:, 1] + f1 * self.P13l[:, 3]) # *b3
                    self.Ploopl = self.Ploopl.at[:, 4].set(self.P22l[:, 2] + f1 * self.P22l[:, 5] + f1**2 * self.P22l[:, 11] + f1**2 * self.P22l[:, 12]) # *b4

                    self.Cloopl = self.Cloopl.at[:, 0].set(f1**2 * self.C22l[:, 6] + f1**2 * self.C22l[:, 13] + f1**2 * self.C22l[:, 14] + f1**3 * self.C22l[:, 17] + f1**3 * self.C22l[:, 18] + f1**4 * self.C22l[:, 19] + f1**4 * self.C22l[:, 20] + f1**4 * self.C22l[:, 21] + \
                        f1 * self.C13l[:, 4] + f1**2 * self.C13l[:, 7] + f1**2 * self.C13l[:, 8] + f1**3 * self.C13l[:, 9] + f1**3 * self.C13l[:, 10]) # *1
                    self.Cloopl = self.Cloopl.at[:, 1].set(self.C22l[:, 0] + f1**2 * self.C22l[:, 7] + f1**2 * self.C22l[:, 8]  + f1**3 * self.C22l[:, 15] + f1**3 * self.C22l[:, 16] + f1 * self.C13l[:, 2] + f1**2 * self.C13l[:, 5] + f1**2 * self.C13l[:, 6]) # *b1
                    self.Cloopl = self.Cloopl.at[:, 2].set(self.C22l[:, 1] + f1 * self.C22l[:, 4] + f1**2 * self.C22l[:, 9] + f1**2 * self.C22l[:, 10]) # *b2
                    self.Cloopl = self.Cloopl.at[:, 3].set(self.C13l[:, 1] + f1 * self.C13l[:, 3]) # *b3
                    self.Cloopl = self.Cloopl.at[:, 4].set(self.C22l[:, 2] + f1 * self.C22l[:, 5] + f1**2 * self.C22l[:, 11] + f1**2 * self.C22l[:, 12]) # *b4

                elif self.co.Nloop == 25:
                    pass

        ### numpy ###
        else:
            if self.co.halohalo:

                if self.co.exact_time:      # config["with_exact_time"] == True
                    if self.co.Nloop == 12: # config["with_time"] == True
                        f1 = self.f

                        ## EdS: Y1 = 0., G1t = 3/7., V12t = 1/7.
                        G1 = self.G1
                        Y1 = self.Y1
                        G1t = self.G1t
                        V12t = self.V12t

                        self.Ploopl[:, 0] = G1**2 * f1**2 * self.P22l[:, 20] + G1 * f1**3 * self.P22l[:, 23] + G1 * f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + \
                            G1 * G1t * f1**2 * self.P22l[:, 32] + G1t * f1**3 * self.P22l[:, 33] + G1t * f1**3 * self.P22l[:, 34] + G1t**2 * f1**2 * self.P22l[:, 35] + \
                            G1**2 * f1**2 * self.P13l[:, 4] + Y1 * f1**2 * self.P13l[:, 7] + G1t * f1**3 * self.P13l[:, 11] + G1t * f1**3 * self.P13l[:, 12] + V12t * f1**2 * self.P13l[:, 14] # *1
                        self.Ploopl[:, 1] = G1**2 * f1 * self.P22l[:, 9] + G1 * f1**2 * self.P22l[:, 14] + G1 * f1**2 * self.P22l[:, 15] + f1**3 * self.P22l[:, 21] + f1**3 * self.P22l[:, 22] + G1 * G1t * f1 * self.P22l[:, 28] + G1t * f1**2 * self.P22l[:, 31] + \
                            G1**2 * f1 * self.P13l[:, 2] + Y1 * f1 * self.P13l[:, 6] + G1t * f1**2 * self.P13l[:, 9] + G1t * f1**2 * self.P13l[:, 10] + V12t * f1 * self.P13l[:, 13] # *b1
                        self.Ploopl[:, 2] = G1 * f1 * self.P22l[:, 10] + f1**2 * self.P22l[:, 16] + f1**2 * self.P22l[:, 17] + G1t * f1 * self.P22l[:, 29] # *b2
                        self.Ploopl[:, 3] = f1 * self.P13l[:, 3] # *b3
                        self.Ploopl[:, 4] = G1 * f1 * self.P22l[:, 11] + f1**2 * self.P22l[:, 18] + f1**2 * self.P22l[:, 19] + G1t * f1 * self.P22l[:, 30] # *b4
                        self.Ploopl[:, 5] = G1**2 * self.P22l[:, 0] + G1 * f1 * self.P22l[:, 6] + f1**2 * self.P22l[:, 12] + f1**2 * self.P22l[:, 13] + G1**2 * self.P13l[:, 0] + Y1 * self.P13l[:, 5] + G1t * f1 * self.P13l[:, 8]  # *b1*b1
                        self.Ploopl[:, 6] = G1 * self.P22l[:, 1] + f1 * self.P22l[:, 7]  # *b1*b2
                        self.Ploopl[:, 7] = self.P13l[:, 1]  # *b1*b3
                        self.Ploopl[:, 8] = G1 * self.P22l[:, 2] + f1 * self.P22l[:, 8]  # *b1*b4
                        self.Ploopl[:, 9] = self.P22l[:, 3]  # *b2*b2
                        self.Ploopl[:, 10] = self.P22l[:, 4]  # *b2*b4
                        self.Ploopl[:, 11] = self.P22l[:, 5]  # *b4*b4

                        self.Cloopl[:, 0] = G1**2 * f1**2 * self.C22l[:, 20] + G1 * f1**3 * self.C22l[:, 23] + G1 * f1**3 * self.C22l[:, 24] + f1**4 * self.C22l[:, 25] + f1**4 * self.C22l[:, 26] + f1**4 * self.C22l[:, 27] + \
                            G1 * G1t * f1**2 * self.C22l[:, 32] + G1t * f1**3 * self.C22l[:, 33] + G1t * f1**3 * self.C22l[:, 34] + G1t**2 * f1**2 * self.C22l[:, 35] + \
                            G1**2 * f1**2 * self.C13l[:, 4] + Y1 * f1**2 * self.C13l[:, 7] + G1t * f1**3 * self.C13l[:, 11] + G1t * f1**3 * self.C13l[:, 12] + V12t * f1**2 * self.C13l[:, 14] # *1
                        self.Cloopl[:, 1] = G1**2 * f1 * self.C22l[:, 9] + G1 * f1**2 * self.C22l[:, 14] + G1 * f1**2 * self.C22l[:, 15] + f1**3 * self.C22l[:, 21] + f1**3 * self.C22l[:, 22] + G1 * G1t * f1 * self.C22l[:, 28] + G1t * f1**2 * self.C22l[:, 31] + \
                            G1**2 * f1 * self.C13l[:, 2] + Y1 * f1 * self.C13l[:, 6] + G1t * f1**2 * self.C13l[:, 9] + G1t * f1**2 * self.C13l[:, 10] + V12t * f1 * self.C13l[:, 13] # *b1
                        self.Cloopl[:, 2] = G1 * f1 * self.C22l[:, 10] + f1**2 * self.C22l[:, 16] + f1**2 * self.C22l[:, 17] + G1t * f1 * self.C22l[:, 29] # *b2
                        self.Cloopl[:, 3] = f1 * self.C13l[:, 3] # *b3
                        self.Cloopl[:, 4] = G1 * f1 * self.C22l[:, 11] + f1**2 * self.C22l[:, 18] + f1**2 * self.C22l[:, 19] + G1t * f1 * self.C22l[:, 30] # *b4
                        self.Cloopl[:, 5] = G1**2 * self.C22l[:, 0] + G1 * f1 * self.C22l[:, 6] + f1**2 * self.C22l[:, 12] + f1**2 * self.C22l[:, 13] + G1**2 * self.C13l[:, 0] + Y1 * self.C13l[:, 5] + G1t * f1 * self.C13l[:, 8]  # *b1*b1
                        self.Cloopl[:, 6] = G1 * self.C22l[:, 1] + f1 * self.C22l[:, 7]  # *b1*b2
                        self.Cloopl[:, 7] = self.C13l[:, 1]  # *b1*b3
                        self.Cloopl[:, 8] = G1 * self.C22l[:, 2] + f1 * self.C22l[:, 8]  # *b1*b4
                        self.Cloopl[:, 9] = self.C22l[:, 3]  # *b2*b2
                        self.Cloopl[:, 10] = self.C22l[:, 4]  # *b2*b4
                        self.Cloopl[:, 11] = self.C22l[:, 5]  # *b4*b4

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl[:, 0] += f1**4 * self.P13l[:, -1] + f1**3 * self.P13l[:, -2] # *1
                            self.Ploopl[:, 1] += f1**3 * self.P13l[:, -3] + f1**2 * self.P13l[:, -4] # *b1
                            self.Ploopl[:, 5] += f1**2 * self.P13l[:, -5] + f1 * self.P13l[:, -6] # *b1*b1

                            self.Cloopl[:, 0] += f1**4 * self.C13l[:, -1] + f1**3 * self.C13l[:, -2] # *1
                            self.Cloopl[:, 1] += f1**3 * self.C13l[:, -3] + f1**2 * self.C13l[:, -4] # *b1
                            self.Cloopl[:, 5] += f1**2 * self.C13l[:, -5] + f1 * self.C13l[:, -6] # *b1*b1

                    elif self.co.Nloop == 35: # config["with_time"] == False
                        self.Ploopl[:, 0] = self.P22l[:, 20] + self.P13l[:, 4]   # *f^2
                        self.Ploopl[:, 1] = self.P22l[:, 32]    # *f^2*G1t
                        self.Ploopl[:, 2] = self.P22l[:, 35]    # *f^2*G1t**2
                        self.Ploopl[:, 3] = self.P13l[:, 7]     # *f^2*Y1
                        self.Ploopl[:, 4] = self.P13l[:, 14]    # *f^2*V12t
                        self.Ploopl[:, 5] = self.P22l[:, 23] + self.P22l[:, 24]   # *f^3
                        self.Ploopl[:, 6] = self.P22l[:, 33] + self.P22l[:, 34] + self.P13l[:, 11] + self.P13l[:, 12] # *f^3*G1t
                        self.Ploopl[:, 7] = self.P22l[:, 25] + self.P22l[:, 26] + self.P22l[:, 27]   # *f^4
                        self.Ploopl[:, 8] = self.P22l[:, 9] + self.P13l[:, 2]  # *b1*f
                        self.Ploopl[:, 9] = self.P22l[:, 28]    # *b1*f*G1t
                        self.Ploopl[:, 10] = self.P13l[:, 6]     # *b1*f*Y1
                        self.Ploopl[:, 11] = self.P13l[:, 13]    # *b1*f*V12t
                        self.Ploopl[:, 12] = self.P22l[:, 14] + self.P22l[:, 15]     # *b1*f^2
                        self.Ploopl[:, 13] = self.P22l[:, 31] + self.P13l[:, 9] + self.P13l[:, 10]   # *b1*f^2*G1t
                        self.Ploopl[:, 14] = self.P22l[:, 21] + self.P22l[:, 22]     # *b1*f^3
                        self.Ploopl[:, 15] = self.P22l[:, 10]    # *b2*f
                        self.Ploopl[:, 16] = self.P22l[:, 29]    # *b2*f*G1t
                        self.Ploopl[:, 17] = self.P22l[:, 16] + self.P22l[:, 17]     # *b2*f^2
                        self.Ploopl[:, 18] = self.P13l[:, 3]     # *b3*f
                        self.Ploopl[:, 19] = self.P22l[:, 11]    # *b4*f
                        self.Ploopl[:, 20] = self.P22l[:, 30]    # *b4*f*G1t
                        self.Ploopl[:, 21] = self.P22l[:, 18] + self.P22l[:, 19]     # *b4*f^2
                        self.Ploopl[:, 22] = self.P22l[:, 0] + self.P13l[:, 0]   # *b1^2
                        self.Ploopl[:, 23] = self.P13l[:, 5]     # *b1^2*Y1
                        self.Ploopl[:, 24] = self.P22l[:, 6]     # *b1^2*f
                        self.Ploopl[:, 25] = self.P13l[:, 8]     # *b1^2*f*G1t
                        self.Ploopl[:, 26] = self.P22l[:, 12] + self.P22l[:, 13]     # *b1^2*f^2
                        self.Ploopl[:, 27] = self.P22l[:, 1]     # *b1*b2
                        self.Ploopl[:, 28] = self.P22l[:, 7]     # *b1*b2*f
                        self.Ploopl[:, 29] = self.P13l[:, 1]     # *b1*b3
                        self.Ploopl[:, 30] = self.P22l[:, 2]     # *b1*b4
                        self.Ploopl[:, 31] = self.P22l[:, 8]     # *b1*b4*f
                        self.Ploopl[:, 32] = self.P22l[:, 3]     # *b2^2
                        self.Ploopl[:, 33] = self.P22l[:, 4]     # *b2*b4
                        self.Ploopl[:, 34] = self.P22l[:, 5]     # *b4^2

                        self.Cloopl[:, 0] = self.C22l[:, 20] + self.C13l[:, 4]   # *f^2
                        self.Cloopl[:, 1] = self.C22l[:, 32]    # *f^2*G1t
                        self.Cloopl[:, 2] = self.C22l[:, 35]    # *f^2*G1t**2
                        self.Cloopl[:, 3] = self.C13l[:, 7]     # *f^2*Y1
                        self.Cloopl[:, 4] = self.C13l[:, 14]    # *f^2*V12t
                        self.Cloopl[:, 5] = self.C22l[:, 23] + self.C22l[:, 24]   # *f^3
                        self.Cloopl[:, 6] = self.C22l[:, 33] + self.C22l[:, 34] + self.C13l[:, 11] + self.C13l[:, 12] # *f^3*G1t
                        self.Cloopl[:, 7] = self.C22l[:, 25] + self.C22l[:, 26] + self.C22l[:, 27]   # *f^4
                        self.Cloopl[:, 8] = self.C22l[:, 9] + self.C13l[:, 2]  # *b1*f
                        self.Cloopl[:, 9] = self.C22l[:, 28]    # *b1*f*G1t
                        self.Cloopl[:, 10] = self.C13l[:, 6]     # *b1*f*Y1
                        self.Cloopl[:, 11] = self.C13l[:, 13]    # *b1*f*V12t
                        self.Cloopl[:, 12] = self.C22l[:, 14] + self.C22l[:, 15]     # *b1*f^2
                        self.Cloopl[:, 13] = self.C22l[:, 31] + self.C13l[:, 9] + self.C13l[:, 10]   # *b1*f^2*G1t
                        self.Cloopl[:, 14] = self.C22l[:, 21] + self.C22l[:, 22]     # *b1*f^3
                        self.Cloopl[:, 15] = self.C22l[:, 10]    # *b2*f
                        self.Cloopl[:, 16] = self.C22l[:, 29]    # *b2*f*G1t
                        self.Cloopl[:, 17] = self.C22l[:, 16] + self.C22l[:, 17]     # *b2*f^2
                        self.Cloopl[:, 18] = self.C13l[:, 3]     # *b3*f
                        self.Cloopl[:, 19] = self.C22l[:, 11]    # *b4*f
                        self.Cloopl[:, 20] = self.C22l[:, 30]    # *b4*f*G1t
                        self.Cloopl[:, 21] = self.C22l[:, 18] + self.C22l[:, 19]     # *b4*f^2
                        self.Cloopl[:, 22] = self.C22l[:, 0] + self.C13l[:, 0]   # *b1^2
                        self.Cloopl[:, 23] = self.C13l[:, 5]     # *b1^2*Y1
                        self.Cloopl[:, 24] = self.C22l[:, 6]     # *b1^2*f
                        self.Cloopl[:, 25] = self.C13l[:, 8]     # *b1^2*f*G1t
                        self.Cloopl[:, 26] = self.C22l[:, 12] + self.C22l[:, 13]     # *b1^2*f^2
                        self.Cloopl[:, 27] = self.C22l[:, 1]     # *b1*b2
                        self.Cloopl[:, 28] = self.C22l[:, 7]     # *b1*b2*f
                        self.Cloopl[:, 29] = self.C13l[:, 1]     # *b1*b3
                        self.Cloopl[:, 30] = self.C22l[:, 2]     # *b1*b4
                        self.Cloopl[:, 31] = self.C22l[:, 8]     # *b1*b4*f
                        self.Cloopl[:, 32] = self.C22l[:, 3]     # *b2^2
                        self.Cloopl[:, 33] = self.C22l[:, 4]     # *b2*b4
                        self.Cloopl[:, 34] = self.C22l[:, 5]     # *b4^2

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl[:, 7] += self.P13l[:, -1] # *f^4
                            self.Ploopl[:, 5] += self.P13l[:, -2] # *f^3
                            self.Ploopl[:, 14] += self.P13l[:, -3] # *b1*f^3
                            self.Ploopl[:, 12] += self.P13l[:, -4] # *b1*f^2
                            self.Ploopl[:, 26] += self.P13l[:, -5] # *b1*b1*f^2
                            self.Ploopl[:, 24] += self.P13l[:, -6] # *b1*b1*f

                            self.Cloopl[:, 7] += self.C13l[:, -1] # *f^4
                            self.Cloopl[:, 5] += self.C13l[:, -2] # *f^3
                            self.Cloopl[:, 14] += self.C13l[:, -3] # *b1*f^3
                            self.Cloopl[:, 12] += self.C13l[:, -4] # *b1*f^2
                            self.Cloopl[:, 26] += self.C13l[:, -5] # *b1*b1*f^2
                            self.Cloopl[:, 24] += self.C13l[:, -6] # *b1*b1*f

                else:                       # config["with_exact_time"] == False
                    if self.co.Nloop == 12: # config["with_time"] == True
                        f1 = self.f

                        self.Ploopl[:, 0] = f1**2 * self.P22l[:, 20] + f1**3 * self.P22l[:, 23] + f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + f1**2 * self.P13l[:, 7] + f1**3 * self.P13l[:, 8] + f1**3 * self.P13l[:, 9]  # *1
                        self.Ploopl[:, 1] = f1 * self.P22l[:, 9] + f1**2 * self.P22l[:, 14] + f1**2 * self.P22l[:, 15] + f1**3 * self.P22l[:, 21] + f1**3 * self.P22l[:, 22] + f1 * self.P13l[:, 3] + f1**2 * self.P13l[:, 5] + f1**2 * self.P13l[:, 6]  # *b1
                        self.Ploopl[:, 2] = f1 * self.P22l[:, 10] + f1**2 * self.P22l[:, 16] + f1**2 * self.P22l[:, 17]  # *b2
                        self.Ploopl[:, 3] = f1 * self.P13l[:, 4]  # *b3
                        self.Ploopl[:, 4] = f1 * self.P22l[:, 11] + f1**2 * self.P22l[:, 18] + f1**2 * self.P22l[:, 19]  # *b4
                        self.Ploopl[:, 5] = self.P22l[:, 0] + f1 * self.P22l[:, 6] + f1**2 * self.P22l[:, 12] + f1**2 * self.P22l[:, 13] + self.P13l[:, 0] + f1 * self.P13l[:, 2]  # *b1*b1
                        self.Ploopl[:, 6] = self.P22l[:, 1] + f1 * self.P22l[:, 7]  # *b1*b2
                        self.Ploopl[:, 7] = self.P13l[:, 1]  # *b1*b3
                        self.Ploopl[:, 8] = self.P22l[:, 2] + f1 * self.P22l[:, 8]  # *b1*b4
                        self.Ploopl[:, 9] = self.P22l[:, 3]  # *b2*b2
                        self.Ploopl[:, 10] = self.P22l[:, 4]  # *b2*b4
                        self.Ploopl[:, 11] = self.P22l[:, 5]  # *b4*b4

                        self.Cloopl[:, 0] = f1**2 * self.C22l[:, 20] + f1**3 * self.C22l[:, 23] + f1**3 * self.C22l[:, 24] + f1**4 * self.C22l[:, 25] + f1**4 * self.C22l[:, 26] + f1**4 * self.C22l[:, 27] + f1**2 * self.C13l[:, 7] + f1**3 * self.C13l[:, 8] + f1**3 * self.C13l[:, 9]  # *1
                        self.Cloopl[:, 1] = f1 * self.C22l[:, 9] + f1**2 * self.C22l[:, 14] + f1**2 * self.C22l[:, 15] + f1**3 * self.C22l[:, 21] + f1**3 * self.C22l[:, 22] + f1 * self.C13l[:, 3] + f1**2 * self.C13l[:, 5] + f1**2 * self.C13l[:, 6]  # *b1
                        self.Cloopl[:, 2] = f1 * self.C22l[:, 10] + f1**2 * self.C22l[:, 16] + f1**2 * self.C22l[:, 17]  # *b2
                        self.Cloopl[:, 3] = f1 * self.C13l[:, 4]  # *b3
                        self.Cloopl[:, 4] = f1 * self.C22l[:, 11] + f1**2 * self.C22l[:, 18] + f1**2 * self.C22l[:, 19]  # *b4
                        self.Cloopl[:, 5] = self.C22l[:, 0] + f1 * self.C22l[:, 6] + f1**2 * self.C22l[:, 12] + f1**2 * self.C22l[:, 13] + self.C13l[:, 0] + f1 * self.C13l[:, 2]  # *b1*b1
                        self.Cloopl[:, 6] = self.C22l[:, 1] + f1 * self.C22l[:, 7]  # *b1*b2
                        self.Cloopl[:, 7] = self.C13l[:, 1]  # *b1*b3
                        self.Cloopl[:, 8] = self.C22l[:, 2] + f1 * self.C22l[:, 8]  # *b1*b4
                        self.Cloopl[:, 9] = self.C22l[:, 3]  # *b2*b2
                        self.Cloopl[:, 10] = self.C22l[:, 4]  # *b2*b4
                        self.Cloopl[:, 11] = self.C22l[:, 5]  # *b4*b4

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl[:, 0] += f1**4 * self.P13l[:, -1] # *1
                            self.Ploopl[:, 1] += f1**3 * self.P13l[:, -2] # *b1
                            self.Ploopl[:, 5] += f1**2 * self.P13l[:, -3] # *b1*b1

                            self.Cloopl[:, 0] += f1**4 * self.C13l[:, -1] # *1
                            self.Cloopl[:, 1] += f1**3 * self.C13l[:, -2] # *b1
                            self.Cloopl[:, 5] += f1**2 * self.C13l[:, -3] # *b1*b1

                        # if self.co.angular: ### this is depreciated
                        #     self.Aloopl[:, 0] = f1**2 * self.A22l[:, 20] + f1**3 * self.A22l[:, 23] + f1**3 * self.A22l[:, 24] + f1**4 * self.A22l[:, 25] + \
                        #     f1**4 * self.A22l[:, 26] + f1**4 * self.A22l[:, 27] + f1**2 * \
                        #     self.A13l[:, 7] + f1**3 * self.A13l[:, 8] + f1**3 * self.A13l[:, 9]  # *1
                        #     self.Aloopl[:, 1] = f1 * self.A22l[:, 9] + f1**2 * self.A22l[:, 14] + f1**2 * self.A22l[:, 15] + f1**3 * self.A22l[:, 21] + f1**3 * self.A22l[:, 22] + f1 * self.A13l[:, 3] + f1**2 * self.A13l[:, 5] + f1**2 * self.A13l[:, 6]  # *b1
                        #     self.Aloopl[:, 2] = f1 * self.A22l[:, 10] + f1**2 * self.A22l[:, 16] + f1**2 * self.A22l[:, 17]  # *b2
                        #     self.Aloopl[:, 3] = f1 * self.A13l[:, 4]  # *b3
                        #     self.Aloopl[:, 4] = f1 * self.A22l[:, 11] + f1**2 * self.A22l[:, 18] + f1**2 * self.A22l[:, 19]  # *b4
                        #     self.Aloopl[:, 5] = self.A22l[:, 0] + f1 * self.A22l[:, 6] + f1**2 * self.A22l[:, 12] + \
                        #         f1**2 * self.A22l[:, 13] + self.A13l[:, 0] + f1 * self.A13l[:, 2]  # *b1*b1
                        #     self.Aloopl[:, 6] = self.A22l[:, 1] + f1 * self.A22l[:, 7]  # *b1*b2
                        #     self.Aloopl[:, 7] = self.A13l[:, 1]  # *b1*b3
                        #     self.Aloopl[:, 8] = self.A22l[:, 2] + f1 * self.A22l[:, 8]  # *b1*b4
                        #     self.Aloopl[:, 9] = self.A22l[:, 3]  # *b2*b2
                        #     self.Aloopl[:, 10] = self.A22l[:, 4]  # *b2*b4
                        #     self.Aloopl[:, 11] = self.A22l[:, 5]  # *b4*b_4

                    elif self.co.Nloop == 22: # config["with_time"] == False
                        self.Ploopl[:, 0] = self.P22l[:, 20] + self.P13l[:, 7]   # *f^2
                        self.Ploopl[:, 1] = self.P22l[:, 23] + self.P22l[:, 24] + self.P13l[:, 8] + self.P13l[:, 9]   # *f^3
                        self.Ploopl[:, 2] = self.P22l[:, 25] + self.P22l[:, 26] + self.P22l[:, 27]   # *f^4
                        self.Ploopl[:, 3] = self.P22l[:, 9] + self.P13l[:, 3]  # *b1*f
                        self.Ploopl[:, 4] = self.P22l[:, 14] + self.P22l[:, 15] + self.P13l[:, 5] + self.P13l[:, 6]   # *b1*f^2
                        self.Ploopl[:, 5] = self.P22l[:, 21] + self.P22l[:, 22]   # *b1*f^3
                        self.Ploopl[:, 6] = self.P22l[:, 10]   # *b2*f
                        self.Ploopl[:, 7] = self.P22l[:, 16] + self.P22l[:, 17]  # *b2*f^2
                        self.Ploopl[:, 8] = self.P13l[:, 4]   # *b3*f
                        self.Ploopl[:, 9] = self.P22l[:, 11]   # *b4*f
                        self.Ploopl[:, 10] = self.P22l[:, 18] + self.P22l[:, 19]   # *b4*f^2
                        self.Ploopl[:, 11] = self.P22l[:, 0] + self.P13l[:, 0]   # *b1*b1
                        self.Ploopl[:, 12] = self.P22l[:, 6] + self.P13l[:, 2]   # *b1*b1*f
                        self.Ploopl[:, 13] = self.P22l[:, 12] + self.P22l[:, 13]   # *b1*b1*f^2
                        self.Ploopl[:, 14] = self.P22l[:, 1]  # *b1*b2
                        self.Ploopl[:, 15] = self.P22l[:, 7]  # *b1*b2*f
                        self.Ploopl[:, 16] = self.P13l[:, 1]  # *b1*b3
                        self.Ploopl[:, 17] = self.P22l[:, 2]  # *b1*b4
                        self.Ploopl[:, 18] = self.P22l[:, 8]  # *b1*b4*f
                        self.Ploopl[:, 19] = self.P22l[:, 3]  # *b2*b2
                        self.Ploopl[:, 20] = self.P22l[:, 4]  # *b2*b4
                        self.Ploopl[:, 21] = self.P22l[:, 5]  # *b4*b4

                        self.Cloopl[:, 0] = self.C22l[:, 20] + self.C13l[:, 7]   # *f^2
                        self.Cloopl[:, 1] = self.C22l[:, 23] + self.C22l[:, 24] + self.C13l[:, 8] + self.C13l[:, 9]   # *f^3
                        self.Cloopl[:, 2] = self.C22l[:, 25] + self.C22l[:, 26] + self.C22l[:, 27]   # *f^4
                        self.Cloopl[:, 3] = self.C22l[:, 9] + self.C13l[:, 3]  # *b1*f
                        self.Cloopl[:, 4] = self.C22l[:, 14] + self.C22l[:, 15] + self.C13l[:, 5] + self.C13l[:, 6]   # *b1*f^2
                        self.Cloopl[:, 5] = self.C22l[:, 21] + self.C22l[:, 22]   # *b1*f^3
                        self.Cloopl[:, 6] = self.C22l[:, 10]   # *b2*f
                        self.Cloopl[:, 7] = self.C22l[:, 16] + self.C22l[:, 17]  # *b2*f^2
                        self.Cloopl[:, 8] = self.C13l[:, 4]   # *b3*f
                        self.Cloopl[:, 9] = self.C22l[:, 11]   # *b4*f
                        self.Cloopl[:, 10] = self.C22l[:, 18] + self.C22l[:, 19]   # *b4*f^2
                        self.Cloopl[:, 11] = self.C22l[:, 0] + self.C13l[:, 0]   # *b1*b1
                        self.Cloopl[:, 12] = self.C22l[:, 6] + self.C13l[:, 2]   # *b1*b1*f
                        self.Cloopl[:, 13] = self.C22l[:, 12] + self.C22l[:, 13]   # *b1*b1*f^2
                        self.Cloopl[:, 14] = self.C22l[:, 1]   # *b1*b2
                        self.Cloopl[:, 15] = self.C22l[:, 7]   # *b1*b2*f
                        self.Cloopl[:, 16] = self.C13l[:, 1]  # *b1*b3
                        self.Cloopl[:, 17] = self.C22l[:, 2]   # *b1*b4
                        self.Cloopl[:, 18] = self.C22l[:, 8]   # *b1*b4*f
                        self.Cloopl[:, 19] = self.C22l[:, 3]  # *b2*b2
                        self.Cloopl[:, 20] = self.C22l[:, 4]  # *b2*b4
                        self.Cloopl[:, 21] = self.C22l[:, 5]  # *b4*b4

                        if self.co.with_uvmatch or self.co.with_irmatch:
                            self.Ploopl[:, 2] += self.P13l[:, -1] # *f^4
                            self.Ploopl[:, 5] += self.P13l[:, -2] # *b1*f^3
                            self.Ploopl[:, 13] += self.P13l[:, -3] # *b1*b1*f^2

                            self.Cloopl[:, 2] += self.C13l[:, -1] # *f^4
                            self.Cloopl[:, 5] += self.C13l[:, -2] # *b1*f^3
                            self.Cloopl[:, 13] += self.C13l[:, -3] # *b1*b1*f^2

                    elif self.co.Nloop == 18: # config["with_tidal_alignements"] = True
                        f1 = self.f

                        self.Ploopl[:, 0] = f1**2 * self.P22l[:, 36] + f1**3 * self.P22l[:, 39] + f1**3 * self.P22l[:, 40] + f1**4 * self.P22l[:, 41] + f1**4 * self.P22l[:, 42] + f1**4 * self.P22l[:, 43] + f1**2 * self.P13l[:, 21] + f1**3 * self.P13l[:, 22] + f1**3 * self.P13l[:, 23]  # *1
                        self.Ploopl[:, 1] = f1 * self.P22l[:, 20] + f1**2 * self.P22l[:, 27] + f1**2 * self.P22l[:, 28] + f1**3 * self.P22l[:, 37] + f1**3 * self.P22l[:, 38] + f1 * self.P13l[:, 12] + f1**2 * self.P13l[:, 16] + f1**2 * self.P13l[:, 17]  # *b1
                        self.Ploopl[:, 2] = f1 * self.P22l[:, 21] + f1**2 * self.P22l[:, 29] + f1**2 * self.P22l[:, 30]  # *b2
                        self.Ploopl[:, 3] = f1 * self.P13l[:, 13]  # *b3
                        self.Ploopl[:, 4] = f1 * self.P22l[:, 22] + f1**2 * self.P22l[:, 31] + f1**2 * self.P22l[:, 32]  # *b4
                        self.Ploopl[:, 5] = self.P22l[:, 5] + f1 * self.P22l[:, 15] + f1**2 * self.P22l[:, 25] + f1**2 * self.P22l[:, 26] + self.P13l[:, 4] + f1 * self.P13l[:, 9]  # *b1*b1
                        self.Ploopl[:, 6] = self.P22l[:, 6] + f1 * self.P22l[:, 16]  # *b1*b2
                        self.Ploopl[:, 7] = self.P13l[:, 5]  # *b1*b3
                        self.Ploopl[:, 8] = self.P22l[:, 7] + f1 * self.P22l[:, 17]  # *b1*b4
                        self.Ploopl[:, 9] = self.P22l[:, 9]  # *b2*b2
                        self.Ploopl[:, 10] = self.P22l[:, 10]  # *b2*b4
                        self.Ploopl[:, 11] = self.P22l[:, 12]  # *b4*b4

                        self.Ploopl[:, 12] = f1 * self.P22l[:, 23] + f1 * self.P22l[:, 24] + f1**2 * self.P22l[:, 33] + f1**2 * self.P22l[:, 34] + f1**2 * self.P22l[:, 35] + f1 * self.P13l[:, 14] + f1 * self.P13l[:, 15] + f1**2 * self.P13l[:, 18] + f1**2 * self.P13l[:, 19] + f1**2 * self.P13l[:, 20] # *bq
                        self.Ploopl[:, 13] = self.P22l[:, 3] + self.P22l[:, 4] + self.P22l[:, 14] + self.P13l[:, 2] + self.P13l[:, 3] + self.P13l[:, 8] # *bq*bq
                        self.Ploopl[:, 14] = self.P22l[:, 0] + self.P22l[:, 8] + f1 * self.P22l[:, 18] + f1 * self.P22l[:, 19] + self.P13l[:, 0] + self.P13l[:, 6] + f1 * self.P13l[:, 10] + f1 * self.P13l[:, 11] # *bq*b1
                        self.Ploopl[:, 15] = self.P22l[:, 1] + self.P22l[:, 11] # *bq*b2
                        self.Ploopl[:, 16] = self.P13l[:, 1] + self.P13l[:, 7] # *bq*b3
                        self.Ploopl[:, 17] = self.P22l[:, 2] + self.P22l[:, 13] # *bq*b4

                        self.Cloopl[:, 0] = f1**2 * self.C22l[:, 36] + f1**3 * self.C22l[:, 39] + f1**3 * self.C22l[:, 40] + f1**4 * self.C22l[:, 41] + f1**4 * self.C22l[:, 42] + f1**4 * self.C22l[:, 43] + f1**2 * self.C13l[:, 21] + f1**3 * self.C13l[:, 22] + f1**3 * self.C13l[:, 23]  # *1
                        self.Cloopl[:, 1] = f1 * self.C22l[:, 20] + f1**2 * self.C22l[:, 27] + f1**2 * self.C22l[:, 28] + f1**3 * self.C22l[:, 37] + f1**3 * self.C22l[:, 38] + f1 * self.C13l[:, 12] + f1**2 * self.C13l[:, 16] + f1**2 * self.C13l[:, 17]  # *b1
                        self.Cloopl[:, 2] = f1 * self.C22l[:, 21] + f1**2 * self.C22l[:, 29] + f1**2 * self.C22l[:, 30]  # *b2
                        self.Cloopl[:, 3] = f1 * self.C13l[:, 13]  # *b3
                        self.Cloopl[:, 4] = f1 * self.C22l[:, 22] + f1**2 * self.C22l[:, 31] + f1**2 * self.C22l[:, 32]  # *b4
                        self.Cloopl[:, 5] = self.C22l[:, 5] + f1 * self.C22l[:, 15] + f1**2 * self.C22l[:, 25] + f1**2 * self.C22l[:, 26] + self.C13l[:, 4] + f1 * self.C13l[:, 9]  # *b1*b1
                        self.Cloopl[:, 6] = self.C22l[:, 6] + f1 * self.C22l[:, 16]  # *b1*b2
                        self.Cloopl[:, 7] = self.C13l[:, 5]  # *b1*b3
                        self.Cloopl[:, 8] = self.C22l[:, 7] + f1 * self.C22l[:, 17]  # *b1*b4
                        self.Cloopl[:, 9] = self.C22l[:, 9]  # *b2*b2
                        self.Cloopl[:, 10] = self.C22l[:, 10]  # *b2*b4
                        self.Cloopl[:, 11] = self.C22l[:, 12]  # *b4*b4

                        self.Cloopl[:, 12] = f1 * self.C22l[:, 23] + f1 * self.C22l[:, 24] + f1**2 * self.C22l[:, 33] + f1**2 * self.C22l[:, 34] + f1**2 * self.C22l[:, 35] + f1 * self.C13l[:, 14] + f1 * self.C13l[:, 15] + f1**2 * self.C13l[:, 18] + f1**2 * self.C13l[:, 19] + f1**2 * self.C13l[:, 20] # *bq
                        self.Cloopl[:, 13] = self.C22l[:, 3] + self.C22l[:, 4] + self.C22l[:, 14] + self.C13l[:, 2] + self.C13l[:, 3] + self.C13l[:, 8] # *bq*bq
                        self.Cloopl[:, 14] = self.C22l[:, 0] + self.C22l[:, 8] + f1 * self.C22l[:, 18] + f1 * self.C22l[:, 19] + self.C13l[:, 0] + self.C13l[:, 6] + f1 * self.C13l[:, 10] + f1 * self.C13l[:, 11] # *bq*b1
                        self.Cloopl[:, 15] = self.C22l[:, 1] + self.C22l[:, 11] # *bq*b2
                        self.Cloopl[:, 16] = self.C13l[:, 1] + self.C13l[:, 7] # *bq*b3
                        self.Cloopl[:, 17] = self.C22l[:, 2] + self.C22l[:, 13] # *bq*b4

                    if self.co.Nloop == self.co.N22 + self.co.N13:
                        self.Ploopl[:, :self.co.N22] = self.P22l
                        self.Ploopl[:, self.co.N22:] = self.P13l
                        self.Cloopl[:, :self.co.N22] = self.C22l
                        self.Cloopl[:, self.co.N22:] = self.C13l

            else: # halo-matter
                if self.co.Nloop == 5:
                    f1 = self.f

                    self.Ploopl[:, 0] = f1**2 * self.P22l[:, 6] + f1**2 * self.P22l[:, 13] + f1**2 * self.P22l[:, 14] + f1**3 * self.P22l[:, 17] + f1**3 * self.P22l[:, 18] + f1**4 * self.P22l[:, 19] + f1**4 * self.P22l[:, 20] + f1**4 * self.P22l[:, 21] + \
                        f1 * self.P13l[:, 4] + f1**2 * self.P13l[:, 7] + f1**2 * self.P13l[:, 8] + f1**3 * self.P13l[:, 9] + f1**3 * self.P13l[:, 10] # *1
                    self.Ploopl[:, 1] = self.P22l[:, 0] + f1**2 * self.P22l[:, 7] + f1**2 * self.P22l[:, 8]  + f1**3 * self.P22l[:, 15] + f1**3 * self.P22l[:, 16] + f1 * self.P13l[:, 2] + f1**2 * self.P13l[:, 5] + f1**2 * self.P13l[:, 6] # *b1
                    self.Ploopl[:, 2] = self.P22l[:, 1] + f1 * self.P22l[:, 4] + f1**2 * self.P22l[:, 9] + f1**2 * self.P22l[:, 10] # *b2
                    self.Ploopl[:, 3] = self.P13l[:, 1] + f1 * self.P13l[:, 3] # *b3
                    self.Ploopl[:, 4] = self.P22l[:, 2] + f1 * self.P22l[:, 5] + f1**2 * self.P22l[:, 11] + f1**2 * self.P22l[:, 12] # *b4

                    self.Cloopl[:, 0] = f1**2 * self.C22l[:, 6] + f1**2 * self.C22l[:, 13] + f1**2 * self.C22l[:, 14] + f1**3 * self.C22l[:, 17] + f1**3 * self.C22l[:, 18] + f1**4 * self.C22l[:, 19] + f1**4 * self.C22l[:, 20] + f1**4 * self.C22l[:, 21] + \
                        f1 * self.C13l[:, 4] + f1**2 * self.C13l[:, 7] + f1**2 * self.C13l[:, 8] + f1**3 * self.C13l[:, 9] + f1**3 * self.C13l[:, 10] # *1
                    self.Cloopl[:, 1] = self.C22l[:, 0] + f1**2 * self.C22l[:, 7] + f1**2 * self.C22l[:, 8]  + f1**3 * self.C22l[:, 15] + f1**3 * self.C22l[:, 16] + f1 * self.C13l[:, 2] + f1**2 * self.C13l[:, 5] + f1**2 * self.C13l[:, 6] # *b1
                    self.Cloopl[:, 2] = self.C22l[:, 1] + f1 * self.C22l[:, 4] + f1**2 * self.C22l[:, 9] + f1**2 * self.C22l[:, 10] # *b2
                    self.Cloopl[:, 3] = self.C13l[:, 1] + f1 * self.C13l[:, 3] # *b3
                    self.Cloopl[:, 4] = self.C22l[:, 2] + f1 * self.C22l[:, 5] + f1**2 * self.C22l[:, 11] + f1**2 * self.C22l[:, 12] # *b4

                elif self.co.Nloop == 25:
                    pass


        self.subtractShotNoise()


    def reducePsCflf(self): # depreciated
        f = self.f

        self.Plooplf[:, 0] = f**2 * self.Ploopl[:, 0] + f**3 * self.Ploopl[:, 1] + f**4 * self.Ploopl[:, 2] # *1
        self.Plooplf[:, 1] = f * self.Ploopl[:, 3] + f**2 * self.Ploopl[:, 4] + f**3 * self.Ploopl[:, 5]    # *b1
        self.Plooplf[:, 2] = f * self.Ploopl[:, 6] + f**2 * self.Ploopl[:, 7]                               # *b2
        self.Plooplf[:, 3] = f * self.Ploopl[:, 8]                                                          # *b3
        self.Plooplf[:, 4] = f * self.Ploopl[:, 9] + f**2 * self.Ploopl[:, 10]                              # *b4
        self.Plooplf[:, 5] = self.Ploopl[:, 11] + f * self.Ploopl[:, 12] + f**2 * self.Ploopl[:, 13]        # *b1*b1
        self.Plooplf[:, 6] = self.Ploopl[:, 14] + f * self.Ploopl[:, 15]                                    # *b1*b2
        self.Plooplf[:, 7] = self.Ploopl[:, 16]                                                             # *b1*b3
        self.Plooplf[:, 8] = self.Ploopl[:, 17] + f * self.Ploopl[:, 18]                                    # *b1*b4
        self.Plooplf[:, 9] = self.Ploopl[:, 19]                                                             # *b2*b2
        self.Plooplf[:, 10] = self.Ploopl[:, 20]                                                            # *b2*b4
        self.Plooplf[:, 11] = self.Ploopl[:, 21]                                                            # *b4*b4

    def setreducePslb(self, bs, what="full"):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the power spectrum multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """

        # PZ: we can change all this function by a sum of all the terms that are asked (instead of calling setfullPs() and constructing an intermediate array that is useless)
        self.setBias(bs)
        self.Ps = [None] * 2

        if "full" in what:
            self.Ps[0] = einsum('b,lbx->lx', self.b11, self.P11l)
            self.Ps[1] = einsum('b,lbx->lx', self.bloop, self.Ploopl) + einsum('b,lbx->lx', self.bct, self.Pctl)
            if self.with_stoch: self.Ps[1] += einsum('b,lbx->lx', self.bst, self.Pstl)
            if self.with_nnlo_counterterm: self.Ps[1] += einsum('b,lbx->lx', self.cnnlo, self.Pnnlol)

        elif "linear" in what: 
            self.Ps[0] = einsum('b,lbx->lx', self.b11, self.P11l)
            self.Ps[1] = zeros_like(self.Ps[0])
        elif "1loop" in what:
            self.Ps[1] = einsum('b,lbx->lx', self.bloop, self.Ploopl) + einsum('b,lbx->lx', self.bct, self.Pctl)
            if self.with_stoch: self.Ps[1] += einsum('b,lbx->lx', self.bst, self.Pstl)
            self.Ps[0] = zeros_like(self.Ps[1])
        self.Ps = array(self.Ps)
        self.setfullPs()

    def setreduceCflb(self, bs, what="full"):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the correlation multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Cf = [None] * 2
        self.Cf[0] = einsum('b,lbx->lx', self.b11, self.C11l)
        self.Cf[1] = einsum('b,lbx->lx', self.bloop, self.Cloopl) + einsum('b,lbx->lx', self.bct, self.Cctl)
        if self.with_stoch: self.Cf[1] += einsum('b,lbx->lx', self.bst, self.Cstl)
        if self.with_nnlo_counterterm: self.Cf[1] = einsum('b,lbx->lx', self.cnnlo, self.Cnnlol)
        self.Cf = array(self.Cf)
        self.setfullCf()

    def subtractShotNoise(self):
        """ For option: which='all'. Subtract the constant stochastic term from the (22-)loop """
        for l in range(self.co.Nl):
            for n in range(self.co.Nloop):
                # pass
                shotnoise = self.Ploopl[l, n, self.co.id_kstable] # self.co.id_kstable = 0 if kmin = 0.001 (default), = 1 if kmin < 0.001 (option)
                if is_jax: self.Ploopl = self.Ploopl.at[l, n].add(-shotnoise)
                else: self.Ploopl[l, n] -= shotnoise

    # def formatTaylor(self):
    #     """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """
    #     allk = concatenate([self.co.k, self.co.k]).reshape(-1, 1)
    #     Plin = flip(einsum('n,lnk->lnk', array([1., 2. * self.f, self.f**2]), self.P11l), axis=1)
    #     Plin = concatenate(einsum('lnk->lkn', Plin), axis=0)
    #     Plin = hstack((allk, Plin))
    #     Ploop1 = concatenate(einsum('lnk->lkn', self.Ploopl), axis=0)
    #     Ploop2 = einsum('n,lnk->lnk', array([2., 2., 2., 2. * self.f, 2. * self.f, 2. * self.f]), self.Pctl)
    #     Ploop2 = concatenate(einsum('lnk->lkn', Ploop2), axis=0)
    #     Ploop = hstack((allk, Ploop1, Ploop2))
    #     return Plin, Ploop

    def concatenate(self):
        """For Taylor expansion of the theory prediction: concatenate in 1D vector"""
        if self.co.with_cf: 
            bird_1D = concatenate((array([self.f or 0., self.H or 0., self.DA or 0.]), self.C11l.reshape(-1), self.Cctl.reshape(-1), self.Cloopl.reshape(-1)))
            if self.with_nnlo_counterterm: bird_1D = concatenate((bird_1D, self.Cnnlol.reshape(-1)))
        else:
            bird_1D = concatenate((array([self.f or 0., self.H or 0., self.DA or 0.]), self.P11l.reshape(-1), self.Pctl.reshape(-1), self.Ploopl.reshape(-1)))
            if self.with_stoch: bird_1D = concatenate((bird_1D, self.Pstl.reshape(-1)))
            if self.with_nnlo_counterterm: bird_1D = concatenate((bird_1D, self.Pnnlol.reshape(-1)))
        return bird_1D

    def _unpack_fields(self, bird_1D, idx, fields):
        for name in fields:
            attr = getattr(self, name)
            size = np.prod(attr.shape)
            setattr(self, name, bird_1D[idx:idx + size].reshape(attr.shape))
            idx += size
        return idx
    
    def unravel(self, bird_1D):
        """For Taylor expansion of the theory prediction: unravel in attributes with native shapes"""
        idx = 0
        size_cosmo = 3; self.f, self.H, self.DA = bird_1D[idx:idx + size_cosmo]; idx += size_cosmo
        if self.co.with_cf:
            idx = self._unpack_fields(bird_1D, idx, ['C11l', 'Cctl', 'Cloopl'])
            if self.with_nnlo_counterterm: idx = self._unpack_fields(bird_1D, idx, ['Cnnlol'])
        else:
            idx = self._unpack_fields(bird_1D, idx, ['P11l', 'Pctl', 'Ploopl'])
            if self.with_stoch: idx = self._unpack_fields(bird_1D, idx, ['Pstl'])
            if self.with_nnlo_counterterm: idx = self._unpack_fields(bird_1D, idx, ['Pnnlol'])
        return

    def setIRPs(self, Q=None):

        if Q is None: Q = self.Q

        if self.with_bias:
            self.fullIRPs = einsum('alpn,apnk->alk', Q, self.IRPs)
        else:
            self.fullIRPs11 = einsum('lpn,pnk,pi->lik', Q[0], self.IRPs11, self.co.l11)
            self.fullIRPsct = einsum('lpn,pnk,pi->lik', Q[1], self.IRPsct, self.co.lct)
            self.fullIRPsloop = einsum('lpn,pink->lik', Q[1], self.IRPsloop)


    def setresumPs(self, setfull=True):

        if self.with_bias:
            # self.Ps[:2] += self.fullIRPs
            self.Ps = self.Ps + self.fullIRPs
            if setfull: self.setfullPs()

        else:
            self.P11l += self.fullIRPs11
            self.Pctl += self.fullIRPsct
            self.Ploopl += self.fullIRPsloop

    def setresumCf(self, setfull=True):

        if self.with_bias:
            self.Cf[:2] += self.fullIRCf
            if setfull: self.setfullCf()

        else:
            self.C11l += self.fullIRCf11
            self.Cctl += self.fullIRCfct
            self.Cloopl += self.fullIRCfloop

    def settime(self, cosmo, co=None):

        if self.co.nonequaltime:
            Dfid = self.D
            self.setcosmo(cosmo)
            D1 = self.D1 / Dfid
            D2 = self.D2 / Dfid
            Dp2 = D1 * D2
            Dp22 = Dp2 * Dp2
            Dp13 = 0.5 * (D1**3*D2 + D2**3*D1)
            tloop = concatenate([ self.co.N22*[Dp22], self.co.N13*[Dp13] ])

            if self.co.with_cf:
                self.C11l *= Dp2
                self.Cctl *= Dp2
                self.Cloopl = einsum('n,lns->lns', tloop, self.Cloopl)
            else:
                self.P11l *= Dp2
                self.Pctl *= Dp2
                self.Ploopl = einsum('n,lnk->lnk', tloop, self.Ploopl)

        else:
            if co: self.co = co # to pass km, kr, nd

            Dfid = self.D

            self.setcosmo(cosmo)
            Dp1 = self.D/Dfid
            Dp2 = Dp1**2


            if self.co.with_cf:
                self.C11l *= Dp2
                self.Cctl *= Dp2
                self.Cloopl *= Dp2**2
            else:
                self.P11l *= Dp2
                self.Pctl *= Dp2
                self.Ploopl *= Dp2**2

            Dp2n = concatenate(( 2*[array(self.co.Na*[Dp2**(n+1)]) for n in range(self.co.NIR)] ))

            self.IRPs11 = einsum('n,lnk->lnk', Dp2*Dp2n, self.IRPs11)
            self.IRPsct = einsum('n,lnk->lnk', Dp2*Dp2n, self.IRPsct)
            self.IRPsloop = einsum('n,lmnk->lmnk', Dp2**2*Dp2n, self.IRPsloop)
