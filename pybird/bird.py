import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from .common import co, mu
from scipy.special import erf

from .greenfunction import GreenFunction


class Bird(object):
    """
    Main class which contains the power spectrum and correlation function, given a cosmology and a set of EFT parameters.
    Bird: Biased tracers in redshift space

    Attributes
    ----------
    co : class, optional
        An object of type Common() used to share data
    which : string, optional
        Options to choose (default: 'full'):
        - 'full': to compute with a given cosmology and a given set of EFT parameters. This is the fastest evaluation.
        - 'all': to compute with a given cosmology only. Bird(object) will store all terms factorized from the EFT parameters.
    f : float
        Growth rate (for redshift space distortion)
    DA : float, optional
        Angular distance (for AP effect)
    H : float, optional
        Hubble parameter (for AP effect)
    z : float, optional
        Redshift (for AP effect)
    kin : array
        k-array on which the input linear power spectrum is evaluated
    Pin : array
        Input linear power spectrum
    Plin : scipy.interpolate.interp1d
        Interpolated function of the linear power spectrum
    P11 : ndarray
        Linear power spectrum evaluated on co.k (the internal k-array on which PyBird evaluates the power spectrum)
    P22 : ndarray
        To store the power spectrum 22-loop terms
    P13 : ndarray
        To store the power spectrum 13-loop terms
    C11 : ndarray
        To store the correlation function multipole linear terms
    C22l : ndarray
        To store the correlation function multipole 22-loop terms
    C13l : ndarray
        To store the correlation function multipole 13-loop terms
    Cct : ndarray
        To store the correlation function multipole counter terms
    Ps : ndarray
        To store the power spectrum multipole full linear part and full loop part (the loop including the counterterms)
    Cf : ndarray
        To store the correlation function multipole full linear part and full loop part (the loop including the counterterms)
    fullPs : ndarray
        To store the full power spectrum multipole (linear + loop)
    b11 : ndarray
        EFT parameters for the linear terms per multipole
    b13 : ndarray
        EFT parameters for the 13-loop terms per multipole
    b22 : ndarray
        EFT parameters for the 22-loop terms per multipole
    bct : ndarray
        EFT parameters for the counter terms per multipole
    """

    def __init__(self, cosmology=None, with_bias=True, eft_basis='eftoflss', with_stoch=False, with_nnlo_counterterm=False, co=co):

        self.co = co

        self.with_bias = with_bias
        self.with_stoch = with_stoch
        self.with_nnlo_counterterm = with_nnlo_counterterm
        self.with_tidal_alignments = self.co.with_tidal_alignments
        self.eft_basis = eft_basis

        if cosmology is not None:
            self.setcosmo(cosmology)

        self.P22 = np.empty(shape=(self.co.N22, self.co.Nk))
        self.P13 = np.empty(shape=(self.co.N13, self.co.Nk))
        self.Ps = np.zeros(shape=(3, self.co.Nl, self.co.Nk))  # 3: linear, 1-loop, NNLO

        self.C11 = np.empty(shape=(self.co.Nl, self.co.Ns))
        self.C22l = np.empty(shape=(self.co.Nl, self.co.N22, self.co.Ns))
        self.C13l = np.empty(shape=(self.co.Nl, self.co.N13, self.co.Ns))
        self.Cct = np.empty(shape=(self.co.Nl, self.co.Ns))
        self.Cf = np.zeros(shape=(3, self.co.Nl, self.co.Ns))  # 3: linear, 1-loop, NNLO

        # if self.with_nnlo_counterterm:
        #     self.cnnlo = np.zeros(shape=(self.co.Nl))
        #     self.Cnnlo = np.empty(shape=(self.co.Nl, self.co.Ns))
        #     self.Cnnlol = np.empty(shape=(self.co.Nl, 1, self.co.Ns))
        #     self.Pnnlo = np.empty(shape=(self.co.Nk))
        #     self.Pnnlol = np.empty(shape=(self.co.Nl, 1, self.co.Nk))

        if not with_bias:
            self.Ploopl = np.empty(shape=(self.co.Nl, self.co.Nloop, self.co.Nk))
            self.P11l = np.empty(shape=(self.co.Nl, self.co.N11, self.co.Nk))
            self.Pctl = np.empty(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
            self.P22l = np.empty(shape=(self.co.Nl, self.co.N22, self.co.Nk))
            self.P13l = np.empty(shape=(self.co.Nl, self.co.N13, self.co.Nk))

            self.Cloopl = np.empty(shape=(self.co.Nl, self.co.Nloop, self.co.Ns))
            self.C11l = np.empty(shape=(self.co.Nl, self.co.N11, self.co.Ns))
            self.Cctl = np.empty(shape=(self.co.Nl, self.co.Nct, self.co.Ns))

            ###
            self.IRPs11 = np.zeros(shape=(self.co.Nl, self.co.Nn, self.co.Nk))
            self.IRPsct = np.zeros(shape=(self.co.Nl, self.co.Nn, self.co.Nk))
            self.IRPsloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nn, self.co.Nk))

            self.IRCf11 = np.zeros(shape=(self.co.Nl, self.co.N11, self.co.Nn, self.co.Ns))
            self.IRCfct = np.zeros(shape=(self.co.Nl, self.co.Nct, self.co.Nn, self.co.Ns))
            self.IRCfloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nn, self.co.Ns))

            self.fullIRPs11 = np.zeros(shape=(self.co.Nl, self.co.N11, self.co.Nk))
            self.fullIRPsct = np.zeros(shape=(self.co.Nl, self.co.Nct, self.co.Nk))
            self.fullIRPsloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nk))

            self.fullIRCf11 = np.zeros(shape=(self.co.Nl, self.co.N11, self.co.Ns))
            self.fullIRCfct = np.zeros(shape=(self.co.Nl, self.co.Nct, self.co.Ns))
            self.fullIRCfloop = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Ns))

        self.IRPs = np.zeros(shape=(2, self.co.Nl, self.co.Nn, self.co.Nk))
        self.IRCf = np.zeros(shape=(2, self.co.Nl, self.co.Nn, self.co.Ns))

        self.fullIRPs = np.empty(shape=(2, self.co.Nl, self.co.Nk))
        self.fullIRCf = np.empty(shape=(2, self.co.Nl, self.co.Ns))

        if self.with_bias:
            self.b11 = np.empty(shape=(self.co.Nl))
            self.b13 = np.empty(shape=(self.co.Nl, self.co.N13))
            self.b22 = np.empty(shape=(self.co.Nl, self.co.N22))
            self.bct = np.empty(shape=(self.co.Nl))
        else:
            self.b11 = np.empty(shape=(self.co.N11))
            self.bct = np.empty(shape=(self.co.Nct))
            self.bloop = np.empty(shape=(self.co.Nloop))

        if self.with_stoch:
            if self.co.with_cf: # no stochastic term for cf in general ; below is the stochastic terms from a Pade expansion of the Fourier-space stochastic terms
                self.bst = np.zeros(shape=(self.co.Nst))
                self.Cstl = np.zeros(shape=(self.co.Nl, self.co.Nst, self.co.Ns))
                self.Cstl[0,0] = np.exp(-self.co.km * self.co.s) * self.co.km**2 / (4.*np.pi*self.co.s) / self.co.nd
                self.Cstl[0,1] = -self.co.km**2*np.exp(-self.co.km * self.co.s) / (4.*np.pi*self.co.s**2) / self.co.nd
                self.Cstl[1,2] = np.exp(-self.co.km * self.co.s) * (3.+3.*self.co.km*self.co.s+self.co.km**2*self.co.s**2) / (4.*np.pi*self.co.s**3) / self.co.nd
            else:
                self.bst = np.zeros(shape=(self.co.Nst))
                self.Pstl = np.zeros(shape=(self.co.Nl, self.co.Nst, self.co.Nk))
                # self.Pstl[0, 0] = self.co.k ** 0 / self.co.nd
                # # self.Pstl[1, 0] = self.co.k ** 0 / self.co.nd
                # self.Pstl[0, 1] = self.co.k ** 2 / self.co.km ** 2 / self.co.nd
                # self.Pstl[1, 2] = self.co.k ** 2 / self.co.km ** 2 / self.co.nd
                self.Pstl[0,0] = self.co.k**0 / self.co.nd
                if self.eft_basis in ["eftoflss", "westcoast"]:
                    self.Pstl[0,1] = self.co.k**2 / self.co.km**2 / self.co.nd
                    self.Pstl[1,2] = self.co.k**2 / self.co.km**2 / self.co.nd
                elif self.eft_basis == 'eastcoast':
                    for i in range(self.co.Nl):
                        self.Pstl[i,1] = mu[0][2*i] * self.co.k**2 / self.co.km**2 / self.co.nd
                        self.Pstl[i,2] = mu[2][2*i] * self.co.k**2 / self.co.km**2 / self.co.nd
            # self.Pstl[0, 0] = self.co.k ** 0 / self.co.nd
            # self.Pstl[0, 1] = self.co.k ** 2 / self.co.km ** 2 / self.co.nd
            # self.Pstl[1, 1] = self.co.k ** 2 / self.co.km ** 2 / self.co.nd
            # self.Pstl[0, 2] = self.co.k ** 2 / self.co.km ** 2 / self.co.nd
            # self.Pstl[1, 2] = self.co.k ** 2 / self.co.km ** 2 / self.co.nd
        else:
            if self.co.with_cf:
                self.Cstl = None
            else:
                self.Pstl = None
                
        if self.with_nnlo_counterterm:
            self.cnnlo = np.zeros(shape=(self.co.Nl))
            self.Cnnlo = np.empty(shape=(self.co.Nl, self.co.Ns))
            self.Cnnlol = np.empty(shape=(self.co.Nl, self.co.Nnnlo, self.co.Ns))
            self.Pnnlo = np.empty(shape=(self.co.Nk))
            self.Pnnlol = np.empty(shape=(self.co.Nl, self.co.Nnnlo, self.co.Nk))
        else: # this was clashing with redshift_bin: True, because for output: 'bpk', it is the correlation that is first computed, so co.with_cf = True at the instatiation of the bird... need to change that # PZ
            self.Pnnlol = None
            self.Cnnlol = None
            # if self.co.with_cf: self.Cnnlol = None
            # else: self.Pnnlol = None

    def setcosmo(self, cosmo):

        self.kin = cosmo["k11"]
        self.Pin = cosmo["P11"]
        # try:
        #     self.Plin = interp1d(self.kin, self.Pin, kind="cubic")
        #     self.P11 = self.Plin(self.co.k)
        # except:
        #     self.Plin = None
        #     self.P11 = None
        
        if self.Pin is not None: 
            self.Plin = interp1d(self.kin, self.Pin, kind='cubic')
            self.P11 = self.Plin(self.co.k)
        else:
            self.Plin = None
            self.P11 = None
            
        if cosmo["pk_lin_2"] is not None: self.Pin_2 = cosmo["pk_lin_2"]
        else: self.Pin_2 = self.Pin
        if self.Pin_2 is not None:
            self.Plin_2 = interp1d(self.kin, self.Pin_2, kind='cubic')
            self.P11_2 = self.Plin_2(self.co.k)

        if not self.co.with_time:
            self.D = cosmo["D"]
        self.f = cosmo["f"]
        self.DA = cosmo["DA"]
        self.H = cosmo["H"]
        
        # print(cosmo["Omega0_m"], cosmo["z"])

        if self.co.exact_time:
            # try:
            # try:
            #     self.w0 = cosmo["w0_fld"]
            # except:
            #     self.w0 = None
            
            if "w0_fld" in cosmo: self.w0 = cosmo["w0_fld"]
            else: self.w0 = None
            
            self.Omega0_m = cosmo["Omega0_m"]
            self.z = cosmo["z"]
            
            # print(self.z, self.Omega0_m)
            
            self.a = 1 / (1.0 + self.z)
            GF = GreenFunction(self.Omega0_m, w=self.w0, quintessence=self.co.quintessence)
            self.Y1 = GF.Y(self.a)
            self.G1t = GF.mG1t(self.a)
            self.V12t = GF.mV12t(self.a)
            if self.co.quintessence:
                self.G1 = GF.G(self.a)
                self.f = GF.fplus(self.a)
            else:
                self.G1 = 1.0
            # print (self.Y1, self.G1t, self.V12t, self.G1, self.f, GF.fplus(self.a))
            # except:
            #     print ("setting EdS time approximation")
            #     self.Y1 = 0.
            #     self.G1t = 3/7.
            #     self.V12t = 1/7.
            #     self.G1 = 1.

        if self.co.nonequaltime:
            self.D = cosmo["D"]
            self.D1 = cosmo["D1"]
            self.D2 = cosmo["D2"]
            self.f1 = cosmo["f1"]
            self.f2 = cosmo["f2"]

    def setBias(self, bias):
        """Given an array of EFT parameters, set them among linear, loops and counter terms, and among multipoles

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """

        f = self.f

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
        # b5 = bias["cct"] / self.co.km ** 2
        # b6 = bias["cr1"] / self.co.km ** 2
        # b7 = bias["cr2"] / self.co.km ** 2
        if self.eft_basis in ["eftoflss", "westcoast"]:
            b5 = bias["cct"] / self.co.km**2
            b6 = bias["cr1"] / self.co.kr**2
            b7 = bias["cr2"] / self.co.kr**2
        elif self.eft_basis == 'eastcoast': # inversion of (2.23) of 2004.10607
            ct0 = bias["c0"] - f/3. * bias["c2"] + 3/35. * f**2 * bias["c4"]
            ct2 = bias["c2"] - 6/7. * f * bias["c4"]
            ct4 = bias["c4"]

        if self.with_stoch:
            self.bst[0] = bias["ce0"]
            self.bst[1] = bias["ce1"]
            self.bst[2] = bias["ce2"]

        if self.co.halohalo:

            if self.with_nnlo_counterterm:
                for i in range(self.co.Nl):
                    self.cnnlo[i] = bias["cnnlo_l%s" % (2 * i)] / self.co.km ** 4
            if self.with_tidal_alignments:
                bq = bias["bq"]

            if self.with_bias:  # evaluation with biases specified
                for i in range(self.co.Nl):
                    l = 2 * i
                    # if self.with_tidal_alignments:
                    #     self.b11[i] = (
                    #         (b1 - bq / 3.0) ** 2 * mu[0][l]
                    #         + 2.0 * (b1 - bq / 3.0) * (f + bq) * mu[2][l]
                    #         + (f + bq) ** 2 * mu[4][l]
                    #     )
                    # else:
                    #     self.b11[i] = b1 ** 2 * mu[0][l] + 2.0 * b1 * f * mu[2][l] + f ** 2 * mu[4][l]
                    # self.bct[i] = 2.0 * b1 * (b5 * mu[0][l] + b6 * mu[2][l] + b7 * mu[4][l]) + 2.0 * f * (
                    #     b5 * mu[2][l] + b6 * mu[4][l] + b7 * mu[6][l]
                    # )
                    
                    if self.with_nnlo_counterterm:
                        if self.eft_basis in ["eftoflss", "westcoast"]: self.cnnlo[i] = 0.25 * ( b1**2 * bias["cr4"] * mu[4][l] + b1 * bias["cr6"] * mu[6][l] ) / self.co.kr**4
                        elif self.eft_basis == "eastcoast": self.cnnlo[i] = - bias["ct"] * f**4 * ( b1**2 * mu[4][l] + 2. * b1 * f * mu[6][l] + f**2 * mu[8][l] )  # these are not divided by kr^4 according to eastcoast definition; the prior is adjusted accordingly
                    if self.with_tidal_alignments: self.b11[i] = (b1-bq/3.)**2 * mu[0][l] + 2. * (b1-bq/3.) * (f+bq) * mu[2][l] + (f+bq)**2 * mu[4][l]
                    else: self.b11[i] = b1**2 * mu[0][l] + 2. * b1 * f * mu[2][l] + f**2 * mu[4][l]
                    if self.eft_basis in ["eftoflss", "westcoast"]: self.bct[i] = 2. * b1 * (b5 * mu[0][l] + b6 * mu[2][l] + b7 * mu[4][l]) + 2. * f * (b5 * mu[2][l] + b6 * mu[4][l] + b7 * mu[6][l])
                    elif self.eft_basis == "eastcoast": self.bct[i] = - 2. * (ct0 * mu[0][l] + ct2 * f * mu[2][l] + ct4 * f**2 * mu[4][l])  # these are not divided by km^2 or kr^2 according to eastcoast definition; the prior is adjusted accordingly
                    
                    if self.co.exact_time:
                        self.b22[i] = np.array(
                            [
                                b1 ** 2 * G1 ** 2 * mu[0][l],
                                b1 * b2 * G1 * mu[0][l],
                                b1 * b4 * G1 * mu[0][l],
                                b2 ** 2 * mu[0][l],
                                b2 * b4 * mu[0][l],
                                b4 ** 2 * mu[0][l],
                                b1 ** 2 * f * G1 * mu[2][l],
                                b1 * b2 * f * mu[2][l],
                                b1 * b4 * f * mu[2][l],
                                b1 * f * G1 ** 2 * mu[2][l],
                                b2 * f * G1 * mu[2][l],
                                b4 * f * G1 * mu[2][l],
                                b1 ** 2 * f ** 2 * mu[2][l],
                                b1 ** 2 * f ** 2 * mu[4][l],
                                b1 * f ** 2 * G1 * mu[2][l],
                                b1 * f ** 2 * G1 * mu[4][l],
                                b2 * f ** 2 * mu[2][l],
                                b2 * f ** 2 * mu[4][l],
                                b4 * f ** 2 * mu[2][l],
                                b4 * f ** 2 * mu[4][l],
                                f ** 2 * G1 ** 2 * mu[4][l],
                                b1 * f ** 3 * mu[4][l],
                                b1 * f ** 3 * mu[6][l],
                                f ** 3 * G1 * mu[4][l],
                                f ** 3 * G1 * mu[6][l],
                                f ** 4 * mu[4][l],
                                f ** 4 * mu[6][l],
                                f ** 4 * mu[8][l],
                                b1 * f * G1 * G1t * mu[2][l],
                                b2 * f * G1t * mu[2][l],
                                b4 * f * G1t * mu[2][l],
                                b1 * f ** 2 * G1t * mu[4][l],
                                f ** 2 * G1 * G1t * mu[4][l],
                                f ** 3 * G1t * mu[4][l],
                                f ** 3 * G1t * mu[6][l],
                                f ** 2 * G1t ** 2 * mu[4][l],
                            ]
                        )
                        if self.co.with_uvmatch: self.b13[i] = np.array([ b1**2*G1**2*mu[0][l], b1*b3*mu[0][l], b1*f*G1**2*mu[2][l], b3*f*mu[2][l], f**2*G1**2*mu[4][l], b1**2*Y1*mu[0][l], b1*f*mu[2][l]*Y1, f**2*mu[4][l]*Y1, b1**2*f*G1t*mu[2][l], b1*f**2*G1t*mu[2][l], b1*f**2*G1t*mu[4][l], f**3*G1t*mu[4][l], f**3*G1t*mu[6][l], b1*f*mu[2][l]*V12t, f**2*mu[4][l]*V12t,
                             b1**2 * f * mu[2][l], b1**2 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], b1 * f**3 * mu[4][l], f**3 * mu[6][l], f**4 * mu[6][l] ])
                        else:
                            self.b13[i] = np.array(
                                [
                                    b1 ** 2 * G1 ** 2 * mu[0][l],
                                    b1 * b3 * mu[0][l],
                                    b1 * f * G1 ** 2 * mu[2][l],
                                    b3 * f * mu[2][l],
                                    f ** 2 * G1 ** 2 * mu[4][l],
                                    b1 ** 2 * Y1 * mu[0][l],
                                    b1 * f * mu[2][l] * Y1,
                                    f ** 2 * mu[4][l] * Y1,
                                    b1 ** 2 * f * G1t * mu[2][l],
                                    b1 * f ** 2 * G1t * mu[2][l],
                                    b1 * f ** 2 * G1t * mu[4][l],
                                    f ** 3 * G1t * mu[4][l],
                                    f ** 3 * G1t * mu[6][l],
                                    b1 * f * mu[2][l] * V12t,
                                    f ** 2 * mu[4][l] * V12t,
                                ]
                            )
                        # similar to above but with G1 = 1
                        # self.b22[i] = np.array([b1**2*mu[0][l], b1*b2*mu[0][l], b1*b4*mu[0][l], b2**2*mu[0][l], b2*b4*mu[0][l], b4**2*mu[0][l], b1**2*f*mu[2][l], b1*b2*f*mu[2][l], b1*b4*f*mu[2][l], b1*f*mu[2][l], b2*f*mu[2][l], b4*f*mu[2][l], b1**2*f**2*mu[2][l], b1**2*f**2*mu[4][l], b1*f**2*mu[2][l], b1*f**2*mu[4][l], b2*f**2*mu[2][l], b2*f**2*mu[4][l], b4*f**2*mu[2][l], b4*f**2*mu[4][l], f**2*mu[4][l], b1*f**3*mu[4][l], b1*f**3*mu[6][l], f**3*mu[4][l], f**3*mu[6][l], f**4*mu[4][l], f**4*mu[6][l], f**4*mu[8][l], b1*f*G1t*mu[2][l], b2*f*G1t*mu[2][l], b4*f*G1t*mu[2][l], b1*f**2*G1t*mu[4][l], f**2*G1t*mu[4][l], f**3*G1t*mu[4][l], f**3*G1t*mu[6][l], f**2*G1t**2*mu[4][l] ])
                        # self.b13[i] = np.array([b1**2*mu[0][l], b1*b3*mu[0][l], b1*f*mu[2][l], b3*f*mu[2][l], f**2*mu[4][l], b1**2*Y1*mu[0][l], b1*f*mu[2][l]*Y1, f**2*mu[4][l]*Y1, b1**2*f*G1t*mu[2][l], b1*f**2*G1t*mu[2][l], b1*f**2*G1t*mu[4][l], f**3*G1t*mu[4][l], f**3*G1t*mu[6][l], b1*f*mu[2][l]*V12t, f**2*mu[4][l]*V12t])
                    elif self.with_tidal_alignments:  # tidal alignments with exact time not coded up
                        self.b22[i] = np.array(
                            [
                                b1 * bq * mu[2][l],
                                b2 * bq * mu[2][l],
                                b4 * bq * mu[2][l],
                                bq ** 2 * mu[2][l],
                                bq ** 2 * mu[4][l],
                                b1 ** 2 * mu[0][l],
                                b1 * b2 * mu[0][l],
                                b1 * b4 * mu[0][l],
                                b1 * bq * mu[0][l],
                                b2 ** 2 * mu[0][l],
                                b2 * b4 * mu[0][l],
                                b2 * bq * mu[0][l],
                                b4 ** 2 * mu[0][l],
                                b4 * bq * mu[0][l],
                                bq ** 2 * mu[0][l],
                                b1 ** 2 * f * mu[2][l],
                                b1 * b2 * f * mu[2][l],
                                b1 * b4 * f * mu[2][l],
                                b1 * bq * f * mu[2][l],
                                b1 * bq * f * mu[4][l],
                                b1 * f * mu[2][l],
                                b2 * f * mu[2][l],
                                b4 * f * mu[2][l],
                                bq * f * mu[2][l],
                                bq * f * mu[4][l],
                                b1 ** 2 * f ** 2 * mu[2][l],
                                b1 ** 2 * f ** 2 * mu[4][l],
                                b1 * f ** 2 * mu[2][l],
                                b1 * f ** 2 * mu[4][l],
                                b2 * f ** 2 * mu[2][l],
                                b2 * f ** 2 * mu[4][l],
                                b4 * f ** 2 * mu[2][l],
                                b4 * f ** 2 * mu[4][l],
                                bq * f ** 2 * mu[2][l],
                                bq * f ** 2 * mu[4][l],
                                bq * f ** 2 * mu[6][l],
                                f ** 2 * mu[4][l],
                                b1 * f ** 3 * mu[4][l],
                                b1 * f ** 3 * mu[6][l],
                                f ** 3 * mu[4][l],
                                f ** 3 * mu[6][l],
                                f ** 4 * mu[4][l],
                                f ** 4 * mu[6][l],
                                f ** 4 * mu[8][l],
                            ]
                        )
                        self.b13[i] = np.array(
                            [
                                b1 * bq * mu[2][l],
                                b3 * bq * mu[2][l],
                                bq ** 2 * mu[2][l],
                                bq ** 2 * mu[4][l],
                                b1 ** 2 * mu[0][l],
                                b1 * b3 * mu[0][l],
                                b1 * bq * mu[0][l],
                                b3 * bq * mu[0][l],
                                bq ** 2 * mu[0][l],
                                b1 ** 2 * f * mu[2][l],
                                b1 * bq * f * mu[2][l],
                                b1 * bq * f * mu[4][l],
                                b1 * f * mu[2][l],
                                b3 * f * mu[2][l],
                                bq * f * mu[2][l],
                                bq * f * mu[4][l],
                                b1 * f ** 2 * mu[2][l],
                                b1 * f ** 2 * mu[4][l],
                                bq * f ** 2 * mu[2][l],
                                bq * f ** 2 * mu[4][l],
                                bq * f ** 2 * mu[6][l],
                                f ** 2 * mu[4][l],
                                f ** 3 * mu[4][l],
                                f ** 3 * mu[6][l],
                            ]
                        )
                    else:  # EdS time approximation, no tidal alignments
                        self.b22[i] = np.array(
                            [
                                b1 ** 2 * mu[0][l],
                                b1 * b2 * mu[0][l],
                                b1 * b4 * mu[0][l],
                                b2 ** 2 * mu[0][l],
                                b2 * b4 * mu[0][l],
                                b4 ** 2 * mu[0][l],
                                b1 ** 2 * f * mu[2][l],
                                b1 * b2 * f * mu[2][l],
                                b1 * b4 * f * mu[2][l],
                                b1 * f * mu[2][l],
                                b2 * f * mu[2][l],
                                b4 * f * mu[2][l],
                                b1 ** 2 * f ** 2 * mu[2][l],
                                b1 ** 2 * f ** 2 * mu[4][l],
                                b1 * f ** 2 * mu[2][l],
                                b1 * f ** 2 * mu[4][l],
                                b2 * f ** 2 * mu[2][l],
                                b2 * f ** 2 * mu[4][l],
                                b4 * f ** 2 * mu[2][l],
                                b4 * f ** 2 * mu[4][l],
                                f ** 2 * mu[4][l],
                                b1 * f ** 3 * mu[4][l],
                                b1 * f ** 3 * mu[6][l],
                                f ** 3 * mu[4][l],
                                f ** 3 * mu[6][l],
                                f ** 4 * mu[4][l],
                                f ** 4 * mu[6][l],
                                f ** 4 * mu[8][l],
                            ]
                        )
                        if self.co.with_uvmatch: self.b13[i] = np.array([ b1**2 * mu[0][l], b1 * b3 * mu[0][l], b1**2 * f * mu[2][l], b1 * f * mu[2][l], b3 * f * mu[2][l], b1 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], f**2 * mu[4][l], f**3 * mu[4][l], f**3 * mu[6][l],
                            b1**2 * f**2 * mu[2][l], b1 * f**3 * mu[4][l], f**4 * mu[6][l] ])
                        else:
                            self.b13[i] = np.array(
                                [
                                    b1 ** 2 * mu[0][l],
                                    b1 * b3 * mu[0][l],
                                    b1 ** 2 * f * mu[2][l],
                                    b1 * f * mu[2][l],
                                    b3 * f * mu[2][l],
                                    b1 * f ** 2 * mu[2][l],
                                    b1 * f ** 2 * mu[4][l],
                                    f ** 2 * mu[4][l],
                                    f ** 3 * mu[4][l],
                                    f ** 3 * mu[6][l],
                                ]
                            )
            else:  # evaluation with biases unspecified
            
                if self.with_nnlo_counterterm:
                    if self.eft_basis in ["eftoflss", "westcoast"]: self.cnnlo = 0.25 * np.array([b1**2 * bias["cr4"], b1 * bias["cr6"]]) / self.co.kr**4
                    elif self.eft_basis == "eastcoast": self.cnnlo = - bias["ct"] * f**4 * np.array([b1**2, 2. * b1 * f, f**2])   # these are not divided by kr^4 according to eastcoast definition; the prior is adjusted accordingly
                    
                if self.with_tidal_alignments:
                    self.b11 = np.array([(b1 - bq / 3.0) ** 2, 2.0 * (b1 - bq / 3.0) * (f + bq), (f + bq) ** 2])
                else:
                    self.b11 = np.array([b1 ** 2, 2.0 * b1 * f, f ** 2])
                    
                # self.bct = np.array(
                #     [2.0 * b1 * b5, 2.0 * b1 * b6, 2.0 * b1 * b7, 2.0 * f * b5, 2.0 * f * b6, 2.0 * f * b7]
                # )
                
                if self.eft_basis in ["eftoflss", "westcoast"]: self.bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
                elif self.eft_basis == "eastcoast": self.bct = - 2. * np.array([ct0, f * ct2, f**2 * ct4]) # these are not divided by km^2 or kr^2 according to eastcoast definition; the prior is adjusted accordingly
                
                if self.co.Nloop is 12:
                    self.bloop = np.array(
                        [1.0, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4]
                    )
                elif self.co.Nloop is 22:
                    self.bloop = np.array(
                        [
                            f ** 2,
                            f ** 3,
                            f ** 4,
                            b1 * f,
                            b1 * f ** 2,
                            b1 * f ** 3,
                            b2 * f,
                            b2 * f ** 2,
                            b3 * f,
                            b4 * f,
                            b4 * f ** 2,
                            b1 ** 2,
                            b1 ** 2 * f,
                            b1 ** 2 * f ** 2,
                            b1 * b2,
                            b1 * b2 * f,
                            b1 * b3,
                            b1 * b4,
                            b1 * b4 * f,
                            b2 ** 2,
                            b2 * b4,
                            b4 ** 2,
                        ]
                    )
                elif self.co.Nloop is 35:
                    self.bloop = np.array(
                        [
                            f ** 2,
                            f ** 2 * G1t,
                            f ** 2 * G1t ** 2,
                            f ** 2 * Y1,
                            f ** 2 * V12t,
                            f ** 3,
                            f ** 3 * G1t,
                            f ** 4,
                            b1 * f,
                            b1 * f * G1t,
                            b1 * f * Y1,
                            b1 * f * V12t,
                            b1 * f ** 2,
                            b1 * f ** 2 * G1t,
                            b1 * f ** 3,
                            b2 * f,
                            b2 * f * G1t,
                            b2 * f ** 2,
                            b3 * f,
                            b4 * f,
                            b4 * f * G1t,
                            b4 * f ** 2,
                            b1 ** 2,
                            b1 ** 2 * Y1,
                            b1 ** 2 * f,
                            b1 ** 2 * f * G1t,
                            b1 ** 2 * f ** 2,
                            b1 * b2,
                            b1 * b2 * f,
                            b1 * b3,
                            b1 * b4,
                            b1 * b4 * f,
                            b2 ** 2,
                            b2 * b4,
                            b4 ** 2,
                        ]
                    )
                elif self.co.Nloop is self.co.N22 + self.co.N13:
                    self.bloop = np.array(
                        [
                            b1 ** 2,
                            b1 * b2,
                            b1 * b4,
                            b2 ** 2,
                            b2 * b4,
                            b4 ** 2,
                            b1 ** 2 * f,
                            b1 * b2 * f,
                            b1 * b4 * f,
                            b1 * f,
                            b2 * f,
                            b4 * f,
                            b1 ** 2 * f ** 2,
                            b1 ** 2 * f ** 2,
                            b1 * f ** 2,
                            b1 * f ** 2,
                            b2 * f ** 2,
                            b2 * f ** 2,
                            b4 * f ** 2,
                            b4 * f ** 2,
                            f ** 2,
                            b1 * f ** 3,
                            b1 * f ** 3,
                            f ** 3,
                            f ** 3,
                            f ** 4,
                            f ** 4,
                            f ** 4,
                            b1 ** 2,
                            b1 * b3,
                            b1 ** 2 * f,
                            b1 * f,
                            b3 * f,
                            b1 * f ** 2,
                            b1 * f ** 2,
                            f ** 2,
                            f ** 3,
                            f ** 3,
                        ]
                    )
                elif self.co.Nloop is 18:
                    self.bloop = np.array(
                        [
                            1.0,
                            b1,
                            b2,
                            b3,
                            b4,
                            b1 * b1,
                            b1 * b2,
                            b1 * b3,
                            b1 * b4,
                            b2 * b2,
                            b2 * b4,
                            b4 * b4,
                            bq,
                            bq * bq,
                            bq * b1,
                            bq * b2,
                            bq * b3,
                            bq * b4,
                        ]
                    )  # with_tidal_alignements

        else:  # halo-matter

            d5 = bias["dct"] / self.co.km ** 2  # matter counterterm
            d6 = bias["dr1"] / self.co.km ** 2  # matter redshift counterterm 1
            d7 = bias["dr2"] / self.co.km ** 2  # matter redshift counterterm 2

            if self.with_bias:
                for i in range(self.co.Nl):
                    l = 2 * i
                    self.b11[i] = b1 * mu[0][l] + b1 * f * mu[2][l] + f * mu[2][l] + f ** 2 * mu[4][l]
                    self.bct[i] = (
                        b1 * (d5 * mu[0][l] + d6 * mu[2][l] + d7 * mu[4][l])
                        + f * (d5 * mu[2][l] + d6 * mu[4][l] + d7 * mu[6][l])
                        + b5 * (mu[0][l] + f * mu[2][l])
                        + b6 * (mu[2][l] + f * mu[4][l])
                        + b7 * (mu[4][l] + f * mu[6][l])
                    )
                    self.b22[i] = np.array(
                        [
                            b1 * mu[0][l],
                            b2 * mu[0][l],
                            b4 * mu[0][l],
                            b1 * f * mu[2][l],
                            b2 * f * mu[2][l],
                            b4 * f * mu[2][l],
                            f * mu[2][l],
                            b1 * f ** 2 * mu[2][l],
                            b1 * f ** 2 * mu[4][l],
                            b2 * f ** 2 * mu[2][l],
                            b2 * f ** 2 * mu[4][l],
                            b4 * f ** 2 * mu[2][l],
                            b4 * f ** 2 * mu[4][l],
                            f ** 2 * mu[2][l],
                            f ** 2 * mu[4][l],
                            b1 * f ** 3 * mu[4][l],
                            b1 * f ** 3 * mu[6][l],
                            f ** 3 * mu[4][l],
                            f ** 3 * mu[6][l],
                            f ** 4 * mu[4][l],
                            f ** 4 * mu[6][l],
                            f ** 4 * mu[8][l],
                        ]
                    )
                    self.b13[i] = np.array(
                        [
                            b1 * mu[0][l],
                            b3 * mu[0][l],
                            b1 * f * mu[2][l],
                            b3 * f * mu[2][l],
                            f * mu[2][l],
                            b1 * f ** 2 * mu[2][l],
                            b1 * f ** 2 * mu[4][l],
                            f ** 2 * mu[2][l],
                            f ** 2 * mu[4][l],
                            f ** 3 * mu[4][l],
                            f ** 3 * mu[6][l],
                        ]
                    )
            else:
                self.b11 = np.array([b1, b1 * f, f, f ** 2])
                self.bct = np.array(
                    [b1 * dct, b1 * dr1, b1 * dr2, f * dct, f * dr1, f * dr2, cct, cr1, cr2, f * cct, f * cr1, f * cr2]
                )
                if self.co.Nloop is 5:
                    self.bloop = np.array([1.0, b1, b2, b3, b4])

    def setPs(self, bs=None, setfull=True):
        """For option: which='full'. Given an array of EFT parameters, multiplies them accordingly to the power spectrum multipole terms and adds the resulting terms together per loop order

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        if bs is not None:
            self.setBias(bs)
        self.Ps[0] = np.einsum("l,x->lx", self.b11, self.P11)
        self.Ps[1] = np.einsum("lb,bx->lx", self.b22, self.P22)
        for l in range(self.co.Nl):
            self.Ps[1, l] -= self.Ps[1, l, 0]
        self.Ps[1] += np.einsum("lb,bx->lx", self.b13, self.P13) + np.einsum(
            "l,x,x->lx", self.bct, self.co.k ** 2, self.P11
        )
        if self.with_stoch:
            self.Ps[1] += np.einsum("b,lbx->lx", self.bst, self.Pstl)
        if self.with_nnlo_counterterm:
            self.Ps[2] = np.einsum("l,x->lx", self.cnnlo, self.Pnnlo)
        if setfull:
            self.setfullPs()

    def setCf(self, bs=None, setfull=True):
        """For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        if bs is not None:
            self.setBias(bs)
        self.Cf[0] = np.einsum("l,lx->lx", self.b11, self.C11)
        self.Cf[1] = (
            np.einsum("lb,lbx->lx", self.b22, self.C22l)
            + np.einsum("lb,lbx->lx", self.b13, self.C13l)
            + np.einsum("l,lx->lx", self.bct, self.Cct)
        )
        if self.with_stoch: self.Cf[1] += np.einsum('b,lbx->lx', self.bst, self.Cstl) # no stochastic term for Cf
        if self.with_nnlo_counterterm:
            self.Cf[2] = np.einsum("l,lx->lx", self.cnnlo, self.Cnnlo)
        if setfull:
            self.setfullCf()

    def setPsCf(self, bs, setfull=True):
        """For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the power spectrum and correlation function multipole terms

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
        self.fullPs = np.sum(self.Ps, axis=0)

    def setfullCf(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full correlation function multipoles """
        self.fullCf = np.sum(self.Cf, axis=0)

    def setPsCfl(self):
        """ For option: which='all'. Creates multipoles for each term weighted accordingly """
        self.P11l = np.einsum("x,ln->lnx", self.P11, self.co.l11)
        self.Pctl = np.einsum("x,x,ln->lnx", self.co.k ** 2, self.P11, self.co.lct)
        self.P22l = np.einsum("nx,ln->lnx", self.P22, self.co.l22)
        self.P13l = np.einsum("nx,ln->lnx", self.P13, self.co.l13)

        self.C11l = np.einsum("lx,ln->lnx", self.C11, self.co.l11)
        self.Cctl = np.einsum("lx,ln->lnx", self.Cct, self.co.lct)
        self.C22l = np.einsum("lnx,ln->lnx", self.C22l, self.co.l22)
        self.C13l = np.einsum("lnx,ln->lnx", self.C13l, self.co.l13)

        if self.with_nnlo_counterterm:
            self.Pnnlol = np.einsum("x,ln->lnx", self.Pnnlo, self.co.lnnlo)
            self.Cnnlol = np.einsum("lx,ln->lnx", self.Cnnlo, self.co.lnnlo)

        self.reducePsCfl()

    def reducePsCfl(self):
        """ For option: which='all'. Regroups terms that share the same EFT parameter(s) (more generally, the same time functions) """

        if self.co.halohalo:

            if self.co.exact_time:  # config["with_exact_time"] == True
                if self.co.Nloop is 12:  # config["with_time"] == True
                    f1 = self.f

                    ## EdS: Y1 = 0., G1t = 3/7., V12t = 1/7.
                    G1 = self.G1
                    Y1 = self.Y1
                    G1t = self.G1t
                    V12t = self.V12t

                    self.Ploopl[:, 0] = (
                        G1 ** 2 * f1 ** 2 * self.P22l[:, 20]
                        + G1 * f1 ** 3 * self.P22l[:, 23]
                        + G1 * f1 ** 3 * self.P22l[:, 24]
                        + f1 ** 4 * self.P22l[:, 25]
                        + f1 ** 4 * self.P22l[:, 26]
                        + f1 ** 4 * self.P22l[:, 27]
                        + G1 * G1t * f1 ** 2 * self.P22l[:, 32]
                        + G1t * f1 ** 3 * self.P22l[:, 33]
                        + G1t * f1 ** 3 * self.P22l[:, 34]
                        + G1t ** 2 * f1 ** 2 * self.P22l[:, 35]
                        + G1 ** 2 * f1 ** 2 * self.P13l[:, 4]
                        + Y1 * f1 ** 2 * self.P13l[:, 7]
                        + G1t * f1 ** 3 * self.P13l[:, 11]
                        + G1t * f1 ** 3 * self.P13l[:, 12]
                        + V12t * f1 ** 2 * self.P13l[:, 14]
                    )  # *1
                    self.Ploopl[:, 1] = (
                        G1 ** 2 * f1 * self.P22l[:, 9]
                        + G1 * f1 ** 2 * self.P22l[:, 14]
                        + G1 * f1 ** 2 * self.P22l[:, 15]
                        + f1 ** 3 * self.P22l[:, 21]
                        + f1 ** 3 * self.P22l[:, 22]
                        + G1 * G1t * f1 * self.P22l[:, 28]
                        + G1t * f1 ** 2 * self.P22l[:, 31]
                        + G1 ** 2 * f1 * self.P13l[:, 2]
                        + Y1 * f1 * self.P13l[:, 6]
                        + G1t * f1 ** 2 * self.P13l[:, 9]
                        + G1t * f1 ** 2 * self.P13l[:, 10]
                        + V12t * f1 * self.P13l[:, 13]
                    )  # *b1
                    self.Ploopl[:, 2] = (
                        G1 * f1 * self.P22l[:, 10]
                        + f1 ** 2 * self.P22l[:, 16]
                        + f1 ** 2 * self.P22l[:, 17]
                        + G1t * f1 * self.P22l[:, 29]
                    )  # *b2
                    self.Ploopl[:, 3] = f1 * self.P13l[:, 3]  # *b3
                    self.Ploopl[:, 4] = (
                        G1 * f1 * self.P22l[:, 11]
                        + f1 ** 2 * self.P22l[:, 18]
                        + f1 ** 2 * self.P22l[:, 19]
                        + G1t * f1 * self.P22l[:, 30]
                    )  # *b4
                    self.Ploopl[:, 5] = (
                        G1 ** 2 * self.P22l[:, 0]
                        + G1 * f1 * self.P22l[:, 6]
                        + f1 ** 2 * self.P22l[:, 12]
                        + f1 ** 2 * self.P22l[:, 13]
                        + G1 ** 2 * self.P13l[:, 0]
                        + Y1 * self.P13l[:, 5]
                        + G1t * f1 * self.P13l[:, 8]
                    )  # *b1*b1
                    self.Ploopl[:, 6] = G1 * self.P22l[:, 1] + f1 * self.P22l[:, 7]  # *b1*b2
                    self.Ploopl[:, 7] = self.P13l[:, 1]  # *b1*b3
                    self.Ploopl[:, 8] = G1 * self.P22l[:, 2] + f1 * self.P22l[:, 8]  # *b1*b4
                    self.Ploopl[:, 9] = self.P22l[:, 3]  # *b2*b2
                    self.Ploopl[:, 10] = self.P22l[:, 4]  # *b2*b4
                    self.Ploopl[:, 11] = self.P22l[:, 5]  # *b4*b4

                    self.Cloopl[:, 0] = (
                        G1 ** 2 * f1 ** 2 * self.C22l[:, 20]
                        + G1 * f1 ** 3 * self.C22l[:, 23]
                        + G1 * f1 ** 3 * self.C22l[:, 24]
                        + f1 ** 4 * self.C22l[:, 25]
                        + f1 ** 4 * self.C22l[:, 26]
                        + f1 ** 4 * self.C22l[:, 27]
                        + G1 * G1t * f1 ** 2 * self.C22l[:, 32]
                        + G1t * f1 ** 3 * self.C22l[:, 33]
                        + G1t * f1 ** 3 * self.C22l[:, 34]
                        + G1t ** 2 * f1 ** 2 * self.C22l[:, 35]
                        + G1 ** 2 * f1 ** 2 * self.C13l[:, 4]
                        + Y1 * f1 ** 2 * self.C13l[:, 7]
                        + G1t * f1 ** 3 * self.C13l[:, 11]
                        + G1t * f1 ** 3 * self.C13l[:, 12]
                        + V12t * f1 ** 2 * self.C13l[:, 14]
                    )  # *1
                    self.Cloopl[:, 1] = (
                        G1 ** 2 * f1 * self.C22l[:, 9]
                        + G1 * f1 ** 2 * self.C22l[:, 14]
                        + G1 * f1 ** 2 * self.C22l[:, 15]
                        + f1 ** 3 * self.C22l[:, 21]
                        + f1 ** 3 * self.C22l[:, 22]
                        + G1 * G1t * f1 * self.C22l[:, 28]
                        + G1t * f1 ** 2 * self.C22l[:, 31]
                        + G1 ** 2 * f1 * self.C13l[:, 2]
                        + Y1 * f1 * self.C13l[:, 6]
                        + G1t * f1 ** 2 * self.C13l[:, 9]
                        + G1t * f1 ** 2 * self.C13l[:, 10]
                        + V12t * f1 * self.C13l[:, 13]
                    )  # *b1
                    self.Cloopl[:, 2] = (
                        G1 * f1 * self.C22l[:, 10]
                        + f1 ** 2 * self.C22l[:, 16]
                        + f1 ** 2 * self.C22l[:, 17]
                        + G1t * f1 * self.C22l[:, 29]
                    )  # *b2
                    self.Cloopl[:, 3] = f1 * self.C13l[:, 3]  # *b3
                    self.Cloopl[:, 4] = (
                        G1 * f1 * self.C22l[:, 11]
                        + f1 ** 2 * self.C22l[:, 18]
                        + f1 ** 2 * self.C22l[:, 19]
                        + G1t * f1 * self.C22l[:, 30]
                    )  # *b4
                    self.Cloopl[:, 5] = (
                        G1 ** 2 * self.C22l[:, 0]
                        + G1 * f1 * self.C22l[:, 6]
                        + f1 ** 2 * self.C22l[:, 12]
                        + f1 ** 2 * self.C22l[:, 13]
                        + G1 ** 2 * self.C13l[:, 0]
                        + Y1 * self.C13l[:, 5]
                        + G1t * f1 * self.C13l[:, 8]
                    )  # *b1*b1
                    self.Cloopl[:, 6] = G1 * self.C22l[:, 1] + f1 * self.C22l[:, 7]  # *b1*b2
                    self.Cloopl[:, 7] = self.C13l[:, 1]  # *b1*b3
                    self.Cloopl[:, 8] = G1 * self.C22l[:, 2] + f1 * self.C22l[:, 8]  # *b1*b4
                    self.Cloopl[:, 9] = self.C22l[:, 3]  # *b2*b2
                    self.Cloopl[:, 10] = self.C22l[:, 4]  # *b2*b4
                    self.Cloopl[:, 11] = self.C22l[:, 5]  # *b4*b4
                    
                    if self.co.with_uvmatch:
                        self.Ploopl[:, 0] += f1**4 * self.P13l[:, -1] + f1**3 * self.P13l[:, -2] # *1
                        self.Ploopl[:, 1] += f1**3 * self.P13l[:, -3] + f1**2 * self.P13l[:, -4] # *b1
                        self.Ploopl[:, 5] += f1**2 * self.P13l[:, -5] + f1 * self.P13l[:, -6] # *b1*b1

                        self.Cloopl[:, 0] += f1**4 * self.C13l[:, -1] + f1**3 * self.C13l[:, -2] # *1
                        self.Cloopl[:, 1] += f1**3 * self.C13l[:, -3] + f1**2 * self.C13l[:, -4] # *b1
                        self.Cloopl[:, 5] += f1**2 * self.C13l[:, -5] + f1 * self.C13l[:, -6] # *b1*b1

                elif self.co.Nloop is 35:  # config["with_time"] == False
                    self.Ploopl[:, 0] = self.P22l[:, 20] + self.P13l[:, 4]  # *f^2
                    self.Ploopl[:, 1] = self.P22l[:, 32]  # *f^2*G1t
                    self.Ploopl[:, 2] = self.P22l[:, 35]  # *f^2*G1t**2
                    self.Ploopl[:, 3] = self.P13l[:, 7]  # *f^2*Y1
                    self.Ploopl[:, 4] = self.P13l[:, 14]  # *f^2*V12t
                    self.Ploopl[:, 5] = self.P22l[:, 23] + self.P22l[:, 24]  # *f^3
                    self.Ploopl[:, 6] = (
                        self.P22l[:, 33] + self.P22l[:, 34] + self.P13l[:, 11] + self.P13l[:, 12]
                    )  # *f^3*G1t
                    self.Ploopl[:, 7] = self.P22l[:, 25] + self.P22l[:, 26] + self.P22l[:, 27]  # *f^4
                    self.Ploopl[:, 8] = self.P22l[:, 9] + self.P13l[:, 2]  # *b1*f
                    self.Ploopl[:, 9] = self.P22l[:, 28]  # *b1*f*G1t
                    self.Ploopl[:, 10] = self.P13l[:, 6]  # *b1*f*Y1
                    self.Ploopl[:, 11] = self.P13l[:, 13]  # *b1*f*V12t
                    self.Ploopl[:, 12] = self.P22l[:, 14] + self.P22l[:, 15]  # *b1*f^2
                    self.Ploopl[:, 13] = self.P22l[:, 31] + self.P13l[:, 9] + self.P13l[:, 10]  # *b1*f^2*G1t
                    self.Ploopl[:, 14] = self.P22l[:, 21] + self.P22l[:, 22]  # *b1*f^3
                    self.Ploopl[:, 15] = self.P22l[:, 10]  # *b2*f
                    self.Ploopl[:, 16] = self.P22l[:, 29]  # *b2*f*G1t
                    self.Ploopl[:, 17] = self.P22l[:, 16] + self.P22l[:, 17]  # *b2*f^2
                    self.Ploopl[:, 18] = self.P13l[:, 3]  # *b3*f
                    self.Ploopl[:, 19] = self.P22l[:, 11]  # *b4*f
                    self.Ploopl[:, 20] = self.P22l[:, 30]  # *b4*f*G1t
                    self.Ploopl[:, 21] = self.P22l[:, 18] + self.P22l[:, 19]  # *b4*f^2
                    self.Ploopl[:, 22] = self.P22l[:, 0] + self.P13l[:, 0]  # *b1^2
                    self.Ploopl[:, 23] = self.P13l[:, 5]  # *b1^2*Y1
                    self.Ploopl[:, 24] = self.P22l[:, 6]  # *b1^2*f
                    self.Ploopl[:, 25] = self.P13l[:, 8]  # *b1^2*f*G1t
                    self.Ploopl[:, 26] = self.P22l[:, 12] + self.P22l[:, 13]  # *b1^2*f^2
                    self.Ploopl[:, 27] = self.P22l[:, 1]  # *b1*b2
                    self.Ploopl[:, 28] = self.P22l[:, 7]  # *b1*b2*f
                    self.Ploopl[:, 29] = self.P13l[:, 1]  # *b1*b3
                    self.Ploopl[:, 30] = self.P22l[:, 2]  # *b1*b4
                    self.Ploopl[:, 31] = self.P22l[:, 8]  # *b1*b4*f
                    self.Ploopl[:, 32] = self.P22l[:, 3]  # *b2^2
                    self.Ploopl[:, 33] = self.P22l[:, 4]  # *b2*b4
                    self.Ploopl[:, 34] = self.P22l[:, 5]  # *b4^2

                    self.Cloopl[:, 0] = self.C22l[:, 20] + self.C13l[:, 4]  # *f^2
                    self.Cloopl[:, 1] = self.C22l[:, 32]  # *f^2*G1t
                    self.Cloopl[:, 2] = self.C22l[:, 35]  # *f^2*G1t**2
                    self.Cloopl[:, 3] = self.C13l[:, 7]  # *f^2*Y1
                    self.Cloopl[:, 4] = self.C13l[:, 14]  # *f^2*V12t
                    self.Cloopl[:, 5] = self.C22l[:, 23] + self.C22l[:, 24]  # *f^3
                    self.Cloopl[:, 6] = (
                        self.C22l[:, 33] + self.C22l[:, 34] + self.C13l[:, 11] + self.C13l[:, 12]
                    )  # *f^3*G1t
                    self.Cloopl[:, 7] = self.C22l[:, 25] + self.C22l[:, 26] + self.C22l[:, 27]  # *f^4
                    self.Cloopl[:, 8] = self.C22l[:, 9] + self.C13l[:, 2]  # *b1*f
                    self.Cloopl[:, 9] = self.C22l[:, 28]  # *b1*f*G1t
                    self.Cloopl[:, 10] = self.C13l[:, 6]  # *b1*f*Y1
                    self.Cloopl[:, 11] = self.C13l[:, 13]  # *b1*f*V12t
                    self.Cloopl[:, 12] = self.C22l[:, 14] + self.C22l[:, 15]  # *b1*f^2
                    self.Cloopl[:, 13] = self.C22l[:, 31] + self.C13l[:, 9] + self.C13l[:, 10]  # *b1*f^2*G1t
                    self.Cloopl[:, 14] = self.C22l[:, 21] + self.C22l[:, 22]  # *b1*f^3
                    self.Cloopl[:, 15] = self.C22l[:, 10]  # *b2*f
                    self.Cloopl[:, 16] = self.C22l[:, 29]  # *b2*f*G1t
                    self.Cloopl[:, 17] = self.C22l[:, 16] + self.C22l[:, 17]  # *b2*f^2
                    self.Cloopl[:, 18] = self.C13l[:, 3]  # *b3*f
                    self.Cloopl[:, 19] = self.C22l[:, 11]  # *b4*f
                    self.Cloopl[:, 20] = self.C22l[:, 30]  # *b4*f*G1t
                    self.Cloopl[:, 21] = self.C22l[:, 18] + self.C22l[:, 19]  # *b4*f^2
                    self.Cloopl[:, 22] = self.C22l[:, 0] + self.C13l[:, 0]  # *b1^2
                    self.Cloopl[:, 23] = self.C13l[:, 5]  # *b1^2*Y1
                    self.Cloopl[:, 24] = self.C22l[:, 6]  # *b1^2*f
                    self.Cloopl[:, 25] = self.C13l[:, 8]  # *b1^2*f*G1t
                    self.Cloopl[:, 26] = self.C22l[:, 12] + self.C22l[:, 13]  # *b1^2*f^2
                    self.Cloopl[:, 27] = self.C22l[:, 1]  # *b1*b2
                    self.Cloopl[:, 28] = self.C22l[:, 7]  # *b1*b2*f
                    self.Cloopl[:, 29] = self.C13l[:, 1]  # *b1*b3
                    self.Cloopl[:, 30] = self.C22l[:, 2]  # *b1*b4
                    self.Cloopl[:, 31] = self.C22l[:, 8]  # *b1*b4*f
                    self.Cloopl[:, 32] = self.C22l[:, 3]  # *b2^2
                    self.Cloopl[:, 33] = self.C22l[:, 4]  # *b2*b4
                    self.Cloopl[:, 34] = self.C22l[:, 5]  # *b4^2
                    
                    if self.co.with_uvmatch:
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

            else:  # config["with_exact_time"] == False
                if self.co.Nloop is 12:  # config["with_time"] == True
                    f1 = self.f

                    self.Ploopl[:, 0] = (
                        f1 ** 2 * self.P22l[:, 20]
                        + f1 ** 3 * self.P22l[:, 23]
                        + f1 ** 3 * self.P22l[:, 24]
                        + f1 ** 4 * self.P22l[:, 25]
                        + f1 ** 4 * self.P22l[:, 26]
                        + f1 ** 4 * self.P22l[:, 27]
                        + f1 ** 2 * self.P13l[:, 7]
                        + f1 ** 3 * self.P13l[:, 8]
                        + f1 ** 3 * self.P13l[:, 9]
                    )  # *1
                    self.Ploopl[:, 1] = (
                        f1 * self.P22l[:, 9]
                        + f1 ** 2 * self.P22l[:, 14]
                        + f1 ** 2 * self.P22l[:, 15]
                        + f1 ** 3 * self.P22l[:, 21]
                        + f1 ** 3 * self.P22l[:, 22]
                        + f1 * self.P13l[:, 3]
                        + f1 ** 2 * self.P13l[:, 5]
                        + f1 ** 2 * self.P13l[:, 6]
                    )  # *b1
                    self.Ploopl[:, 2] = (
                        f1 * self.P22l[:, 10] + f1 ** 2 * self.P22l[:, 16] + f1 ** 2 * self.P22l[:, 17]
                    )  # *b2
                    self.Ploopl[:, 3] = f1 * self.P13l[:, 4]  # *b3
                    self.Ploopl[:, 4] = (
                        f1 * self.P22l[:, 11] + f1 ** 2 * self.P22l[:, 18] + f1 ** 2 * self.P22l[:, 19]
                    )  # *b4
                    self.Ploopl[:, 5] = (
                        self.P22l[:, 0]
                        + f1 * self.P22l[:, 6]
                        + f1 ** 2 * self.P22l[:, 12]
                        + f1 ** 2 * self.P22l[:, 13]
                        + self.P13l[:, 0]
                        + f1 * self.P13l[:, 2]
                    )  # *b1*b1
                    self.Ploopl[:, 6] = self.P22l[:, 1] + f1 * self.P22l[:, 7]  # *b1*b2
                    self.Ploopl[:, 7] = self.P13l[:, 1]  # *b1*b3
                    self.Ploopl[:, 8] = self.P22l[:, 2] + f1 * self.P22l[:, 8]  # *b1*b4
                    self.Ploopl[:, 9] = self.P22l[:, 3]  # *b2*b2
                    self.Ploopl[:, 10] = self.P22l[:, 4]  # *b2*b4
                    self.Ploopl[:, 11] = self.P22l[:, 5]  # *b4*b4

                    self.Cloopl[:, 0] = (
                        f1 ** 2 * self.C22l[:, 20]
                        + f1 ** 3 * self.C22l[:, 23]
                        + f1 ** 3 * self.C22l[:, 24]
                        + f1 ** 4 * self.C22l[:, 25]
                        + f1 ** 4 * self.C22l[:, 26]
                        + f1 ** 4 * self.C22l[:, 27]
                        + f1 ** 2 * self.C13l[:, 7]
                        + f1 ** 3 * self.C13l[:, 8]
                        + f1 ** 3 * self.C13l[:, 9]
                    )  # *1
                    self.Cloopl[:, 1] = (
                        f1 * self.C22l[:, 9]
                        + f1 ** 2 * self.C22l[:, 14]
                        + f1 ** 2 * self.C22l[:, 15]
                        + f1 ** 3 * self.C22l[:, 21]
                        + f1 ** 3 * self.C22l[:, 22]
                        + f1 * self.C13l[:, 3]
                        + f1 ** 2 * self.C13l[:, 5]
                        + f1 ** 2 * self.C13l[:, 6]
                    )  # *b1
                    self.Cloopl[:, 2] = (
                        f1 * self.C22l[:, 10] + f1 ** 2 * self.C22l[:, 16] + f1 ** 2 * self.C22l[:, 17]
                    )  # *b2
                    self.Cloopl[:, 3] = f1 * self.C13l[:, 4]  # *b3
                    self.Cloopl[:, 4] = (
                        f1 * self.C22l[:, 11] + f1 ** 2 * self.C22l[:, 18] + f1 ** 2 * self.C22l[:, 19]
                    )  # *b4
                    self.Cloopl[:, 5] = (
                        self.C22l[:, 0]
                        + f1 * self.C22l[:, 6]
                        + f1 ** 2 * self.C22l[:, 12]
                        + f1 ** 2 * self.C22l[:, 13]
                        + self.C13l[:, 0]
                        + f1 * self.C13l[:, 2]
                    )  # *b1*b1
                    self.Cloopl[:, 6] = self.C22l[:, 1] + f1 * self.C22l[:, 7]  # *b1*b2
                    self.Cloopl[:, 7] = self.C13l[:, 1]  # *b1*b3
                    self.Cloopl[:, 8] = self.C22l[:, 2] + f1 * self.C22l[:, 8]  # *b1*b4
                    self.Cloopl[:, 9] = self.C22l[:, 3]  # *b2*b2
                    self.Cloopl[:, 10] = self.C22l[:, 4]  # *b2*b4
                    self.Cloopl[:, 11] = self.C22l[:, 5]  # *b4*b4
                    
                    if self.co.with_uvmatch:
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

                elif self.co.Nloop is 22:  # config["with_time"] == False
                    self.Ploopl[:, 0] = self.P22l[:, 20] + self.P13l[:, 7]  # *f^2
                    self.Ploopl[:, 1] = self.P22l[:, 23] + self.P22l[:, 24] + self.P13l[:, 8] + self.P13l[:, 9]  # *f^3
                    self.Ploopl[:, 2] = self.P22l[:, 25] + self.P22l[:, 26] + self.P22l[:, 27]  # *f^4
                    self.Ploopl[:, 3] = self.P22l[:, 9] + self.P13l[:, 3]  # *b1*f
                    self.Ploopl[:, 4] = (
                        self.P22l[:, 14] + self.P22l[:, 15] + self.P13l[:, 5] + self.P13l[:, 6]
                    )  # *b1*f^2
                    self.Ploopl[:, 5] = self.P22l[:, 21] + self.P22l[:, 22]  # *b1*f^3
                    self.Ploopl[:, 6] = self.P22l[:, 10]  # *b2*f
                    self.Ploopl[:, 7] = self.P22l[:, 16] + self.P22l[:, 17]  # *b2*f^2
                    self.Ploopl[:, 8] = self.P13l[:, 4]  # *b3*f
                    self.Ploopl[:, 9] = self.P22l[:, 11]  # *b4*f
                    self.Ploopl[:, 10] = self.P22l[:, 18] + self.P22l[:, 19]  # *b4*f^2
                    self.Ploopl[:, 11] = self.P22l[:, 0] + self.P13l[:, 0]  # *b1*b1
                    self.Ploopl[:, 12] = self.P22l[:, 6] + self.P13l[:, 2]  # *b1*b1*f
                    self.Ploopl[:, 13] = self.P22l[:, 12] + self.P22l[:, 13]  # *b1*b1*f^2
                    self.Ploopl[:, 14] = self.P22l[:, 1]  # *b1*b2
                    self.Ploopl[:, 15] = self.P22l[:, 7]  # *b1*b2*f
                    self.Ploopl[:, 16] = self.P13l[:, 1]  # *b1*b3
                    self.Ploopl[:, 17] = self.P22l[:, 2]  # *b1*b4
                    self.Ploopl[:, 18] = self.P22l[:, 8]  # *b1*b4*f
                    self.Ploopl[:, 19] = self.P22l[:, 3]  # *b2*b2
                    self.Ploopl[:, 20] = self.P22l[:, 4]  # *b2*b4
                    self.Ploopl[:, 21] = self.P22l[:, 5]  # *b4*b4

                    self.Cloopl[:, 0] = self.C22l[:, 20] + self.C13l[:, 7]  # *f^2
                    self.Cloopl[:, 1] = self.C22l[:, 23] + self.C22l[:, 24] + self.C13l[:, 8] + self.C13l[:, 9]  # *f^3
                    self.Cloopl[:, 2] = self.C22l[:, 25] + self.C22l[:, 26] + self.C22l[:, 27]  # *f^4
                    self.Cloopl[:, 3] = self.C22l[:, 9] + self.C13l[:, 3]  # *b1*f
                    self.Cloopl[:, 4] = (
                        self.C22l[:, 14] + self.C22l[:, 15] + self.C13l[:, 5] + self.C13l[:, 6]
                    )  # *b1*f^2
                    self.Cloopl[:, 5] = self.C22l[:, 21] + self.C22l[:, 22]  # *b1*f^3
                    self.Cloopl[:, 6] = self.C22l[:, 10]  # *b2*f
                    self.Cloopl[:, 7] = self.C22l[:, 16] + self.C22l[:, 17]  # *b2*f^2
                    self.Cloopl[:, 8] = self.C13l[:, 4]  # *b3*f
                    self.Cloopl[:, 9] = self.C22l[:, 11]  # *b4*f
                    self.Cloopl[:, 10] = self.C22l[:, 18] + self.C22l[:, 19]  # *b4*f^2
                    self.Cloopl[:, 11] = self.C22l[:, 0] + self.C13l[:, 0]  # *b1*b1
                    self.Cloopl[:, 12] = self.C22l[:, 6] + self.C13l[:, 2]  # *b1*b1*f
                    self.Cloopl[:, 13] = self.C22l[:, 12] + self.C22l[:, 13]  # *b1*b1*f^2
                    self.Cloopl[:, 14] = self.C22l[:, 1]  # *b1*b2
                    self.Cloopl[:, 15] = self.C22l[:, 7]  # *b1*b2*f
                    self.Cloopl[:, 16] = self.C13l[:, 1]  # *b1*b3
                    self.Cloopl[:, 17] = self.C22l[:, 2]  # *b1*b4
                    self.Cloopl[:, 18] = self.C22l[:, 8]  # *b1*b4*f
                    self.Cloopl[:, 19] = self.C22l[:, 3]  # *b2*b2
                    self.Cloopl[:, 20] = self.C22l[:, 4]  # *b2*b4
                    self.Cloopl[:, 21] = self.C22l[:, 5]  # *b4*b4
                    
                    if self.co.with_uvmatch:
                        self.Ploopl[:, 2] += self.P13l[:, -1] # *f^4
                        self.Ploopl[:, 5] += self.P13l[:, -2] # *b1*f^3
                        self.Ploopl[:, 13] += self.P13l[:, -3] # *b1*b1*f^2

                        self.Cloopl[:, 2] += self.C13l[:, -1] # *f^4
                        self.Cloopl[:, 5] += self.C13l[:, -2] # *b1*f^3
                        self.Cloopl[:, 13] += self.C13l[:, -3] # *b1*b1*f^2

                elif self.co.Nloop is 18:  # config["with_tidal_alignements"] = True
                    f1 = self.f

                    self.Ploopl[:, 0] = (
                        f1 ** 2 * self.P22l[:, 36]
                        + f1 ** 3 * self.P22l[:, 39]
                        + f1 ** 3 * self.P22l[:, 40]
                        + f1 ** 4 * self.P22l[:, 41]
                        + f1 ** 4 * self.P22l[:, 42]
                        + f1 ** 4 * self.P22l[:, 43]
                        + f1 ** 2 * self.P13l[:, 21]
                        + f1 ** 3 * self.P13l[:, 22]
                        + f1 ** 3 * self.P13l[:, 23]
                    )  # *1
                    self.Ploopl[:, 1] = (
                        f1 * self.P22l[:, 20]
                        + f1 ** 2 * self.P22l[:, 27]
                        + f1 ** 2 * self.P22l[:, 28]
                        + f1 ** 3 * self.P22l[:, 37]
                        + f1 ** 3 * self.P22l[:, 38]
                        + f1 * self.P13l[:, 12]
                        + f1 ** 2 * self.P13l[:, 16]
                        + f1 ** 2 * self.P13l[:, 17]
                    )  # *b1
                    self.Ploopl[:, 2] = (
                        f1 * self.P22l[:, 21] + f1 ** 2 * self.P22l[:, 29] + f1 ** 2 * self.P22l[:, 30]
                    )  # *b2
                    self.Ploopl[:, 3] = f1 * self.P13l[:, 13]  # *b3
                    self.Ploopl[:, 4] = (
                        f1 * self.P22l[:, 22] + f1 ** 2 * self.P22l[:, 31] + f1 ** 2 * self.P22l[:, 32]
                    )  # *b4
                    self.Ploopl[:, 5] = (
                        self.P22l[:, 5]
                        + f1 * self.P22l[:, 15]
                        + f1 ** 2 * self.P22l[:, 25]
                        + f1 ** 2 * self.P22l[:, 26]
                        + self.P13l[:, 4]
                        + f1 * self.P13l[:, 9]
                    )  # *b1*b1
                    self.Ploopl[:, 6] = self.P22l[:, 6] + f1 * self.P22l[:, 16]  # *b1*b2
                    self.Ploopl[:, 7] = self.P13l[:, 5]  # *b1*b3
                    self.Ploopl[:, 8] = self.P22l[:, 7] + f1 * self.P22l[:, 17]  # *b1*b4
                    self.Ploopl[:, 9] = self.P22l[:, 9]  # *b2*b2
                    self.Ploopl[:, 10] = self.P22l[:, 10]  # *b2*b4
                    self.Ploopl[:, 11] = self.P22l[:, 12]  # *b4*b4

                    self.Ploopl[:, 12] = (
                        f1 * self.P22l[:, 23]
                        + f1 * self.P22l[:, 24]
                        + f1 ** 2 * self.P22l[:, 33]
                        + f1 ** 2 * self.P22l[:, 34]
                        + f1 ** 2 * self.P22l[:, 35]
                        + f1 * self.P13l[:, 14]
                        + f1 * self.P13l[:, 15]
                        + f1 ** 2 * self.P13l[:, 18]
                        + f1 ** 2 * self.P13l[:, 19]
                        + f1 ** 2 * self.P13l[:, 20]
                    )  # *bq
                    self.Ploopl[:, 13] = (
                        self.P22l[:, 3]
                        + self.P22l[:, 4]
                        + self.P22l[:, 14]
                        + self.P13l[:, 2]
                        + self.P13l[:, 3]
                        + self.P13l[:, 8]
                    )  # *bq*bq
                    self.Ploopl[:, 14] = (
                        self.P22l[:, 0]
                        + self.P22l[:, 8]
                        + f1 * self.P22l[:, 18]
                        + f1 * self.P22l[:, 19]
                        + self.P13l[:, 0]
                        + self.P13l[:, 6]
                        + f1 * self.P13l[:, 10]
                        + f1 * self.P13l[:, 11]
                    )  # *bq*b1
                    self.Ploopl[:, 15] = self.P22l[:, 1] + self.P22l[:, 11]  # *bq*b2
                    self.Ploopl[:, 16] = self.P13l[:, 1] + self.P13l[:, 7]  # *bq*b3
                    self.Ploopl[:, 17] = self.P22l[:, 2] + self.P22l[:, 13]  # *bq*b4

                    self.Cloopl[:, 0] = (
                        f1 ** 2 * self.C22l[:, 36]
                        + f1 ** 3 * self.C22l[:, 39]
                        + f1 ** 3 * self.C22l[:, 40]
                        + f1 ** 4 * self.C22l[:, 41]
                        + f1 ** 4 * self.C22l[:, 42]
                        + f1 ** 4 * self.C22l[:, 43]
                        + f1 ** 2 * self.C13l[:, 21]
                        + f1 ** 3 * self.C13l[:, 22]
                        + f1 ** 3 * self.C13l[:, 23]
                    )  # *1
                    self.Cloopl[:, 1] = (
                        f1 * self.C22l[:, 20]
                        + f1 ** 2 * self.C22l[:, 27]
                        + f1 ** 2 * self.C22l[:, 28]
                        + f1 ** 3 * self.C22l[:, 37]
                        + f1 ** 3 * self.C22l[:, 38]
                        + f1 * self.C13l[:, 12]
                        + f1 ** 2 * self.C13l[:, 16]
                        + f1 ** 2 * self.C13l[:, 17]
                    )  # *b1
                    self.Cloopl[:, 2] = (
                        f1 * self.C22l[:, 21] + f1 ** 2 * self.C22l[:, 29] + f1 ** 2 * self.C22l[:, 30]
                    )  # *b2
                    self.Cloopl[:, 3] = f1 * self.C13l[:, 13]  # *b3
                    self.Cloopl[:, 4] = (
                        f1 * self.C22l[:, 22] + f1 ** 2 * self.C22l[:, 31] + f1 ** 2 * self.C22l[:, 32]
                    )  # *b4
                    self.Cloopl[:, 5] = (
                        self.C22l[:, 5]
                        + f1 * self.C22l[:, 15]
                        + f1 ** 2 * self.C22l[:, 25]
                        + f1 ** 2 * self.C22l[:, 26]
                        + self.C13l[:, 4]
                        + f1 * self.C13l[:, 9]
                    )  # *b1*b1
                    self.Cloopl[:, 6] = self.C22l[:, 6] + f1 * self.C22l[:, 16]  # *b1*b2
                    self.Cloopl[:, 7] = self.C13l[:, 5]  # *b1*b3
                    self.Cloopl[:, 8] = self.C22l[:, 7] + f1 * self.C22l[:, 17]  # *b1*b4
                    self.Cloopl[:, 9] = self.C22l[:, 9]  # *b2*b2
                    self.Cloopl[:, 10] = self.C22l[:, 10]  # *b2*b4
                    self.Cloopl[:, 11] = self.C22l[:, 12]  # *b4*b4

                    self.Cloopl[:, 12] = (
                        f1 * self.C22l[:, 23]
                        + f1 * self.C22l[:, 24]
                        + f1 ** 2 * self.C22l[:, 33]
                        + f1 ** 2 * self.C22l[:, 34]
                        + f1 ** 2 * self.C22l[:, 35]
                        + f1 * self.C13l[:, 14]
                        + f1 * self.C13l[:, 15]
                        + f1 ** 2 * self.C13l[:, 18]
                        + f1 ** 2 * self.C13l[:, 19]
                        + f1 ** 2 * self.C13l[:, 20]
                    )  # *bq
                    self.Cloopl[:, 13] = (
                        self.C22l[:, 3]
                        + self.C22l[:, 4]
                        + self.C22l[:, 14]
                        + self.C13l[:, 2]
                        + self.C13l[:, 3]
                        + self.C13l[:, 8]
                    )  # *bq*bq
                    self.Cloopl[:, 14] = (
                        self.C22l[:, 0]
                        + self.C22l[:, 8]
                        + f1 * self.C22l[:, 18]
                        + f1 * self.C22l[:, 19]
                        + self.C13l[:, 0]
                        + self.C13l[:, 6]
                        + f1 * self.C13l[:, 10]
                        + f1 * self.C13l[:, 11]
                    )  # *bq*b1
                    self.Cloopl[:, 15] = self.C22l[:, 1] + self.C22l[:, 11]  # *bq*b2
                    self.Cloopl[:, 16] = self.C13l[:, 1] + self.C13l[:, 7]  # *bq*b3
                    self.Cloopl[:, 17] = self.C22l[:, 2] + self.C22l[:, 13]  # *bq*b4

                if self.co.Nloop is self.co.N22 + self.co.N13:
                    self.Ploopl[:, : self.co.N22] = self.P22l
                    self.Ploopl[:, self.co.N22 :] = self.P13l
                    self.Cloopl[:, : self.co.N22] = self.C22l
                    self.Cloopl[:, self.co.N22 :] = self.C13l

        else:  # halo-matter
            if self.co.Nloop is 5:
                f1 = self.f

                self.Ploopl[:, 0] = (
                    f1 ** 2 * self.P22l[:, 6]
                    + f1 ** 2 * self.P22l[:, 13]
                    + f1 ** 2 * self.P22l[:, 14]
                    + f1 ** 3 * self.P22l[:, 17]
                    + f1 ** 3 * self.P22l[:, 18]
                    + f1 ** 4 * self.P22l[:, 19]
                    + f1 ** 4 * self.P22l[:, 20]
                    + f1 ** 4 * self.P22l[:, 21]
                    + f1 * self.P13l[:, 4]
                    + f1 ** 2 * self.P13l[:, 7]
                    + f1 ** 2 * self.P13l[:, 8]
                    + f1 ** 3 * self.P13l[:, 9]
                    + f1 ** 3 * self.P13l[:, 10]
                )  # *1
                self.Ploopl[:, 1] = (
                    self.P22l[:, 0]
                    + f1 ** 2 * self.P22l[:, 7]
                    + f1 ** 2 * self.P22l[:, 8]
                    + f1 ** 3 * self.P22l[:, 15]
                    + f1 ** 3 * self.P22l[:, 16]
                    + f1 * self.P13l[:, 2]
                    + f1 ** 2 * self.P13l[:, 5]
                    + f1 ** 2 * self.P13l[:, 6]
                )  # *b1
                self.Ploopl[:, 2] = (
                    self.P22l[:, 1] + f1 * self.P22l[:, 4] + f1 ** 2 * self.P22l[:, 9] + f1 ** 2 * self.P22l[:, 10]
                )  # *b2
                self.Ploopl[:, 3] = self.P13l[:, 1] + f1 * self.P13l[:, 3]  # *b3
                self.Ploopl[:, 4] = (
                    self.P22l[:, 2] + f1 * self.P22l[:, 5] + f1 ** 2 * self.P22l[:, 11] + f1 ** 2 * self.P22l[:, 12]
                )  # *b4

                self.Cloopl[:, 0] = (
                    f1 ** 2 * self.C22l[:, 6]
                    + f1 ** 2 * self.C22l[:, 13]
                    + f1 ** 2 * self.C22l[:, 14]
                    + f1 ** 3 * self.C22l[:, 17]
                    + f1 ** 3 * self.C22l[:, 18]
                    + f1 ** 4 * self.C22l[:, 19]
                    + f1 ** 4 * self.C22l[:, 20]
                    + f1 ** 4 * self.C22l[:, 21]
                    + f1 * self.C13l[:, 4]
                    + f1 ** 2 * self.C13l[:, 7]
                    + f1 ** 2 * self.C13l[:, 8]
                    + f1 ** 3 * self.C13l[:, 9]
                    + f1 ** 3 * self.C13l[:, 10]
                )  # *1
                self.Cloopl[:, 1] = (
                    self.C22l[:, 0]
                    + f1 ** 2 * self.C22l[:, 7]
                    + f1 ** 2 * self.C22l[:, 8]
                    + f1 ** 3 * self.C22l[:, 15]
                    + f1 ** 3 * self.C22l[:, 16]
                    + f1 * self.C13l[:, 2]
                    + f1 ** 2 * self.C13l[:, 5]
                    + f1 ** 2 * self.C13l[:, 6]
                )  # *b1
                self.Cloopl[:, 2] = (
                    self.C22l[:, 1] + f1 * self.C22l[:, 4] + f1 ** 2 * self.C22l[:, 9] + f1 ** 2 * self.C22l[:, 10]
                )  # *b2
                self.Cloopl[:, 3] = self.C13l[:, 1] + f1 * self.C13l[:, 3]  # *b3
                self.Cloopl[:, 4] = (
                    self.C22l[:, 2] + f1 * self.C22l[:, 5] + f1 ** 2 * self.C22l[:, 11] + f1 ** 2 * self.C22l[:, 12]
                )  # *b4

            elif self.co.Nloop is 25:
                pass

        self.subtractShotNoise()

    def reducePsCflf(self):  # depreciated
        f = self.f

        self.Plooplf[:, 0] = f ** 2 * self.Ploopl[:, 0] + f ** 3 * self.Ploopl[:, 1] + f ** 4 * self.Ploopl[:, 2]  # *1
        self.Plooplf[:, 1] = f * self.Ploopl[:, 3] + f ** 2 * self.Ploopl[:, 4] + f ** 3 * self.Ploopl[:, 5]  # *b1
        self.Plooplf[:, 2] = f * self.Ploopl[:, 6] + f ** 2 * self.Ploopl[:, 7]  # *b2
        self.Plooplf[:, 3] = f * self.Ploopl[:, 8]  # *b3
        self.Plooplf[:, 4] = f * self.Ploopl[:, 9] + f ** 2 * self.Ploopl[:, 10]  # *b4
        self.Plooplf[:, 5] = self.Ploopl[:, 11] + f * self.Ploopl[:, 12] + f ** 2 * self.Ploopl[:, 13]  # *b1*b1
        self.Plooplf[:, 6] = self.Ploopl[:, 14] + f * self.Ploopl[:, 15]  # *b1*b2
        self.Plooplf[:, 7] = self.Ploopl[:, 16]  # *b1*b3
        self.Plooplf[:, 8] = self.Ploopl[:, 17] + f * self.Ploopl[:, 18]  # *b1*b4
        self.Plooplf[:, 9] = self.Ploopl[:, 19]  # *b2*b2
        self.Plooplf[:, 10] = self.Ploopl[:, 20]  # *b2*b4
        self.Plooplf[:, 11] = self.Ploopl[:, 21]  # *b4*b4

    # def setreducePslb(self, bs):
    #     """For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the power spectrum multipole regrouped terms and adds the resulting terms together per loop order.

    #     Parameters
    #     ----------
    #     bs : array
    #         An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
    #     """
    #     self.setBias(bs)
    #     self.Ps = np.zeros(
    #         shape=(3, self.P11l.shape[0], self.P11l.shape[-1])
    #     )  # using P11l.shape[0] since it can be either M multipoles or N wedges, and using P11l.shape[-1] since it can be binned or evaluated on the asked output k array
    #     self.Ps[0] = np.einsum("b,lbx->lx", self.b11, self.P11l)
    #     self.Ps[1] = np.einsum("b,lbx->lx", self.bloop, self.Ploopl) + np.einsum("b,lbx->lx", self.bct, self.Pctl)
    #     if self.with_stoch:
    #         self.Ps[1] += np.einsum("b,lbx->lx", self.bst, self.Pstl)
    #     if self.with_nnlo_counterterm:
    #         self.Ps[2] = np.einsum("l,lbx->lx", self.cnnlo, self.Pnnlol)
    #     self.setfullPs()
    
    def setreducePslb(self, bs, what="full"):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the power spectrum multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Ps = [None] * 3
        if "full" in what:
            self.Ps[0] = np.einsum('b,lbx->lx', self.b11, self.P11l)
            self.Ps[1] = np.einsum('b,lbx->lx', self.bloop, self.Ploopl) + np.einsum('b,lbx->lx', self.bct, self.Pctl)
            if self.with_stoch: self.Ps[1] += np.einsum('b,lbx->lx', self.bst, self.Pstl)
            if self.with_nnlo_counterterm: self.Ps[2] = np.einsum('b,lbx->lx', self.cnnlo, self.Pnnlol)
        else:
            if "linear" in what: self.Ps[0] = np.einsum('b,lbx->lx', self.b11, self.P11l)
            if "sptloop" in what:
                if "sptloop22" in what and self.co.keep_loop_pieces_independent: self.Ps[1] = np.einsum('b,lbx->lx', self.bloop[:self.co.N22], self.Ploopl[:, :self.co.N22])
                elif "sptloop13" in what and self.co.keep_loop_pieces_independent: self.Ps[1] = np.einsum('b,lbx->lx', self.bloop[self.co.N22:], self.Ploopl[:, self.co.N22:])
                else: self.Ps[1] = np.einsum('b,lbx->lx', self.bloop, self.Ploopl)
            if "spt" not in what or "counterterm" in what: self.Ps[1] += np.einsum('b,lbx->lx', self.bct, self.Pctl)
            if "stochastic" in what and self.with_stoch: self.Ps[1] += np.einsum('b,lbx->lx', self.bst, self.Pstl)
            if "nnlo_counterterm" in what and self.with_nnlo_counterterm: self.Ps[2] = np.einsum('b,lbx->lx', self.cnnlo, self.Pnnlol)
            if "ir_correction" in what:
                self.Ps[0] = np.einsum('b,lbx->lx', self.b11, self.fullIRPs11)
                self.Ps[1] = np.einsum('b,lbx->lx', self.bloop, self.fullIRPsloop) + np.einsum('b,lbx->lx', self.bct, self.fullIRPsct)
        if self.Ps[2] is None:
            self.Ps[2] = np.zeros_like(self.Ps[0])
        self.Ps = np.array(self.Ps)
        self.setfullPs()

    # def setreduceCflb(self, bs):
    #     """For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the correlation multipole regrouped terms and adds the resulting terms together per loop order.

    #     Parameters
    #     ----------
    #     bs : array
    #         An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
    #     """
    #     self.setBias(bs)
    #     self.Cf = np.zeros(shape=(3, self.C11l.shape[0], self.C11l.shape[-1]))
    #     self.Cf[0] = np.einsum("b,lbx->lx", self.b11, self.C11l)
    #     self.Cf[1] = np.einsum("b,lbx->lx", self.bloop, self.Cloopl) + np.einsum("b,lbx->lx", self.bct, self.Cctl)
    #     if self.with_stoch:
    #         self.Cf[1] += np.einsum("b,lbx->lx", self.bst, self.Cstl)
    #     if self.with_nnlo_counterterm:
    #         self.Cf[2] = np.einsum("l,lbx->lx", self.cnnlo, self.Cnnlol)
    #     self.setfullCf()

    #     self.setreducePslb(bs)  # PZ NNLO
    
    def setreduceCflb(self, bs, what="full"):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the correlation multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Cf = [None] * 3
        self.Cf[0] = np.einsum('b,lbx->lx', self.b11, self.C11l)
        self.Cf[1] = np.einsum('b,lbx->lx', self.bloop, self.Cloopl) + np.einsum('b,lbx->lx', self.bct, self.Cctl)
        if self.with_stoch: self.Cf[1] += np.einsum('b,lbx->lx', self.bst, self.Cstl)
        if self.with_nnlo_counterterm: self.Cf[2] = np.einsum('b,lbx->lx', self.cnnlo, self.Cnnlol)
        if self.Cf[2] is None:
            self.Cf[2] = np.zeros_like(self.Cf[0])
        self.Cf = np.array(self.Cf)
        self.setfullCf()

        self.setreducePslb(bs) # PZ NNLO

    def subtractShotNoise(self):
        """ For option: which='all'. Subtract the constant stochastic term from the (22-)loop """
        for l in range(self.co.Nl):
            for n in range(self.co.Nloop):
                shotnoise = self.Ploopl[l, n, 0]
                self.Ploopl[l, n] -= shotnoise

    # def formatTaylorPs(self, kdata=None, Ps=None):
    #     """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """

    #     allk = (
    #         np.concatenate([self.co.k for i in range(self.co.Nl)]).reshape(-1, 1)
    #         if kdata is None
    #         else np.concatenate([[kdata for i in range(self.co.Nl)]]).reshape(-1, 1)
    #     )
    #     if Ps is None:
    #         P11l, Ploopl, Pctl, Pstl = self.P11l, self.Ploopl, self.Pctl, self.Pstl
    #     else:
    #         P11l, Ploopl, Pctl, Pstl = Ps

    #     Plin = np.flip(np.einsum("n,lnk->lnk", np.array([1.0, 2.0 * self.f, self.f ** 2]), P11l), axis=1)
    #     Plin = np.concatenate(np.einsum("lnk->lkn", Plin), axis=0)
    #     Plin = np.hstack((allk, Plin))
    #     if self.co.Nloop is 12:
    #         Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploopl), axis=0)
    #     elif self.co.Nloop is 22:
    #         Ploop1 = np.empty(shape=(self.co.Nl, 12, np.shape(Ploopl)[-1]))
    #         Ploop1[:, 0] = (
    #             self.f ** 2 * self.Ploopl[:, 0] + self.f ** 3 * self.Ploopl[:, 1] + self.f ** 4 * self.Ploopl[:, 2]
    #         )  # *1
    #         Ploop1[:, 1] = (
    #             self.f * self.Ploopl[:, 3] + self.f ** 2 * self.Ploopl[:, 4] + self.f ** 3 * self.Ploopl[:, 5]
    #         )  # *b1
    #         Ploop1[:, 2] = self.f * self.Ploopl[:, 6] + self.f ** 2 * self.Ploopl[:, 7]  # *b2
    #         Ploop1[:, 3] = self.f * self.Ploopl[:, 8]  # *b3
    #         Ploop1[:, 4] = self.f * self.Ploopl[:, 9] + self.f ** 2 * self.Ploopl[:, 10]  # *b4
    #         Ploop1[:, 5] = self.Ploopl[:, 11] + self.f * self.Ploopl[:, 12] + self.f ** 2 * self.Ploopl[:, 13]  # *b1*b1
    #         Ploop1[:, 6] = self.Ploopl[:, 14] + self.f * self.Ploopl[:, 15]  # *b1*b2
    #         Ploop1[:, 7] = self.Ploopl[:, 16]  # *b1*b3
    #         Ploop1[:, 8] = self.Ploopl[:, 17] + self.f * self.Ploopl[:, 18]  # *b1*b4
    #         Ploop1[:, 9] = self.Ploopl[:, 19]  # *b2*b2
    #         Ploop1[:, 10] = self.Ploopl[:, 20]  # *b2*b4
    #         Ploop1[:, 11] = self.Ploopl[:, 21]  # *b4*b4
    #         Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploop1), axis=0)
    #     Ploop2 = np.einsum("n,lnk->lnk", np.array([2.0, 2.0, 2.0, 2.0 * self.f, 2.0 * self.f, 2.0 * self.f]), Pctl)
    #     Ploop2 = np.concatenate(np.einsum("lnk->lkn", Ploop2), axis=0)
    #     Ploop3 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, 1.0]), Pstl)
    #     Ploop3 = np.concatenate(np.einsum("lnk->lkn", Ploop3), axis=0)
    #     Ploop = np.hstack((allk, Ploop1, Ploop2, Ploop3))

    #     return Plin, Ploop
    
    def formatTaylorPs(self, kdata=None, Ps=None):
        """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """

        allk = (
            np.concatenate([self.co.k for i in range(self.co.Nl)]).reshape(-1, 1)
            if kdata is None
            else np.concatenate([[kdata for i in range(self.co.Nl)]]).reshape(-1, 1)
        )
        # if Ps is None:
        #     P11l, Ploopl, Pctl = self.P11l, self.Ploopl, self.Pctl
        # else:
        #     P11l, Ploopl, Pctl = Ps
        
        if Ps is None:
            P11l, Ploopl, Pctl, Pstl = self.P11l, self.Ploopl, self.Pctl, self.Pstl
        else:
            P11l, Ploopl, Pctl, Pstl = Ps
            
        # print(np.shape(P11l), np.shape(Ploopl), np.shape(Pctl))
        
        if isinstance(self.f, float) == False:
            self.f = self.f[0]
        
        Plin = np.flip(np.einsum("n,lnk->lnk", np.array([1.0, 2.0 * self.f, self.f ** 2]), P11l), axis=1)
        # Plin = np.flip(np.einsum("n,lnk->lnk", np.array([1.0, 1.0 * self.f, self.f ** 2]), P11l), axis=1)
        Plin = np.concatenate(np.einsum("lnk->lkn", Plin), axis=0)
        Plin = np.hstack((allk, Plin))
        
        if self.co.Nloop == 12:
            Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploopl), axis=0)
        elif self.co.Nloop == 22:
            Ploop1 = np.zeros(shape=(self.co.Nl, 12, np.shape(Ploopl)[-1]))
            # Ploop1[:, 0] = (
            #     self.f ** 2 * self.Ploopl[:, 0] + self.f ** 3 * self.Ploopl[:, 1] + self.f ** 4 * self.Ploopl[:, 2]
            # )  # *1
            # Ploop1[:, 1] = (
            #     self.f * self.Ploopl[:, 3] + self.f ** 2 * self.Ploopl[:, 4] + self.f ** 3 * self.Ploopl[:, 5]
            # )  # *b1
            # Ploop1[:, 2] = self.f * self.Ploopl[:, 6] + self.f ** 2 * self.Ploopl[:, 7]  # *b2
            # Ploop1[:, 3] = self.f * self.Ploopl[:, 8]  # *b3
            # Ploop1[:, 4] = self.f * self.Ploopl[:, 9] + self.f ** 2 * self.Ploopl[:, 10]  # *b4
            # Ploop1[:, 5] = self.Ploopl[:, 11] + self.f * self.Ploopl[:, 12] + self.f ** 2 * self.Ploopl[:, 13]  # *b1*b1
            # Ploop1[:, 6] = self.Ploopl[:, 14] + self.f * self.Ploopl[:, 15]  # *b1*b2
            # Ploop1[:, 7] = self.Ploopl[:, 16]  # *b1*b3
            # Ploop1[:, 8] = self.Ploopl[:, 17] + self.f * self.Ploopl[:, 18]  # *b1*b4
            # Ploop1[:, 9] = self.Ploopl[:, 19]  # *b2*b2
            # Ploop1[:, 10] = self.Ploopl[:, 20]  # *b2*b4
            # Ploop1[:, 11] = self.Ploopl[:, 21]  # *b4*b4
            Ploop1[:, 0] = (
                self.f ** 2 * Ploopl[:, 0] + self.f ** 3 * Ploopl[:, 1] + self.f ** 4 * Ploopl[:, 2]
            )  # *1
            Ploop1[:, 1] = (
                self.f * Ploopl[:, 3] + self.f ** 2 * Ploopl[:, 4] + self.f ** 3 * Ploopl[:, 5]
            )  # *b1
            Ploop1[:, 2] = self.f * Ploopl[:, 6] + self.f ** 2 * Ploopl[:, 7]  # *b2
            Ploop1[:, 3] = self.f * Ploopl[:, 8]  # *b3
            Ploop1[:, 4] = self.f * Ploopl[:, 9] + self.f ** 2 * Ploopl[:, 10]  # *b4
            Ploop1[:, 5] = Ploopl[:, 11] + self.f * Ploopl[:, 12] + self.f ** 2 * Ploopl[:, 13]  # *b1*b1
            Ploop1[:, 6] = Ploopl[:, 14] + self.f * Ploopl[:, 15]  # *b1*b2
            Ploop1[:, 7] = Ploopl[:, 16]  # *b1*b3
            Ploop1[:, 8] = Ploopl[:, 17] + self.f * Ploopl[:, 18]  # *b1*b4
            Ploop1[:, 9] = Ploopl[:, 19]  # *b2*b2
            Ploop1[:, 10] = Ploopl[:, 20]  # *b2*b4
            Ploop1[:, 11] = Ploopl[:, 21]  # *b4*b4
            Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploop1), axis=0)
        Ploop2 = np.einsum("n,lnk->lnk", np.array([2.0, 2.0, 2.0, 2.0 * self.f, 2.0 * self.f, 2.0 * self.f]), Pctl)
        # Ploop2 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, 1.0, 1.0 * self.f, 1.0 * self.f, 1.0 * self.f]), Pctl)

        Ploop2 = np.concatenate(np.einsum("lnk->lkn", Ploop2), axis=0)
        Ploop3 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, 1.0]), Pstl)
        # Ploop3 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, self.f]), Pstl)
        Ploop3 = np.concatenate(np.einsum("lnk->lkn", Ploop3), axis=0)
        Ploop = np.hstack((allk, Ploop1, Ploop2, Ploop3))

        return Plin, Ploop
    
    def formatTaylorCf(self, sdata=None, CF=None):
        """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """

        allk = (
            np.concatenate([self.co.s for i in range(self.co.Nl)]).reshape(-1, 1)
            if sdata is None
            else np.concatenate([[sdata for i in range(self.co.Nl)]]).reshape(-1, 1)
        )
        
        if CF is None:
            C11l, Cloopl, Cctl, Cstl = self.C11l, self.Cloopl, self.Cctl, self.Cstl
        else:
            C11l, Cloopl, Cctl, Cstl = CF

        Plin = np.flip(np.einsum("n,lnk->lnk", np.array([1.0, 2.0 * self.f, self.f ** 2]), C11l), axis=1)
        Plin = np.concatenate(np.einsum("lnk->lkn", Plin), axis=0)
        Plin = np.hstack((allk, Plin))
        if self.co.Nloop == 12:
            Ploop1 = np.concatenate(np.einsum("lnk->lkn", Cloopl), axis=0)
        elif self.co.Nloop == 22:
            # Ploop1 = np.einsum(
            #     "n,lnk->lnk",
            #     np.array(
            #         [
            #             self.f ** 2,
            #             self.f ** 3,
            #             self.f ** 4,
            #             self.f,
            #             self.f ** 2,
            #             self.f ** 3,
            #             self.f,
            #             self.f ** 2,
            #             self.f,
            #             self.f,
            #             self.f ** 2,
            #             1.0,
            #             self.f,
            #             self.f ** 2,
            #             1.0,
            #             self.f,
            #             1.0,
            #             1.0,
            #             self.f,
            #             1.0,
            #             1.0,
            #             1.0,
            #         ]
            #     ),
            #     Cloopl,
            # )
            
            Ploop1 = np.zeros((self.co.Nl, 12, np.shape(Cloopl)[-1]))
            Ploop1[:, 0] = (
                self.f ** 2 * Cloopl[:, 0] + self.f ** 3 * Cloopl[:, 1] + self.f ** 4 * Cloopl[:, 2]
            )  # *1
            Ploop1[:, 1] = (
                self.f * Cloopl[:, 3] + self.f ** 2 * Cloopl[:, 4] + self.f ** 3 * Cloopl[:, 5]
            )  # *b1
            Ploop1[:, 2] = self.f * Cloopl[:, 6] + self.f ** 2 * Cloopl[:, 7]  # *b2
            Ploop1[:, 3] = self.f * Cloopl[:, 8]  # *b3
            Ploop1[:, 4] = self.f * Cloopl[:, 9] + self.f ** 2 * Cloopl[:, 10]  # *b4
            Ploop1[:, 5] = Cloopl[:, 11] + self.f * Cloopl[:, 12] + self.f ** 2 * Cloopl[:, 13]  # *b1*b1
            Ploop1[:, 6] = Cloopl[:, 14] + self.f * Cloopl[:, 15]  # *b1*b2
            Ploop1[:, 7] = Cloopl[:, 16]  # *b1*b3
            Ploop1[:, 8] = Cloopl[:, 17] + self.f * Cloopl[:, 18]  # *b1*b4
            Ploop1[:, 9] = Cloopl[:, 19]  # *b2*b2
            Ploop1[:, 10] = Cloopl[:, 20]  # *b2*b4
            Ploop1[:, 11] = Cloopl[:, 21]  # *b4*b4
            
            
            Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploop1), axis=0)
        Ploop2 = np.einsum("n,lnk->lnk", np.array([2.0, 2.0, 2.0, 2.0 * self.f, 2.0 * self.f, 2.0 * self.f]), Cctl)
        Ploop2 = np.concatenate(np.einsum("lnk->lkn", Ploop2), axis=0)
        
        Ploop3 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, 1.0]), Cstl)
        # Ploop3 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, self.f]), Pstl)
        Ploop3 = np.concatenate(np.einsum("lnk->lkn", Ploop3), axis=0)
        Ploop = np.hstack((allk, Ploop1, Ploop2, Ploop3))

        # Ploop = np.hstack((allk, Ploop1, Ploop2))
        return Plin, Ploop

    # def formatShapefitPs(self, kdata=None, Ps=None):
    #     """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """

    #     allk = (
    #         np.concatenate([self.co.k for i in range(self.co.Nl)]).reshape(-1, 1, 1)
    #         if kdata is None
    #         else np.concatenate([[kdata for i in range(self.co.Nl)]]).reshape(-1, 1, 1)
    #     )
    #     # if Ps is None:
    #     #     P11l, Ploopl, Pctl = self.P11l, self.Ploopl, self.Pctl
    #     # else:
    #     #     P11l, Ploopl, Pctl = Ps
        
    #     if Ps is None:
    #         P11l, Ploopl, Pctl, Pstl = self.P11l, self.Ploopl, self.Pctl, self.Pstl
    #     else:
    #         P11l, Ploopl, Pctl, Pstl = Ps
            
    #     # print(np.shape(P11l), np.shape(Ploopl), np.shape(Pctl))
        
    #     num = len(self.f)
        
    #     # Plin = np.flip(np.einsum("n,lnk->lnk", np.array([1.0, 2.0 * self.f, self.f ** 2]), P11l), axis=1)
    #     # Plin = np.concatenate(np.einsum("lnk->lkn", Plin), axis=0)
    #     # Plin = np.hstack((allk, Plin))
        
    #     Plin = np.flip(np.einsum("na, alnk->alnk", np.array([np.ones(num), 2.0 * self.f, self.f ** 2]), P11l), axis=2)
    #     Plin = np.concatenate(np.einsum("alnk-> lkna", Plin), axis = 0)
    #     Plin = np.hstack((np.tile(allk, [1, 1, num]), Plin))
        
        
    #     Ploop2 = np.einsum("na,alnk->alnk", np.array([2.0*np.ones(num), 2.0*np.ones(num), 2.0*np.ones(num), 2.0 * self.f, 2.0 * self.f, 2.0 * self.f]), Pctl)
    #     # Ploop2 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, 1.0, 1.0 * self.f, 1.0 * self.f, 1.0 * self.f]), Pctl)

    #     Ploop2 = np.concatenate(np.einsum("alnk->lkna", Ploop2), axis=0)
        
    #     if self.co.Nloop == 12:
    #         Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploopl), axis=0)
    #     elif self.co.Nloop == 22:
    #         # Ploop1 = np.empty(shape=(self.co.Nl, 12, np.shape(Ploopl)[-1]))
    #         Ploop1 = np.zeros((num, self.co.Nl, 12, np.shape(Ploopl)[-1]))
    #         # Ploop1[:, 0] = (
    #         #     self.f ** 2 * self.Ploopl[:, 0] + self.f ** 3 * self.Ploopl[:, 1] + self.f ** 4 * self.Ploopl[:, 2]
    #         # )  # *1
    #         # Ploop1[:, 1] = (
    #         #     self.f * self.Ploopl[:, 3] + self.f ** 2 * self.Ploopl[:, 4] + self.f ** 3 * self.Ploopl[:, 5]
    #         # )  # *b1
    #         # Ploop1[:, 2] = self.f * self.Ploopl[:, 6] + self.f ** 2 * self.Ploopl[:, 7]  # *b2
    #         # Ploop1[:, 3] = self.f * self.Ploopl[:, 8]  # *b3
    #         # Ploop1[:, 4] = self.f * self.Ploopl[:, 9] + self.f ** 2 * self.Ploopl[:, 10]  # *b4
    #         # Ploop1[:, 5] = self.Ploopl[:, 11] + self.f * self.Ploopl[:, 12] + self.f ** 2 * self.Ploopl[:, 13]  # *b1*b1
    #         # Ploop1[:, 6] = self.Ploopl[:, 14] + self.f * self.Ploopl[:, 15]  # *b1*b2
    #         # Ploop1[:, 7] = self.Ploopl[:, 16]  # *b1*b3
    #         # Ploop1[:, 8] = self.Ploopl[:, 17] + self.f * self.Ploopl[:, 18]  # *b1*b4
    #         # Ploop1[:, 9] = self.Ploopl[:, 19]  # *b2*b2
    #         # Ploop1[:, 10] = self.Ploopl[:, 20]  # *b2*b4
    #         # Ploop1[:, 11] = self.Ploopl[:, 21]  # *b4*b4
            
    #         self.f = self.f.reshape((num, 1, 1))
            
    #         Ploop1[:, :, 0] = (
    #             self.f ** 2 * Ploopl[:, :, 0] + self.f ** 3 * Ploopl[:, :, 1] + self.f ** 4 * Ploopl[:, :, 2]
    #         )  # *1
    #         Ploop1[:, :, 1] = (
    #             self.f * Ploopl[:, :, 3] + self.f ** 2 * Ploopl[:, :, 4] + self.f ** 3 * Ploopl[:, :, 5]
    #         )  # *b1
    #         Ploop1[:, :, 2] = self.f * Ploopl[:, :, 6] + self.f ** 2 * Ploopl[:, :, 7]  # *b2
    #         Ploop1[:, :, 3] = self.f * Ploopl[:, :, 8]  # *b3
    #         Ploop1[:, :, 4] = self.f * Ploopl[:, :, 9] + self.f ** 2 * Ploopl[:, :, 10]  # *b4
    #         Ploop1[:, :, 5] = Ploopl[:, :, 11] + self.f * Ploopl[:, :, 12] + self.f ** 2 * Ploopl[:, :, 13]  # *b1*b1
    #         Ploop1[:, :, 6] = Ploopl[:, :, 14] + self.f * Ploopl[:, :, 15]  # *b1*b2
    #         Ploop1[:, :, 7] = Ploopl[:, :, 16]  # *b1*b3
    #         Ploop1[:, :, 8] = Ploopl[:, :, 17] + self.f * Ploopl[:, :, 18]  # *b1*b4
    #         Ploop1[:, :, 9] = Ploopl[:, :, 19]  # *b2*b2
    #         Ploop1[:, :, 10] = Ploopl[:, :, 20]  # *b2*b4
    #         Ploop1[:, :, 11] = Ploopl[:, :, 21]  # *b4*b4
    #         # Ploop1 = np.concatenate(np.einsum("lnk->lkn", Ploop1), axis=0)
    #         Ploop1 = np.concatenate(np.einsum("alnk->lkna", Ploop1), axis=0)
            
    #     Ploop3 = np.einsum("na,alnk->alnk", np.array([np.ones(num), np.ones(num), np.ones(num)]), Pstl)
    #     # Ploop3 = np.einsum("n,lnk->lnk", np.array([1.0, 1.0, self.f]), Pstl)
        
    #     Ploop3 = np.concatenate(np.einsum("alnk->lkna", Ploop3), axis=0)
    #     Ploop = np.hstack((np.tile(allk, [1, 1, num]), Ploop1, Ploop2, Ploop3))
        
    #     Plin = np.swapaxes(np.reshape(Plin, (self.co.Nl, Plin.shape[-3]//self.co.Nl, Plin.shape[-2], num))[:, :, 1:], axis1=1, axis2=2)
    #     Ploop = np.swapaxes(np.reshape(Ploop, (self.co.Nl, Ploop.shape[-3]//self.co.Nl, Ploop.shape[-2], num))[:, :, 1:], axis1=1, axis2=2)

    #     return Plin[:3], Ploop[:3]

    
    def setIRPs(self, Q=None):

        if Q is None:
            Q = self.Q

        if self.with_bias:
            self.fullIRPs = np.einsum("alpn,apnk->alk", Q, self.IRPs)
        else:
            self.fullIRPs11 = np.einsum("lpn,pnk,pi->lik", Q[0], self.IRPs11, self.co.l11)
            self.fullIRPsct = np.einsum("lpn,pnk,pi->lik", Q[1], self.IRPsct, self.co.lct)
            self.fullIRPsloop = np.einsum("lpn,pink->lik", Q[1], self.IRPsloop)

    def setresumPs(self, setfull=True):

        if self.with_bias:
            self.Ps[:2] += self.fullIRPs
            if setfull is True:
                self.setfullPs()

        else:
            # np.save('check_resum.npy', [self.P11l, self.fullIRPs11])
            self.P11l += self.fullIRPs11
            self.Pctl += self.fullIRPsct
            self.Ploopl += self.fullIRPsloop

    def setresumCf(self, setfull=True):

        if self.with_bias:
            self.Cf[:2] += self.fullIRCf
            if setfull is True:
                self.setfullCf()

        else:
            self.C11l += self.fullIRCf11
            self.Cctl += self.fullIRCfct
            self.Cloopl += self.fullIRCfloop

    def settime(self, cosmo):

        if self.co.nonequaltime:
            Dfid = self.D
            self.setcosmo(cosmo)
            D1 = self.D1 / Dfid
            D2 = self.D2 / Dfid
            Dp2 = D1 * D2
            Dp22 = Dp2 * Dp2
            Dp13 = 0.5 * (D1 ** 3 * D2 + D2 ** 3 * D1)
            tloop = np.concatenate([self.co.N22 * [Dp22], self.co.N13 * [Dp13]])

            if self.co.with_cf:
                self.C11l *= Dp2
                self.Cctl *= Dp2
                self.Cloopl = np.einsum("n,lns->lns", tloop, self.Cloopl)
            else:
                self.P11l *= Dp2
                self.Pctl *= Dp2
                self.Ploopl = np.einsum("n,lnk->lnk", tloop, self.Ploopl)

        else:
            Dfid = self.D
            self.setcosmo(cosmo)
            Dp1 = self.D / Dfid
            Dp2 = Dp1 ** 2

            if self.co.with_cf:
                self.C11l *= Dp2
                self.Cctl *= Dp2
                self.Cloopl *= Dp2 ** 2
            else:
                self.P11l *= Dp2
                self.Pctl *= Dp2
                self.Ploopl *= Dp2 ** 2

            Dp2n = np.concatenate((2 * [self.co.Na * [Dp2 ** (n + 1)] for n in range(self.co.NIR)]))

            self.IRPs11 = np.einsum("n,lnk->lnk", Dp2 * Dp2n, self.IRPs11)
            self.IRPsct = np.einsum("n,lnk->lnk", Dp2 * Dp2n, self.IRPsct)
            self.IRPsloop = np.einsum("n,lmnk->lmnk", Dp2 ** 2 * Dp2n, self.IRPsloop)
            
    def setShapefit(self, factor_m, xdata=None, factor_a = 0.6, factor_kp = 0.03, sigma8_ratio = 1.0, IRPs_all = None, power = 1.0, resum = None):
        
        kmode = xdata
        
        # print(factor_kp)
        
        # factor_kp = 0.0337
        # ratio = np.exp(np.einsum("j, k->jk", factor_m, 1.0/factor_a*np.tanh(factor_a*np.log(kmode/factor_kp))))*sigma8_ratio
        # ratio = np.exp(np.sqrt(np.pi)/2.0*factor_m/factor_a*erf(factor_a*np.log(kmode/factor_kp)))*sigma8_ratio
        # ratio = np.exp(factor_m/factor_a*(factor_a*np.log(kmode/factor_kp))/np.sqrt((factor_a*np.log(kmode/factor_kp))**2+1.0))*sigma8_ratio
        # ratio = np.exp(factor_m/factor_a*np.arctan(factor_a*np.log(kmode/factor_kp)))*sigma8_ratio
        # ratio = np.exp(factor_m/factor_a*np.tanh(factor_a*(1.0-0.5*factor_m)*np.log(kmode/(factor_kp+0.017*factor_m))))*sigma8_ratio
        # ratio = np.exp(factor_m/(factor_a)*np.tanh(factor_a*np.log(kmode/factor_kp)))*np.heaviside(factor_kp-kmode, 0.5) + np.exp(factor_m/(factor_a)*np.arctan(factor_a*np.log(kmode/factor_kp)))*np.heaviside(kmode - factor_kp, 0.5)

        
        ratio = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(kmode/factor_kp)))*sigma8_ratio
        
        # ratio_dash = (factor_m*np.exp((factor_m*np.tanh(factor_a*np.log(kmode/factor_kp)))/factor_a)*np.cosh(factor_a*np.log(kmode/factor_kp))**(-2))/kmode

        # ratio = np.exp(np.einsum('l, k -> lk', factor_m, 1.0/factor_a*np.tanh(factor_a*np.log(kmode/factor_kp))))
        # lambda_IR = 1.0 
        # damping = np.exp(-kmode**2/lambda_IR**2)
        
        # P11l = np.einsum("jk, mnk->jmnk", ratio, self.P11l)
        # Pctl = np.einsum("jk, mnk->jmnk", ratio, self.Pctl)
        # Ploopl = np.einsum("jk, mnk->jmnk", ratio, self.Ploopl)
    
        P11l = self.P11l*ratio
        Pctl = self.Pctl*ratio
        Ploopl = self.Ploopl*ratio ** 2
        # Ploopl = self.Ploopl*ratio * (ratio - (np.exp(factor_m/factor_a)-np.exp(-factor_m/factor_a)))
        
        # P11l = np.einsum('abc, dc -> dabc', self.P11l, ratio)
        # Pctl = np.einsum('abc, dc -> dabc', self.Pctl, ratio)
        # Ploopl = np.einsum('abc, dc -> dabc', self.Ploopl, ratio)
        
        # P11l = np.einsum('abc, c->abc', self.P11l, ratio)
        # Pctl = np.einsum('abc, c->abc', self.Pctl, ratio)
        # Ploopl = np.einsum('abc, c->abc', self.Ploopl, ratio**2)
        # print(np.shape(Ploopl), np.shape(ratio))
        
        # print(np.max(self.Ploopl), np.min(self.Ploopl))
        
        # ratio_dash = (factor_m*np.exp((factor_m*np.tanh(factor_a*np.log(kmode/factor_kp)))/factor_a)*np.cosh(factor_a*np.log(kmode/factor_kp))**(-2))/kmode


        ratio_2n = np.concatenate((2 * [self.co.Na * [ratio ** (n + 1)] for n in range(self.co.NIR)]))
        # ratio_2n = np.concatenate((2 * [self.co.Na * [(ratio-(np.exp(factor_m/factor_a)-np.exp(-factor_m/factor_a))) ** (n+1)] for n in range(self.co.NIR)]))

        
        IRPs11 = np.einsum("nk,lnk->lnk", ratio * ratio_2n, self.IRPs11)
        IRPsct = np.einsum("nk,lnk->lnk", ratio * ratio_2n, self.IRPsct)
        IRPsloop = np.einsum("nk,lmnk->lmnk", (ratio ** 2) * ratio_2n, self.IRPsloop)
        # IRPs11 = np.einsum("nk,lnk->lnk", (ratio-(np.exp(factor_m/factor_a)-np.exp(-factor_m/factor_a))) * ratio_2n, self.IRPs11)
        # IRPsct = np.einsum("nk,lnk->lnk", (ratio-(np.exp(factor_m/factor_a)-np.exp(-factor_m/factor_a))) * ratio_2n, self.IRPsct)
        # IRPsloop = np.einsum("nk,lmnk->lmnk", ((ratio-(np.exp(factor_m/factor_a)-np.exp(-factor_m/factor_a))) ** 2) * ratio_2n, self.IRPsloop)
        # IRPsloop = np.einsum("nk,lmnk->lmnk", (ratio *(ratio - 2.0*ratio_dash)) * ratio_2n, self.IRPsloop)
        
        # IRPs11 = self.IRPs11_interp(factor_m)*self.IRPs11
        # IRPsct = self.IRPsct_interp(factor_m)*self.IRPsct
        # IRPsloop = self.IRPsloop_interp(factor_m)*self.IRPsloop
        
        # IRPs11_in = np.zeros(shape=(self.co.Nl, self.co.Nn, self.co.Nk))
        # IRPsct_in = np.zeros(shape=(self.co.Nl, self.co.Nn, self.co.Nk))
        # IRPsloop_in = np.zeros(shape=(self.co.Nl, self.co.Nloop, self.co.Nn, self.co.Nk))
        
        # self.ratio_new = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(self.kin/factor_kp)))*sigma8_ratio
        # IRPs11, IRPsct, IRPsloop = resum.IRPs(self, IRPs_all = [IRPs11_in, IRPsct_in, IRPsloop_in])

        
        return P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop
    
    