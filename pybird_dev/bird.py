import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from common import co, mu

from greenfunction import GreenFunction

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

    def __init__(self, cosmology=None, with_bias=True, with_stoch=False, with_assembly_bias=False, co=co):
        self.co = co

        self.with_bias = with_bias
        self.with_stoch = with_stoch
        self.with_assembly_bias = with_assembly_bias

        if cosmology is not None: self.setcosmo(cosmology)

        self.P22 = np.empty(shape=(self.co.N22, self.co.Nk))
        self.P13 = np.empty(shape=(self.co.N13, self.co.Nk))
        self.Ps = np.empty(shape=(2, self.co.Nl, self.co.Nk))
        
        self.C11 = np.empty(shape=(self.co.Nl, self.co.Ns))
        self.C22l = np.empty(shape=(self.co.Nl, self.co.N22, self.co.Ns))
        self.C13l = np.empty(shape=(self.co.Nl, self.co.N13, self.co.Ns))
        self.Cct = np.empty(shape=(self.co.Nl, self.co.Ns))
        self.Cf = np.empty(shape=(2, self.co.Nl, self.co.Ns))

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
            self.bst = np.zeros(shape=(self.co.Nl, self.co.Nst))
            self.Pstl = np.zeros(shape=(self.co.Nl, self.co.Nst, self.co.Nk))
            self.Pstl[0,0] = self.co.k**0
            self.Pstl[0,1] = self.co.k**2
            self.Pstl[1,2] = self.co.k**2

    def setcosmo(self, cosmo):


        self.kin = cosmo["k11"]
        self.Pin = cosmo["P11"]
        try: 
            self.Plin = interp1d(self.kin, self.Pin, kind='cubic')
            self.P11 = self.Plin(self.co.k)
        except: 
            self.Plin = None
            self.P11 = None

        if not self.co.with_time: self.D = cosmo["D"]
        self.f = cosmo["f"]
        self.DA = cosmo["DA"]
        self.H = cosmo["H"]

        if self.co.exact_time:
            try:
                try: self.w0 = cosmo["w0_fld"]
                except: self.w0 = None
                self.Omega0_m = cosmo["Omega0_m"]
                self.z = cosmo["z"]
                self.a = 1/(1.+self.z)
                GF = GreenFunction(self.Omega0_m, w=self.w0)
                self.f = GF.fplus(self.a)
                self.Y1 = GF.Y(self.a)
                self.G1t = GF.mG1t(self.a)
                self.V12t = GF.mV12t(self.a)
            except:
                print ("setting EdS time approximation")
                self.Y1 = 0.
                self.G1t = 3/7.
                self.V12t = 1/7.

    def setBias(self, bias):
        """ Given an array of EFT parameters, set them among linear, loops and counter terms, and among multipoles

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """

        f = self.f

        b1 = bias["b1"]
        b2 = bias["b2"]
        b3 = bias["b3"]
        b4 = bias["b4"]
        b5 = bias["cct"] / self.co.km**2
        b6 = bias["cr1"] / self.co.km**2
        b7 = bias["cr2"] / self.co.km**2

        if self.with_assembly_bias: bq = bias["bq"]

        if self.with_bias:
            for i in range(self.co.Nl):
                l = 2 * i
                
                if self.with_assembly_bias: (b1-bq/3.)**2 * mu[0][l] + 2. * (b1-bq/3.) * (f+bq) * mu[2][l] + (f+bq)**2 * mu[4][l]
                else: self.b11[i] = b1**2 * mu[0][l] + 2. * b1 * f * mu[2][l] + f**2 * mu[4][l]
                self.bct[i] = 2. * b1 * (b5 * mu[0][l] + b6 * mu[2][l] + b7 * mu[4][l]) + 2. * f * (b5 * mu[2][l] + b6 * mu[4][l] + b7 * mu[6][l])
                
                if self.co.exact_time:
                    ## EdS: Y1 = 0., G1t = 3/7., V12t = 1/7.
                    Y1 = self.Y1
                    G1t = self.G1t
                    V12t = self.V12t
                    self.b22[i] = np.array([b1**2*mu[0][l], b1*b2*mu[0][l], b1*b4*mu[0][l], b2**2*mu[0][l], b2*b4*mu[0][l], b4**2*mu[0][l], b1**2*f*mu[2][l], b1*b2*f*mu[2][l], b1*b4*f*mu[2][l], b1*f*mu[2][l], b2*f*mu[2][l], b4*f*mu[2][l], b1**2*f**2*mu[2][l], b1**2*f**2*mu[4][l], b1*f**2*mu[2][l], b1*f**2*mu[4][l], b2*f**2*mu[2][l], b2*f**2*mu[4][l], b4*f**2*mu[2][l], b4*f**2*mu[4][l], f**2*mu[4][l], b1*f**3*mu[4][l], b1*f**3*mu[6][l], f**3*mu[4][l], f**3*mu[6][l], f**4*mu[4][l], f**4*mu[6][l], f**4*mu[8][l], b1*f*G1t*mu[2][l], b2*f*G1t*mu[2][l], b4*f*G1t*mu[2][l], b1*f**2*G1t*mu[4][l], f**2*G1t*mu[4][l], f**3*G1t*mu[4][l], f**3*G1t*mu[6][l], f**2*G1t**2*mu[4][l] ])
                    self.b13[i] = np.array([b1**2*mu[0][l], b1*b3*mu[0][l], b1*f*mu[2][l], b3*f*mu[2][l], f**2*mu[4][l], b1**2*Y1*mu[0][l], b1*f*mu[2][l]*Y1, f**2*mu[4][l]*Y1, b1**2*f*G1t*mu[2][l], b1*f**2*G1t*mu[2][l], b1*f**2*G1t*mu[4][l], f**3*G1t*mu[4][l], f**3*G1t*mu[6][l], b1*f*mu[2][l]*V12t, f**2*mu[4][l]*V12t])

                else:
                    self.b22[i] = np.array([b1**2 * mu[0][l], b1 * b2 * mu[0][l], b1 * b4 * mu[0][l], b2**2 * mu[0][l], b2 * b4 * mu[0][l], b4**2 * mu[0][l], b1**2 * f * mu[2][l], b1 * b2 * f * mu[2][l], b1 * b4 * f * mu[2][l], b1 * f * mu[2][l], b2 * f * mu[2][l], b4 * f * mu[2][l], b1**2 * f**2 * mu[2][l], b1**2 * f**2 * mu[4][l], b1 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], b2 * f**2 * mu[2][l], b2 * f**2 * mu[4][l], b4 * f**2 * mu[2][l], b4 * f**2 * mu[4][l], f**2 * mu[4][l], b1 * f**3 * mu[4][l], b1 * f**3 * mu[6][l], f**3 * mu[4][l], f**3 * mu[6][l], f**4 * mu[4][l], f**4 * mu[6][l], f**4 * mu[8][l]])
                    self.b13[i] = np.array([b1**2 * mu[0][l], b1 * b3 * mu[0][l], b1**2 * f * mu[2][l], b1 * f * mu[2][l], b3 * f * mu[2][l], b1 * f**2 * mu[2][l], b1 * f**2 * mu[4][l], f**2 * mu[4][l], f**3 * mu[4][l], f**3 * mu[6][l]])
                
        
        else:
            if self.with_assembly_bias: self.b11 = np.array([(b1-bq/3.)**2, 2. * (b1-bq/3.) * (f+bq), (f+bq)**2])
            else: self.b11 = np.array([b1**2, 2. * b1 * f, f**2])
            self.bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * f * b5, 2. * f * b6, 2. * f * b7])
            if self.co.Nloop is 12: self.bloop = np.array([1., b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
            elif self.co.Nloop is 22: self.bloop = np.array([f**2, f**3, f**4, b1*f, b1*f**2, b1*f**3, b2*f, b2*f**2, b3*f, b4*f, b4*f**2, b1**2, b1**2*f, b1**2*f**2, b1*b2, b1*b2*f, b1*b3, b1*b4, b1*b4*f, b2**2, b2*b4, b4**2])
            elif self.co.Nloop is self.co.N22+self.co.N13: self.bloop = np.array([
                b1**2, b1 * b2, b1 * b4, b2**2, b2 * b4, b4**2, 
                b1**2 * f, b1 * b2 * f, b1 * b4 * f, b1 * f, b2 * f, b4 * f, 
                b1**2 * f**2, b1**2 *f**2, b1 * f**2, b1 * f**2, b2 * f**2, b2 * f**2, b4 * f**2, b4 * f**2, f**2, 
                b1 * f**3, b1 * f**3, f**3, f**3, f**4, f**4, f**4,
                b1**2, b1 * b3, b1**2 * f, b1 * f, b3 * f, b1 * f**2, b1 * f**2, f**2, f**3, f**3])

        if self.with_stoch:
            self.bst[0] = np.array([ bias["ce0"], bias["ce1"] / self.co.km**2, 0. ]) / self.co.nd
            self.bst[1] = np.array([ 0., 0., bias["ce2"] / self.co.km**2 ]) / self.co.nd

    def setPs(self, bs, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiplies them accordingly to the power spectrum multipole terms and adds the resulting terms together per loop order

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
        for l in range(self.co.Nl): self.Ps[1, l] -= self.Ps[1, l, 0]
        self.Ps[1] += np.einsum('lb,bx->lx', self.b13, self.P13) + np.einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11)
        if self.with_stoch: self.Ps[1] += np.einsum('lb,lbx->lx', self.bst, self.Pstl)
        if setfull: self.setfullPs()

    def setCf(self, bs, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)
        self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22l) + np.einsum('lb,lbx->lx', self.b13, self.C13l) + np.einsum('l,lx->lx', self.bct, self.Cct)
        if setfull: self.setfullCf()

    def setPsCf(self, bs, setfull=True):
        """ For option: which='full'. Given an array of EFT parameters, multiply them accordingly to the power spectrum and correlation function multipole terms

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)

        self.Ps[0] = np.einsum('l,x->lx', self.b11, self.P11)
        self.Ps[1] = np.einsum('lb,bx->lx', self.b22, self.P22)
        for l in range(self.co.Nl): self.Ps[1, l] -= self.Ps[1, l, 0]
        self.Ps[1] += np.einsum('lb,bx->lx', self.b13, self.P13) + np.einsum('l,x,x->lx', self.bct, self.co.k**2, self.P11)
        if self.with_stoch: self.Ps[1] += np.einsum('lb,lbx->lx', self.bst, self.Pstl)
        if setfull: self.setfullPs()

        self.Cf[0] = np.einsum('l,lx->lx', self.b11, self.C11)
        self.Cf[1] = np.einsum('lb,lbx->lx', self.b22, self.C22l) + np.einsum('lb,lbx->lx', self.b13, self.C13l) + np.einsum('l,lx->lx', self.bct, self.Cct)
        if setfull: self.setfullCf()

    def setfullPs(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full power spectrum multipoles """
        self.fullPs = np.sum(self.Ps, axis=0)

    def setfullCf(self):
        """ For option: which='full'. Adds together the linear and the loop parts to get the full correlation function multipoles """
        self.fullCf = np.sum(self.Cf, axis=0)

    def setPsCfl(self):
        """ For option: which='all'. Creates multipoles for each term weighted accordingly """
        self.P11l = np.einsum('x,ln->lnx', self.P11, self.co.l11)
        self.Pctl = np.einsum('x,x,ln->lnx', self.co.k**2, self.P11, self.co.lct)
        self.P22l = np.einsum('nx,ln->lnx', self.P22, self.co.l22)
        self.P13l = np.einsum('nx,ln->lnx', self.P13, self.co.l13)

        self.C11l = np.einsum('lx,ln->lnx', self.C11, self.co.l11)
        self.Cctl = np.einsum('lx,ln->lnx', self.Cct, self.co.lct)
        self.C22l = np.einsum('lnx,ln->lnx', self.C22l, self.co.l22)
        self.C13l = np.einsum('lnx,ln->lnx', self.C13l, self.co.l13)

        self.reducePsCfl()

    def reducePsCfl(self):
        """ For option: which='all'. Regroups terms that share the same EFT parameter(s) """

        # if self.co.Nloop is self.co.N22 + self.co.N13:
        #     self.Ploopl[:, :self.co.N22] = self.P22l
        #     self.Ploopl[:, self.co.N22:] = self.P13l
        #     self.Cloopl[:, :self.co.N22] = self.C22l
        #     self.Cloopl[:, self.co.N22:] = self.C13l

        if self.co.exact_time:
            if self.co.Nloop is 12:

                f1 = self.f

                ## EdS: Y1 = 0., G1t = 3/7., V12t = 1/7.
                Y1 = self.Y1
                G1t = self.G1t
                V12t = self.V12t

                self.Ploopl[:, 0] = f1**2 * self.P22l[:, 20] + f1**3 * self.P22l[:, 23] + f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + \
                    G1t * f1**2 * self.P22l[:, 32] + G1t * f1**3 * self.P22l[:, 33] + G1t * f1**3 * self.P22l[:, 34] + G1t**2 * f1**2 * self.P22l[:, 35] + \
                    f1**2 * self.P13l[:, 4] + Y1 * f1**2 * self.P13l[:, 7] + G1t * f1**3 * self.P13l[:, 11] + G1t * f1**3 * self.P13l[:, 12] + V12t * f1**2 * self.P13l[:, 14] # *1
                self.Ploopl[:, 1] = f1 * self.P22l[:, 9] + f1**2 * self.P22l[:, 14] + f1**2 * self.P22l[:, 15] + f1**3 * self.P22l[:, 21] + f1**3 * self.P22l[:, 22] + G1t * f1 * self.P22l[:, 28] + G1t * f1**2 * self.P22l[:, 31] + \
                    f1 * self.P13l[:, 2] + Y1 * f1 * self.P13l[:, 6] + G1t * f1**2 * self.P13l[:, 9] + G1t * f1**2 * self.P13l[:, 10] + V12t * f1 * self.P13l[:, 13] # *b1
                self.Ploopl[:, 2] = f1 * self.P22l[:, 10] + f1**2 * self.P22l[:, 16] + f1**2 * self.P22l[:, 17] + G1t * f1 * self.P22l[:, 29] # *b2
                self.Ploopl[:, 3] = f1 * self.P13l[:, 3] # *b3
                self.Ploopl[:, 4] = f1 * self.P22l[:, 11] + f1**2 * self.P22l[:, 18] + f1**2 * self.P22l[:, 19] + G1t * f1 * self.P22l[:, 30] # *b4
                self.Ploopl[:, 5] = self.P22l[:, 0] + f1 * self.P22l[:, 6] + f1**2 * self.P22l[:, 12] + f1**2 * self.P22l[:, 13] + self.P13l[:, 0] + Y1 * self.P13l[:, 5] + G1t * f1 * self.P13l[:, 8]  # *b1*b1
                self.Ploopl[:, 6] = self.P22l[:, 1] + f1 * self.P22l[:, 7]  # *b1*b2
                self.Ploopl[:, 7] = self.P13l[:, 1]  # *b1*b3
                self.Ploopl[:, 8] = self.P22l[:, 2] + f1 * self.P22l[:, 8]  # *b1*b4
                self.Ploopl[:, 9] = self.P22l[:, 3]  # *b2*b2
                self.Ploopl[:, 10] = self.P22l[:, 4]  # *b2*b4
                self.Ploopl[:, 11] = self.P22l[:, 5]  # *b4*b4

                self.Cloopl[:, 0] = f1**2 * self.C22l[:, 20] + f1**3 * self.C22l[:, 23] + f1**3 * self.C22l[:, 24] + f1**4 * self.C22l[:, 25] + f1**4 * self.C22l[:, 26] + f1**4 * self.C22l[:, 27] + \
                    G1t * f1**2 * self.C22l[:, 32] + G1t * f1**3 * self.C22l[:, 33] + G1t * f1**3 * self.C22l[:, 34] + G1t**2 * f1**2 * self.C22l[:, 35] + \
                    f1**2 * self.C13l[:, 4] + Y1 * f1**2 * self.C13l[:, 7] + G1t * f1**3 * self.C13l[:, 11] + G1t * f1**3 * self.C13l[:, 12] + V12t * f1**2 * self.C13l[:, 14] # *1
                self.Cloopl[:, 1] = f1 * self.C22l[:, 9] + f1**2 * self.C22l[:, 14] + f1**2 * self.C22l[:, 15] + f1**3 * self.C22l[:, 21] + f1**3 * self.C22l[:, 22] + G1t * f1 * self.C22l[:, 28] + G1t * f1**2 * self.C22l[:, 31] + \
                    f1 * self.C13l[:, 2] + Y1 * f1 * self.C13l[:, 6] + G1t * f1**2 * self.C13l[:, 9] + G1t * f1**2 * self.C13l[:, 10] + V12t * f1 * self.C13l[:, 13] # *b1
                self.Cloopl[:, 2] = f1 * self.C22l[:, 10] + f1**2 * self.C22l[:, 16] + f1**2 * self.C22l[:, 17] + G1t * f1 * self.C22l[:, 29] # *b2
                self.Cloopl[:, 3] = f1 * self.C13l[:, 3] # *b3
                self.Cloopl[:, 4] = f1 * self.C22l[:, 11] + f1**2 * self.C22l[:, 18] + f1**2 * self.C22l[:, 19] + G1t * f1 * self.C22l[:, 30] # *b4
                self.Cloopl[:, 5] = self.C22l[:, 0] + f1 * self.C22l[:, 6] + f1**2 * self.C22l[:, 12] + f1**2 * self.C22l[:, 13] + self.C13l[:, 0] + Y1 * self.C13l[:, 5] + G1t * f1 * self.C13l[:, 8]  # *b1*b1
                self.Cloopl[:, 6] = self.C22l[:, 1] + f1 * self.C22l[:, 7]  # *b1*b2
                self.Cloopl[:, 7] = self.C13l[:, 1]  # *b1*b3
                self.Cloopl[:, 8] = self.C22l[:, 2] + f1 * self.C22l[:, 8]  # *b1*b4
                self.Cloopl[:, 9] = self.C22l[:, 3]  # *b2*b2
                self.Cloopl[:, 10] = self.C22l[:, 4]  # *b2*b4
                self.Cloopl[:, 11] = self.C22l[:, 5]  # *b4*b4

        else:

            if self.co.Nloop is 12:
                f1 = self.f

                self.Ploopl[:, 0] = f1**2 * self.P22l[:, 20] + f1**3 * self.P22l[:, 23] + f1**3 * self.P22l[:, 24] + f1**4 * self.P22l[:, 25] + \
                    f1**4 * self.P22l[:, 26] + f1**4 * self.P22l[:, 27] + f1**2 * self.P13l[:, 7] + f1**3 * self.P13l[:, 8] + f1**3 * self.P13l[:, 9]  # *1
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

                self.Cloopl[:, 0] = f1**2 * self.C22l[:, 20] + f1**3 * self.C22l[:, 23] + f1**3 * self.C22l[:, 24] + f1**4 * self.C22l[:, 25] + \
                    f1**4 * self.C22l[:, 26] + f1**4 * self.C22l[:, 27] + f1**2 * \
                    self.C13l[:, 7] + f1**3 * self.C13l[:, 8] + f1**3 * self.C13l[:, 9]  # *1
                self.Cloopl[:, 1] = f1 * self.C22l[:, 9] + f1**2 * self.C22l[:, 14] + f1**2 * self.C22l[:, 15] + f1**3 * self.C22l[:, 21] + f1**3 * self.C22l[:, 22] + f1 * self.C13l[:, 3] + f1**2 * self.C13l[:, 5] + f1**2 * self.C13l[:, 6]  # *b1
                self.Cloopl[:, 2] = f1 * self.C22l[:, 10] + f1**2 * self.C22l[:, 16] + f1**2 * self.C22l[:, 17]  # *b2
                self.Cloopl[:, 3] = f1 * self.C13l[:, 4]  # *b3
                self.Cloopl[:, 4] = f1 * self.C22l[:, 11] + f1**2 * self.C22l[:, 18] + f1**2 * self.C22l[:, 19]  # *b4
                self.Cloopl[:, 5] = self.C22l[:, 0] + f1 * self.C22l[:, 6] + f1**2 * self.C22l[:, 12] + \
                    f1**2 * self.C22l[:, 13] + self.C13l[:, 0] + f1 * self.C13l[:, 2]  # *b1*b1
                self.Cloopl[:, 6] = self.C22l[:, 1] + f1 * self.C22l[:, 7]  # *b1*b2
                self.Cloopl[:, 7] = self.C13l[:, 1]  # *b1*b3
                self.Cloopl[:, 8] = self.C22l[:, 2] + f1 * self.C22l[:, 8]  # *b1*b4
                self.Cloopl[:, 9] = self.C22l[:, 3]  # *b2*b2
                self.Cloopl[:, 10] = self.C22l[:, 4]  # *b2*b4
                self.Cloopl[:, 11] = self.C22l[:, 5]  # *b4*b4

            elif self.co.Nloop is 22:
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

        self.subtractShotNoise()

    def reducePsCflf(self):
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

    def setreducePslb(self, bs):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the power spectrum multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)

        # self.Ps[0] = np.einsum('b,lbx->lx', b11, self.P11l)
        # self.Ps[1] = np.einsum('b,lbx->lx', bloop, self.Ploopl)
        # for l in range(self.co.Nl): self.Ps[1,l] -= self.Ps[1,l,0]
        # self.Ps[1] += np.einsum('b,lbx->lx', bct, self.Pctl)

        Ps0 = np.einsum('b,lbx->lx', self.b11, self.P11l)
        Ps1 = np.einsum('b,lbx->lx', self.bloop, self.Ploopl) + np.einsum('b,lbx->lx', self.bct, self.Pctl)

        if self.with_stoch: Ps1 += np.einsum('lb,lbx->lx', self.bst, self.Pstl)

        self.fullPs = Ps0 + Ps1

    def setreduceCflb(self, bs):
        """ For option: which='all'. Given an array of EFT parameters, multiply them accordingly to the correlation multipole regrouped terms and adds the resulting terms together per loop order.

        Parameters
        ----------
        bs : array
            An array of 7 EFT parameters: b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c_{r,2}/k_{m}^2
        """
        self.setBias(bs)

        Cf0 = np.einsum('b,lbx->lx', self.b11, self.C11l)
        Cf1 = np.einsum('b,lbx->lx', self.bloop, self.Cloopl) + np.einsum('b,lbx->lx', self.bct, self.Cctl)
        self.fullCf = Cf0 + Cf1

    def subtractShotNoise(self):
        """ For option: which='all'. Subtract the constant stochastic term from the (22-)loop """
        for l in range(self.co.Nl):
            for n in range(self.co.Nloop):
                shotnoise = self.Ploopl[l, n, 0]
                self.Ploopl[l, n] -= shotnoise

    def formatTaylor(self):
        """ An auxiliary to pipe PyBird with TBird: puts Bird(object) power spectrum multipole terms into the right shape for TBird """
        allk = np.concatenate([self.co.k, self.co.k]).reshape(-1, 1)
        Plin = np.flip(np.einsum('n,lnk->lnk', np.array([1., 2. * self.f, self.f**2]), self.P11l), axis=1)
        Plin = np.concatenate(np.einsum('lnk->lkn', Plin), axis=0)
        Plin = np.hstack((allk, Plin))
        Ploop1 = np.concatenate(np.einsum('lnk->lkn', self.Ploopl), axis=0)
        Ploop2 = np.einsum('n,lnk->lnk', np.array([2., 2., 2., 2. * self.f, 2. * self.f, 2. * self.f]), self.Pctl)
        Ploop2 = np.concatenate(np.einsum('lnk->lkn', Ploop2), axis=0)
        Ploop = np.hstack((allk, Ploop1, Ploop2))
        return Plin, Ploop

    def setIRPs(self, Q=None):

        if Q is None: Q = self.Q

        if self.with_bias:
            self.fullIRPs = np.einsum('alpn,apnk->alk', Q, self.IRPs)

        else:
            self.fullIRPs11 = np.einsum('lpn,pnk,pi->lik', Q[0], self.IRPs11, self.co.l11)
            self.fullIRPsct = np.einsum('lpn,pnk,pi->lik', Q[1], self.IRPsct, self.co.lct)
            self.fullIRPsloop = np.einsum('lpn,pink->lik', Q[1], self.IRPsloop)

    def setresumPs(self, setfull=True):

        if self.with_bias:
            self.Ps += self.fullIRPs
            if setfull is True: self.setfullPs()

        else:
            self.P11l += self.fullIRPs11
            self.Pctl += self.fullIRPsct
            self.Ploopl += self.fullIRPsloop

    def setresumCf(self, setfull=True):

        if self.with_bias:
            self.Cf += self.fullIRCf
            if setfull is True: self.setfullCf()

        else:
            self.C11l += self.fullIRCf11
            self.Cctl += self.fullIRCfct
            self.Cloopl += self.fullIRCfloop

    def settime(self, cosmo):

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

        Dp2n = np.concatenate(( 2*[self.co.Na*[Dp2**(n+1)] for n in range(self.co.NIR)] ))

        self.IRPs11 = np.einsum('n,lnk->lnk', Dp2*Dp2n, self.IRPs11)
        self.IRPsct = np.einsum('n,lnk->lnk', Dp2*Dp2n, self.IRPsct)
        self.IRPsloop = np.einsum('n,lmnk->lmnk', Dp2**2*Dp2n, self.IRPsloop)

    def setw(self, bs):
        
        self.setBias(bs)

        # b1, b2, b3, b4, b5, b6, b7 = bs

        # self.b11 = np.array([b1**2, 2. * b1, 1.])
        # self.bct = np.array([2. * b1 * b5, 2. * b1 * b6, 2. * b1 * b7, 2. * b5, 2. * b6, 2. * b7])
        # self.bloop = np.array([
        #         b1**2, b1 * b2, b1 * b4, b2**2, b2 * b4, b4**2, 
        #         b1**2, b1 * b2, b1 * b4, b1, b2, b4, 
        #         b1**2, b1**2, b1, b1, b2, b2, b4, b4, 1., 
        #         b1, b1, 1., 1., 1., 1., 1.,
        #         b1**2, b1 * b3, b1**2, b1, b3, b1, b1, 1., 1, 1.])

        #self.bct = np.zeros(shape=(self.wct.shape[0]))
        #self.bloop = np.zeros(shape=(self.wloop.shape[0]))

        self.w = np.einsum('n,nx->x', self.b11, self.wlin) + np.einsum('n,nx->x', self.bct, self.wct) + np.einsum('n,nx->x', self.bloop, self.wloop)

