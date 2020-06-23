import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from scipy.special import gamma
from fftlog import FFTLog, MPC
from nonlinear import M13a, M22a
from common import co

#### LOOP OVER nlens and nsource !!!

class Limber(object):
    """
    ...

    Attributes
    ----------
    co : class
        An object of type Common() used to share data
    """

    def __init__(self, theta, z, nlens, nsource, gg=True, load=True, save=True, path='./', NFFT=256, km=1.):

        self.gg = gg
        self.km = km

        self.z = z
        self.theta, _ = np.meshgrid(theta, z, indexing='ij')

        self.nlens = np.asarray(nlens)
        self.nsource = np.asarray(nsource)
        self.Ng = self.nlens.shape[0]
        self.Ns = self.nsource.shape[0]
        self.Nss = self.Ns*(self.Ns+1)//2
        self.Nsg = self.Ns*self.Ng
        self.Ngg = self.Ng
        self.N = max([self.Nss, self.Nsg])
        
        self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1.e3, bias=-1.6)
        self.fft = FFTLog(**self.fftsettings)

        if self.gg: self.pyegg = os.path.join(path, 'pyegg%s_limber.npz') % (NFFT)
        else: self.pyegg = os.path.join(path, 'pyegg%s_limber_nogg.npz') % (NFFT)

        if load is True:
            try:
                L = np.load( self.pyegg )
                if (self.fft.Pow - L['Pow']).any():
                    print ('Loaded loop matrices do not correspond to asked FFTLog configuration. \n Computing new matrices.')
                    load = False
                else:
                    self.M11, self.M22, self.M13, self.Mct = L['M11'], L['M22'], L['M13'], L['Mct']
                    save = False
            except:
                print ('Can\'t load loop matrices at %s. \n Computing new matrices.' % path)
                load = False

        if load is False:
            self.setM()
            self.setM11()
            self.setMct()
            self.setM13()
            self.setM22()

        if save is True:
            try: np.savez(self.pyegg, Pow=self.fft.Pow, M11=self.M11, M22=self.M22, M13=self.M13, Mct=self.Mct)
            except: print ('Can\'t save loop matrices at %s.' % path)

        self.setsPow()

        # To speed-up matrix multiplication:
        self.optipath13 = np.einsum_path('ns,ms,bnm->bs', self.sPow, self.sPow, self.M22, optimize='optimal')[0]
        self.optipath22 = np.einsum_path('ns,ms,bnm->bs', self.sPow, self.sPow, self.M13, optimize='optimal')[0]

    def setsPow(self):
        """ Compute the r's to the powers of the FFTLog to evaluate the loop 'ular' correlation function. Called at the instantiation of the class. """
        #slog = np.geomspace(1e-4, 3., 40)
        #slin = np.arange(3, 200., 1)
        #slog2 = np.geomspace(200, 1e4, 20)
        #self.s = np.unique(np.concatenate([slog, slin, slog2]))
        self.s = np.geomspace(1.e-4, 1.e3, 200)
        self.sPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3. - 0.5, log(self.s)))

    def setM(self):
        """ Compute the power spectrum to 'ular' correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        M = np.empty(shape=(3, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        for l in range(3):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    self.M[l, u, v] = (2*pi)**.5 * MPC(2 * l - 0.5, n1 + n2 - 1.5)

    def setM22(self):
        """ Compute the 22-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mbb22 = np.empty(shape=(6, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        self.Mbm22 = np.empty(shape=(3, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        self.Mmm22 = np.empty(shape=(2, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        Ma = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex') # common piece of M22
        Mmm = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex') # matter-matter M22
        for u, n1 in enumerate(-0.5 * self.fft.Pow):
            for v, n2 in enumerate(-0.5 * self.fft.Pow):
                Ma[u, v] = M22a(n1, n2)
                Mmm[u, v] = M22mm[0](n1, n2)
        for i in range(6):
            Mbb = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
            Mbm = np.empty(shape=(self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    Mbb[u, v] = M22bb[i](n1, n2)
                    if i < 3: Mbm[u, v] = M22bb[i](n1, n2)
            self.Mbb22[i] = Mbb
            if i < 3: self.Mbm22[i] = Mbm
        self.Mbb22 = np.einsum('nm,nm,bnm->bnm', self.M[0], Ma, self.Mbb22)
        self.Mbm22 = np.einsum('nm,nm,bnm->bnm', self.M[1], Ma, self.Mbm22)
        self.Mmm22 = np.einsum('lnm,nm,nm->lnm', self.M[[0,2]], Ma, Mmm)
        if self.gg: self.M22 = np.hstack([self.Mmm22, self.Mbm22, self.Mbb22])
        else: self.M22 = np.hstack([self.Mmm22, self.Mbm22])

    def setM13(self):
        """ Compute the 13-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mbb13 = np.empty(shape=(2, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        self.Mbm13 = np.empty(shape=(2, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        self.Mmm13 = np.empty(shape=(2, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        Ma = M13a(-0.5 * self.fft.Pow)
        Mmm = M13mm[0](-0.5 * self.fft.Pow)
        for i in range(2): 
            self.Mbb13[i] = M13bb[i](-0.5 * self.fft.Pow)
            self.Mbm13[i] = M13bm[i](-0.5 * self.fft.Pow)
        self.Mbb13 = np.einsum('nm,n,bn->bnm', self.M[0], Ma, self.Mbb13)
        self.Mbm13 = np.einsum('nm,n,bn->bnm', self.M[1], Ma, self.Mbm13)
        self.Mmm13 = np.einsum('lnm,n,n->lnm', self.M[[0,2]], Ma, Mmm)
        if self.gg: self.M13 = np.hstack([self.Mmm13, self.Mbm13, self.Mbb13])
        else: self.M22 = np.hstack([self.Mmm13, self.Mbm13])

    def setM11(self):
        """ Compute the linear matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M11 = np.empty(shape=(3, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(3): self.M11[l] = (2*pi)**.5 * MPC(2 * l - 0.5, -0.5 * self.fft.Pow)

    def setMct(self):
        """ Compute the counterterm matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mct = np.empty(shape=(3, self.fft.Pow.shape[0]), dtype='complex')
        for l in range(3): self.Mct[l, u] = (2*pi)**.5 * MPC(2 * l - 0.5, -0.5 * self.fft.Pow - 1.)

    def getA11(self, CoefsPow):
        """ Perform the linear correlation function matrix multiplications """
        A11 = np.real(np.einsum('ns,ln->ls', CoefsPow, self.M11))
        if self.gg: return np.array([A11[0], A11[2], A11[1], A11[0]])
        else: return np.array([A11[0], A11[2], A11[1]])

    def getAct(self, CoefsPow):
        """ Perform the counterterm correlation function matrix multiplications """
        Act = self.s**-2 * np.real(np.einsum('ns,ln->ls', CoefsPow, self.Mct))
        if self.gg: return np.array([Act[0], Act[2], Act[1], Act[0]])
        else: return np.array([Act[0], Act[2], Act[1]])

    def getA22(self, CoefsPow):
        """ Perform the 22-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.M22, optimize=self.optipath22))

    def getA13(self, CoefsPow):
        """ Perform the 13-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.M13, optimize=self.optipath13))

    def getA(self, bird, window=None):
        coef = self.fft.Coef(bird.kin, bird.kin**-0.5 * self.Pin, window=.2)
        coefsPow = np.einsum('n,ns->ns', Coef, self.sPow)
        A11 = self.getA11(coefsPow, bird)
        Act = self.getAct(coefsPow, bird)
        A22 = self.getA22(coefsPow, bird)
        A13 = self.getA13(coefsPow, bird)
        return A11, Act, A22, A13

    def Xi(self, bird, rz, dz_by_dr, Dz, Dfid, h, Omega0_m):

        Dp2 = Dz**2 / Dfid**2
        Dp4 = D2**2

        lensing_factor = 1.5/conts.c**2 * h**2 * 1e10 * Omega0_m
        r1, _ = np.meshgrid(rz, self.z, indexing='ij')
        def lensing_efficiency(nz):
            return lensing_factor * rz * (1+z) * np.trapz(np.heaviside(rz-r1, 0.) * nz * (rz-r1)/rz, x=self.z, axis=-1)
        
        qshear = np.empty_like(self.nsource)
        qgal = np.empty_like(self.nlens)
        for i, ns in enumerate(self.nsource): qshear[i] = self.lensing_efficiency(ns)
        for i, nl in enumerate(self.nlens): qgal[i] = dz_by_dr * nl
        
        qsqs = np.zeros(shape=(self.N))
        for i, qi in enumerate(qshear):
            for j, qj in enumerate(qshear):
                if qj <= qi: qsqs[i+j] = qi*qj

        qsqg = np.zeros(shape=(self.N))
        for i, qi in enumerate(qshear):
            for j, qj in enumerate(qgal):
                qsqg[i+j] = qi*qj

        if self.gg: 
            qgqg = np.zeros(shape=(self.N))
            for i, qi in enumerate(qgal):
                qgqg[i] = qi**2

        if self.gg:
            qq11 = np.array([qsqs, qsqs, qsqd, qgqg])
            qq13 = np.array([qsqs, qsqs, qsqd, qsqd, qgqg, qgqg])
            qq22 = np.array([qsqs, qsqs, qsqd, qsqd, qgqg, qgqg, qgqg, qgqg, qgqg, qgqg])
        else:
            qq11 = np.array([qsqs, qsqs, qsqd])
            qq13 = np.array([qsqs, qsqs, qsqd, qsqd])
            qq22 = np.array([qsqs, qsqs, qsqd, qsqd])

        def time_integral(qq, DD, A):
            A1 = interp1d(self.s, A, kind='cubic', axis=-1)(self.theta * rz)
            return np.trapz(np.einsum('biz,z,btz->bitz', qq, DD, A1), x=rz, axis=-1)
        
        A11 = time_integral(qq11, Dp2, A11)
        Act = time_integral(qq11, Dp2, Act)
        A13 = time_integral(qq13, Dp4, A13)
        A22 = time_integral(qq22, Dp4, A22)

        self.Assp = np.array([A11[0], Act[0], A13[0], A22[0]])[:,:self.Nss]
        self.Assm = np.array([A11[1], Act[1], A13[1], A22[1]])[:,:self.Nss]
        self.Asg = np.array([A11[2], Act[2], A13[2], A13[3], A22[2], A22[3], A22[4]])[:,self.Nsg]
        if self.gg: self.Agg = np.array([A11[3], Act[3], A13[4], A13[5], A22[5], A22[6], A22[7], A22[8], A22[9], A22[10]])[:,:self.Ngg]

    def setBias(self, bias):

        b1 = bias["b1"]
        b2 = bias["b2"]
        b3 = bias["b3"]
        b4 = bias["b4"]
        css = bias["css"] / self.km**2
        csg = bias["csg"] / self.km**2
        if self.gg: cgg = bias["cgg"] / self.km**2

        self.bss = np.array([1., 2.*css, 1., 1.])
        self.Xssp = np.einsum('b,bitz->itz', bss, self.Assp)
        self.Xssm = np.einsum('b,bitz->itz', bss, self.Assm)

        self.bsg = np.array([b1, 2.*csg, b1, b3, b1, b2, b4])
        self.Xsg = np.einsum('b,bitz->itz', bsg, self.Asg)

        if self.gg: 
            bgg = np.array([b1**2 + 2.*b1*cgg, b1**2, b1*b3, b1**2, b1*b2, b1*b4, b2**2, b2*b4, b4**2])
            self.Xgg = np.einsum('b,bitz->itz', bgg, self.Agg)
    
M22bb = { # galaxy-galaxy
    0: lambda n1, n2: (6 + n1**4 * (4 - 24 * n2) - 7 * n2 + 8 * n1**5 * n2 - 13 * n2**2 + 4 * n2**3 + 4 * n2**4 + n1**2 * (-13 + 38 * n2 + 12 * n2**2 - 8 * n2**3) + 2 * n1**3 * (2 - 5 * n2 - 4 * n2**2 + 8 * n2**3) + n1 * (-7 - 6 * n2 + 38 * n2**2 - 10 * n2**3 - 24 * n2**4 + 8 * n2**5)) / (4. * n1 * (1 + n1) * (-1 + 2 * n1) * n2 * (1 + n2) * (-1 + 2 * n2)),
    1: lambda n1, n2: (-18 + n1**2 * (1 - 11 * n2) - 12 * n2 + n2**2 + 10 * n2**3 + 2 * n1**3 * (5 + 7 * n2) + n1 * (-12 - 38 * n2 - 11 * n2**2 + 14 * n2**3)) / (7. * n1 * (1 + n1) * n2 * (1 + n2)),
    2: lambda n1, n2: (-3 * n1 + 2 * n1**2 + n2 * (-3 + 2 * n2)) / (n1 * n2),
    3: lambda n1, n2: (-4 * (-24 + n2 + 10 * n2**2) + 2 * n1 * (-2 + 51 * n2 + 21 * n2**2) + n1**2 * (-40 + 42 * n2 + 98 * n2**2)) / (49. * n1 * (1 + n1) * n2 * (1 + n2)),
    4: lambda n1, n2: (4 * (3 - 2 * n2 + n1 * (-2 + 7 * n2))) / (7. * n1 * n2),
    5: lambda n1, n2: 2.
} # b1**2, b1*b2, b1*b4, b2**2, b2*b4, b4**2

M13bb = { # galaxy-galaxy
    0: lambda n1: 1.125,
    1: lambda n1: -(1 / (1. + n1))
} # b1**2, b1*b3

M13bm = { # galaxy-matter
    0: lambda n1: (5 + 9*n1)/(8. + 8*n1),
    1: lambda n1: -(1/(2. + 2*n1))
} # b1, b3

M22bm = { # galaxy-matter
    0: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-2 + n1*(-1 + (13 - 6*n1)*n1) - n2 + 2*n1*(-3 + 2*n1)*(-9 + n1*(3 + 7*n1))*n2 + (13 + 2*n1*(-27 + 14*(-1 + n1)*n1))*n2**2 + 2*(-3 + n1*(-15 + 14*n1))*n2**3 + 28*n1*n2**4))/(28.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2)),
    1: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-22 + 7*n1**2*(5 + 7*n2) + n2*(16 + 35*n2) + n1*(16 + 7*n2*(6 + 7*n2))))/(98.*n1*(1 + n1)*n2*(1 + n2)),
    2: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-4 + 7*n1 + 7*n2))/(14.*n1*n2)
} # b1, b2, b4

M22mm = { # matter-matter
    0: lambda n1, n2: ((-3 + 2*n1 + 2*n2)*(-1 + 2*n1 + 2*n2)*(58 + 98*n1**3*n2 + (3 - 91*n2)*n2 + 7*n1**2*(-13 - 2*n2 + 28*n2**2) + n1*(3 + 2*n2*(-73 + 7*n2*(-1 + 7*n2)))))/(196.*n1*(1 + n1)*(-1 + 2*n1)*n2*(1 + n2)*(-1 + 2*n2))
}

M13mm = { # matter-matter
    0: lambda n1: 1.125 - 1./(1. + n1)
}
