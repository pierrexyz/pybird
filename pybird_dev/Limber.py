import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from scipy.special import legendre
from scipy.integrate import quad, dblquad, simps
from common import co
import scipy.constants as conts

from fftlog import FFTLog, MPC
from nonlinear import M13a, M22a
import scipy.constants as conts


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

class Limber(object):
    
    def __init__(self, theta, z, nlens, nsource, load=True, save=True, path='./', NFFT=256, km=1., nnlo=None):
        
        self.km = km

        self.z = z
        self.theta, _ = np.meshgrid(theta, z, indexing='ij')
        self.Nz = len(self.z)
        self.Nt = len(theta)

        self.nlens = np.asarray(nlens)
        self.nsource = np.asarray(nsource)
        
        self.Ng = self.nlens.shape[0]
        self.Ns = self.nsource.shape[0]
        self.Nss = self.Ns*(self.Ns+1)//2
        self.Ngs = self.Ns*self.Ng
        self.Ngg = self.Ng
        self.N = max([self.Nss, self.Ngs])
        
        self.Nbin = 2*self.Nss + self.Ngs + self.Ngg
        
        self.fft11settings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1.e3, bias=-1.7)
        self.fft11 = FFTLog(**self.fft11settings)
        
        self.fftctsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1.e3, bias=-2.9)
        self.fftct = FFTLog(**self.fftctsettings)
        
        self.fftsettings = dict(Nmax=NFFT, xmin=1.5e-5, xmax=1.e3, bias=-1.7)#bias=-2.3)
        self.fft = FFTLog(**self.fftsettings)

        self.pyegg = os.path.join(path, 'pyegg%s_limber.npz') % (NFFT)

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

        # nnlo DESY1
        if nnlo is not None:
            # X2L = np.load( nnlo )
            # self.X2Lssp, self.X2Lssm, self.X2Lgs, self.X2Lgg = X2L['X2Lssp'], X2L['X2Lssm'], X2L['X2Lgs'], X2L['X2Lgg']
            self.biasfid = np.load( nnlo ) # np.load('synthy1/synthy1_bias.npy')

    def setsPow(self):
        """ Compute the r's to the powers of the FFTLog to evaluate the loop 'ular' correlation function. Called at the instantiation of the class. """
        #slog = np.geomspace(1e-4, 3., 40)
        #slin = np.arange(3, 200., 1)
        #slog2 = np.geomspace(200, 1e4, 20)
        #self.s = np.unique(np.concatenate([slog, slin, slog2]))
        self.s = np.geomspace(1.e-2, 1.e3, 200)
        self.sPow = exp(np.einsum('n,s->ns', -self.fft.Pow - 3 + 0.25, log(self.s))) # loop
        self.sPow11 = exp(np.einsum('n,s->ns', -self.fft11.Pow - 3. + 0.5, log(self.s)))
        self.sPowct = exp(np.einsum('n,s->ns', -self.fftct.Pow - 3. + 0.5 - 2., log(self.s)))

    def setM(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M = np.empty(shape=(3, self.fft.Pow.shape[0], self.fft.Pow.shape[0]), dtype='complex')
        for l in range(3):
            for u, n1 in enumerate(-0.5 * self.fft.Pow):
                for v, n2 in enumerate(-0.5 * self.fft.Pow):
                    self.M[l, u, v] = (2*pi)**.5 * MPC(2 * l - 0.5, n1 + n2 - 1.25)

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
                    if i < 3: Mbm[u, v] = M22bm[i](n1, n2)
            self.Mbb22[i] = Mbb
            if i < 3: self.Mbm22[i] = Mbm
        self.Mbb22 = np.einsum('nm,nm,bnm->bnm', self.M[0], Ma, self.Mbb22)
        self.Mbm22 = np.einsum('nm,nm,bnm->bnm', self.M[1], Ma, self.Mbm22)
        self.Mmm22 = np.einsum('lnm,nm,nm->lnm', self.M[[0,2]], Ma, Mmm)
        self.M22 = np.vstack([self.Mmm22, self.Mbm22, self.Mbb22])

    def setM13(self):
        """ Compute the 13-loop matrices. Called at the instantiation of the class if the matrices are not loaded. """
        Ma = M13a(-0.5 * self.fft.Pow)
        Mmm = M13mm[0](-0.5 * self.fft.Pow)
        Mbm = np.empty(shape=(2, self.fft.Pow.shape[0]), dtype='complex')
        Mbb = np.empty(shape=(2, self.fft.Pow.shape[0]), dtype='complex')
        for i in range(2): 
            Mbb[i] = M13bb[i](-0.5 * self.fft.Pow)
            Mbm[i] = M13bm[i](-0.5 * self.fft.Pow)
        self.Mbb13 = np.einsum('nm,n,bn->bnm', self.M[0], Ma, Mbb)
        self.Mbm13 = np.einsum('nm,n,bn->bnm', self.M[1], Ma, Mbm)
        self.Mmm13 = np.einsum('lnm,n,n->lnm', self.M[[0,2]], Ma, Mmm)
        self.M13 = np.vstack([self.Mmm13, self.Mbm13, self.Mbb13])

    def setM11(self):
        """ Compute the linear matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.M11 = np.empty(shape=(3, self.fft11.Pow.shape[0]), dtype='complex')
        for l in range(3): self.M11[l] = (2*pi)**.5 * MPC(2 * l - 0.5, -0.5 * self.fft11.Pow)

    def setMct(self):
        """ Compute the counterterm matrices. Called at the instantiation of the class if the matrices are not loaded. """
        self.Mct = np.empty(shape=(3, self.fftct.Pow.shape[0]), dtype='complex')
        for l in range(3): self.Mct[l] = (2*pi)**.5 * MPC(2 * l - 0.5, -0.5 * self.fftct.Pow - 1.)

    def getA11(self, CoefsPow):
        """ Perform the linear correlation function matrix multiplications """
        A11 = np.real(np.einsum('ns,ln->ls', CoefsPow, self.M11))
        return np.array([A11[0], A11[2], A11[1], A11[0]])

    def getAct(self, CoefsPow):
        """ Perform the counterterm correlation function matrix multiplications """
        Act = np.real(np.einsum('ns,ln->ls', CoefsPow, self.Mct))
        return np.array([Act[0], Act[2], Act[1], Act[0]])

    def getA22(self, CoefsPow):
        """ Perform the 22-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.M22, optimize=self.optipath22))

    def getA13(self, CoefsPow):
        """ Perform the 13-loop correlation function matrix multiplications """
        return np.real(np.einsum('ns,ms,bnm->bs', CoefsPow, CoefsPow, self.M13, optimize=self.optipath13))

    def getA(self, kin, Pin, Pnl=None, window=None):
        coef11 = self.fft11.Coef(kin, kin**-0.5*Pin, window=.2)
        coefsPow11 = np.einsum('n,ns->ns', coef11, self.sPow11)
        coefct = self.fftct.Coef(kin, kin**-0.5*Pin, window=.2)
        coefsPowct = np.einsum('n,ns->ns', coefct, self.sPowct)
        coef = self.fft.Coef(kin, Pin, window=.2)
        coefsPow = np.einsum('n,ns->ns', coef, self.sPow)
        A11 = self.getA11(coefsPow11)
        Act = self.getAct(coefsPowct)
        A22 = self.getA22(coefsPow)
        A13 = self.getA13(coefsPow)
        
        # halofit
        # coefhf = self.fft11.Coef(kin, kin**-0.5 * Pnl, window=.2)
        # coefsPowhf = np.einsum('n,ns->ns', coefhf, self.sPow11)
        # Ahf = self.getA11(coefsPowhf)
        
        return A11, Act, A13, A22 #, Ahf
    
    def getTime(self, rz, dz_by_dr, Dz, Dfid, h, Omega0_m, A=0., alpha=1.):
        ### line-of-sight functions
        self.Dp2 = Dz**2 / Dfid**2
        self.Dp4 = self.Dp2**2
        self.Dp6 = self.Dp2**3
        self.rz = rz

        lensing_factor = 1.5/conts.c**2 * h**2 * 1e10 * Omega0_m
        r1, _ = np.meshgrid(rz, self.z, indexing='ij')
        
        def lensing_efficiency(nz):
            return lensing_factor * rz * (1+self.z) * np.trapz(np.heaviside(rz-r1, 0.) * nz * (rz-r1)/rz, x=self.z, axis=-1)
        
        def intrinsic_alignments(A=A, alpha=alpha, z0=0.62, C1_rho_crit=0.0134):
            return - A * ( (1+self.z)/(1+z0) )**alpha * C1_rho_crit * Omega0_m / Dz
        
        qshear = np.array([lensing_efficiency(ns) + intrinsic_alignments(A, alpha)*dz_by_dr*ns for ns in self.nsource])
        qgal = np.array([dz_by_dr * nl for nl in self.nlens])
        
        ### line-of-sight kernels
        qsqs = np.zeros(shape=(self.N, self.Nz)) # we pad with zeros such that all arrays, shear-shear, 
        qsqs[:self.Nss] = np.array([qi*qj for i, qi in enumerate(qshear) for j, qj in enumerate(qshear) if i <= j])
        
        qgqs = np.zeros(shape=(self.N, self.Nz)) # galaxy-shear, etc. have same size for the einsum below:
        qgqs[:self.Ngs] = np.array([qi*qj for qi in qgal for qj in qshear])
        
        qgqg = np.zeros(shape=(self.N, self.Nz)) # it makes the code much more concise.
        qgqg[:self.Ng] = np.array([qi**2 for qi in qgal])
        
        self.qsqs = qsqs[:self.Nss] # this is however for nnlo, 
        self.qgqs = qgqs[:self.Ngs] # for which each observable
        self.qgqg = qgqg[:self.Ngg] # is computed separately

        self.qq11 = np.array([qsqs, qsqs, qgqs, qgqg]) 
        self.qq13 = np.array([qsqs, qsqs, qgqs, qgqs, qgqg, qgqg])
        self.qq22 = np.array([qsqs, qsqs, qgqs, qgqs, qgqs, qgqg, qgqg, qgqg, qgqg, qgqg, qgqg])
            
    def make_observables(self, A11, Act, A13, A22):
        ### ss+, ss=, gs, gg pieces
        Assp = np.array([A11[0], Act[0], A13[0], A22[0]])
        Assm = np.array([A11[1], Act[1], A13[1], A22[1]])
        Ags = np.array([A11[2], Act[2], A13[2], A13[3], A22[2], A22[3], A22[4]])
        Agg = np.array([A11[3], Act[3], A13[4], A13[5], A22[5], A22[6], A22[7], A22[8], A22[9], A22[10]])
        return Assp, Assm, Ags, Agg
        
    def computeXi(self, kin, Pin, Pnl, rz, dz_by_dr, Dz, Dfid, h, Omega0_m, A=0., alpha=1.):
        
        self.getTime(rz, dz_by_dr, Dz, Dfid, h, Omega0_m, A=A, alpha=alpha)
        
        ### correlation functions
        self.A11, Act, A13, A22 = self.getA(kin, Pin, Pnl)
        self.Bssp, self.Bssm, self.Bgs, self.Bgg = self.make_observables(np.zeros_like(self.A11), np.zeros_like(Act), A13, A22) # for nnlo: not integrated over the line of sight
        
        ### line-of-sight integration 
        def time_integral(qq, DD, A):
            A1 = interp1d(self.s, A, kind='cubic', axis=-1)(self.theta * rz)
            return np.trapz(np.einsum('biz,z,btz->bitz', qq, DD, A1), x=rz, axis=-1) 
            # b: [ss+, ss-, gs, gg] ; i: ss/gs/gg bins ; z: redshift bins ; t: theta bins
        
        A11 = time_integral(self.qq11, self.Dp2, self.A11)
        Act = time_integral(self.qq11, self.Dp2, Act)
        A13 = time_integral(self.qq13, self.Dp4, A13)
        A22 = time_integral(self.qq22, self.Dp4, A22)
        
        self.Assp, self.Assm, self.Ags, self.Agg = self.make_observables(A11, Act, A13, A22)
        self.Assp = self.Assp[:,:self.Nss]
        self.Assm = self.Assm[:,:self.Nss]
        self.Ags = self.Ags[:,:self.Ngs]
        self.Agg = self.Agg[:,:self.Ngg]
    
    def formatBias(self, bias):
            
        bgg, cgg, cgs, cssp, cssm = bias
        
        bssp = np.ones(shape=(self.Nss, 4))
        bssm = np.ones(shape=(self.Nss, 4))
        bssp[:,1] = 2./self.km**2 * cssp
        bssm[:,1] = 2./self.km**2 * cssm
        
        cgs = 2./self.km**2 * cgs.reshape(self.Ng, self.Ns)
        # b1*11, c*ct, b1*13, b3*13, b1*22, b2*22, b4*22
        bgs = np.array([[bs[0], ci, bs[0], bs[2], bs[0], bs[1], bs[3]] for bs, cs in zip(bgg.T, cgs) for ci in cs])
        
        b1, b2, b3, b4 = bgg
        bgg = np.array([b1**2, 2.*b1*cgg/self.km**2, b1**2, b1*b3, b1**2, b1*b2, b1*b4, b2**2, b2*b4, b4**2])
        
        b1gs = np.array([bs[0] for bs, cs in zip(bgg.T, cgs) for ci in cs]) # for nnlo
        
        return bssp, bssm, bgs, bgg, b1gs, b1
        
    def setBias(self, bias):
        
        bssp, bssm, bgs, bgg, b1gs, b1 = self.formatBias(bias)
        
        self.Xssp = np.einsum('ib,bit->it', bssp, self.Assp)
        self.Xssm = np.einsum('ib,bit->it', bssm, self.Assm) 
        self.Xgs = np.einsum('ib,bit->it', bgs, self.Ags)
        self.Xgg = np.einsum('bi,bit->it', bgg, self.Agg)
    
    def setnnlo(self, bnnlo): ##### nnlo (2L: 2-loop) estimate
        
        bssp, bssm, bgs, bgg, b1gs, b1 = self.formatBias(self.biasfid)
        
        A2Lssp = np.einsum('ib,bs->is', bssp, self.Bssp)**2 / self.A11[0]
        A2Lssm = np.einsum('ib,bs->is', bssm, self.Bssm)**2 / self.A11[1]
        A1Lgs = np.einsum('ib,bs->is', bgs, self.Bgs)
        A2Lgs = np.einsum('is,i,s->is', A1Lgs**2,  1/b1gs, 1/self.A11[2])
        A1Lgg = np.einsum('bi,bs->is', bgg, self.Bgg)
        A2Lgg = np.einsum('is,i,s->is', A1Lgg**2,  1/b1**2, 1/self.A11[3])
        
        def time_integral_twoloop(qq, DD, A):
            A1 = interp1d(self.s, A, kind='cubic', axis=-1)(self.theta * self.rz)
            return np.trapz(np.einsum('iz,z,itz->itz', qq, DD, A1), x=self.rz, axis=-1) 
            # i: ss/gs/gg bins ; z: redshift bins ; t: theta bins
        
        self.X2Lssp = np.einsum( 'i,it->it', bnnlo[:self.Nss] , time_integral_twoloop(self.qsqs, self.Dp6, A2Lssp) )
        self.X2Lssm = np.einsum( 'i,it->it', bnnlo[self.Nss:2*self.Nss] , time_integral_twoloop(self.qsqs, self.Dp6, A2Lssm) )
        self.X2Lgs = np.einsum( 'i,it->it', bnnlo[2*self.Nss:2*self.Nss+self.Ngs] , time_integral_twoloop(self.qgqs, self.Dp6, A2Lgs) )
        self.X2Lgg = np.einsum( 'i,it->it', bnnlo[2*self.Nss+self.Ngs:] , time_integral_twoloop(self.qgqg, self.Dp6, A2Lgg) )

    def getmarg(self, b1, external_gg_counterterm=None, external_gg_b3=None, nnlo=False):
        
        if nnlo: marg = np.zeros(shape=(2*self.Nbin+self.Ng, self.Nt*self.Nbin)) # (Nbin [counterterms] + Ng [b3] + Nbin [nnlo], Nt * Nbin)
        else: marg = np.zeros(shape=(self.Nbin+self.Ng, self.Nt*self.Nbin)) # (Gaussian parameters, data points) = (Nbin [counterterms] + Ng [b3], Nt * Nbin) 
        
        # counterterms: one per bin pair (i, j)
        for i in range(self.Nss): marg[i, i*self.Nt:(i+1)*self.Nt] = self.Assp[1,i] * 2/self.km**2 # ss+
        for i in np.arange(self.Nss, 2*self.Nss): marg[i, i*self.Nt:(i+1)*self.Nt] = self.Assm[1,i-self.Nss] * 2/self.km**2 # ss-
        for i in np.arange(2*self.Nss, 2*self.Nss+self.Ngs): marg[i, i*self.Nt:(i+1)*self.Nt] = self.Ags[1,i-2*self.Nss] * 2/self.km**2 # gs
        for i in np.arange(2*self.Nss+self.Ngs, self.Nbin): # gg
            if external_gg_counterterm is None: marg[i, i*self.Nt:(i+1)*self.Nt] = self.Agg[1,i-(2*self.Nss+self.Ngs)] * 2*b1[i-(2*self.Nss+self.Ngs)]/self.km**2
            else: marg[i, i*self.Nt:(i+1)*self.Nt] = external_gg_counterterm[i-(2*self.Nss+self.Ngs)]
        
        # b3 
        for i in range(self.Ng):
            u = self.Nbin + i  # gs
            for j in range(self.Ns):
                v = 2*self.Nss + i*self.Ns + j
                marg[u, v*self.Nt:(v+1)*self.Nt] = self.Ags[3,i*self.Ns + j]
            w = 2*self.Nss+self.Ngs + i # gg
            if external_gg_b3 is None: marg[u, w*self.Nt:(w+1)*self.Nt] = b1[i]*self.Agg[3,i]
            else: marg[u, w*self.Nt:(w+1)*self.Nt] = external_gg_b3[i]

        # nnlo: one per bin pair (i, j)
        if nnlo: 
            ii = self.Nbin+self.Ng
            for i in range(self.Nss): marg[i+ii, i*self.Nt:(i+1)*self.Nt] = self.X2Lssp[i] # ss+
            for i in np.arange(self.Nss, 2*self.Nss): marg[i+ii, i*self.Nt:(i+1)*self.Nt] = self.X2Lssm[i-self.Nss] # ss-
            for i in np.arange(2*self.Nss, 2*self.Nss+self.Ngs): marg[i+ii, i*self.Nt:(i+1)*self.Nt] = self.X2Lgs[i-2*self.Nss] # gs
            for i in np.arange(2*self.Nss+self.Ngs, self.Nbin): marg[i+ii, i*self.Nt:(i+1)*self.Nt] = self.X2Lgg[i-(2*self.Nss+self.Ngs)] # gg

        return marg
