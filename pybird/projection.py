import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import legendre, spherical_jn, j1
from pybird.fftlog import FFTLog, MPC
from pybird.common import co
from pybird.greenfunction import GreenFunction
from pybird.fourier import FourierTransform

# import importlib, sys
# importlib.reload(sys.modules['greenfunction'])
# from greenfunction import GreenFunction

# from time import time

def cH(Om, a):
    """ LCDM growth rate auxiliary function """
    return np.sqrt(Om / a + a**2 * (1 - Om))

def DgN(Om, a):
    """ LCDM growth rate auxiliary function """
    return 5. / 2 * Om * cH(Om, a) / a * quad(lambda x: cH(Om, x)**-3, 0, a)[0]

def fN(Om, z):
    """ LCDM growth rate """
    a = 1. / (1. + z)
    return (Om * (5 * a - 3 * DgN(Om, a))) / (2. * (a**3 * (1 - Om) + Om) * DgN(Om, a))

def Hubble(Om, z):
    """ LCDM AP parameter auxiliary function """
    return ((Om) * (1 + z)**3. + (1 - Om))**0.5

def DA(Om, z):
    """ LCDM AP parameter auxiliary function """
    r = quad(lambda x: 1. / Hubble(Om, x), 0, z)[0]
    return r / (1 + z)

def W2D(x):
    """ Fiber collision effective window method auxiliary function  """
    return (2. * j1(x)) / x

def Hllp(l, lp, x):
    """ Fiber collision effective window method auxiliary function  """
    if l == 2 and lp == 0:
        return x ** 2 - 1.
    if l == 4 and lp == 0:
        return 1.75 * x**4 - 2.5 * x**2 + 0.75
    if l == 4 and lp == 2:
        return x**4 - x**2
    if l == 6 and lp == 0:
        return 4.125 * x**6 - 7.875 * x**4 + 4.375 * x**2 - 0.625
    if l == 6 and lp == 2:
        return 2.75 * x**6 - 4.5 * x**4 + 7. / 4. * x**2
    if l == 6 and lp == 4:
        return x**6 - x**4
    else:
        return x * 0.

def fllp_IR(l, lp, k, q, Dfc):
    """ Fiber collision effective window method auxiliary function  """
    # IR q < k
    # q is an array, k is a scalar
    if l == lp:
        return (q / k) * W2D(q * Dfc) * (q / k)**l
    else:
        return (q / k) * W2D(q * Dfc) * (2. * l + 1.) / 2. * Hllp(max(l, lp), min(l, lp), q / k)

def fllp_UV(l, lp, k, q, Dfc):
    """ Fiber collision effective window method auxiliary function  """
    # UV q > k
    # q is an array, k is a scalar
    if l == lp:
        return W2D(q * Dfc) * (k / q)**l
    else:
        return W2D(q * Dfc) * (2. * l + 1.) / 2. * Hllp(max(l, lp), min(l, lp), k / q)

class Projection(object):
    """
    A class to apply projection effects:
    - Alcock-Pascynski (AP) effect
    - Window functions (survey masks)
    - k-binning or interpolation over the data k-array
    - Fiber collision corrections
    - Wedges
    """
    def __init__(self, xout, 
        with_ap=False, H_fid=None, D_fid=None, 
        with_survey_mask=False, survey_mask_arr_p=None, survey_mask_mat_kp=None, 
        with_binning=False, binsize=None, 
        fibcol=False, 
        with_wedge=0, wedge_mat_wl=None, 
        with_redshift_bin=False, redshift_bin_zz=None, redshift_bin_nz=None, 
        co=co):

        self.co = co
        self.cf = self.co.with_cf
        self.xout = xout

        if with_ap: 
            self.H_fid, self.D_fid, = H_fid, D_fid
            self.muacc = np.linspace(0., 1., 100)
            self.sgrid, self.musgrid = np.meshgrid(self.co.s, self.muacc, indexing='ij')
            self.kgrid, self.mukgrid = np.meshgrid(self.co.k, self.muacc, indexing='ij')
            self.arrayLegendremusgrid = np.array([(2*2*l+1)/2.*legendre(2*l)(self.musgrid) for l in range(self.co.Nl)])
            self.arrayLegendremukgrid = np.array([(2*2*l+1)/2.*legendre(2*l)(self.mukgrid) for l in range(self.co.Nl)])

        if with_survey_mask: self.arr_p, self.mat_kp = survey_mask_arr_p, survey_mask_mat_kp
        if with_binning: self.loadBinning(self.xout, binsize)
        if with_wedge: self.wedge_mat_wl = wedge_mat_wl
        
        # redshift bin evolution
        if with_redshift_bin:
            self.zz, self.nz = redshift_bin_zz, redshift_bin_nz
            mu = np.linspace(0, 1, 60)
            self.s, self.z1, self.mu = np.meshgrid(self.co.s, self.zz, mu, indexing='ij')
            self.n1 = self.mesheval1d(self.zz, self.z1, nz)
            self.L = np.array([legendre(2*l)(self.mu) for l in range(self.co.Nl)]) # Legendre to reconstruct the 3D 2pt function
            self.Lp = 2. * np.array([(4*l+1)/2. * legendre(2*l)(self.mu) for l in range(self.co.Nl)]) # Legendre in the integrand to get the multipoles ; factor 2 in front because mu integration goes from 0 to 1
            self.ft = FourierTransform(co=self.co)

    def get_AP_param(self, bird=None, DA=None, H=None):
        """
        Compute the AP parameters
        """
        if bird is not None: qpar, qperp = self.H_fid / bird.H, bird.DA / self.D_fid
        elif DA is not None and H is not None: qpar, qperp = self.H_fid / H, DA / self.D_fid
        return qperp, qpar

    def integrAP(self, k, Pk, kp, arrayLegendremup):
        """
        AP integration
        Credit: Jerome Gleyzes
        """
        if self.cf:
            mugrid = self.musgrid
            arrayLegendremugrid = self.arrayLegendremusgrid
        else:
            mugrid = self.mukgrid
            arrayLegendremugrid = self.arrayLegendremukgrid
        Pkint = interp1d(k, Pk, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        Pkmu = np.einsum('l...km,lkm->...km', Pkint(kp), arrayLegendremup)
        Integrandmu = np.einsum('...km,lkm->l...km', Pkmu, arrayLegendremugrid)
        return 2 * np.trapz(Integrandmu, x=mugrid, axis=-1)

    def AP(self, bird=None, q=None):
        """
        Apply the AP effect to the bird power spectrum or correlation function
        Credit: Jerome Gleyzes
            """
        if q is None: qperp, qpar = self.get_AP_param(bird)
        else: qperp, qpar = q

        if self.cf:
            G = (self.musgrid**2 * qpar**2 + (1-self.musgrid**2) * qperp**2)**0.5
            sp = self.sgrid * G
            mup = self.musgrid * qpar / G
            arrayLegendremup = np.array([legendre(2*l)(mup) for l in range(self.co.Nl)])

            if bird.with_bias:
                bird.fullCf = self.integrAP(self.co.s, bird.fullCf, sp, arrayLegendremup)
            else:
                bird.C11l = self.integrAP(self.co.s, bird.C11l, sp, arrayLegendremup)
                bird.Cctl = self.integrAP(self.co.s, bird.Cctl, sp, arrayLegendremup)
                bird.Cloopl = self.integrAP(self.co.s, bird.Cloopl, sp, arrayLegendremup)
                if bird.with_nnlo_counterterm: bird.Cnnlol = self.integrAP(self.co.s, bird.Cnnlol, sp, arrayLegendremup)
        else:
            F = qpar / qperp
            kp = self.kgrid / qperp * (1 + self.mukgrid**2 * (F**-2 - 1))**0.5
            mup = self.mukgrid / F * (1 + self.mukgrid**2 * (F**-2 - 1))**-0.5
            arrayLegendremup = np.array([legendre(2*l)(mup) for l in range(self.co.Nl)])

            if bird.with_bias:
                bird.fullPs = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.fullPs, kp, arrayLegendremup)
            else:
                bird.P11l = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.P11l, kp, arrayLegendremup)
                bird.Pctl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Pctl, kp, arrayLegendremup)
                bird.Ploopl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Ploopl, kp, arrayLegendremup)
                if bird.with_nnlo_counterterm: bird.Pnnlol = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Pnnlol, kp, arrayLegendremup)      

    def integrWindow(self, P):
        """
        Convolve the window functions to a power spectrum P
        """
        Pk = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)(self.arr_p)
        return np.einsum('alkp,l...p->a...k', self.mat_kp, Pk) # (multipole a, multipole l, k, p) , (multipole l, power pectra s, p)

    def Window(self, bird):
        """
        Apply the survey window function to the bird power spectrum 
        """
        if bird.with_bias:
            bird.fullPs = self.integrWindow(bird.fullPs)
        else:
            bird.P11l = self.integrWindow(bird.P11l)
            bird.Pctl = self.integrWindow(bird.Pctl)
            bird.Ploopl = self.integrWindow(bird.Ploopl)
            if bird.with_stoch: bird.Pstl = self.integrWindow(bird.Pstl)
            if bird.with_nnlo_counterterm: bird.Pnnlol = self.integrWindow(bird.Pnnlol)

    def dPuncorr(self, xout, fs=0.6, Dfc=0.43 / 0.6777):
        """
        Compute the uncorrelated contribution of fiber collisions

        kPS : a cbird wavenumber output, typically a (39,) np array
        fs : fraction of the survey affected by fiber collisions
        Dfc : angular distance of the fiber channel Dfc(z = 0.55) = 0.43Mpc

        Credit: Thomas Colas
        """
        dPunc = np.zeros((3, len(xout)))
        for l in [0, 2, 4]:
            dPunc[int(l / 2)] = - fs * pi * Dfc**2. * (2. * pi / xout) * (2. * l + 1.) / \
                2. * special.legendre(l)(0) * (1. - (xout * Dfc)**2 / 8.)
        return dPunc

    def dPcorr(self, xout, kPS, PS, many=False, ktrust=0.25, fs=0.6, Dfc=0.43 / 0.6777):
        """
        Compute the correlated contribution of fiber collisions

        kPS : a cbird wavenumber output, typically a (39,) np array
        PS : a cbird power spectrum output, typically a (3, 39) np array
        ktrust : a UV cutoff
        fs : fraction of the survey affected by fiber collisions
        Dfc : angular distance of the fiber channel Dfc(z = 0.55) = 0.43Mpc

        Credit: Thomas Colas
        """
        q_ref = np.geomspace(min(kPS), ktrust, num=1024)
        # create log bin
        dq_ref = q_ref[1:] - q_ref[:-1]
        dq_ref = np.concatenate([[0], dq_ref])

        PS_interp = interp1d(kPS, PS, axis=-1, bounds_error=False, fill_value='extrapolate')(q_ref)

        if many:
            dPcorr = np.zeros(shape=(PS.shape[0], PS.shape[1], len(xout)))
            for j in range(PS.shape[1]):
                for l in range(self.co.Nl):
                    for lp in range(self.co.Nl):
                        for i, k in enumerate(xout):
                            if lp <= l:
                                maskIR = (q_ref < k)
                                dPcorr[l, j, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskIR],
                                                                                   dq_ref[maskIR], PS_interp[lp, j, maskIR], fllp_IR(2 * l, 2 * lp, k, q_ref[maskIR], Dfc))
                            if lp >= l:
                                maskUV = ((q_ref > k) & (q_ref < ktrust))
                                dPcorr[l, j, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskUV],
                                                                                   dq_ref[maskUV], PS_interp[lp, j, maskUV], fllp_UV(2 * l, 2 * lp, k, q_ref[maskUV], Dfc))
        else:
            dPcorr = np.zeros(shape=(PS.shape[0], len(xout)))
            for l in range(self.co.Nl):
                for lp in range(self.co.Nl):
                    for i, k in enumerate(xout):
                        if lp <= l:
                            maskIR = (q_ref < k)
                            dPcorr[l, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskIR],
                                                                            dq_ref[maskIR], PS_interp[lp, maskIR], fllp_IR(2 * l, 2 * lp, k, q_ref[maskIR], Dfc))
                        if lp >= l:
                            maskUV = ((q_ref > k) & (q_ref < ktrust))
                            dPcorr[l, i] += - 0.5 * fs * Dfc**2 * np.einsum('q,q,q,q->', q_ref[maskUV],
                                                                            dq_ref[maskUV], PS_interp[lp, maskUV], fllp_UV(2 * l, 2 * lp, k, q_ref[maskUV], Dfc))
        return dPcorr

    def fibcolWindow(self, bird):
        """
        Apply window effective method correction to fiber collisions to the bird power spectrum
        """
        if not bird.with_bias:
            bird.P11l += self.dPcorr(self.co.k, self.co.k, bird.P11l, many=True)
            bird.Pctl += self.dPcorr(self.co.k, self.co.k, bird.Pctl, many=True)
            bird.Ploopl += self.dPcorr(self.co.k, self.co.k, bird.Ploopl, many=True)
            # bird.Pnnlol += self.dPcorr(self.co.k, self.co.k, bird.Pnnlol, many=True)

    def loadBinning(self, xout, binsize):
        """
        Create the bins of the data k's
        """
        kcentral = (xout[-1] - binsize * np.arange(len(xout)))[::-1] # in case the provided ks are not the central ones but effective ones...
        binmin = kcentral - binsize / 2.
        binmax = kcentral + binsize / 2.
        self.binvol = np.array([quad(lambda k: k**2, kbinmin, kbinmax)[0] for (kbinmin, kbinmax) in zip(binmin, binmax)])
        self.points = [np.linspace(kbinmin, kbinmax, 100) for (kbinmin, kbinmax) in zip(binmin, binmax)]

    def integrBinning(self, P):
        """
        Integrate over each bin of the data k's
        """
        if self.cf: integrand = interp1d(self.co.s, P, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)
        else: integrand = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)
        res = np.array([np.trapz(integrand(pts) * pts**2, x=pts) for pts in self.points])
        return np.moveaxis(res, 0, -1) / self.binvol

    def xbinning(self, bird):
        """
        Apply binning in k-space for linear-spaced data k-array
        """
        if self.cf:
            if bird.with_bias:
                bird.fullCf = self.integrBinning(bird.fullCf)
            else:
                bird.C11l = self.integrBinning(bird.C11l)
                bird.Cctl = self.integrBinning(bird.Cctl)
                bird.Cloopl = self.integrBinning(bird.Cloopl)
                if bird.with_nnlo_counterterm: bird.Cnnlol = self.integrBinning(bird.Cnnlol)
        else:
            if bird.with_bias:
                bird.fullPs = self.integrBinning(bird.fullPs)
            else:
                bird.P11l = self.integrBinning(bird.P11l)
                bird.Pctl = self.integrBinning(bird.Pctl)
                bird.Ploopl = self.integrBinning(bird.Ploopl)
                if bird.with_stoch: bird.Pstl = self.integrBinning(bird.Pstl)
                if bird.with_nnlo_counterterm: bird.Pnnlol = self.integrBinning(bird.Pnnlol)

    def xdata(self, bird):
        """
        Interpolate the bird power spectrum on the data k-array
        """
        if self.cf:
            if bird.with_bias:
                bird.fullCf = interp1d(self.co.s, bird.fullCf, axis=-1, kind='cubic', bounds_error=False)(self.xout)
            else:
                bird.C11l = interp1d(self.co.s, bird.C11l, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                bird.Cctl = interp1d(self.co.s, bird.Cctl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                bird.Cloopl = interp1d(self.co.s, bird.Cloopl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                if bird.with_nnlo_counterterm: bird.Cnnlol = interp1d(self.co.s, bird.Cnnlol, axis=-1, kind='cubic', bounds_error=False)(self.xout)
        else:
            if bird.with_bias:
                bird.fullPs = interp1d(self.co.k, bird.fullPs, axis=-1, kind='cubic', bounds_error=False)(self.xout)
            else:
                bird.P11l = interp1d(self.co.k, bird.P11l, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                bird.Pctl = interp1d(self.co.k, bird.Pctl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                bird.Ploopl = interp1d(self.co.k, bird.Ploopl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                if bird.with_stoch: bird.Pstl = interp1d(self.co.k, bird.Pstl, kind='cubic', bounds_error=False)(self.xout)
                if bird.with_nnlo_counterterm: bird.Pnnlol = interp1d(self.co.k, bird.Pnnlol, axis=-1, kind='cubic', bounds_error=False)(self.xout)

    def integrWedges(self, P):
        w = np.einsum('l...k,wl->w...k', P, self.wedge_mat_wl)
        return w

    def Wedges(self, bird):
        """
        Rotate multipoles to wedges
        """
        if self.cf:
            if bird.with_bias:
                bird.fullCf = self.integrWedges(bird.fullCf)
            else:
                bird.C11l = self.integrWedges(bird.C11l)
                bird.Cctl = self.integrWedges(bird.Cctl)
                bird.Cloopl = self.integrWedges(bird.Cloopl)
                if bird.with_nnlo_counterterm: bird.Cnnlol = self.integrWedges(bird.Cnnlol)
        else:
            if bird.with_bias:
                bird.fullPs = self.integrWedges(bird.fullPs)
            else:
                bird.P11l = self.integrWedges(bird.P11l)
                bird.Pctl = self.integrWedges(bird.Pctl)
                bird.Ploopl = self.integrWedges(bird.Ploopl)
                if bird.with_stoch: bird.Pstl = self.integrWedges(bird.Pstl)
                if bird.with_nnlo_counterterm: bird.Pnnlol = self.integrWedges(bird.Pnnlol)
    
    def Wedges_external(self, P):
        return self.integrWedges(P, many=False)

    def mesheval1d(self, z1d, zm, func):
        ifunc = interp1d(z1d, func, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)
        return ifunc(zm)

    def redshift(self, bird, rz, Dz, fz, pk='Pk'):
        
        if 'Pk' in pk: # for the Pk, we use the endpoint LOS. We first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            D1 = self.mesheval1d(self.zz, self.z1, Dz/bird.D) 
            f1 = self.mesheval1d(self.zz, self.z1, fz/bird.f) 
            s1 = self.mesheval1d(self.zz, self.z1, rz) 
            s2 = (self.s**2 + s1**2 + 2*self.s*s1*self.mu)**0.5
            n2 = self.mesheval1d(rz, s2, self.nz)  
            D2 = self.mesheval1d(rz, s2, Dz/bird.D) 
            f2 = self.mesheval1d(rz, s2, fz/bird.f) 
            # in principle, 13-type and 22-type loops have different time dependence, however, using the time dependence D1^2 x D^2 for both 22 and 13 gives a ~1e-4 relative difference ; similarly, we do some approximations in powers of f ; 
            # if self.co.nonequaltime:
            #     Dp2 = D1 * D2
            #     Dp22 = Dp2 * Dp2
            #     Dp13 = Dp22 # 0.5 * (D1**2 + D2**2) * Dp2 
            #     fp0 = np.ones_like(f1)  # f1**0
            #     fp1 = 0.5 * (f1 + f2)   # this one is exact, 
            #     fp2 = fp1**2            # but this one is approximate, since f**2 = f1 * f2 or 0.5 * (f1**2+f2**2), instead we use mean f approximation
            #     fp3 = fp1 * fp2         # and similar here
            #     fp4 = f1**2 * f2**2     # however this one is exact
            #     f11 = np.array([fp0, fp1, fp2])
            #     fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
            #     floop = np.concatenate([6*[fp0], 6*[fp1], 9*[fp2], 4*[fp3], 3*[fp4], 2*[fp0],  3*[fp1], 3*[fp2], 2*[fp3]])
            #     tlin = np.einsum('n...,...->n...', f11, Dp2 * self.n1 * n2)
            #     tct = np.einsum('n...,...->n...', fct, Dp2 * self.n1 * n2)
            #     tloop = np.empty_like(floop) 
            #     tloop[:self.co.N22] = np.einsum('n...,...->n...', floop[:self.co.N22], Dp22 * self.n1 * n2)
            #     tloop[self.co.N22:] = np.einsum('n...,...->n...', floop[self.co.N22:], Dp13 * self.n1 * n2)
            # else:
            Dp2 = D1 * D2
            Dp4 = Dp2**2
            fp0 = np.ones_like(f1) 
            fp1 = 0.5 * (f1 + f2)
            fp2 = fp1**2
            fp3 = fp1 * fp2
            fp4 = f1**2 * f2**2
            f11 = np.array([fp0, fp1, fp2])
            fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
            floop = np.array([fp2, fp3, fp4, fp1, fp2, fp3, fp1, fp2, fp1, fp1, fp2, fp0, fp1, fp2, fp0, fp1, fp0, fp0, fp1, fp0, fp0, fp0])
            tlin = np.einsum('n...,...->n...', f11, Dp2 * self.n1 * n2)
            tct = np.einsum('n...,...->n...', fct, Dp2 * self.n1 * n2)
            tloop = np.einsum('n...,...->n...', floop, Dp4 * self.n1 * n2)
            
            norm = np.trapz(self.nz**2 * rz**2, x=rz) # FKP normalization
            # norm = np.trapz(np.trapz(self.n1 * n2 * s1**2, x=self.mu, axis=-1), x=rz, axis=-1) # for CF with endpoint LOS
            def integrand(t, c): 
                cmesh = self.mesheval1d(self.co.s, self.s, c)  
                return np.einsum('p...,l...,ln...,n...,...->pn...', self.Lp, self.L, cmesh, t, s1**2) # p: legendre polynomial order, l: multipole, n: number of linear/loop terms, (s, z1, mu)
            def integration(t, c):
                return np.trapz(np.trapz(integrand(t, c), x=self.mu, axis=-1), x=rz, axis=-1) / norm

            bird.C11l = integration(tlin, bird.C11l)
            bird.Cctl = integration(tct, bird.Cctl)
            bird.Cloopl = integration(tloop, bird.Cloopl)
            self.cf = False # This is a hack, such that later on when another function from the projection class is called, it is evaluated for the Pk instead of the Cf
            self.ft.Cf2Ps(bird)

        else: # for CF, we use the mean LOS
            r = self.mesheval1d(self.zz, self.z1, rz)
            s1 = (r**2 + (.5*self.s)**2 - self.s*r*self.mu)**0.5
            s2 = (r**2 + (.5*self.s)**2 + self.s*r*self.mu)**0.5
            D1 = self.mesheval1d(rz, s1, Dz/bird.D)
            D2 = self.mesheval1d(rz, s2, Dz/bird.D)
            f1 = self.mesheval1d(rz, s1, fz/bird.f)
            f2 = self.mesheval1d(rz, s2, fz/bird.f)
            n1 = self.mesheval1d(rz, s1, self.nz)  
            n2 = self.mesheval1d(rz, s2, self.nz)  

            Dp2 = D1 * D2
            Dp4 = Dp2**2
            fp0 = np.ones_like(f1) 
            fp1 = 0.5 * (f1 + f2)
            fp2 = fp1**2
            fp3 = fp1 * fp2
            fp4 = f1**2 * f2**2
            f11 = np.array([fp0, fp1, fp2])
            fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
            floop = np.array([fp2, fp3, fp4, fp1, fp2, fp3, fp1, fp2, fp1, fp1, fp2, fp0, fp1, fp2, fp0, fp1, fp0, fp0, fp1, fp0, fp0, fp0])
            tlin = np.einsum('n...,...->n...', f11, Dp2 * n1 * n2)
            tct = np.einsum('n...,...->n...', fct, Dp2 * n1 * n2)
            tloop = np.einsum('n...,...->n...', floop, Dp4 * n1 * n2)
            
            norm = np.trapz(np.trapz(n1 * n2 * r**2, x=self.mu, axis=-1), x=rz, axis=-1)
            #norm = np.trapz(self.nz**2 * rz**2, x=rz)
            def integrand(t, c): 
                cmesh = self.mesheval1d(self.co.s, self.s, c)  
                return np.einsum('p...,l...,ln...,n...,...->pn...', self.Lp, self.L, cmesh, t, r**2) # p: legendre polynomial order, l: multipole, n: number of linear/loop terms, (s, z1, mu)
            def integration(t, c):
                return np.trapz(np.trapz(integrand(t, c), x=self.mu, axis=-1), x=rz, axis=-1) / norm

            bird.C11l = integration(tlin, bird.C11l)
            bird.Cctl = integration(tct, bird.Cctl)
            bird.Cloopl = integration(tloop, bird.Cloopl)

        

            

