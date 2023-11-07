import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, splev, splrep
from scipy.integrate import quad
from scipy.special import legendre, spherical_jn, j1
from .fftlog import FFTLog, MPC
from .common import co
from .greenfunction import GreenFunction
from .fourier import FourierTransform
from scipy import special

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
        if with_binning: 
            # self.loadBinning(self.xout, binsize)
            self.getxbin_mat()
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

    def integrAP(self, k, Pk, kp, arrayLegendremup, many=False):
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
        Pkint = interp1d(k, Pk, axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")
        #
        if many:
            Pkmu = np.einsum("lpkm,lkm->pkm", Pkint(kp), arrayLegendremup)

            Integrandmu = np.einsum("pkm,lkm->lpkm", Pkmu, arrayLegendremugrid)
 
        else:
            Pkmu = np.einsum("lkm,lkm->km", Pkint(kp), arrayLegendremup)
            Integrandmu = np.einsum("km,lkm->lkm", Pkmu, arrayLegendremugrid)
        return 2 * np.trapz(Integrandmu, x=mugrid, axis=-1)

    def AP(self, bird=None, q=None, overwrite=True, xdata=None, PS=None):
        """
        Apply the AP effect to the bird power spectrum or correlation function
        Credit: Jerome Gleyzes
        """
        if q is None:
            qperp, qpar = self.get_AP_param(bird)
        else:
            qperp, qpar = q
            
        if xdata is not None:
            kdata = xdata
        else:
            kdata = self.co.k

        if self.cf:
            G = (self.musgrid ** 2 * qpar ** 2 + (1 - self.musgrid ** 2) * qperp ** 2) ** 0.5
            sp = self.sgrid * G
            mup = self.musgrid * qpar / G
            arrayLegendremup = np.array([legendre(2 * l)(mup) for l in range(self.co.Nl)])

            if bird.with_bias:
                if overwrite:
                    bird.fullCf = self.integrAP(self.co.s, bird.fullCf, sp, arrayLegendremup, many=False)
                else:
                    return self.integrAP(self.co.s, bird.fullCf, sp, arrayLegendremup, many=False)
            else:
                C11l_AP = self.integrAP(self.co.s, bird.C11l, sp, arrayLegendremup, many=True)
                Cctl_AP = self.integrAP(self.co.s, bird.Cctl, sp, arrayLegendremup, many=True)
                Cloopl_AP = self.integrAP(self.co.s, bird.Cloopl, sp, arrayLegendremup, many=True)
                if bird.with_nnlo_counterterm: Cnnlol_AP = self.integrAP(self.co.s, bird.Cnnlol, sp, arrayLegendremup, many=True)
                if overwrite:
                    bird.C11l, bird.Cctl, bird.Cloopl = C11l_AP, Cctl_AP, Cloopl_AP
                else:
                    return C11l_AP, Cctl_AP, Cloopl_AP
        else:
            F = qpar / qperp
            kp = self.kgrid / qperp * (1.0 + self.mukgrid ** 2 * ((F ** -2) - 1.0)) ** 0.5
            mup = self.mukgrid / F * (1.0 + self.mukgrid ** 2 * ((F ** -2) - 1.0)) ** -0.5
            
            
            arrayLegendremup = np.array([legendre(2 * l)(mup) for l in range(self.co.Nl)])

            if bird.with_bias:
                if overwrite:
                    bird.fullPs = (
                        1.0
                        / (qperp ** 2 * qpar)
                        * self.integrAP(kdata, bird.fullPs, kp, arrayLegendremup, many=False)
                    )
                else:
                    return (
                        1.0
                        / (qperp ** 2 * qpar)
                        * self.integrAP(kdata, bird.fullPs, kp, arrayLegendremup, many=False)
                    )
            else:
                if PS is None:
                    P11l = bird.P11l
                    Pctl = bird.Pctl
                    Ploopl = bird.Ploopl
                    Pstl = bird.Pstl
                else:
                    P11l, Ploopl, Pctl, Pstl = PS
                    
                P11l_AP = (
                    1.0 / (qperp ** 2 * qpar) * self.integrAP(kdata, P11l, kp, arrayLegendremup, many=True)
                )
                Pctl_AP = (
                    1.0 / (qperp ** 2 * qpar) * self.integrAP(kdata, Pctl, kp, arrayLegendremup, many=True)
                )
                
                Ploopl_AP = (
                    1.0 / (qperp ** 2 * qpar) * self.integrAP(kdata, Ploopl, kp, arrayLegendremup, many=True)
                )
                
                Pstl_AP = Pstl
                
                if overwrite:
                    bird.P11l, bird.Pctl, bird.Ploopl, bird.Pstl = P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP

                else:
                    return P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP

    def integrWindow(self, P):
        """
        Convolve the window functions to a power spectrum P
        """
        Pk = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)(self.arr_p)
        return np.einsum('alkp,l...p->a...k', self.mat_kp, Pk) # (multipole a, multipole l, k, p) , (multipole l, power pectra s, p)

    # def Window(self, bird):
    #     """
    #     Apply the survey window function to the bird power spectrum 
    #     """
    #     if bird.with_bias:
    #         bird.fullPs = self.integrWindow(bird.fullPs)
    #     else:
    #         bird.P11l = self.integrWindow(bird.P11l)
    #         bird.Pctl = self.integrWindow(bird.Pctl)
    #         bird.Ploopl = self.integrWindow(bird.Ploopl)
    #         if bird.with_stoch: bird.Pstl = self.integrWindow(bird.Pstl)
    #         if bird.with_nnlo_counterterm: bird.Pnnlol = self.integrWindow(bird.Pnnlol)
            
    def Window(self, bird, PS=None):
        """
        Apply the survey window function to the bird power spectrum
        """
            
        # if self.cf:
        #     if bird.with_bias:
        #         bird.fullCf = np.einsum("als,ls->as", self.Qal, bird.fullCf)
        #     else:
        #         bird.C11l = np.einsum("als,lns->ans", self.Qal, bird.C11l)
        #         bird.Cctl = np.einsum("als,lns->ans", self.Qal, bird.Cctl)
        #         bird.Cloopl = np.einsum("als,lns->ans", self.Qal, bird.Cloopl)
        #         if bird.with_nnlo_counterterm:
        #             bird.Cnnlol = np.einsum("als,lns->ans", self.Qal, bird.Cnnlol)
        
        # else:
        if bird.with_bias:
            bird.fullPs = self.integrWindow(bird.fullPs)
        else:
            if PS is None:
                bird.P11l = self.integrWindow(bird.P11l)
                bird.Pctl = self.integrWindow(bird.Pctl)
                bird.Ploopl = self.integrWindow(bird.Ploopl)
                if bird.with_stoch:
                    bird.Pstl = self.integrWindow(bird.Pstl)
                if bird.with_nnlo_counterterm:
                    bird.Pnnlol = self.integrWindow(bird.Pnnlol)
            else:
                
                P11l_in, Ploopl_in, Pctl_in, Pstl_in = PS
                P11l = self.integrWindow(P11l_in)
                Pctl = self.integrWindow(Pctl_in)
                Ploopl = self.integrWindow(Ploopl_in)
                if bird.with_stoch:
                    Pstl = self.integrWindow(Pstl_in)
                    
                    return P11l, Ploopl, Pctl, Pstl
                
                elif bird.with_nnlo_counterterm:
                    raise Exception('Pnnlol not avaliable for Shapefit yet.')
                    Pnnlol = self.integrWindow(Pnnlol_in)
                else:
                    return P11l, Ploopl, Pctl
                    

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

            
    def getxbin_mat(self):
        
        try:
            dist_input = self.co.dist
            print('Converting power spectrum to correlation function.')
            self.corr_convert = True
        except:
            dist_input = None
            self.corr_convert = False
        
        if (self.cf == False and dist_input is None):
        
            ks = self.xout
            
            dk = ks[-1] - ks[-2]
            ks_input = self.co.k
            
            self.kmat = ks_input
           
    
            binmat = np.zeros((len(ks), len(ks_input)))
            for ii in range(len(ks_input)):
    
                # Define basis vector
                pkvec = np.zeros_like(ks_input)
                pkvec[ii] = 1
                # print(pkvec)
    
                # Define the spline:
                pkvec_spline = splrep(ks_input, pkvec)
    
                # Now compute binned basis vector:
                tmp = np.zeros_like(ks)
                for i, kk in enumerate(ks):
                    if i == 0 or i == len(ks) - 1:
                        kl = kk - dk / 2
                        kr = kk + dk / 2
                    else:
                        kl = (kk + ks[i-1])/2.0
                        kr = (ks[i+1] + kk)/2.0
                    
                    kin = np.linspace(kl, kr, 100)
                    tmp[i] = np.trapz(kin**2 * splev(kin, pkvec_spline, ext=0), x=kin) * 3 / (kr**3 - kl**3)
                    
                binmat[:, ii] = tmp
            
            
            self.xbin_mat = binmat
        else:
            ss = self.xout
            
            ds = ss[-1] - ss[-2]
            if dist_input is None:
                ss_input = self.co.s
            else:
                ss_input = dist_input
            
            self.smat = ss_input
            # print(self.kmat)
            
    
            binmat = np.zeros((len(ss), len(ss_input)))
            for ii in range(len(ss_input)):
    
                # Define basis vector
                cfvec = np.zeros_like(ss_input)
                cfvec[ii] = 1
    
                # Define the spline:
                cfvec_spline = splrep(ss_input, cfvec)
    
                # Now compute binned basis vector:
                tmp = np.zeros_like(ss)
                for i, sk in enumerate(ss):
                    if i == 0 or i == len(ss) - 1:
                        sl = sk - ds / 2
                        sr = sk + ds / 2
                    else:
                        sl = (sk + ss[i-1])/2.0
                        sr = (ss[i+1] + sk)/2.0
                    
                    s_in = np.linspace(sl, sr, 100)
                    tmp[i] = np.trapz(s_in**2 * splev(s_in, cfvec_spline, ext=2), x=s_in) * 3 / (sr**3 - sl**3)
                    
                binmat[:, ii] = tmp
            
            
            
            self.xbin_mat = binmat

    def loadBinning(self, setxout):
        """
        Create the bins of the data k's
        """
        
        delta_k = setxout[-1] - setxout[-2]
        kcentral = (setxout[-1] - delta_k * np.arange(len(setxout)))[::-1]
        binmin = kcentral - delta_k / 2
        binmax = kcentral + delta_k / 2
        
        binmin = np.where(binmin<0.0, 0.0, binmin)
        
        # binmin = []
        # binmax = []
        # for i in range(len(setxout)):
        #     if i != len(setxout) - 1:
        #         delta_k = setxout[i+1]-setxout[i]
        #     else:
        #         delta_k = setxout[-1]-setxout[-2]
        #     kcentral = setxout[i]
        #     binmin.append(kcentral - delta_k/2.0)
        #     binmax.append(kcentral + delta_k/2.0)
            
        # print('New binning routine')
            
        # binmin = np.array(binmin)
        # binmax = np.array(binmax)
        
        self.binvol = np.array(
            [quad(lambda k: k ** 2, kbinmin, kbinmax)[0] for (kbinmin, kbinmax) in zip(binmin, binmax)]
        )

        self.points = [np.linspace(kbinmin, kbinmax, 100) for (kbinmin, kbinmax) in zip(binmin, binmax)]

    def integrBinning(self, P):
        """
        Integrate over each bin of the data k's
        """
        if self.cf:
            integrand = interp1d(self.co.s, P, axis=-1, kind="cubic", bounds_error=True)
        else:
            integrand = interp1d(self.co.k, P, axis=-1, kind="cubic", bounds_error=True)
        
        res = np.array([np.trapz(integrand(pts) * pts ** 2, x=pts) for pts in self.points])
        
        # result_all = []
        # for i in range(len(self.xout)):
        #     n = 1000
        #     xp = (self.kcentral[i] + self.delta_k*(np.linspace(0, n-1, n) - n)/(2.0*n))
        #     result = np.sum(xp**2*integrand(xp)*self.delta_k/np.float64(n))/self.binvol[i]
        #     result_all.append(result)
            
        # result_all = np.array(result_all)
            
        return np.moveaxis(res, 0, -1) / self.binvol

    def xbinning(self, bird, PS_all = None, CF_all = None):
        """
        Apply binning in k-space for linear-spaced data k-array
        """
        if (self.cf or self.corr_convert):
            if CF_all is None:
                if bird.with_bias:
                    bird.fullCf = np.einsum("abc, dc -> abd", bird.fullCf, self.xbin_mat)
                else:
                    bird.C11l = np.einsum("abc, dc -> abd", bird.C11l, self.xbin_mat)
                    bird.Cctl = np.einsum("abc, dc -> abd", bird.Cctl, self.xbin_mat)
                    bird.Cloopl = np.einsum("abc, dc -> abd", bird.Cloopl, self.xbin_mat)
                    if bird.with_stoch: bird.Cstl = np.einsum("abc, dc -> abd", bird.Cstl, self.xbin_mat)
                    if bird.with_nnlo_counterterm:
                        bird.Cnnlol = self.integrBinning(bird.Cnnlol)
            else:
                C11l_in, Cloopl_in, Cctl_in, Cstl_in = CF_all
                
                C11l_new = np.einsum("abc, dc -> abd", C11l_in, self.xbin_mat)
                Cctl_new = np.einsum("abc, dc -> abd", Cctl_in, self.xbin_mat)
                Cloopl_new = np.einsum("abc, dc -> abd", Cloopl_in, self.xbin_mat)
                Cstl_new = np.einsum("abc, dc -> abd", Cstl_in, self.xbin_mat)
                
                return C11l_new, Cloopl_new, Cctl_new, Cstl_new
        
        else:
            
            if PS_all is None:
                if bird.with_bias:
                    bird.fullPs = np.einsum("abc, dc -> abd", bird.fullPs, self.xbin_mat)
                else:
                    bird.P11l = np.einsum("abc, dc -> abd", bird.P11l, self.xbin_mat)
                    bird.Pctl = np.einsum("abc, dc -> abd", bird.Pctl, self.xbin_mat)
                    bird.Ploopl = np.einsum("abc, dc -> abd", bird.Ploopl, self.xbin_mat)
                    if bird.with_stoch:
                        bird.Pstl = np.einsum("abc, dc -> abd", bird.Pstl, self.xbin_mat)
                    if bird.with_nnlo_counterterm:
                        bird.Pnnlol = np.einsum("abc, dc -> abd", bird.Pnnlol, self.xbin_mat)
            else:
                    P11l_in, Ploopl_in, Pctl_in, Pstl_in = PS_all
                    P11l_out = np.einsum("abc, dc -> abd", P11l_in, self.xbin_mat)
                    Pctl_out = np.einsum("abc, dc -> abd", Pctl_in, self.xbin_mat)
                    Ploopl_out = np.einsum("abc, dc -> abd", Ploopl_in, self.xbin_mat)
                    Pstl_out = np.einsum("abc, dc -> abd", Pstl_in, self.xbin_mat)
        
                    return P11l_out, Ploopl_out, Pctl_out, Pstl_out
        

    def xdata(self, bird, PS=None, CF = None):
        """
        Interpolate the bird power spectrum on the data k-array
        """
        if self.cf:
            if self.corr_convert:
                dist = self.co.dist
            else:
                dist = self.co.s
            
            if CF is None:
                if bird.with_bias:
                    bird.fullCf = interp1d(dist, bird.fullCf, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                else:
                    bird.C11l = interp1d(dist, bird.C11l, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                    bird.Cctl = interp1d(dist, bird.Cctl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                    bird.Cloopl = interp1d(dist, bird.Cloopl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                    if bird.with_stoch: bird.Cstl = interp1d(dist, bird.Cstl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                    if bird.with_nnlo_counterterm:
                        bird.Cnnlol = interp1d(dist, bird.Cnnlol, axis=-1, kind="cubic", bounds_error=False)(self.xout)
            else:
                C11l_in, Cloopl_in, Cctl_in, Cstl_in = CF
                C11l_new = interp1d(dist, C11l_in, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                Cctl_new = interp1d(dist, Cctl_in, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                Cloopl_new = interp1d(dist, Cloopl_in, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                Cstl_new = interp1d(dist, Cstl_in, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                
                return C11l_new, Cloopl_new, Cctl_new, Cstl_new
        
        else:
            if PS is None:
                bird.P11l = interp1d(self.co.k, bird.P11l, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                bird.Pctl = interp1d(self.co.k, bird.Pctl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                bird.Ploopl = interp1d(self.co.k, bird.Ploopl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                if bird.with_stoch:
                    bird.Pstl = interp1d(self.co.k, bird.Pstl, kind="cubic", bounds_error=False)(self.xout)
                if bird.with_nnlo_counterterm:
                    bird.Pnnlol = interp1d(self.co.k, bird.Pnnlol, axis=-1, kind="cubic", bounds_error=False)(self.xout)
            else:
                if bird.with_stoch:
                    P11l, Pctl, Ploopl, Pstl = PS
                else:
                    P11l, Pctl, Ploopl = PS
                P11l_new = interp1d(self.co.k, P11l, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                Pctl_new = interp1d(self.co.k, Pctl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                Ploopl_new = interp1d(self.co.k, Ploopl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
                if bird.with_stoch:
                    Pstl_new = interp1d(self.co.k, Pstl, kind="cubic", bounds_error=False)(self.xout)
                    return P11l_new, Pctl_new, Ploopl_new, Pstl_new
                else:
                    return P11l_new, Pctl_new, Ploopl_new

    def IntegralLegendre(self, l, a, b):
        if l == 0:
            return 1.0
        if l == 2:
            return 0.5 * (b ** 3 - b - a ** 3 + a) / (b - a)
        if l == 4:
            return (
                0.25
                * (-3 / 2.0 * a + 5 * a ** 3 - 7 / 2.0 * a ** 5 + 3 / 2.0 * b - 5 * b ** 3 + 7 / 2.0 * b ** 5)
                / (b - a)
            )

    def IntegralLegendreArray(self, Nw=3, Nl=2, bounds=None):
        deltamu = 1.0 / float(Nw)
        if bounds is None:
            boundsmu = np.arange(0.0, 1.0 + deltamu, deltamu)
        else:
            boundsmu = bounds
        IntegrLegendreArray = np.empty(shape=(Nw, Nl))
        for w in range(Nw):
            for l in range(Nl):
                IntegrLegendreArray[w, l] = self.IntegralLegendre(2 * l, boundsmu[w], boundsmu[w + 1])
        return IntegrLegendreArray

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