import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import legendre, spherical_jn, j1
from copy import deepcopy 
from fftlog import FFTLog, MPC
from common import co

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
    def __init__(self, xout, Om_AP=None, z_AP=None, nbinsmu=100, 
        window_fourier_name=None, path_to_window=None, window_configspace_file=None, 
        binning=False, fibcol=False, Nwedges=0, 
        zz=None, nz=None, co=co):

        self.co = co
        self.cf = self.co.with_cf
        self.xout = xout

        if Om_AP is not None and z_AP is not None:
            self.Om_AP = Om_AP
            self.z_AP = z_AP

            self.DA = DA(self.Om_AP, self.z_AP)
            self.H = Hubble(self.Om_AP, self.z_AP)
            self.muacc = np.linspace(0., 1., nbinsmu)
            if self.cf: self.sgrid, self.mugrid = np.meshgrid(self.co.s, self.muacc, indexing='ij')
            else: self.kgrid, self.mugrid = np.meshgrid(self.co.k, self.muacc, indexing='ij')
            self.arrayLegendremugrid = np.array([(2*2*l+1)/2.*legendre(2*l)(self.mugrid) for l in range(self.co.Nl)])

        self.with_window = False
        if window_configspace_file is not None: 
            if window_fourier_name is not None:
                self.path_to_window = path_to_window
                self.window_fourier_name = window_fourier_name
            self.window_configspace_file = window_configspace_file
            self.setWindow(Nl=self.co.Nl)
            self.with_window = True

        if binning:
            self.loadBinning(self.xout)

        # wedges
        if Nwedges is not 0:
            self.Nw = Nwedges
            self.IL = self.IntegralLegendreArray(Nw=self.Nw, Nl=self.co.Nl)

        if zz is not None and nz is not None:
            self.zz = zz
            self.nz = nz
            self.z1, self.z2 = np.meshgrid(self.zz, self.zz, indexing='ij')
            self.n1, self.n2 = self.mesheval2d(self.zz, self.z1, self.z2, self.nz)
            self.np2 = self.n1 * self.n2
            self.zm = 0.5 * (self.z1 + self.z2)

    def get_AP_param(self, bird=None, DA=None, H=None):
        """
        Compute the AP parameters
        """
        if bird is not None:
            qperp = bird.DA / self.DA
            qpar = self.H / bird.H
        elif DA is not None and H is not None:
            qperp = DA / self.DA
            qpar = self.H / H
        return qperp, qpar

    def integrAP(self, k, Pk, kp, arrayLegendremup, many=False):
        """
        AP integration
        Credit: Jerome Gleyzes
        """
        Pkint = interp1d(k, Pk, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        if many:
            Pkmu = np.einsum('lpkm,lkm->pkm', Pkint(kp), arrayLegendremup)
            Integrandmu = np.einsum('pkm,lkm->lpkm', Pkmu, self.arrayLegendremugrid)
        else:
            Pkmu = np.einsum('lkm,lkm->km', Pkint(kp), arrayLegendremup)
            Integrandmu = np.einsum('km,lkm->lkm', Pkmu, self.arrayLegendremugrid)
        return 2 * np.trapz(Integrandmu, x=self.mugrid, axis=-1)

    def AP(self, bird=None, q=None):
        """
        Apply the AP effect to the bird power spectrum or correlation function
        Credit: Jerome Gleyzes
            """
        if q is None: qperp, qpar = self.get_AP_param(bird)
        else: qperp, qpar = q

        if self.cf:
            G = (self.mugrid**2 * qpar**2 + (1-self.mugrid**2) * qperp**2)**0.5
            sp = self.sgrid * G
            mup = self.mugrid * qpar / G
            arrayLegendremup = np.array([legendre(2*l)(mup) for l in range(self.co.Nl)])

            if bird.with_bias:
                bird.fullCf = self.integrAP(self.co.s, bird.fullCf, sp, arrayLegendremup, many=False)
            else:
                bird.C11l = self.integrAP(self.co.s, bird.C11l, sp, arrayLegendremup, many=True)
                bird.Cctl = self.integrAP(self.co.s, bird.Cctl, sp, arrayLegendremup, many=True)
                bird.Cloopl = self.integrAP(self.co.s, bird.Cloopl, sp, arrayLegendremup, many=True)
            
        
        else:
            F = qpar / qperp
            kp = self.kgrid / qperp * (1 + self.mugrid**2 * (F**-2 - 1))**0.5
            mup = self.mugrid / F * (1 + self.mugrid**2 * (F**-2 - 1))**-0.5
            arrayLegendremup = np.array([legendre(2*l)(mup) for l in range(self.co.Nl)])

            if bird.with_bias:
                bird.fullPs = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.fullPs, kp, arrayLegendremup, many=False)
            else:
                bird.P11l = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.P11l, kp, arrayLegendremup, many=True)
                bird.Pctl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Pctl, kp, arrayLegendremup, many=True)
                bird.Ploopl = 1. / (qperp**2 * qpar) * self.integrAP(self.co.k, bird.Ploopl, kp, arrayLegendremup, many=True)
            

    def setWindow(self, load=True, save=True, Nl=3, withmask=True, windowk=0.05):
        """
        Pre-load the window function to apply to the power spectrum by convolution in Fourier space
        Wal is an array of shape l,l',k',k where so that $P_a(k) = \int dk' \sum_l W_{a l}(k,k') P_l(k')$
        If it cannot find the file, it will compute a new one from a provided mask in configuration space.

        Inputs
        ------
        withmask: whether to only do the convolution over a small range around k
        windowk: the size of said range
        """

        compute = True

        if self.cf:
            try:
                swindow_config_space = np.loadtxt(self.window_configspace_file)
            except:
                print ('Error: can\'t load mask file: %s.'%self.window_configspace_file)
                compute = False

        else:
            self.p = np.concatenate([ np.geomspace(1e-5, 0.015, 100, endpoint=False) , np.arange(0.015, self.co.kmax, 1e-3) ])
            window_fourier_file = os.path.join(self.path_to_window, '%s_Nl%s_kmax%.2f.npy') % (self.window_fourier_name, self.co.Nl, self.co.kmax)
            
            if load:
                try:
                    self.Wal = np.load(window_fourier_file)
                    print ('Loaded mask: %s' % window_fourier_file)
                    save = False
                    compute = False
                except:
                    print ('Can\'t load mask: %s \n instead,' % window_fourier_file )
                    load = False

            if not load: # do not change to else
                print ('Computing new mask.')
                if self.window_configspace_file is None:
                    print ('Error: please specify a configuration-space mask file.')
                    compute = False
                else:    
                    try:
                        swindow_config_space = np.loadtxt(self.window_configspace_file)
                    except:
                        print ('Error: can\'t load mask file: %s.'%self.window_configspace_file)
                        compute = False
        
        if compute is True:
            Calp = np.array([ 
                [ [1., 0., 0.],
                  [0., 1/5., 0.],
                  [0., 0., 1/9.] ],
                [ [0., 1., 0.],
                  [1., 2/7., 2/7.],
                  [0., 2/7., 100/693.] ],
                [ [0., 0., 1.],
                  [0., 18/35., 20/77.],
                  [1., 20/77., 162/1001.] ],
                ])

            sw = swindow_config_space[:,0]
            Qp = np.moveaxis(swindow_config_space[:,1:].reshape(-1,3), 0, -1 )[:Nl]
            Qal = np.einsum('alp,ps->als', Calp[:Nl,:Nl,:Nl], Qp)

            if self.cf: 
                self.Qal = interp1d(sw, Qal, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(self.co.s)

            else:
                self.fftsettings = dict(Nmax=4096, xmin=sw[0], xmax=sw[-1]*100., bias=-1.6) # 1e-2 - 1e6 [Mpc/h]
                self.fft = FFTLog(**self.fftsettings)
                self.pPow = exp(np.einsum('n,s->ns', -self.fft.Pow-3., log(self.p)))
                self.M = np.empty(shape=(Nl, self.fft.Pow.shape[0]), dtype='complex')
                for l in range(Nl): self.M[l] = 4*pi * MPC(2*l, -0.5*self.fft.Pow)

                self.Coef = np.empty(shape=(self.co.Nl, Nl, self.co.Nk, self.fft.Pow.shape[0]), dtype='complex')
                for a in range(self.co.Nl):
                    for l in range(Nl):
                        for i,k in enumerate(self.co.k):
                            self.Coef[a,l,i] = (-1j)**(2*a) * 1j**(2*l) * self.fft.Coef(sw, Qal[a,l]*spherical_jn(2*a, k*sw), extrap = 'padding')

                self.Wal = self.p**2 * np.real( np.einsum('alkn,np,ln->alkp', self.Coef, self.pPow, self.M) )

                if save: 
                    print ( 'Saving mask: %s' % window_fourier_file )
                    np.save(window_fourier_file, self.Wal)

        if not self.cf:
            self.Wal = self.Wal[:,:self.co.Nl]

            # Apply masking centered around the value of k
            if withmask:
                kpgrid, kgrid = np.meshgrid(self.p, self.co.k, indexing='ij')
                mask = (kpgrid < kgrid + windowk) & (kpgrid > kgrid - windowk)
                Wal_masked = np.einsum('alkp,pk->alkp', self.Wal, mask)

            # the spacing (needed to do the convolution as a sum)
            deltap = self.p[1:] - self.p[:-1]
            deltap = np.concatenate([[0], deltap])
            self.Waldk = np.einsum('alkp,p->alkp', Wal_masked, deltap)

    def integrWindow(self, P, many=False):
        """
        Convolve the window functions to a power spectrum P
        """
        Pk = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(self.p)
        # (multipole l, multipole ' p, k, k' m) , (multipole ', power pectra s, k' m)
        #print (self.Qlldk.shape, Pk.shape)
        if many:
            return np.einsum('alkp,lsp->ask', self.Waldk, Pk)
        else:
            return np.einsum('alkp,lp->ak', self.Waldk, Pk)

    def Window(self, bird):
        """
        Apply the survey window function to the bird power spectrum 
        """
        if self.with_window:
            if self.cf:
                if bird.with_bias:
                    bird.fullCf = np.einsum('als,ls->as', self.Qal, bird.fullCf)
                else:
                    bird.C11l = np.einsum('als,lns->ans', self.Qal, bird.C11l)
                    bird.Cctl = np.einsum('als,lns->ans', self.Qal, bird.Cctl)
                    bird.Cloopl = np.einsum('als,lns->ans', self.Qal, bird.Cloopl)

            else:
                if bird.with_bias:
                    bird.fullPs = self.integrWindow(bird.fullPs, many=False)
                else:
                    bird.P11l = self.integrWindow(bird.P11l, many=True)
                    bird.Pctl = self.integrWindow(bird.Pctl, many=True)
                    bird.Ploopl = self.integrWindow(bird.Ploopl, many=True)

            

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

    def loadBinning(self, setxout):
        """
        Create the bins of the data k's
        """
        delta_k = np.round(setxout[-1] - setxout[-2], 2)
        kcentral = (setxout[-1] - delta_k * np.arange(len(setxout)))[::-1]
        binmin = kcentral - delta_k / 2
        binmax = kcentral + delta_k / 2
        self.binvol = np.array([quad(lambda k: k**2, kbinmin, kbinmax)[0]
                                for (kbinmin, kbinmax) in zip(binmin, binmax)])

        self.points = [np.linspace(kbinmin, kbinmax, 100) for (kbinmin, kbinmax) in zip(binmin, binmax)]

    def integrBinning(self, P):
        """
        Integrate over each bin of the data k's
        """
        Pkint = interp1d(self.co.k, P, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        res = np.array([np.trapz(Pkint(pts) * pts**2, x=pts) for pts in self.points])
        return np.moveaxis(res, 0, -1) / self.binvol

    def kbinning(self, bird):
        """
        Apply binning in k-space for linear-spaced data k-array
        """
        if not bird.with_bias:
            bird.P11l = self.integrBinning(bird.P11l)
            bird.Pctl = self.integrBinning(bird.Pctl)
            bird.Ploopl = self.integrBinning(bird.Ploopl)
            if bird.with_stoch: bird.Pstl = self.integrBinning(bird.Pstl)

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
            
        else:
            if bird.with_bias:
                bird.fullPs = interp1d(self.co.k, bird.fullPs, axis=-1, kind='cubic', bounds_error=False)(self.xout)
            else:
                bird.P11l = interp1d(self.co.k, bird.P11l, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                bird.Pctl = interp1d(self.co.k, bird.Pctl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                bird.Ploopl = interp1d(self.co.k, bird.Ploopl, axis=-1, kind='cubic', bounds_error=False)(self.xout)
                if bird.with_stoch: bird.Pstl = interp1d(self.co.k, bird.Pstl, kind='cubic', bounds_error=False)(self.xout)
            

    def IntegralLegendre(self, l, a, b):
        if l == 0: return 1.
        if l == 2: return 0.5*(b**3-b-a**3+a)/(b-a)
        if l == 4: return 0.25*(-3/2.*a+5*a**3-7/2.*a**5+3/2.*b-5*b**3+7/2.*b**5)/(b-a)

    def IntegralLegendreArray(self, Nw=3, Nl=2):
        deltamu = 1./float(Nw)
        boundsmu = np.arange(0., 1., deltamu)
        IntegrLegendreArray = np.empty(shape=(Nw, Nl))
        for w, boundmu in enumerate(boundsmu):
            for l in range(Nl):
                IntegrLegendreArray[w,l] = self.IntegralLegendre(2*l, boundmu, boundmu+deltamu)
        return IntegrLegendreArray

    def integrWedges(self, P, many=False):
        if many: w = np.einsum('lpk,wl->wpk', P, self.IL)
        else: w = np.einsum('lk,wl->wk', P, self.IL)
        return w

    def Wedges(self, bird):
        """
        Produce wedges
        """
        if bird.with_bias:
            bird.fullPs = self.integrWedges(bird.fullPs, many=False)
        else:
            bird.P11l = self.integrWedges(bird.P11l, many=True)
            bird.Pctl = self.integrWedges(bird.Pctl, many=True)
            bird.Ploopl = self.integrWedges(bird.Ploopl, many=True)
            if bird.with_stoch: bird.Pstl = self.integrWedges(bird.Pstl, many=True)

    def Wedges_external(self, P):
        return self.integrWedges(P, many=False)

    def mesheval1d(self, z1d, zm, func):
        ifunc = interp1d(z1d, func, axis=-1, kind='cubic')
        return ifunc(zm)

    def mesheval2d(self, z1d, z1, z2, func):
        ifunc = interp1d(z1d, func, axis=-1, kind='cubic')
        return ifunc(z1), ifunc(z2)

    def redshift(self, bird, Dz, fz, DAz, Hz):

        Dm = self.mesheval1d(self.zz, self.zm, Dz/bird.D)
        fm = self.mesheval1d(self.zz, self.zm, fz/bird.f)
        
        Dp2 = Dm**2
        Dp4 = Dp2**2
        fp0 = fm**0
        fp1 = fm
        fp2 = fm**2
        fp3 = fp2*fm
        fp4 = fp2**2

        f11 = np.array([fp0, fp1, fp2])
        fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
        floop = np.array([fp2, fp3, fp4, fp1, fp2, fp3, fp1, fp2, fp1, fp1, fp2, fp0, fp1, fp2, fp0, fp1, fp0, fp0, fp1, fp0, fp0, fp0])

        tlin = np.einsum('n...,...->n...', f11, Dp2 * self.np2)
        tct = np.einsum('n...,...->n...', fct, Dp2 * self.np2)
        tloop = np.einsum('n...,...->n...', floop, Dp4 * self.np2)

        if self.co.with_cf:
            C11l = np.empty(shape=(len(self.zz), self.co.Nl, self.co.N11, self.co.Ns))
            Cctl = np.empty(shape=(len(self.zz), self.co.Nl, self.co.Nct, self.co.Ns))
            Cloopl = np.empty(shape=(len(self.zz), self.co.Nl, self.co.Nloop, self.co.Ns))
        else:
            P11l = np.empty(shape=(len(self.zz), self.co.Nl, self.co.N11, self.co.Nk))
            Pctl = np.empty(shape=(len(self.zz), self.co.Nl, self.co.Nct, self.co.Nk))
            Ploopl = np.empty(shape=(len(self.zz), self.co.Nl, self.co.Nloop, self.co.Nk))
        
        for i, (zi, DAi, Hi) in enumerate(zip(self.zz, DAz, Hz)):

            self.DA = DA(self.Om_AP, zi)
            self.H = Hubble(self.Om_AP, zi)
            qperp, qpar = self.get_AP_param(DA=DAi, H=Hi)
            birdi = deepcopy(bird)
            self.AP(birdi, q=(qperp, qpar))
            
            if self.co.with_cf:
                C11l[i] = birdi.C11l
                Cctl[i] = birdi.Cctl
                Cloopl[i] = birdi.Cloopl
            else:
                P11l[i] = birdi.P11l
                Pctl[i] = birdi.Pctl
                Ploopl[i] = birdi.Ploopl

        if self.co.with_cf:
            C11l = np.einsum('nyz,yzlnk->lnkyz', tlin, interp1d(self.zz, C11l, axis=0, kind='cubic')(self.zm))
            bird.C11l = np.trapz(np.trapz(C11l, x=self.z2, axis=-1), x=self.zz, axis=-1)

            Cctl = np.einsum('nyz,yzlnk->lnkyz', tct, interp1d(self.zz, Cctl, axis=0, kind='cubic')(self.zm))
            bird.Cctl = np.trapz(np.trapz(Cctl, x=self.z2, axis=-1), x=self.zz, axis=-1)

            Cloopl = np.einsum('nyz,yzlnk->lnkyz', tloop, interp1d(self.zz, Cloopl, axis=0, kind='cubic')(self.zm))
            bird.Cloopl = np.trapz(np.trapz(Cloopl, x=self.z2, axis=-1), x=self.zz, axis=-1)
        else:
            P11l = np.einsum('nyz,yzlnk->lnkyz', tlin, interp1d(self.zz, P11l, axis=0, kind='cubic')(self.zm))
            bird.P11l = np.trapz(np.trapz(P11l, x=self.z2, axis=-1), x=self.zz, axis=-1)

            Pctl = np.einsum('nyz,yzlnk->lnkyz', tct, interp1d(self.zz, Pctl, axis=0, kind='cubic')(self.zm))
            bird.Pctl = np.trapz(np.trapz(Pctl, x=self.z2, axis=-1), x=self.zz, axis=-1)

            Ploopl = np.einsum('nyz,yzlnk->lnkyz', tloop, interp1d(self.zz, Ploopl, axis=0, kind='cubic')(self.zm))
            bird.Ploopl = np.trapz(np.trapz(Ploopl, x=self.z2, axis=-1), x=self.zz, axis=-1)

