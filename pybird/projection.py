import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, splev, splrep, RegularGridInterpolator, interpn
from scipy.integrate import quad
from scipy.special import legendre, spherical_jn, j1
from .fftlog import FFTLog, MPC
from .common import co
from .fourier import FourierTransform
from scipy.linalg import block_diag

# import importlib, sys
# importlib.reload(sys.modules['greenfunction'])
# from greenfunction import GreenFunction

# from time import time


def cH(Om, a):
    """ LCDM growth rate auxiliary function """
    return np.sqrt(Om / a + a ** 2 * (1 - Om))


def DgN(Om, a):
    """ LCDM growth rate auxiliary function """
    return 5.0 / 2 * Om * cH(Om, a) / a * quad(lambda x: cH(Om, x) ** -3, 0, a)[0]


def fN(Om, z):
    """ LCDM growth rate """
    a = 1.0 / (1.0 + z)
    return (Om * (5 * a - 3 * DgN(Om, a))) / (2.0 * (a ** 3 * (1 - Om) + Om) * DgN(Om, a))


def Hubble(Om, z):
    """ LCDM AP parameter auxiliary function """
    return ((Om) * (1 + z) ** 3.0 + (1 - Om)) ** 0.5


def DA(Om, z):
    """ LCDM AP parameter auxiliary function """
    r = quad(lambda x: 1.0 / Hubble(Om, x), 0, z)[0]
    return r / (1 + z)


def W2D(x):
    """ Fiber collision effective window method auxiliary function  """
    return (2.0 * j1(x)) / x


def Hllp(l, lp, x):
    """ Fiber collision effective window method auxiliary function  """
    if l == 2 and lp == 0:
        return x ** 2 - 1.0
    if l == 4 and lp == 0:
        return 1.75 * x ** 4 - 2.5 * x ** 2 + 0.75
    if l == 4 and lp == 2:
        return x ** 4 - x ** 2
    if l == 6 and lp == 0:
        return 4.125 * x ** 6 - 7.875 * x ** 4 + 4.375 * x ** 2 - 0.625
    if l == 6 and lp == 2:
        return 2.75 * x ** 6 - 4.5 * x ** 4 + 7.0 / 4.0 * x ** 2
    if l == 6 and lp == 4:
        return x ** 6 - x ** 4
    else:
        return x * 0.0


def fllp_IR(l, lp, k, q, Dfc):
    """ Fiber collision effective window method auxiliary function  """
    # IR q < k
    # q is an array, k is a scalar
    if l == lp:
        return (q / k) * W2D(q * Dfc) * (q / k) ** l
    else:
        return (q / k) * W2D(q * Dfc) * (2.0 * l + 1.0) / 2.0 * Hllp(max(l, lp), min(l, lp), q / k)


def fllp_UV(l, lp, k, q, Dfc):
    """ Fiber collision effective window method auxiliary function  """
    # UV q > k
    # q is an array, k is a scalar
    if l == lp:
        return W2D(q * Dfc) * (k / q) ** l
    else:
        return W2D(q * Dfc) * (2.0 * l + 1.0) / 2.0 * Hllp(max(l, lp), min(l, lp), k / q)


class Projection(object):
    """
    A class to apply projection effects:
    - Alcock-Pascynski (AP) effect
    - Window functions (survey masks)
    - k-binning or interpolation over the data k-array
    - Fiber collision corrections
    - Wedges
    """

    def __init__(
        self,
        xout,
        # DA_AP=None,
        # H_AP=None,
        with_AP = False,
        D_fid = None,
        H_fid = None,
        nbinsmu=100,
        window_fourier_name=None,
        path_to_window=None,
        window_configspace_file=None,
        binning=False,
        fibcol=False,
        Nwedges=0,
        wedges_bounds=None,
        zz=None,
        nz=None,
        co=co,
    ):

        self.co = co
        self.cf = self.co.with_cf
        self.xout = np.float64(xout)

        # if DA_AP is not None and H_AP is not None:
        if with_AP == True:
            # self.DA = DA_AP
            # self.H = H_AP
            self.DA = D_fid
            self.H = H_fid

            self.muacc = np.linspace(0.0, 1.0, nbinsmu)
            # self.muacc = np.linspace(-1.0, 1.0, nbinsmu)
            self.sgrid, self.musgrid = np.meshgrid(self.co.s, self.muacc, indexing="ij")
            self.kgrid, self.mukgrid = np.meshgrid(self.co.k, self.muacc, indexing="ij")
            self.arrayLegendremusgrid = np.array(
                [(2 * 2 * l + 1) / 2.0 * legendre(2 * l)(self.musgrid) for l in range(self.co.Nl)]
            )
            self.arrayLegendremukgrid = np.array(
                [(2 * 2 * l + 1) / 2.0 * legendre(2 * l)(self.mukgrid) for l in range(self.co.Nl)]
            )

        self.with_window = False
        if window_configspace_file is not None:
            if window_fourier_name is not None:
                self.path_to_window = path_to_window
                self.window_fourier_name = window_fourier_name
            self.window_configspace_file = window_configspace_file
            self.setWindow(Nl=self.co.Nl)
            self.with_window = True

        if binning:
            # self.loadBinning(self.xout)
            # self.loadBinning(self.co.k)
            self.getxbin_mat()
            # self.vel_binmat()

        # wedges
        if Nwedges != 0:
            self.Nw = Nwedges
            self.IL = self.IntegralLegendreArray(Nw=self.Nw, Nl=self.co.Nl, bounds=wedges_bounds)

        # redshift bin evolution
        if zz is not None and nz is not None:
            self.zz = zz
            self.nz = nz
            mu = np.linspace(0, 1, 100)
            self.s, self.z1, self.mu = np.meshgrid(self.co.s, self.zz, mu, indexing="ij")
            self.n1 = self.mesheval1d(self.zz, self.z1, nz)
            self.L = np.array(
                [legendre(2 * l)(self.mu) for l in range(self.co.Nl)]
            )  # Legendre to reconstruct the 3D 2pt function
            self.Lp = 2.0 * np.array(
                [(4 * l + 1) / 2.0 * legendre(2 * l)(self.mu) for l in range(self.co.Nl)]
            )  # Legendre in the integrand to get the multipoles ; factor 2 in front because mu integration goes from 0 to 1
            self.ft = FourierTransform(co=self.co)

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
        if self.cf:
            mugrid = self.musgrid
            arrayLegendremugrid = self.arrayLegendremusgrid
        else:
            mugrid = self.mukgrid
            arrayLegendremugrid = self.arrayLegendremukgrid
        Pkint = interp1d(k, Pk, axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")
        # Pkint = interp1d(k, Pk, axis=-1, kind="cubic", bounds_error=True)
        # print(np.shape(Pk), np.shape(kp))
        # print(np.shape(Pkint(kp)), np.shape(Pk))
        if many:
            # grid_new = np.array([arrayLegendremugrid[0]/0.5, arrayLegendremugrid[1]/2.5, arrayLegendremugrid[2]/4.5])
            Pkmu = np.einsum("lpkm,lkm->pkm", Pkint(kp), arrayLegendremup)
            # Pkmu = np.einsum("lpkm,lkm->pkm", Pkint(self.kgrid), grid_new)
            # print(np.shape(np.sum(Pkmu, axis = 0)))
            
            # Pkmu_new = interpn((self.co.k, self.muacc), np.transpose(Pkmu, axes=[1, 2, 0]), np.array([kp.flatten(), self.munew.flatten()]).T, method='linear', bounds_error=False, fill_value = None)
            
            # Pkmu_new = np.reshape(Pkmu_new, (np.shape(Pkmu_new)[-1], len(self.co.k), len(self.muacc)))
            
            Integrandmu = np.einsum("pkm,lkm->lpkm", Pkmu, arrayLegendremugrid)
            # Integrandmu = np.einsum("pkm,lkm->lpkm", Pkmu_new, arrayLegendremugrid)
            
            # Pk_mono = interp1d(k, Pk[0], axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")
            # Pk_quad = interp1d(k, Pk[1], axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")
            # Pk_hexa = interp1d(k, Pk[2], axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")
            
            # # P_mono = Pk_mono(arrayLegendremugrid[0]/0.5)
            # # P_quad = Pk_quad(arrayLegendremugrid[1]/2.5)
            # # P_hexa = Pk_hexa(arrayLegendremugrid[2]/4.5)
            
            # P_mono = np.einsum('pkm, km -> pkm', Pk_mono(self.kgrid), arrayLegendremugrid[0]/0.5)
            # P_quad = np.einsum('pkm, km -> pkm', Pk_quad(self.kgrid), arrayLegendremugrid[1]/2.5)
            # P_hexa = np.einsum('pkm, km -> pkm', Pk_hexa(self.kgrid), arrayLegendremugrid[2]/4.5)
            
            # P = P_mono + P_quad + P_hexa
            
            # # P_new = RegularGridInterpolator((self.co.k, self.muacc), np.transpose(P, axes = [1, 2, 0]), bounds_error=False, fill_value = None)(np.array([kp.flatten(), self.munew.flatten()]).T)
            
            # # # print(np.shape(P_new))
            
            # # P_new = np.reshape(P_new, (np.shape(P_new)[-1], len(self.co.k), len(self.muacc)))
            
            # Integrandmu = np.einsum("pkm,lkm->lpkm", P, arrayLegendremugrid)
            
            
        else:
            Pkmu = np.einsum("lkm,lkm->km", Pkint(kp), arrayLegendremup)
            # Pkmu = np.einsum("lkm,lkm->km", Pkint(kp), arrayLegendremugrid)
            Integrandmu = np.einsum("km,lkm->lkm", Pkmu, arrayLegendremugrid)
        return 2 * np.trapz(Integrandmu, x=mugrid, axis=-1)
        # return np.trapz(Integrandmu, x=mugrid, axis=-1)

    def AP(self, bird=None, q=None, overwrite=True, xdata=None, PS=None):
        """
        Apply the AP effect to the bird power spectrum or correlation function
        Credit: Jerome Gleyzes
        """
        if q is None:
            qperp, qpar = self.get_AP_param(bird)
            # print(qperp, qpar)
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
            # arrayLegendremup = np.array([legendre(2 * l)(self.musgrid) for l in range(self.co.Nl)])

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
            
            # self.munew = mup
            
            arrayLegendremup = np.array([legendre(2 * l)(mup) for l in range(self.co.Nl)])
            # arrayLegendremup = np.array([legendre(2 * l)(self.mukgrid) for l in range(self.co.Nl)])

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
                    # P11l, Ploopl, Pctl = PS
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
                
                # test = (
                #     1.0 / (qperp ** 2 * qpar) * self.integrAP(bird.kin, bird.Pin, kp, arrayLegendremup, many=True)
                # )
                # np.save('test_AP.npy', test)
                # Pstl_AP = (
                #     1.0 / (qperp ** 2 * qpar) * self.integrAP(kdata, Pstl, kp, arrayLegendremup, many=True)
                # )
                
                # Pnlol_AP = (
                #    1.0 / (qperp ** 2 * qpar) * self.integrAP(kdata, bird.Pnlol, kp, arrayLegendremup, many=True)
                # )
                
                # factor_ce1 = 1.0/(qperp**2*qpar)
                # factor_ce2_mono = (2.0*qpar**2 + qperp**2)/(3.0*qperp**4*qpar**3)
                # factor_ce2_quad = -(2.0*(qpar**2-qperp**2))/(3.0*qperp**4*qpar**3)
                # factor_ce3_mono = 1.0/(3.0*qperp**2*qpar**3)
                # factor_ce3_quad = 2.0/(3.0*qperp**2*qpar**3)
                
                # Pstl_AP = np.zeros(shape=(self.co.Nl, self.co.Nst, self.co.Nk))
                # Pstl_AP[0, 0] = factor_ce1*Pstl[0, 0]
                # Pstl_AP[0, 1] = factor_ce2_mono*Pstl[0, 1]
                # Pstl_AP[1, 1] = factor_ce2_quad*Pstl[1, 1]
                # Pstl_AP[0, 2] = factor_ce3_mono*Pstl[0, 2]
                # Pstl_AP[1, 2] = factor_ce3_quad*Pstl[1, 2]
                
                Pstl_AP = Pstl
                
                if overwrite:
                    # bird.P11l, bird.Pctl, bird.Ploopl = P11l_AP, Pctl_AP, Ploopl_AP
                    bird.P11l, bird.Pctl, bird.Ploopl, bird.Pstl = P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP

                else:
                    # return P11l_AP, Pctl_AP, Ploopl_AP
                    return P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP

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

        # if self.cf: # no window for cf estimator
        #     try:
        #         swindow_config_space = np.loadtxt(self.window_configspace_file)
        #     except:
        #         print ('Error: can\'t load mask file: %s.'%self.window_configspace_file)
        #         compute = False

        # else:
        self.p = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])
        window_fourier_file = os.path.join(self.path_to_window, "%s_Nl%s_kmax%.2f.npy") % (
            self.window_fourier_name,
            self.co.Nl,
            self.co.kmax,
        )

        if load:
            try:
                self.Wal = np.load(window_fourier_file)
                print("Loaded mask: %s" % window_fourier_file)
                save = False
                compute = False
            except:
                print("Can't load mask: %s \n instead," % window_fourier_file)
                load = False

        if not load:  # do not change to else
            print("Computing new mask.")
            if self.window_configspace_file is None:
                print("Error: please specify a configuration-space mask file.")
                compute = False
            else:
                try:
                    swindow_config_space = np.loadtxt(self.window_configspace_file)
                except:
                    print("Error: can't load mask file: %s." % self.window_configspace_file)
                    compute = False

        if compute is True:
            Calp = np.array(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1 / 5.0, 0.0], [0.0, 0.0, 1 / 9.0]],
                    [[0.0, 1.0, 0.0], [1.0, 2 / 7.0, 2 / 7.0], [0.0, 2 / 7.0, 100 / 693.0]],
                    [[0.0, 0.0, 1.0], [0.0, 18 / 35.0, 20 / 77.0], [1.0, 20 / 77.0, 162 / 1001.0]],
                ]
            )

            sw = swindow_config_space[:, 0]
            Qp = np.moveaxis(swindow_config_space[:, 1:].reshape(-1, 3), 0, -1)[:Nl]
            Qal = np.einsum("alp,ps->als", Calp[:Nl, :Nl, :Nl], Qp)

            # if self.cf: # no window for cf estimator
            #     self.Qal = interp1d(sw, Qal, axis=-1, kind='cubic', bounds_error=False, fill_value='extrapolate')(self.co.s)

            # else:
            self.fftsettings = dict(Nmax=4096, xmin=sw[0], xmax=sw[-1] * 100.0, bias=-1.6)  # 1e-2 - 1e6 [Mpc/h]
            self.fft = FFTLog(**self.fftsettings)
            self.pPow = exp(np.einsum("n,s->ns", -self.fft.Pow - 3.0, log(self.p)))
            self.M = np.empty(shape=(Nl, self.fft.Pow.shape[0]), dtype="complex")
            for l in range(Nl):
                self.M[l] = 4 * pi * MPC(2 * l, -0.5 * self.fft.Pow)

            self.Coef = np.empty(shape=(self.co.Nl, Nl, self.co.Nk, self.fft.Pow.shape[0]), dtype="complex")
            for a in range(self.co.Nl):
                for l in range(Nl):
                    for i, k in enumerate(self.co.k):
                        self.Coef[a, l, i] = (
                            (-1j) ** (2 * a)
                            * 1j ** (2 * l)
                            * self.fft.Coef(sw, Qal[a, l] * spherical_jn(2 * a, k * sw), extrap="padding")
                        )

            self.Wal = self.p ** 2 * np.real(np.einsum("alkn,np,ln->alkp", self.Coef, self.pPow, self.M))

            if save:
                print("Saving mask: %s" % window_fourier_file)
                np.save(window_fourier_file, self.Wal)

        # if not self.cf: # no window for cf estimator
        self.Wal = self.Wal[:, : self.co.Nl]

        # Apply masking centered around the value of k
        if withmask:
            kpgrid, kgrid = np.meshgrid(self.p, self.co.k, indexing="ij")
            mask = (kpgrid < kgrid + windowk) & (kpgrid > kgrid - windowk)
            Wal_masked = np.einsum("alkp,pk->alkp", self.Wal, mask)

        # the spacing (needed to do the convolution as a sum)
        deltap = self.p[1:] - self.p[:-1]
        deltap = np.concatenate([[0], deltap])
        self.Waldk = np.einsum("alkp,p->alkp", Wal_masked, deltap)

    def integrWindow(self, P, many=False, kmode=None):
        """
        Convolve the window functions to a power spectrum P
        """
        if kmode is not None:
            Pk = interp1d(kmode, P, axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")(self.p)
        else:
            Pk = interp1d(self.co.k, P, axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate")(self.p)
        # (multipole l, multipole ' p, k, k' m) , (multipole ', power pectra s, k' m)
        # print (self.Qlldk.shape, Pk.shape)
        if many:
            return np.einsum("alkp,lsp->ask", self.Waldk, Pk)
        else:
            return np.einsum("alkp,lp->ak", self.Waldk, Pk)

    def Window(self, bird, PS=None, xdata=None):
        """
        Apply the survey window function to the bird power spectrum
        """
        
        if xdata is not None:
            kmode = xdata
        else:
            kmode = self.co.k
            
        if self.with_window:
            if self.cf:
                if bird.with_bias:
                    bird.fullCf = np.einsum("als,ls->as", self.Qal, bird.fullCf)
                else:
                    bird.C11l = np.einsum("als,lns->ans", self.Qal, bird.C11l)
                    bird.Cctl = np.einsum("als,lns->ans", self.Qal, bird.Cctl)
                    bird.Cloopl = np.einsum("als,lns->ans", self.Qal, bird.Cloopl)
                    if bird.with_nnlo_counterterm:
                        bird.Cnnlol = np.einsum("als,lns->ans", self.Qal, bird.Cnnlol)

            # else:
            #     if bird.with_bias:
            #         bird.fullPs = self.integrWindow(bird.fullPs, many=False)
            #     else:
            #         bird.P11l = self.integrWindow(bird.P11l, many=True)
            #         bird.Pctl = self.integrWindow(bird.Pctl, many=True)
            #         bird.Ploopl = self.integrWindow(bird.Ploopl, many=True)
            #         if bird.with_stoch:
            #             bird.Pstl = self.integrWindow(bird.Pstl, many=True)
            #         if bird.with_nnlo_counterterm:
            #             bird.Pnnlol = self.integrWindow(bird.Pnnlol, many=True)
            
            else:
                if bird.with_bias:
                    bird.fullPs = self.integrWindow(bird.fullPs, many=False)
                else:
                    if PS is None:
                        bird.P11l = self.integrWindow(bird.P11l, many=True)
                        bird.Pctl = self.integrWindow(bird.Pctl, many=True)
                        bird.Ploopl = self.integrWindow(bird.Ploopl, many=True)
                        if bird.with_stoch:
                            bird.Pstl = self.integrWindow(bird.Pstl, many=True)
                        if bird.with_nnlo_counterterm:
                            bird.Pnnlol = self.integrWindow(bird.Pnnlol, many=True)
                    else:
                        # P11l_in, Ploopl_in, Pctl_in = PS
                        # P11l = self.integrWindow(P11l_in, many=True)
                        # Pctl = self.integrWindow(Pctl_in, many=True)
                        # Ploopl = self.integrWindow(Ploopl_in, many=True)
                        # if bird.with_stoch:
                        #     if (np.int32(np.shape(bird.Pstl)[-1]) == len(self.co.k)):
                        #         kmode = self.co.k
                        #     else:
                        #         kmode = xdata
                        #     Pstl = self.integrWindow(bird.Pstl, many=True, kmode=kmode)
                            
                        #     return P11l, Ploopl, Pctl, Pstl
                        # else:
                        #     return P11l, Ploopl, Pctl
                        
                        P11l_in, Ploopl_in, Pctl_in, Pstl_in = PS
                        P11l = self.integrWindow(P11l_in, many=True)
                        Pctl = self.integrWindow(Pctl_in, many=True)
                        Ploopl = self.integrWindow(Ploopl_in, many=True)
                        if bird.with_stoch:
                            if (np.int32(np.shape(Pstl_in)[-1]) == len(self.co.k)):
                                kmode = self.co.k
                            else:
                                kmode = xdata
                            Pstl = self.integrWindow(Pstl_in, many=True, kmode=kmode)
                            
                            return P11l, Ploopl, Pctl, Pstl
                        else:
                            return P11l, Ploopl, Pctl
                        
                        if bird.with_nnlo_counterterm:
                            bird.Pnnlol = self.integrWindow(bird.Pnnlol, many=True, kmode=kmode)
                        
                        

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
            dPunc[int(l / 2)] = (
                -fs
                * pi
                * Dfc ** 2.0
                * (2.0 * pi / xout)
                * (2.0 * l + 1.0)
                / 2.0
                * legendre(l)(0)
                * (1.0 - (xout * Dfc) ** 2 / 8.0)
            )
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

        PS_interp = interp1d(kPS, PS, axis=-1, bounds_error=False, fill_value="extrapolate")(q_ref)

        if many:
            dPcorr = np.zeros(shape=(PS.shape[0], PS.shape[1], len(xout)))
            for j in range(PS.shape[1]):
                for l in range(self.co.Nl):
                    for lp in range(self.co.Nl):
                        for i, k in enumerate(xout):
                            if lp <= l:
                                maskIR = q_ref < k
                                dPcorr[l, j, i] += (
                                    -0.5
                                    * fs
                                    * Dfc ** 2
                                    * np.einsum(
                                        "q,q,q,q->",
                                        q_ref[maskIR],
                                        dq_ref[maskIR],
                                        PS_interp[lp, j, maskIR],
                                        fllp_IR(2 * l, 2 * lp, k, q_ref[maskIR], Dfc),
                                    )
                                )
                            if lp >= l:
                                maskUV = (q_ref > k) & (q_ref < ktrust)
                                dPcorr[l, j, i] += (
                                    -0.5
                                    * fs
                                    * Dfc ** 2
                                    * np.einsum(
                                        "q,q,q,q->",
                                        q_ref[maskUV],
                                        dq_ref[maskUV],
                                        PS_interp[lp, j, maskUV],
                                        fllp_UV(2 * l, 2 * lp, k, q_ref[maskUV], Dfc),
                                    )
                                )
        else:
            dPcorr = np.zeros(shape=(PS.shape[0], len(xout)))
            for l in range(self.co.Nl):
                for lp in range(self.co.Nl):
                    for i, k in enumerate(xout):
                        if lp <= l:
                            maskIR = q_ref < k
                            dPcorr[l, i] += (
                                -0.5
                                * fs
                                * Dfc ** 2
                                * np.einsum(
                                    "q,q,q,q->",
                                    q_ref[maskIR],
                                    dq_ref[maskIR],
                                    PS_interp[lp, maskIR],
                                    fllp_IR(2 * l, 2 * lp, k, q_ref[maskIR], Dfc),
                                )
                            )
                        if lp >= l:
                            maskUV = (q_ref > k) & (q_ref < ktrust)
                            dPcorr[l, i] += (
                                -0.5
                                * fs
                                * Dfc ** 2
                                * np.einsum(
                                    "q,q,q,q->",
                                    q_ref[maskUV],
                                    dq_ref[maskUV],
                                    PS_interp[lp, maskUV],
                                    fllp_UV(2 * l, 2 * lp, k, q_ref[maskUV], Dfc),
                                )
                            )
        return dPcorr

    def fibcolWindow(self, bird):
        """
        Apply window effective method correction to fiber collisions to the bird power spectrum
        """
        if not bird.with_bias:
            bird.P11l += self.dPcorr(self.co.k, self.co.k, bird.P11l, many=True)
            bird.Pctl += self.dPcorr(self.co.k, self.co.k, bird.Pctl, many=True)
            bird.Ploopl += self.dPcorr(self.co.k, self.co.k, bird.Ploopl, many=True)
            
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
            # dk = ks[1] - ks[0]
            ks_input = self.co.k
            # ks_input = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])
            
            self.kmat = ks_input
            # print(self.kmat)
            
            # self.p = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])
    
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
            
            
            # if self.co.Nl == 2:
            #     self.xbin_mat = block_diag(*[binmat, binmat])
            # else:
            #     self.xbin_mat = block_diag(*[binmat, binmat, binmat])
            self.xbin_mat = binmat
        else:
            ss = self.xout
            
            ds = ss[-1] - ss[-2]
            # dk = ks[1] - ks[0]
            if dist_input is None:
                ss_input = self.co.s
            else:
                ss_input = dist_input
            # ks_input = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])
            
            self.smat = ss_input
            # print(self.kmat)
            
            # self.p = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])
    
            binmat = np.zeros((len(ss), len(ss_input)))
            for ii in range(len(ss_input)):
    
                # Define basis vector
                cfvec = np.zeros_like(ss_input)
                cfvec[ii] = 1
                # print(pkvec)
    
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
            
            
            # if self.co.Nl == 2:
            #     self.xbin_mat = block_diag(*[binmat, binmat])
            # else:
            #     self.xbin_mat = block_diag(*[binmat, binmat, binmat])
            self.xbin_mat = binmat
            
    # def vel_binmat(self):
    #     self.corr_convert = False
        
    #     self.k_thy = np.linspace(self.co.kmin, self.co.kmax, 1000)
    #     dk_thy = self.k_thy[-1] - self.k_thy[-2]

    #     ko = self.xout
        
    #     self.vel_m = np.zeros((len(ko),len(self.k_thy)))
        
    #     # for i,ki in enumerate(ko):
    #     #     norm = (1./3.)* ( (k_thy[5*i + 4])**3 - (k_thy[5*i])**3 )
    #     #     for j in range(5):
    #     #         m[i,5*i + j] = (k_thy[5*i + j]**2)*0.001 / norm
        
    #     dk = self.xout[-1] - self.xout[-2]
    #     for i, ki in enumerate(ko):
    #         if i == 0 or i == len(ko) - 1:
    #             kl = ki - dk/2.0
    #             kr = ki + dk/2.0
    #         else:
    #             kl = (ki + ko[i-1])/2.0
    #             kr = (ki + ko[i+1])/2.0
            
    #         norm = (1.0/3.0)*(kl**3 - kr**3)
    #         for j, kj in enumerate(self.k_thy):
    #             if (kj >= kl) and (kj < kr):
    #                 self.vel_m[i, j] = kj**2*dk_thy / norm
                
        

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
            
                
                # bird.C11l = self.integrBinning(bird.C11l)
                # bird.Cctl = self.integrBinning(bird.Cctl)
                # bird.Cloopl = self.integrBinning(bird.Cloopl)
                # # if bird.with_stoch: bird.Cstl = self.integrBinning(bird.Cstl)
                # if bird.with_nnlo_counterterm:
                #     bird.Cnnlol = self.integrBinning(bird.Cnnlol)
        # else:
        #     if PS_all is None:
        #         if bird.with_bias:
        #             bird.fullPs = self.integrBinning(bird.fullPs)
        #         else:
        #             bird.P11l = self.integrBinning(bird.P11l)
        #             bird.Pctl = self.integrBinning(bird.Pctl)
        #             bird.Ploopl = self.integrBinning(bird.Ploopl)
        #             if bird.with_stoch:
        #                 bird.Pstl = self.integrBinning(bird.Pstl)
        #             if bird.with_nnlo_counterterm:
        #                 bird.Pnnlol = self.integrBinning(bird.Pnnlol)
        #     else:
        #         P11l_in, Ploopl_in, Pctl_in, Pstl_in = PS_all
        #         P11l_out = self.integrBinning(P11l_in)
        #         Pctl_out = self.integrBinning(Pctl_in)
        #         Ploopl_out = self.integrBinning(Ploopl_in)
        #         Pstl_out = self.integrBinning(Pstl_in)
                
        #         return P11l_out, Ploopl_out, Pctl_out, Pstl_out
        
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
        
            # if PS_all is None:
            #     if bird.with_bias:
            #         bird.fullPs = np.einsum("abc, dc -> abd", bird.fullPs, self.xbin_mat)
            #     else:
            #         bird.P11l = np.einsum("abc, dc -> abd", bird.P11l, self.xbin_mat)
            #         bird.Pctl = np.einsum("abc, dc -> abd", bird.Pctl, self.xbin_mat)
            #         bird.Ploopl = np.einsum("abc, dc -> abd", bird.Ploopl, self.xbin_mat)
            #         if bird.with_stoch:
            #             bird.Pstl = np.einsum("abc, dc -> abd", bird.Pstl, self.xbin_mat)
            #         if bird.with_nnlo_counterterm:
            #             bird.Pnnlol = np.einsum("abc, dc -> abd", bird.Pnnlol, self.xbin_mat)
            # else:
            #         P11l_in, Ploopl_in, Pctl_in, Pstl_in = PS_all
            #         P11l_out = np.einsum("abc, dc -> abd", interp1d(self.co.k, P11l_in, axis = -1, kind = 'cubic', bounds_error=False, fill_value="extrapolate")(self.k_thy) , self.vel_m)
            #         Pctl_out = np.einsum("abc, dc -> abd", interp1d(self.co.k, Pctl_in, axis = -1, kind = 'cubic', bounds_error=False, fill_value="extrapolate")(self.k_thy) , self.vel_m)
            #         Ploopl_out = np.einsum("abc, dc -> abd", interp1d(self.co.k, Ploopl_in, axis = -1, kind = 'cubic', bounds_error=False, fill_value="extrapolate")(self.k_thy) , self.vel_m)
            #         Pstl_out = np.einsum("abc, dc -> abd", interp1d(self.co.k, Pstl_in, axis = -1, kind = 'cubic', bounds_error=False, fill_value="extrapolate")(self.k_thy) , self.vel_m)
            #         return P11l_out, Ploopl_out, Pctl_out, Pstl_out
                    

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
        # else:
        #     if bird.with_bias:
        #         bird.fullPs = interp1d(self.co.k, bird.fullPs, axis=-1, kind="cubic", bounds_error=False)(self.xout)
        #     else:
        #         bird.P11l = interp1d(self.co.k, bird.P11l, axis=-1, kind="cubic", bounds_error=False)(self.xout)
        #         bird.Pctl = interp1d(self.co.k, bird.Pctl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
        #         bird.Ploopl = interp1d(self.co.k, bird.Ploopl, axis=-1, kind="cubic", bounds_error=False)(self.xout)
        #         if bird.with_stoch:
        #             bird.Pstl = interp1d(self.co.k, bird.Pstl, kind="cubic", bounds_error=False)(self.xout)
        #         if bird.with_nnlo_counterterm:
        #             bird.Pnnlol = interp1d(self.co.k, bird.Pnnlol, axis=-1, kind="cubic", bounds_error=False)(self.xout)
        
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

    def integrWedges(self, P, many=False):
        if many:
            w = np.einsum("lpk,wl->wpk", P, self.IL)
        else:
            w = np.einsum("lk,wl->wk", P, self.IL)
        return w

    def Wedges(self, bird):
        """
        Produce wedges
        """
        if self.cf:
            if bird.with_bias:
                bird.fullCf = self.integrWedges(bird.fullCf, many=False)
            else:
                bird.C11l = self.integrWedges(bird.C11l, many=True)
                bird.Cctl = self.integrWedges(bird.Cctl, many=True)
                bird.Cloopl = self.integrWedges(bird.Cloopl, many=True)
                if bird.with_stoch:
                    bird.Cstl = self.integrWedges(bird.Cstl, many=True)
        else:
            if bird.with_bias:
                bird.fullPs = self.integrWedges(bird.fullPs, many=False)
            else:
                bird.P11l = self.integrWedges(bird.P11l, many=True)
                bird.Pctl = self.integrWedges(bird.Pctl, many=True)
                bird.Ploopl = self.integrWedges(bird.Ploopl, many=True)
                if bird.with_stoch:
                    bird.Pstl = self.integrWedges(bird.Pstl, many=True)

    def Wedges_external(self, P):
        return self.integrWedges(P, many=False)

    def mesheval1d(self, z1d, zm, func):
        ifunc = interp1d(z1d, func, axis=-1, kind="cubic", bounds_error=False, fill_value=0.0)
        return ifunc(zm)

    def redshift(self, bird, rz, Dz, fz, pk="Pk"):

        if (
            "Pk" in pk
        ):  # for the Pk, we use the endpoint LOS. We first do the line-of-sight integral in configuration space, then Fourier transform the integrated Cf to get the integrated Pk
            D1 = self.mesheval1d(self.zz, self.z1, Dz / bird.D)
            f1 = self.mesheval1d(self.zz, self.z1, fz / bird.f)
            s1 = self.mesheval1d(self.zz, self.z1, rz)
            s2 = (self.s ** 2 + s1 ** 2 + 2 * self.s * s1 * self.mu) ** 0.5
            n2 = self.mesheval1d(rz, s2, self.nz)
            D2 = self.mesheval1d(rz, s2, Dz / bird.D)
            f2 = self.mesheval1d(rz, s2, fz / bird.f)
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
            Dp4 = Dp2 ** 2
            fp0 = np.ones_like(f1)
            fp1 = 0.5 * (f1 + f2)
            fp2 = fp1 ** 2
            fp3 = fp1 * fp2
            fp4 = f1 ** 2 * f2 ** 2
            f11 = np.array([fp0, fp1, fp2])
            fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
            floop = np.array(
                [
                    fp2,
                    fp3,
                    fp4,
                    fp1,
                    fp2,
                    fp3,
                    fp1,
                    fp2,
                    fp1,
                    fp1,
                    fp2,
                    fp0,
                    fp1,
                    fp2,
                    fp0,
                    fp1,
                    fp0,
                    fp0,
                    fp1,
                    fp0,
                    fp0,
                    fp0,
                ]
            )
            tlin = np.einsum("n...,...->n...", f11, Dp2 * self.n1 * n2)
            tct = np.einsum("n...,...->n...", fct, Dp2 * self.n1 * n2)
            tloop = np.einsum("n...,...->n...", floop, Dp4 * self.n1 * n2)

            norm = np.trapz(self.nz ** 2 * rz ** 2, x=rz)  # FKP normalization
            # norm = np.trapz(np.trapz(self.n1 * n2 * s1**2, x=self.mu, axis=-1), x=rz, axis=-1) # for CF with endpoint LOS
            def integrand(t, c):
                cmesh = self.mesheval1d(self.co.s, self.s, c)
                return np.einsum(
                    "p...,l...,ln...,n...,...->pn...", self.Lp, self.L, cmesh, t, s1 ** 2
                )  # p: legendre polynomial order, l: multipole, n: number of linear/loop terms, (s, z1, mu)

            def integration(t, c):
                return np.trapz(np.trapz(integrand(t, c), x=self.mu, axis=-1), x=rz, axis=-1) / norm

            bird.C11l = integration(tlin, bird.C11l)
            bird.Cctl = integration(tct, bird.Cctl)
            bird.Cloopl = integration(tloop, bird.Cloopl)

            self.cf = False  # This is a hack, such that later on when another function from the projection class is called, it is evaluated for the Pk instead of the Cf
            self.ft.Cf2Ps(bird)

        else:  # for CF, we use the mean LOS
            r = self.mesheval1d(self.zz, self.z1, rz)
            s1 = (r ** 2 + (0.5 * self.s) ** 2 - self.s * r * self.mu) ** 0.5
            s2 = (r ** 2 + (0.5 * self.s) ** 2 + self.s * r * self.mu) ** 0.5
            D1 = self.mesheval1d(rz, s1, Dz / bird.D)
            D2 = self.mesheval1d(rz, s2, Dz / bird.D)
            f1 = self.mesheval1d(rz, s1, fz / bird.f)
            f2 = self.mesheval1d(rz, s2, fz / bird.f)
            n1 = self.mesheval1d(rz, s1, self.nz)
            n2 = self.mesheval1d(rz, s2, self.nz)

            Dp2 = D1 * D2
            Dp4 = Dp2 ** 2
            fp0 = np.ones_like(f1)
            fp1 = 0.5 * (f1 + f2)
            fp2 = fp1 ** 2
            fp3 = fp1 * fp2
            fp4 = f1 ** 2 * f2 ** 2
            f11 = np.array([fp0, fp1, fp2])
            fct = np.array([fp0, fp0, fp0, fp1, fp1, fp1])
            floop = np.array(
                [
                    fp2,
                    fp3,
                    fp4,
                    fp1,
                    fp2,
                    fp3,
                    fp1,
                    fp2,
                    fp1,
                    fp1,
                    fp2,
                    fp0,
                    fp1,
                    fp2,
                    fp0,
                    fp1,
                    fp0,
                    fp0,
                    fp1,
                    fp0,
                    fp0,
                    fp0,
                ]
            )
            tlin = np.einsum("n...,...->n...", f11, Dp2 * n1 * n2)
            tct = np.einsum("n...,...->n...", fct, Dp2 * n1 * n2)
            tloop = np.einsum("n...,...->n...", floop, Dp4 * n1 * n2)

            norm = np.trapz(np.trapz(n1 * n2 * r ** 2, x=self.mu, axis=-1), x=rz, axis=-1)
            # norm = np.trapz(self.nz**2 * rz**2, x=rz)
            def integrand(t, c):
                cmesh = self.mesheval1d(self.co.s, self.s, c)
                return np.einsum(
                    "p...,l...,ln...,n...,...->pn...", self.Lp, self.L, cmesh, t, r ** 2
                )  # p: legendre polynomial order, l: multipole, n: number of linear/loop terms, (s, z1, mu)

            def integration(t, c):
                return np.trapz(np.trapz(integrand(t, c), x=self.mu, axis=-1), x=rz, axis=-1) / norm

            bird.C11l = integration(tlin, bird.C11l)
            bird.Cctl = integration(tct, bird.Cctl)
            bird.Cloopl = integration(tloop, bird.Cloopl)
