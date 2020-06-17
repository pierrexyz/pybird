import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz

# powers of mu to Legendre polynomials
mu = {
    0: {0: 1., 2: 0., 4: 0.},
    2: {0: 1. / 3., 2: 2. / 3., 4: 0.},
    4: {0: 1. / 5., 2: 4. / 7., 4: 8. / 35.},
    6: {0: 1. / 7., 2: 10. / 21., 4: 24. / 77.},
    8: {0: 1. / 9., 2: 40. / 99., 4: 48. / 148.}
}

kbird = np.array([0.001, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02])
sbird = np.array([1.000e+00, 1.124e+00, 1.264e+00, 1.421e+00, 1.597e+00, 1.796e+00, 2.019e+00, 2.270e+00, 2.551e+00, 2.868e+00, 3.225e+00, 3.625e+00, 4.075e+00, 4.582e+00, 5.151e+00, 5.790e+00, 6.510e+00, 7.318e+00, 8.227e+00, 9.249e+00, 1.040e+01, 1.169e+01, 1.314e+01, 1.477e+01, 1.661e+01, 1.867e+01, 2.099e+01, 2.360e+01, 2.653e+01, 2.982e+01, 3.353e+01, 3.769e+01, 4.238e+01, 4.764e+01, 5.356e+01, 6.000e+01, 6.021e+01, 6.526e+01, 6.769e+01,
                  7.053e+01, 7.579e+01, 7.609e+01, 8.105e+01, 8.555e+01, 8.632e+01, 9.158e+01, 9.617e+01, 9.684e+01, 1.021e+02, 1.074e+02, 1.081e+02, 1.126e+02, 1.179e+02, 1.215e+02, 1.232e+02, 1.284e+02, 1.337e+02, 1.366e+02, 1.389e+02, 1.442e+02, 1.495e+02, 1.536e+02, 1.547e+02, 1.600e+02, 1.727e+02, 1.941e+02, 2.183e+02, 2.454e+02, 2.759e+02, 3.101e+02, 3.486e+02, 3.919e+02, 4.406e+02, 4.954e+02, 5.569e+02, 6.261e+02, 7.038e+02, 7.912e+02, 8.895e+02, 1.000e+03])


class Common(object):
    """
    A class to share data among different objects
    
    Attributes
    ----------
    Nl : int
        The maximum multipole to calculate (default 2)
    """

    def __init__(self, Nl=2, kmin=0.001, kmax=0.25, km=1., nd=3e-4, halohalo=True, with_cf=False, with_time=True, accboost=1., optiresum=False, orderresum=16, exact_time=False, quintessence=False, angular=False):
        
        self.halohalo = halohalo
        self.nd = nd
        self.km = km


        self.optiresum = optiresum
        self.with_time = with_time
        self.exact_time = exact_time
        self.quintessence = quintessence
        if self.quintessence: self.exact_time = True

        self.angular = angular

        if self.angular: 
            self.Ng = 3
            rlog = np.geomspace(0.01, 1000., 100) ### Do not change the min max ; the damping windows in the FFTLog of the IR-corrections are depending on those
            rlin = np.arange(1./accboost, 200., 1./accboost)
            rlogmask = np.where((rlog > rlin[-1]) | (rlog < rlin[0] ))[0]
            self.r = np.unique( np.sort( np.concatenate((rlog[rlogmask], rlin)) ) )
            self.Nr = self.r.shape[0]

        if Nl is 0: self.Nl = 1
        elif Nl > 0: self.Nl = Nl
        
        self.Nst = 3  # number of stochastic terms

        if self.halohalo:
            
            self.N11 = 3  # number of linear terms
            self.Nct = 6  # number of counterterms

            if self.exact_time or self.quintessence:
                self.N22 = 36  # number of 22-loops
                self.N13 = 15  # number of 13-loops
            else:
                self.N22 = 28  # number of 22-loops
                self.N13 = 10  # number of 13-loops
            
            if self.with_time: self.Nloop = 12
            else: self.Nloop = 22
            #elif self.redshift is 'geom': self.Nloop = self.N13+self.N22
            #else: self.Nloop = 12

        else: # halo-matter
            self.N11 = 2  # number of linear terms
            self.Nct = 6  # number of counterterms
            self.N22 = 22
            self.N13 = 11
            if self.with_time: self.Nloop = 5
            else: self.Nloop = 5 ###

        self.with_cf = False
        if with_cf:
            self.with_cf = with_cf
            kmax = 0.6
            #self.optiresum = True
            slog = np.geomspace(1., 1000., 100) ### Do not change the min max ; the damping windows in the FFTLog of the IR-corrections are depending on those
            slin = np.arange(1./accboost, 200., 1./accboost)
            slogmask = np.where((slog > slin[-1]) | (slog < slin[0] ))[0]
            self.s = np.unique( np.sort( np.concatenate((slog[slogmask], slin)) ) )
        else:
            if self.optiresum is True: self.s = np.arange(40., 200., 1./accboost)
            else: self.s = sbird
        self.Ns = self.s.shape[0]
        
        if kmax is not None:
            self.kmin = kmin # no support for kmin: keep default
            self.kmax = kmax
            self.k = kbird
            if self.kmax > kbird[-1]:
                kextra = np.arange(kbird[-1], 0.3+1e-3, 0.005/accboost)
                self.k = np.concatenate([self.k, kextra[1:]])
            if self.kmax > 0.3:
                kextra = np.arange(0.3, self.kmax+1e-3, 0.01/accboost)
                self.k = np.concatenate([self.k, kextra[1:]])

            self.Nk = self.k.shape[0]

        # for resummation
        if self.with_cf: self.NIR = 20
        elif self.Nl is 3 or self.kmax > 0.25: self.NIR = 16
        else: self.NIR = 8
        
        if self.NIR is 16: self.Na = 3
        elif self.NIR is 20: self.Na = 3
        elif self.NIR is 8: self.Na = 2

        self.Nn = self.NIR * self.Na * 2

        self.l11 = np.empty(shape=(self.Nl, self.N11))
        self.lct = np.empty(shape=(self.Nl, self.Nct))
        self.l22 = np.empty(shape=(self.Nl, self.N22))
        self.l13 = np.empty(shape=(self.Nl, self.N13))
        
        for i in range(self.Nl):
            l = 2 * i
            self.l11[i] = np.array([mu[0][l], mu[2][l], mu[4][l]])
            self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l]])
            
            if self.exact_time or self.quintessence:
                self.l22[i] = np.array([ 6 * [mu[0][l]] + 7 * [mu[2][l]] + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]] 
                    + 3 * [mu[4][l]] + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]] 
                    + 3 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l], mu[4][l]] ])
                self.l13[i] = np.array([ 2 * [mu[0][l]] + 2 * [mu[2][l]] + [mu[4][l], mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[2][l], mu[4][l]] ])
            else:
                self.l22[i] = np.array([ 6 * [mu[0][l]] + 7 * [mu[2][l]] + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]] 
                    + 3 * [mu[4][l]] + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]] ])
                self.l13[i] = np.array([2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]]])

co = Common()
