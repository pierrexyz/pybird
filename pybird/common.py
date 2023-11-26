import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz

# powers of mu to Legendre polynomials
mu = {
    0: {0: 1., 2: 0., 4: 0.},
    2: {0: 1. / 3., 2: 2. / 3., 4: 0.},
    4: {0: 1. / 5., 2: 4. / 7., 4: 8. / 35.},
    6: {0: 1. / 7., 2: 10. / 21., 4: 24. / 77.},
    8: {0: 1. / 9., 2: 40. / 99., 4: 48. / 143.}
}

kbird = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03])
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

    def __init__(self, Nl=2, kmin=0.001, kmax=0.25, km=1., kr=1., nd=3e-4, eft_basis='eftoflss',
        halohalo=True, with_cf=False, with_time=True, accboost=1., optiresum=False, orderresum=16, 
        with_uvmatch=False, with_irmatch=False, exact_time=False, quintessence=False, with_tidal_alignments=False, nonequaltime=False, keep_loop_pieces_independent=False):
        
        self.eft_basis = eft_basis
        self.halohalo = halohalo
        self.nd = nd
        self.km = km                            # b1/2! * {cr1~1} / kr^2 ~ 8 / km^2                             ~~> 1 / kr^2 ~ 8 / km^2
        self.kr = kr # I've put back the factor 1/4 in front of the NNLO term
        # self.kr4 = self.km**4 / 16.**1           # 3(perm.)/4! * {cr4~1} / kr^4 ~ 1/8 * 8^2 * {cr4~1} / km^4     ~~> 1 / kmr4 ~ 8 / km^4 # there is another factor 2 ~~> 1 / kmr4 ~ 16 / km^4
        self.optiresum = optiresum
        self.with_time = with_time
        self.with_uvmatch = with_uvmatch
        self.with_irmatch = with_irmatch
        self.exact_time = exact_time
        self.quintessence = quintessence
        # if self.quintessence: self.exact_time = True
        self.with_tidal_alignments = with_tidal_alignments
        self.nonequaltime = nonequaltime
        self.keep_loop_pieces_independent = keep_loop_pieces_independent

        if Nl == 0: self.Nl = 1
        elif Nl > 0: self.Nl = Nl
        
        self.Nst = 3  # number of stochastic terms

        if self.halohalo:
            
            self.N11 = 3  # number of linear termss
            if self.eft_basis in ["eftoflss", "westcoast"]: self.Nct, self.Nnnlo = 6, 2  # number of counterterms k^2 P11, number of NNLO counterterms k^4 P11
            elif self.eft_basis == "eastcoast": self.Nct, self.Nnnlo = 3, 3
            if self.exact_time:
                self.N22 = 36  # number of 22-loops
                self.N13 = 15  # number of 13-loops
            elif self.with_tidal_alignments:
                self.N22 = 44
                self.N13 = 24
            else:
                self.N22 = 28  # number of 22-loops
                self.N13 = 10  # number of 13-loops
            if self.with_uvmatch or self.with_irmatch: 
                if self.exact_time: self.N13 += 6
                else: self.N13 += 3 
            

            if self.keep_loop_pieces_independent: 
                self.Nloop = self.N13+self.N22
            elif self.with_time: # giving f (and other time functions e.g. Y if != EdS)
                if self.with_tidal_alignments: self.Nloop = 18          
                else: self.Nloop = 12
            else: 
                if self.exact_time: self.Nloop = 35 # giving nothing, however, more terms than in EdS
                elif self.nonequaltime: self.Nloop = self.N13+self.N22
                else: self.Nloop = 22                                    # giving nothing (this is EdS)
                

        else: # halo-matter
            self.N11 = 4  # number of linear terms
            self.Nct = 12  # number of counterterms
            self.N22 = 22
            self.N13 = 11
            if self.with_time: self.Nloop = 5
            else: self.Nloop = 5 ###

        self.with_cf = False
        if with_cf:
            self.with_cf = with_cf
            kmax = 0.6 # Do not change this: the IR-corrections are computed up to kmax = 0.6. If less, the BAO are not fully resummed; if more, numerical instabilities might appear ; so make sure to provide a linear power spectrum up to k > 0.6
            #self.optiresum = True
            slog = np.geomspace(1., 1000., 100) ### Do not change the min max ; the damping windows in the FFTLog of the IR-corrections are depending on those
            slin = np.arange(1./accboost, 200., 1./accboost)
            slogmask = np.where((slog > slin[-1]) | (slog < slin[0] ))[0]
            self.s = np.unique( np.sort( np.concatenate((slog[slogmask], slin)) ) )
        else:
            if self.optiresum: self.s = np.arange(40., 200., 1./accboost)
            else: self.s = sbird # accuracy for Pk resummation here could be boosted, however compared to thin s computation the relative diffs. on ell=0,2 at ~0.02%, 0.1%, resp., up to k ~ 0.5
        self.Ns = self.s.shape[0]
        
        if kmax: 
            self.kmax = kmax
            self.k = kbird
            if self.kmax > kbird[-1]: 
                kextra = np.arange(kbird[-1], 0.3+1e-3, 0.005/accboost)
                self.k = np.concatenate([self.k, kextra[1:]])
            if self.kmax > 0.3:
                kextra = np.arange(0.3, self.kmax+1e-3, 0.01/accboost)
                self.k = np.concatenate([self.k, kextra[1:]])
            self.Nk = self.k.shape[0]

        if kmin: 
            if kmin >= kbird[0]: self.kmin = kbird[0] # enforce kmin = kbird[0] = 0.001 for numerical stability when user asks kmin > 0.001 
            else: self.kmin = kmin 
            self.id_kstable = 0      
            if self.kmin < kbird[0]: # kbird[0] = 0.001 
                # self.id_kstable = 1  # placeholder to know if kmin < 0.001
                # self.k = np.concatenate(([self.kmin], self.k))
                self.id_kstable = 5
                kextra = np.geomspace(self.kmin, kbird[0], num=self.id_kstable, endpoint=False)
                self.k = np.concatenate([kextra, self.k])
            self.Nk = self.k.shape[0]

        # for resummation
        if self.with_cf: self.NIR = 20
        elif self.Nl == 3 or self.kmax > 0.25: 
            if self.kmax < 0.45: self.NIR = 16
            else: self.NIR = 20
        else: self.NIR = 8
        
        if self.NIR == 16: self.Na = 3
        elif self.NIR == 20: self.Na = 3
        elif self.NIR == 8: self.Na = 2

        self.Nn = self.NIR * self.Na * 2

        self.l11 = np.empty(shape=(self.Nl, self.N11))
        self.lct = np.empty(shape=(self.Nl, self.Nct))
        self.l22 = np.empty(shape=(self.Nl, self.N22))
        self.l13 = np.empty(shape=(self.Nl, self.N13))
        self.lnnlo = np.empty(shape=(self.Nl, self.Nnnlo))
        
        for i in range(self.Nl):
            l = 2 * i
            if self.halohalo:
                self.l11[i] = np.array([mu[0][l], mu[2][l], mu[4][l]])
                if self.eft_basis in ["eftoflss", "westcoast"]: 
                    self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l]])
                    self.lnnlo[i] = np.array([mu[4][l], mu[6][l]])
                elif self.eft_basis == "eastcoast": 
                    self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l]])
                    self.lnnlo[i] = np.array([mu[4][l], mu[6][l], mu[8][l]])
                if self.exact_time:
                    self.l22[i] = np.array([ 6 * [mu[0][l]] + 7 * [mu[2][l]] + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]] + 3 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l], mu[4][l]] ])
                    if self.with_uvmatch or self.with_irmatch: self.l13[i] = np.array([ 2 * [mu[0][l]] + 2 * [mu[2][l]] + [mu[4][l], mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[2][l], mu[4][l]]
                        + [mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[6][l]] ])
                    else: self.l13[i] = np.array([ 2 * [mu[0][l]] + 2 * [mu[2][l]] + [mu[4][l], mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[2][l], mu[4][l]] ])
                elif self.with_tidal_alignments:
                    self.l22[i] = np.array([ mu[2][l], mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[2][l], mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l], mu[4][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l] ])
                    self.l13[i] = np.array([ mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[0][l], mu[2][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l], mu[4][l], mu[4][l], mu[6][l] ])
                else:
                    self.l22[i] = np.array([ 6 * [mu[0][l]] + 7 * [mu[2][l]] + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]] ])
                    if self.with_uvmatch or self.with_irmatch: self.l13[i] = np.array([ 2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]]
                        + [mu[2][l], mu[4][l], mu[6][l]] ])
                    else: self.l13[i] = np.array([ 2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]] ])

            else: # halo-matter
                self.l11[i] = np.array([ mu[0][l], mu[2][l], mu[2][l], mu[4][l] ])
                self.lct[i] = np.array([ mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l], mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l] ])
                self.l22[i] = np.array([ mu[0][l], mu[0][l], mu[0][l], mu[2][l], mu[2][l], mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l] ])
                self.l13[i] = np.array([ mu[0][l], mu[0][l], mu[2][l], mu[2][l], mu[2][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l] ])

co = Common()
