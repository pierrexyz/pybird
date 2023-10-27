import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz

# powers of mu to Legendre polynomials
# mu = {
#     0: {0: 1.0, 2: 0.0, 4: 0.0},
#     2: {0: 1.0 / 3.0, 2: 2.0 / 3.0, 4: 0.0},
#     4: {0: 1.0 / 5.0, 2: 4.0 / 7.0, 4: 8.0 / 35.0},
#     6: {0: 1.0 / 7.0, 2: 10.0 / 21.0, 4: 24.0 / 77.0},
#     8: {0: 1.0 / 9.0, 2: 40.0 / 99.0, 4: 48.0 / 148.0},
# }

# mu = {
#     0: {0: 1.0, 2: 0.0, 4: 0.0},
#     2: {0: 1.0 / 3.0, 2: 2.0 / 3.0, 4: 0.0},
#     4: {0: 1.0 / 5.0, 2: 4.0 / 7.0, 4: 8.0 / 35.0},
#     6: {0: 1.0 / 7.0, 2: 10.0 / 21.0, 4: 24.0 / 77.0},
#     8: {0: 1.0 / 9.0, 2: 40.0 / 99.0, 4: 48.0 / 143.0},
# }

mu = {
    0: {0: 1.0, 2: 0.0, 4: 0.0, 6: 0.0, 8: 0.0},
    2: {0: 1.0 / 3.0, 2: 2.0 / 3.0, 4: 0.0, 6: 0.0, 8: 0.0},
    4: {0: 1.0 / 5.0, 2: 4.0 / 7.0, 4: 8.0 / 35.0, 6: 0.0, 8: 0.0},
    6: {0: 1.0 / 7.0, 2: 10.0 / 21.0, 4: 24.0 / 77.0, 6: 16.0/231.0, 8: 0.0},
    8: {0: 1.0 / 9.0, 2: 40.0 / 99.0, 4: 48.0 / 143.0, 6: 64.0/495.0, 8: 128.0/6435.0},
}

# kbird = np.array([0.001, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02])
kbird = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03])

sbird = np.array([1.000e+00, 1.124e+00, 1.264e+00, 1.421e+00, 1.597e+00, 1.796e+00, 2.019e+00, 2.270e+00, 2.551e+00, 2.868e+00, 3.225e+00, 3.625e+00, 4.075e+00, 4.582e+00, 5.151e+00, 5.790e+00, 6.510e+00, 7.318e+00, 8.227e+00, 9.249e+00, 1.040e+01, 1.169e+01, 1.314e+01, 1.477e+01, 1.661e+01, 1.867e+01, 2.099e+01, 2.360e+01, 2.653e+01, 2.982e+01, 3.353e+01, 3.769e+01, 4.238e+01, 4.764e+01, 5.356e+01, 6.000e+01, 6.021e+01, 6.526e+01, 6.769e+01,
7.053e+01, 7.579e+01, 7.609e+01, 8.105e+01, 8.555e+01, 8.632e+01, 9.158e+01, 9.617e+01, 9.684e+01, 1.021e+02, 1.074e+02, 1.081e+02, 1.126e+02, 1.179e+02, 1.215e+02, 1.232e+02, 1.284e+02, 1.337e+02, 1.366e+02, 1.389e+02, 1.442e+02, 1.495e+02, 1.536e+02, 1.547e+02, 1.600e+02, 1.727e+02, 1.941e+02, 2.183e+02, 2.454e+02, 2.759e+02, 3.101e+02, 3.486e+02, 3.919e+02, 4.406e+02, 4.954e+02, 5.569e+02, 6.261e+02, 7.038e+02, 7.912e+02, 8.895e+02, 1.000e+03])
# sbird = np.array(
#     [
#         1.000e00,
#         1.124e00,
#         1.264e00,
#         1.421e00,
#         1.597e00,
#         1.796e00,
#         2.019e00,
#         2.270e00,
#         2.551e00,
#         2.868e00,
#         3.225e00,
#         3.625e00,
#         4.075e00,
#         4.582e00,
#         5.151e00,
#         5.790e00,
#         6.510e00,
#         7.318e00,
#         8.227e00,
#         9.249e00,
#         1.040e01,
#         1.169e01,
#         1.314e01,
#         1.477e01,
#         1.661e01,
#         1.867e01,
#         2.099e01,
#         2.360e01,
#         2.653e01,
#         2.982e01,
#         3.353e01,
#         3.769e01,
#         4.238e01,
#         4.764e01,
#         5.356e01,
#         6.000e01,
#         6.021e01,
#         6.526e01,
#         6.769e01,
#         7.053e01,
#         7.579e01,
#         7.609e01,
#         8.105e01,
#         8.555e01,
#         8.632e01,
#         9.158e01,
#         9.617e01,
#         9.684e01,
#         1.021e02,
#         1.074e02,
#         1.081e02,
#         1.126e02,
#         1.179e02,
#         1.215e02,
#         1.232e02,
#         1.284e02,
#         1.337e02,
#         1.366e02,
#         1.389e02,
#         1.442e02,
#         1.495e02,
#         1.536e02,
#         1.547e02,
#         1.600e02,
#         1.727e02,
#         1.941e02,
#         2.183e02,
#         2.454e02,
#         2.759e02,
#         3.101e02,
#         3.486e02,
#         3.919e02,
#         4.406e02,
#         4.954e02,
#         5.569e02,
#         6.261e02,
#         7.038e02,
#         7.912e02,
#         8.895e02,
#         1.000e03,
#     ]
# )


class Common(object):
    """
    A class to share data among different objects

    Attributes
    ----------
    Nl : int
        The maximum multipole to calculate (default 2)
    """

    def __init__(
        self,
        Nl=2,
        kmin=0.001,
        kmax=0.25,
        km=1.0,
        nd=3e-4,
        halohalo=True,
        with_cf=False,
        with_time=True,
        accboost=1.0,
        optiresum=False,
        orderresum=16,
        exact_time=False,
        quintessence=False,
        with_tidal_alignments=False,
        angular=False,
        nonequaltime=False,
        corr_convert=False,
        eft_basis='eftoflss',
        kr = 1.0,
        with_uvmatch=False,
        keep_loop_pieces_independent = False,
    ):

        self.halohalo = halohalo
        self.eft_basis = eft_basis
        self.nd = nd
        self.km = km
        if with_cf == True:
            self.km = 0.7
            print(self.km)
        if corr_convert==True:
            self.dist = np.logspace(0.0, np.log10(200.0), 5000)
            print('Hankel transform power spectrum to correlation function.')
            
        self.kr = kr # I've put back the factor 1/4 in front of the NNLO term
        # self.kr4 = self.km**4 / 16.**1           # 3(perm.)/4! * {cr4~1} / kr^4 ~ 1/8 * 8^2 * {cr4~1} / km^4     ~~> 1 / kmr4 ~ 8 / km^4 # there is another factor 2 ~~> 1 / kmr4 ~ 16 / km^4
        
        self.optiresum = optiresum
        self.with_time = with_time
        self.exact_time = exact_time
        self.quintessence = quintessence
        # if self.quintessence: self.exact_time = True
        self.with_tidal_alignments = with_tidal_alignments
        self.angular = angular
        self.nonequaltime = nonequaltime
        self.with_uvmatch = with_uvmatch
        self.keep_loop_pieces_independent = keep_loop_pieces_independent

        if self.angular:
            self.Ng = 3
            rlog = np.geomspace(
                0.01, 1000.0, 100
            )  ### Do not change the min max ; the damping windows in the FFTLog of the IR-corrections are depending on those
            rlin = np.arange(1.0 / accboost, 200.0, 1.0 / accboost)
            rlogmask = np.where((rlog > rlin[-1]) | (rlog < rlin[0]))[0]
            self.r = np.unique(np.sort(np.concatenate((rlog[rlogmask], rlin))))
            self.Nr = self.r.shape[0]

        if Nl is 0:
            self.Nl = 1
        elif Nl > 0:
            self.Nl = Nl
            # self.Nl = 5

        self.Nst = 3  # number of stochastic terms

        if self.halohalo:

            self.N11 = 3  # number of linear terms
            # self.Nct = 6  # number of counterterms
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
                
            if self.with_uvmatch: 
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
                else: self.Nloop = 22

            # if self.with_time:  # giving f (and other time functions e.g. Y if != EdS)
            #     if self.with_tidal_alignments:
            #         self.Nloop = 18
            #     else:
            #         self.Nloop = 12
            # else:
            #     if self.exact_time:
            #         self.Nloop = 35  # giving nothing, however, more terms than in EdS
            #     elif self.nonequaltime:
            #         self.Nloop = self.N13 + self.N22
            #     else:
            #         self.Nloop = 22  # giving nothing (this is EdS)

        else:  # halo-matter
            self.N11 = 4  # number of linear terms
            self.Nct = 12  # number of counterterms
            self.N22 = 22
            self.N13 = 11
            if self.with_time:
                self.Nloop = 5
            else:
                self.Nloop = 5  ###

        self.with_cf = False
        if with_cf:
            self.with_cf = with_cf
            kmax = 0.6  # Do not change this: the IR-corrections are computed up to kmax = 0.6. If less, the BAO are not fully resummed; if more, numerical instabilities might appear ; so make sure to provide a linear power spectrum up to k > 0.6
            # self.optiresum = True
            slog = np.geomspace(
                1.0, 1000.0, 100
            )  ### Do not change the min max ; the damping windows in the FFTLog of the IR-corrections are depending on those
            slin = np.arange(1.0 / accboost, 200.0, 1.0 / accboost)
            slogmask = np.where((slog > slin[-1]) | (slog < slin[0]))[0]
            self.s = np.unique(np.sort(np.concatenate((slog[slogmask], slin))))
        else:
            if self.optiresum is True:
                self.s = np.arange(40.0, 200.0, 1.0 / accboost)
            else:
                self.s = sbird
        self.Ns = self.s.shape[0]

        if kmax is not None:
            self.kmin = kmin  # no support for kmin: keep default
            self.kmax = kmax
            self.k = kbird
            if self.kmax > kbird[-1]:
                kextra = np.arange(kbird[-1], 0.3 + 1e-3, 0.005 / accboost)
                self.k = np.concatenate([self.k, kextra[1:]])
            if self.kmax > 0.3:
                kextra = np.arange(0.3, self.kmax + 1e-3, 0.01 / accboost)
                self.k = np.concatenate([self.k, kextra[1:]])

            self.Nk = self.k.shape[0]

        # for resummation
        if self.with_cf:
            self.NIR = 20
        elif self.Nl is 3 or self.kmax > 0.25:
            # self.NIR = 16
            if self.kmax < 0.45: self.NIR = 16
            else: self.NIR = 20
        else:
            self.NIR = 8

        if self.NIR is 16:
            self.Na = 3
        elif self.NIR is 20:
            self.Na = 3
        elif self.NIR is 8:
            self.Na = 2

        self.Nn = self.NIR * self.Na * 2

        self.l11 = np.empty(shape=(self.Nl, self.N11))
        self.lct = np.empty(shape=(self.Nl, self.Nct))
        self.l22 = np.empty(shape=(self.Nl, self.N22))
        self.l13 = np.empty(shape=(self.Nl, self.N13))
        # self.lnnlo = np.empty(shape=(self.Nl, 1))
        self.lnnlo = np.empty(shape=(self.Nl, self.Nnnlo))

        for i in range(self.Nl):
            l = 2 * i
            if self.halohalo:
                self.l11[i] = np.array([mu[0][l], mu[2][l], mu[4][l]])
                # self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l]])
                # self.lnnlo[i] = np.array([1.0])
                if self.eft_basis in ["eftoflss", "westcoast"]: 
                    self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[6][l]])
                    self.lnnlo[i] = np.array([mu[4][l], mu[6][l]])
                elif self.eft_basis == "eastcoast": 
                    self.lct[i] = np.array([mu[0][l], mu[2][l], mu[4][l]])
                    self.lnnlo[i] = np.array([mu[4][l], mu[6][l], mu[8][l]])
                    
                if self.exact_time:
                    self.l22[i] = np.array(
                        [
                            6 * [mu[0][l]]
                            + 7 * [mu[2][l]]
                            + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]]
                            + 3 * [mu[4][l]]
                            + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]]
                            + 3 * [mu[2][l]]
                            + 3 * [mu[4][l]]
                            + [mu[6][l], mu[4][l]]
                        ]
                    )
                    # self.l13[i] = np.array(
                    #     [
                    #         2 * [mu[0][l]]
                    #         + 2 * [mu[2][l]]
                    #         + [
                    #             mu[4][l],
                    #             mu[0][l],
                    #             mu[2][l],
                    #             mu[4][l],
                    #             mu[2][l],
                    #             mu[2][l],
                    #             mu[4][l],
                    #             mu[4][l],
                    #             mu[6][l],
                    #             mu[2][l],
                    #             mu[4][l],
                    #         ]
                    #     ]
                    # )
                    if self.with_uvmatch: self.l13[i] = np.array([ 2 * [mu[0][l]] + 2 * [mu[2][l]] + [mu[4][l], mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[2][l], mu[4][l]]
                        + [mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[6][l]] ])
                    else: self.l13[i] = np.array([ 2 * [mu[0][l]] + 2 * [mu[2][l]] + [mu[4][l], mu[0][l], mu[2][l], mu[4][l], mu[2][l], mu[2][l], mu[4][l], mu[4][l], mu[6][l], mu[2][l], mu[4][l]] ])
                    
                elif self.with_tidal_alignments:
                    self.l22[i] = np.array(
                        [
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[4][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[6][l],
                            mu[4][l],
                            mu[4][l],
                            mu[6][l],
                            mu[4][l],
                            mu[6][l],
                            mu[4][l],
                            mu[6][l],
                            mu[8][l],
                        ]
                    )
                    self.l13[i] = np.array(
                        [
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[4][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[0][l],
                            mu[2][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[2][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[2][l],
                            mu[4][l],
                            mu[6][l],
                            mu[4][l],
                            mu[4][l],
                            mu[6][l],
                        ]
                    )
                else:
                    self.l22[i] = np.array(
                        [
                            6 * [mu[0][l]]
                            + 7 * [mu[2][l]]
                            + [mu[4][l], mu[2][l], mu[4][l], mu[2][l], mu[4][l], mu[2][l]]
                            + 3 * [mu[4][l]]
                            + [mu[6][l], mu[4][l], mu[6][l], mu[4][l], mu[6][l], mu[8][l]]
                        ]
                    )
                    # self.l13[i] = np.array([2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]]])
                    if self.with_uvmatch: self.l13[i] = np.array([ 2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]]
                        + [mu[2][l], mu[4][l], mu[6][l]] ])
                    else: self.l13[i] = np.array([ 2 * [mu[0][l]] + 4 * [mu[2][l]] + 3 * [mu[4][l]] + [mu[6][l]] ])
            else:  # halo-matter
                self.l11[i] = np.array([mu[0][l], mu[2][l], mu[2][l], mu[4][l]])
                self.lct[i] = np.array(
                    [
                        mu[0][l],
                        mu[2][l],
                        mu[4][l],
                        mu[2][l],
                        mu[4][l],
                        mu[6][l],
                        mu[0][l],
                        mu[2][l],
                        mu[4][l],
                        mu[2][l],
                        mu[4][l],
                        mu[6][l],
                    ]
                )
                self.l22[i] = np.array(
                    [
                        mu[0][l],
                        mu[0][l],
                        mu[0][l],
                        mu[2][l],
                        mu[2][l],
                        mu[2][l],
                        mu[2][l],
                        mu[2][l],
                        mu[4][l],
                        mu[2][l],
                        mu[4][l],
                        mu[2][l],
                        mu[4][l],
                        mu[2][l],
                        mu[4][l],
                        mu[4][l],
                        mu[6][l],
                        mu[4][l],
                        mu[6][l],
                        mu[4][l],
                        mu[6][l],
                        mu[8][l],
                    ]
                )
                self.l13[i] = np.array(
                    [
                        mu[0][l],
                        mu[0][l],
                        mu[2][l],
                        mu[2][l],
                        mu[2][l],
                        mu[2][l],
                        mu[4][l],
                        mu[2][l],
                        mu[4][l],
                        mu[4][l],
                        mu[6][l],
                    ]
                )


co = Common()
