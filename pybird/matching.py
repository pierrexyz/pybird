import numpy as np
from pybird.common import co

def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0: return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

class Matching(object):
    """
    A class implementing 

    Attributes
    ----------
    Nmax : int
        this is a quantity that

    Methods
    -------
    prout()
        do something cool
    """

    def __init__(self, nl, co=co):
        self.co = co
        self.fft = nl.fft
        self.q = np.logspace(-1, 3, 100) # internal q on which to evaluate the reconstructed Plin from FFTLog

        if self.co.exact_time:
            self.uv13 = np.array([1/105., -64/315., 2/105., -64/315., 1/105., 8/5., 16/5., 8/5., 8/5., -16/15., 32/15., -16/15., 8/15., -64/15., -64/15.,
                -2/3., -1/3., -4/3., -2/3., -2/3., -1/3.]) # on this line corresponds the finite contributions with power-law integrand in q set to 0 by dim. reg. 
            self.uv22 = np.array([-2/3., 10/7., 2/3., -16/21., -16/21., 0., -2/3., 2/3., 2/3., 2., -2., -2., 0., 0., 2/3., 0., -2/3., 0., -2/3., 0., 0., 0., 0., 0., 0., 0., 0., 0., -8/3., 8/3., 8/3., 0., 0., 0., 0., 0.])
        else:
            self.uv13 = np.array([1/105., -64/315., 2/105., -62/105., -64/315., -16/35., -44/105., -3/5., -16/35., -46/105., 
                -1/3., -2/3., -1/3.]) # on this line corresponds the finite contributions with power-law integrand in q set to 0 by dim. reg. 
            self.uv22 = np.array([-2/3., 10/7., 2/3., -16/21., -16/21., 0., -2/3., 2/3., 2/3., 6/7., -6/7., -6/7., 0., 0., 2/3., 0., -2/3., 0., -2/3., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def Ps(self, bird):

        pq_1 = self.fft.sumCoefxPow(bird.kin, bird.Pin, self.q, window=.2) 
        pq_2 = self.fft.sumCoefxPow(bird.kin, bird.Pin_2, self.q, window=.2) 

        delta_sigmavv = (2*np.pi**2)**-1 * np.trapz(pq_1 - pq_2, x=self.q) 
        delta_sigmavvdd = (2*np.pi**2)**-1 * np.trapz(pq_1**2 - pq_2**2, x=self.q) 

        # we pad as the Nonlinear class replace bird.P13 = P13_FFTLog that has shape 10 = 13 - 3 as some of 13-pieces as pure power law were set to 0 by dim. reg. 
        if bird.P13.shape[0] != self.co.N13: bird.P13 = pad_along_axis(bird.P13, self.co.N13, axis=0) # back to shape (13, Nk)
        bird.P13 += delta_sigmavv * np.einsum('n,k->nk', self.uv13, self.co.k**2 * bird.P11) 
        bird.P22 += delta_sigmavvdd * np.einsum('n,k->nk', self.uv22, self.co.k**2) 

        # as the correlation function used only for resummation, we neglect the matching here and just pad with 0 C13l so it has the correct shape
        if bird.C13l.shape[1] != self.co.N13: bird.C13l = pad_along_axis(bird.C13l, self.co.N13, axis=1)




