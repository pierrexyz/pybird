from pybird.module import *
from pybird.common import co

def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0: return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return pad(array, pad_width=npad, mode='constant', constant_values=0)

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
        self.qUV = logspace(-1, 3, 100) # internal q on which to evaluate the reconstructed Plin from FFTLog
        self.qIR = logspace(-4, -1, 100) # internal q on which to evaluate the reconstructed Plin from massive propagator basis

        if self.co.exact_time:
            self.uv13 = array([1/105., -64/315., 2/105., -64/315., 1/105., 8/5., 16/5., 8/5., 8/5., -16/15., 32/15., -16/15., 8/15., -64/15., -64/15.,
                -2/3., -1/3., -4/3., -2/3., -2/3., -1/3.]) # on this line corresponds the finite contributions with power-law integrand in q set to 0 by dim. reg. 
            self.uv22 = array([-2/3., 10/7., 2/3., -16/21., -16/21., 0., -2/3., 2/3., 2/3., 2., -2., -2., 0., 0., 2/3., 0., -2/3., 0., -2/3., 0., 0., 0., 0., 0., 0., 0., 0., 0., -8/3., 8/3., 8/3., 0., 0., 0., 0., 0.])
        
        else:
            self.uv13 = array([1/105., -64/315., 2/105., -62/105., -64/315., -16/35., -44/105., -3/5., -16/35., -46/105., 
                -1/3., -2/3., -1/3.]) # on this line corresponds the finite contributions with power-law integrand in q set to 0 by dim. reg. 
            self.uv22 = array([-2/3., 10/7., 2/3., -16/21., -16/21., 0., -2/3., 2/3., 2/3., 6/7., -6/7., -6/7., 0., 0., 2/3., 0., -2/3., 0., -2/3., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

            self.ir13 = array([-1., 0., -2., -2., 0., 0., -4., -1., 0., -2., 
                  -1., -2., -1.]) # on this line corresponds the finite contributions with power-law integrand in q set to 0 by dim. reg. 
            self.ir22 = array([1., 0., 0., 0., 0., 0., 2., 0., 0., 2., 0., 0., 1., 0., 0., 4., 0., 0., 0., 0., 1., 2., 0., 0., 2., 0., 1., 0.])
            # self.ir13 = array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            #    -1., -2., -1.])



    def UVPsCf(self, bird):

        pq_1 = self.fft.sumCoefxPow(bird.kin, bird.Pin, self.qUV, window=.2) 
        pq_2 = self.fft.sumCoefxPow(bird.kin, bird.Pin_2, self.qUV, window=.2) 

        delta_sigmavv = (2*pi**2)**-1 * trapz(pq_1 - pq_2, x=self.qUV) 
        delta_sigmavvdd = (2*pi**2)**-1 * trapz(pq_1**2 - pq_2**2, x=self.qUV) 

        # we pad as the Nonlinear class replace bird.P13 = P13_FFTLog that has shape 10 = 13 - 3 as some of 13-pieces as pure power law were set to 0 by dim. reg. 
        if bird.P13.shape[0] != self.co.N13: bird.P13 = pad_along_axis(bird.P13, self.co.N13, axis=0) # back to shape (13, Nk)
        bird.P13 += delta_sigmavv * einsum('n,k->nk', self.uv13, self.co.k**2 * bird.P11) 
        bird.P22 += delta_sigmavvdd * einsum('n,k->nk', self.uv22, self.co.k**2) 

        # as the correlation function used only for resummation, we also add something; note that we don't add anything to 22 since the Fourier transform of a power law is just a delta
        if bird.C13l.shape[1] != self.co.N13: bird.C13l = pad_along_axis(bird.C13l, self.co.N13, axis=1)
        bird.C13l += delta_sigmavv * einsum('n,lk->lnk', self.uv13, bird.Cct) 

    def IRPsCf(self, bird): 

        pq_1 = self.fft.sumCoefxPow(bird.kin, bird.Pin, self.qIR, window=.2) 
        pq_2 = self.fft.sumCoefxPow(bird.kin, bird.Pin_2, self.qIR, window=.2) 

        delta_sigmavv = (6*pi**2)**-1 * trapz(pq_1 - pq_2, x=self.qIR) 

        if bird.P13.shape[0] != self.co.N13: bird.P13 = pad_along_axis(bird.P13, self.co.N13, axis=0) # back to shape (13, Nk)
        bird.P13 += delta_sigmavv * einsum('n,k->nk', self.ir13, self.co.k**2 * bird.P11) 
        bird.P22 += delta_sigmavv * einsum('n,k->nk', self.ir22, self.co.k**2 * bird.P11) 

        if bird.C13l.shape[1] != self.co.N13: bird.C13l = pad_along_axis(bird.C13l, self.co.N13, axis=1)
        bird.C13l += delta_sigmavv * einsum('n,lk->lnk', self.ir13, bird.Cct) 
        bird.C22l += delta_sigmavv * einsum('n,lk->lnk', self.ir22, bird.Cct) 







