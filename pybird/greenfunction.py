from pybird.module import *

class GreenFunction(object):
    """A class to compute cosmological Green's functions and growth factors.
    
    The GreenFunction class handles calculations related to linear perturbation theory,
    including growth factors, growth rates, and Green's functions for different
    cosmological models (ΛCDM, wCDM, quintessence). It provides the time-dependent
    functions needed for perturbation theory calculations.
    
    Attributes:
        Omega0_m (float): Present-day matter density parameter.
        w (float): Dark energy equation of state parameter.
        quintessence (bool): Whether to include clustering quintessence.
        Omega0_k (float): Present-day curvature density parameter.
        vectorize (bool): Whether to vectorize calculations for arrays of parameters.
        OmegaL_by_Omega_m (float): Ratio of dark energy to matter density.
        wcdm (bool): Whether using wCDM cosmology.
        epsrel (float): Relative tolerance for numerical integrations.
    
    Methods:
        C(): Compute the time-dependent function C(a) for quintessence models.
        H(): Compute the conformal Hubble parameter H(a).
        H3(): Compute H(a)^-3.
        Omega_m(): Compute the time-dependent matter density parameter Ω_m(a).
        D(): Compute the linear growth factor D(a).
        DD(): Compute the derivative of the growth factor D'(a).
        fplus(): Compute the growth rate f+(a) = aD'(a)/D(a).
        Dminus(): Compute the decay factor D-(a).
        DDminus(): Compute the derivative of the decay factor D-'(a).
        fminus(): Compute the decay rate f-(a) = aD-'(a)/D-(a).
        W(): Compute the Wronskian W(a) = D'(a)D-(a) - D(a)D-'(a).
        
        G1d(), G2d(), G1t(), G2t(): Compute Green's functions for perturbation theory.
        I1d(), I2d(), I1t(), I2t(): Compute second order coefficients.
        mG1d(), mG2d(), mG1t(), mG2t(): Compute second order time integrals.
        G(): Compute quintessence time function.
        IU1d(), IU2d(), IU1t(), IU2t(): Compute third order coefficients.
        IV11d(), IV12d(), IV21d(), IV22d(): Compute third order coefficients.
        IV11t(), IV12t(), IV21t(), IV22t(): Compute third order coefficients.
        mU1d(), mU2d(), mU1t(), mU2t(): Compute third order time integrals.
        mV11d(), mV12d(), mV21d(), mV22d(): Compute third order time integrals.
        mV11t(), mV12t(), mV21t(), mV22t(): Compute third order time integrals.
        Y(): Compute third order time function.
    """

    def __init__(self, Omega0_m, w=None, quintessence=False, Omega0_k=0., vectorize=False):
        self.vectorize = vectorize
        self.Omega0_m = Omega0_m
        if self.vectorize:
            self.Omega0_m = np.array(Omega0_m)
            self.Omega0_k = np.array(Omega0_k)
            self.w = np.array(w) if w is not None else None
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m-Omega0_k)/self.Omega0_m
        self.wcdm = False
        self.quintessence = False
        if w is not None:
            self.w = w
            if quintessence: self.quintessence = True
            else: self.wcdm = True

        self.epsrel = 1e-4

    def C(self, a):
        if self.quintessence: return 1. + (1.+self.w) * self.OmegaL_by_Omega_m * a**(-3.*self.w)
        else: return 1.

    def H(self, a):
        """Conformal Hubble"""
        if self.wcdm or self.quintessence: return ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w)) )**.5
        else: return (self.Omega0_m/a + (1.-self.Omega0_m)*a**2)**.5

    def H3(self, a):
        return self.C(a)/self.H(a)**3

    def Omega_m(self, a):
        return self.Omega0_m / (self.H(a)**2 * a)

    def D(self, a):
        """Growth factor"""
        if self.wcdm: return a*hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-(a**(-3*self.w))*self.OmegaL_by_Omega_m)
        else:
            if self.vectorize:
                I = quad_vec(self.H3, 0, a, epsrel=self.epsrel)[0]
                return 5 * self.Omega0_m * I * self.H(a) / (2.*a)
            else:
                I = quad(self.H3, 0, a, epsrel=self.epsrel)[0]
            return 5 * self.Omega0_m * I * self.H(a) / (2.*a)

    def DD(self, a):
        """Derivative of growth factor"""
        if self.wcdm: return -(a**(-3.*self.w))*self.OmegaL_by_Omega_m*((3*(self.w-1))/(6.*self.w-5.))*hyp2f1(1.5-0.5*(1/self.w),1-(1/(3.*self.w)),2-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)+hyp2f1((self.w-1)/(2.*self.w),-1/(3.*self.w),1-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        else: return (2.5-(1.5*self.D(a)/a)) * self.Omega_m(a) * self.C(a)

    def fplus(self, a):
        """Growth rate"""
        return a * self.DD(a) / self.D(a)

    def Dminus(self, a):
        """Decay factor"""
        if self.wcdm: return a**(-3/2.)*hyp2f1(1/(2.*self.w),(1/2.)+(1/(3.*self.w)),1+(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        else: return self.H(a) / (a*self.Omega0_m**.5)

    def DDminus(self, a):
        """Derivative of decay factor"""
        if self.wcdm: return ((-1+3.*self.w)*hyp2f1(0.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m))-(2+3.*self.w)*hyp2f1(1.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m)))/(2*(a**(5/2.)))
        else: return -1.5 * self.Omega_m(a) * self.Dminus(a) / a * self.C(a)

    def fminus(self, a):
        """Decay rate"""
        return a * self.DDminus(a) / self.Dminus(a)

    def W(self, a):
        """Wronskian"""
        return self.DDminus(a) * self.D(a) - self.DD(a) * self.Dminus(a)

    #greens functions
    def G1d(self, a, ai):
        return(self.DDminus(ai)*self.D(a)-self.DD(ai)*self.Dminus(a))/(ai*self.W(ai))
    def G2d(self, a, ai):
        return self.fplus(ai)*(self.Dminus(a)*self.D(ai)-self.D(a)*self.Dminus(ai))/(ai*ai*self.W(ai))
    def G1t(self, a, ai):
        return a*(self.DDminus(ai)*self.DD(a)-self.DD(ai)*self.DDminus(a))/(self.fplus(a)*ai*self.W(ai))
    def G2t(self, a, ai):
        return a*self.fplus(ai)*(self.DDminus(a)*self.D(ai)-self.DD(a)*self.Dminus(ai))/(self.fplus(a)*ai*ai*self.W(ai))

    # second order coefficients
    def I1d(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(ai)
    def I2d(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2 / self.C(ai)
    def I1t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2 / self.C(ai)
    def I2t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2t(a,ai)/self.D(a)**2 / self.C(ai)

    # second order time integrals
    def mG1d(self, a):
        if self.vectorize:
            return quad_vec(self.I1d,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I1d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2d(self, a):
        if self.vectorize:
            return quad_vec(self.I2d,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I2d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a):
        if self.vectorize:
            return quad_vec(self.I1t,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I1t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a):
        if self.vectorize:
            return quad_vec(self.I2t,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I2t,0,a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IU2d(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IU1t(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IU2t(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)

    def IV11d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV12d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV21d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV22d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)

    def IV11t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV12t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV21t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV22t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
   
    # third order time integrals
    def mU1d(self, a):
        if self.vectorize:
            return quad_vec(self.IU1d,0,a,args=(a,), epsrel=self.epsrel)[0]

        else:
            return quad(self.IU1d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2d(self, a):
        if self.vectorize:
            return quad_vec(self.IU2d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU1t(self, a):
        if self.vectorize:
            return quad_vec(self.IU1t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU1t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2t(self, a):
        if self.vectorize:
            return quad_vec(self.IU2t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2t,0,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11d(self, a):
        if self.vectorize:
            return quad_vec(self.IV11d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12d(self, a):
        if self.vectorize:
            return quad_vec(self.IV12d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV12d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a):
        if self.vectorize:
            return quad_vec(self.IV21d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22d(self, a):
        if self.vectorize:
            return quad_vec(self.IV22d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22d,0,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11t(self, a):
        if self.vectorize:
            return quad_vec(self.IV11t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12t(self, a):
        if self.vectorize:
            return quad_vec(self.IV12t,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-5)[0]
        else:
            return quad(self.IV12t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21t(self, a):
        if self.vectorize:
            return quad_vec(self.IV21t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22t(self, a):
        if self.vectorize:
            return quad_vec(self.IV22t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22t,0,a,args=(a,), epsrel=self.epsrel)[0]

    def Y(self, a):
        if self.quintessence: return -3/14.*self.G(a)**2 + self.mV11d(a) + self.mV12d(a)
        else: return -3/14. + self.mV11d(a) + self.mV12d(a)