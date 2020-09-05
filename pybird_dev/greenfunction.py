from scipy.integrate import quad
from scipy.special import hyp2f1

class GreenFunction(object):
    
    def __init__(self, Omega0_m, w=None, quintessence=False):
        self.Omega0_m = Omega0_m
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m)/self.Omega0_m
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
        return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(a)
    def I2d(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2 / self.C(a)
    def I1t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2 / self.C(a)
    def I2t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2t(a,ai)/self.D(a)**2 / self.C(a)

    # second order time integrals
    def mG1d(self, a): 
        return quad(self.I1d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2d(self, a): 
        return quad(self.I2d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a): 
        return quad(self.I1t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a): 
        return quad(self.I2t,0,a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU2d(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU1t(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU2t(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def IV11d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV21d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV22d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def IV11t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV21t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV22t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    
    # third order time integrals
    def mU1d(self, a): 
        return quad(self.IU1d,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mU2d(self, a): 
        return quad(self.IU2d,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mU1t(self, a): 
        return quad(self.IU1t,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mU2t(self, a): 
        return quad(self.IU2t,0,a,args=(a,), epsrel=self.epsrel)[0] 

    def mV11d(self, a): 
        return quad(self.IV11d,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mV12d(self, a): 
        return quad(self.IV12d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a): 
        return quad(self.IV21d,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mV22d(self, a): 
        return quad(self.IV22d,0,a,args=(a,), epsrel=self.epsrel)[0] 

    def mV11t(self, a): 
        return quad(self.IV11t,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mV12t(self, a): 
        return quad(self.IV12t,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mV21t(self, a): 
        return quad(self.IV21t,0,a,args=(a,), epsrel=self.epsrel)[0] 
    def mV22t(self, a): 
        return quad(self.IV22t,0,a,args=(a,), epsrel=self.epsrel)[0] 
    
    def Y(self, a):
        return -3/14. + self.mV11d(a) + self.mV12d(a)
    