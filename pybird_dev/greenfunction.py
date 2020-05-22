from scipy.integrate import quad
from scipy.special import hyp2f1

class GreenFunction(object):
    
    def __init__(self, Omega0_m, w=None):
        self.Omega0_m = Omega0_m
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m)/self.Omega0_m
        if w is None: self.wcdm = False
        else: 
            self.w = w
            self.wcdm = True

    def H(self, a):
        """Conformal Hubble"""  
        if self.wcdm: return ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w)) )**.5
        else: return (self.Omega0_m/a + (1.-self.Omega0_m)*a**2)**.5
    
    def H3(self, a):
        return 1/self.H(a)**3

    def Omega_m(self, a):
        return self.Omega0_m / (self.H(a)**2 * a)

    def D(self, a): 
        """Growth factor"""
        if self.wcdm: return a*hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-(a**(-3*self.w))*self.OmegaL_by_Omega_m)
        else:
            I = quad(self.H3, 0, a)[0]
            return 5 * self.Omega0_m * I * self.H(a) / (2.*a)
    
    def DD(self, a):
        """Derivative of growth factor"""
        if self.wcdm: return -(a**(-3.*self.w))*self.OmegaL_by_Omega_m*((3*(self.w-1))/(6.*self.w-5.))*hyp2f1(1.5-0.5*(1/self.w),1-(1/(3.*self.w)),2-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)+hyp2f1((self.w-1)/(2.*self.w),-1/(3.*self.w),1-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        else: return (2.5-(1.5*self.D(a)/a)) * self.Omega_m(a)

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
        else: return -1.5 * self.Omega_m(a) * self.Dminus(a) / a

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
        return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2
    def I2d(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2
    def I1t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2
    def I2t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2t(a,ai)/self.D(a)**2

    # second order time integrals
    def mG1d(self, a): 
        return quad(self.I1d,0,a,args=(a,))[0]
    def mG2d(self, a): 
        return quad(self.I2d,0,a,args=(a,))[0]
    def mG1t(self, a): 
        return quad(self.I1t,0,a,args=(a,))[0]
    def mG2t(self, a): 
        return quad(self.I2t,0,a,args=(a,))[0]

    # third order coefficients
    def IU1d(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3
    def IU2d(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3
    def IU1t(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3
    def IU2t(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3

    def IV11d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3
    def IV12d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3
    def IV21d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3
    def IV22d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3

    def IV11t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3
    def IV12t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3
    def IV21t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3
    def IV22t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3
    
    # third order time integrals
    def mU1d(self, a): 
        return quad(self.IU1d,0,a,args=(a,))[0] 
    def mU2d(self, a): 
        return quad(self.IU2d,0,a,args=(a,))[0] 
    def mU1t(self, a): 
        return quad(self.IU1t,0,a,args=(a,))[0] 
    def mU2t(self, a): 
        return quad(self.IU2t,0,a,args=(a,))[0] 

    def mV11d(self, a): 
        return quad(self.IV11d,0,a,args=(a,))[0] 
    def mV12d(self, a): 
        return quad(self.IV12d,0,a,args=(a,))[0]
    def mV21d(self, a): 
        return quad(self.IV21d,0,a,args=(a,))[0] 
    def mV22d(self, a): 
        return quad(self.IV22d,0,a,args=(a,))[0] 

    def mV11t(self, a): 
        return quad(self.IV11t,0,a,args=(a,))[0] 
    def mV12t(self, a): 
        return quad(self.IV12t,0,a,args=(a,))[0] 
    def mV21t(self, a): 
        return quad(self.IV21t,0,a,args=(a,))[0] 
    def mV22t(self, a): 
        return quad(self.IV22t,0,a,args=(a,))[0] 
    
    def Y(self, a):
        return -3/14. + self.mV11d(a) + self.mV12d(a)
    