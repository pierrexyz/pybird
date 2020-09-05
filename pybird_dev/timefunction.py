from scipy.interpolate import interp1d
from numpy import trapz, meshgrid

class TimeFunction(object):

	def Y(self, a): # 3rd-order exact time function
		return -3/14. + self.mV11d(a) + self.mV12d(a)

    def G(self, a): # quintessence 2nd-order time function
        return self.mG1d(a) + self.mG2d(a)

	def mV12t(self, a): ####
        return quad(self.IV12t,0,a,args=(a,))[0]
    def mV11d(self, a): ####
        return quad(self.IV11d,0,a,args=(a,))[0] 
    def mV12d(self, a): ####
        return quad(self.IV12d,0,a,args=(a,))[0]

    def IV12t(self, ai,a): ###
        return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV11d(self, ai, a): ###
        return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12d(self, ai, a): ###
        return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def mG1d(self, a): ###
        return quad(self.I1d,0,a,args=(a,))[0]
    def mG2d(self, a): ###
        return quad(self.I2d,0,a,args=(a,))[0]
    def mG1t(self, a): #### ## #
        return quad(self.I1t,0,a,args=(a,))[0]
    
    def I1d(self, ai, a): ##
        return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(a)
    def I2d(self, ai, a): ##
        return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2 / self.C(a)
    def I1t(self, ai, a): #
        return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2 / self.C(a)

    def G1d(self, a, ai): #
        return(self.DDminus(ai)*self.D(a)-self.DD(ai)*self.Dminus(a))/(ai*self.W(ai))
    def G2d(self, a, ai): #
        return self.fplus(ai)*(self.Dminus(a)*self.D(ai)-self.D(a)*self.Dminus(ai))/(ai*ai*self.W(ai))
    def G1t(self, a, ai): 
        return a*(self.DDminus(ai)*self.DD(a)-self.DD(ai)*self.DDminus(a))/(self.fplus(a)*ai*self.W(ai))
    def G2t(self, a, ai):
        return a*self.fplus(ai)*(self.DDminus(a)*self.D(ai)-self.DD(a)*self.Dminus(ai))/(self.fplus(a)*ai*ai*self.W(ai))

    def integrate(self, func2d):
    	return trapz(func, axis=-1, x=self.aai)

    def makemG(self, a):
    	self.I1d = self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(a)
    	self.mG1d = self.integrate(self.I1d)

    def eval(self, func, x1d):
    	return func(x1d)

    def mesheval(self, func, x1d, x2d):
        ifunc = interp1d(x1d, func, axis=-1, kind='cubic')
        return ifunc(x2d)

    def makeGrowth(self, zmax):
    	"""
    	a: array
    	"""
    	a = np.linspace(0, 1/(1/+zmax), num=200)
    	self.a1, self.a2 = meshgrid(a, a, indexing='ij')

	    if self.wcdm or self.quintessence: self.H = ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w)) )**.5
		else: self.H = (self.Omega0_m/a + (1.-self.Omega0_m)*a**2)**.5
		self.C = 1.
		if quintessence: self.C += (1.+self.w) * self.OmegaL_by_Omega_m * a**(-3.*self.w)
		
	    self.H3 = self.C / self.H**3
        self.Omega_m = self.Omega0_m / (self.H**2*a)
    	def D(a): # Growth factor
        	I = quad(self.H3, 0, a)[0]
        	return 5 * self.Omega0_m * I * H(a) / (2.*a)

        self.mesheval(self.H3, a, )


        def DD(self, a): # Growth factor derivative
       		return (2.5-(1.5*D(a)/a)) * self.Omega_m(a) * self.C(a)
       	def fplus(self, a): # Growth rate
        	return a * self.DD(a) / self.D(a)
        def Dminus(self, a): # Decay factor
        	return self.H(a) / (a*self.Omega0_m**.5)
        def DDminus(self, a): # Decay factor derivative
        	return -1.5 * self.Omega_m(a) * self.Dminus(a) / a * self.C(a)
        def fminus(self, a): # Decay rate
        	return a * self.DDminus(a) / self.Dminus(a)
        def W(self, a): # Wronskian
            return self.DDminus(a) * self.D(a) - self.DD(a) * self.Dminus(a)




    # def eval2d(self, x1d, x1, x2, func):
    #     ifunc = interp1d(x1d, func, axis=-1, kind='cubic')
    #     return ifunc(x1), ifunc(x2)

    

    




    
    
    