from pybird.module import * 
from pybird.symbolic_pofk_linear import plin_emulated, sigma8_to_As, As_to_sigma8

def cH(Om,a,w0=-1.,wa=0.): return sqrt(Om/a + (1-Om)*a**(-3.*(1.+w0+wa)+2)*exp(-3.*wa*(1.-a)))
def _D(Om,a,w0=-1.,wa=0.): 
    aa = linspace(1e-5, a, num=30, endpoint=True) # fine-tuned small number of points, precision OK
    return 5/2. * Om * cH(Om,a,w0,wa)/a * trapz(cH(Om, aa, w0, wa)**-3, x=aa)
def D(Om,z,w0=-1.,wa=0.): return _D(Om,1/(1+z),w0,wa)/_D(Om,1,w0,wa)

def f(Om,z,w0=-1.,wa=0.): 
    a = 1/(1+z)
    def _logD(a):
        return log(_D(Om, a, w0, wa))
    if is_jax and (w0 !=-1. or wa != 0.): return a*grad(_logD)(a)
    else: return (Om*(5*a - 3*_D(Om,a,-1,0)))/(2.*(a**3*(1 - Om) + Om)*_D(Om,a,-1,0)) # LCDM only!

def Hubble(Om,z,w0=-1.,wa=0.): return ((Om)*(1+z)**3.+(1-Om)*(1+z)**(3.*(1.+w0+wa))*exp(-3.*wa*z/(1.+z)))**0.5
def DA(Om,z,w0=-1.,wa=0.):
    zz = linspace(1e-5, z, num=30, endpoint=True)
    return trapz(1/Hubble(Om,zz,w0,wa), x=zz) / (1+z)

class Symbolic():
    def __init__(self, max_precision=False, smooth_de=True):
        self.cosmo_name = ['omega_b', 'omega_cdm', 'h', 'ln10^{10}A_s', 'n_s', 'Omega_b', 'Omega_m', 'A_s', 'sigma_8', 'm_ncdm', 'w0_fld', 'wa_fld']
        self.max_precision, self.smooth_de = max_precision, smooth_de

    def set(self, cosmo):
        self.c = {k: v for k, v in cosmo.items() if k in self.cosmo_name}
        if 'w0_fld' in self.c and not is_jax: sys.exit("Need to turn on jax to do dynamical dark energy with Symbolic!")
        if 'm_ncdm' not in self.c: self.c['m_ncdm'] = 0.
        if 'w0_fld' not in self.c: self.c['w0_fld'] = -1.
        if 'wa_fld' not in self.c: self.c['wa_fld'] = 0.
        if 'Omega_b' not in self.c: self.c['Omega_b'] = self.c['omega_b'] / self.c['h']**2
        if 'Omega_m' not in self.c: self.c['Omega_m'] = (self.c['omega_cdm'] + self.c['omega_b'] + self.c['m_ncdm']/93.14) / self.c['h']**2
        if 'sigma_8' not in self.c: 
            if 'A_s' not in self.c: self.c['A_s'] = 1e-10 * exp(self.c['ln10^{10}A_s']) 
            self.c['sigma_8'] = sqrt(1e9) * As_to_sigma8(*(self.c[key] for key in ['A_s', 'Omega_m', 'Omega_b', 'h', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld']), max_precision=self.max_precision)
        return

    def compute(self, k, z, emulator='fiducial'): 
        self.k, self.z = k, z
        self.D, self.f = D(self.c['Omega_m'], self.z, self.c['w0_fld'], self.c['wa_fld']), f(self.c['Omega_m'], self.z, self.c['w0_fld'], self.c['wa_fld']) # for RSD
        self.H, self.DA = Hubble(self.c['Omega_m'], self.z, self.c['w0_fld'], self.c['wa_fld']), DA(self.c['Omega_m'], self.z, self.c['w0_fld'], self.c['wa_fld']) # for AP effect
        if self.smooth_de: # we evolve the power spectrum from LCDM deep inside matter domination with the growth factor in w0waCDM
            zmd = 3. # redshift in matter domination (currently max redshift supported by Symbolic)
            pk_lin_zmd = 1e9 * plin_emulated(self.k, *(self.c[key] for key in ['A_s', 'Omega_m', 'Omega_b', 'h', 'n_s', 'm_ncdm']), -1., 0., 1/(1.+zmd), max_precision=self.max_precision) 
            D_zmd = D(self.c['Omega_m'], zmd, -1., 0.)
            self.pk_lin = pk_lin_zmd * self.D**2 / D_zmd**2
        else: 
            self.pk_lin = 1e9 * plin_emulated(self.k, *(self.c[key] for key in ['A_s', 'Omega_m', 'Omega_b', 'h', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld']), 1/(1.+z), max_precision=self.max_precision)
        return 

