import numpy as np

class EisensteinHu(object):
    """
    Linear power spectrum using the Eisenstein & Hu (1998) fitting formula
    without BAO wiggles.

    Parameters
    ----------
    cosmo : ...

    References
    ----------
    Eisenstein & Hu, "Baryonic Features in the Matter Transfer Function", 1998
    """
    def __init__(self, cosmo):
        
        self.Obh2      = cosmo["Omega0_b"] * cosmo["h"] ** 2
        self.Omh2      = cosmo["Omega0_m"] * cosmo["h"] ** 2
        self.h         = cosmo["h"]
        self.A_s       = cosmo["A_s"]
        self.n_s       = cosmo["n_s"]
        self.f_baryon  = cosmo["Omega0_b"] / cosmo["Omega0_m"]
        self.T_cmb     = cosmo["T_cmb"] / 2.7
        self.D         = cosmo["D"]

        # wavenumber of equality
        self.k_eq = 0.0746 * self.Omh2 * self.T_cmb ** (-2) # units of 1/Mpc

        self.sound_horizon = self.h * 44.5 * np.log(9.83/self.Omh2) / \
                            np.sqrt(1 + 10 * self.Obh2** 0.75) # in Mpc/h
        self.alpha_gamma = 1 - 0.328 * np.log(431*self.Omh2) * self.f_baryon + \
                            0.38* np.log(22.3*self.Omh2) * self.f_baryon ** 2


    def __call__(self, kk):
        r"""
        Return the Eisenstein-Hu transfer function without BAO wiggles.

        Parameters
        ---------
        kk : float, array_like
            the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`

        Returns
        -------
        Pk : float, array_like
            
        """

        # only compute k > 0 modes
        k = np.asarray(kk)
        valid = k > 0.

        k = k[valid] * self.h # in 1/Mpc now
        ks = k * self.sound_horizon / self.h
        q = k / (13.41*self.k_eq)

        gamma_eff = self.Omh2 * (self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43*ks) ** 4))
        q_eff = q * self.Omh2 / gamma_eff
        L0 = np.log(2*np.e + 1.8 * q_eff)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)

        T0 = np.ones(valid.shape)
        T0[valid] = L0 / (L0 + C0 * q_eff**2)
        
        return 2*np.pi**2 * self.A_s * kk / (100/299792.458)**4 * (k/0.05)**(self.n_s-1) * (T0*self.D)**2
