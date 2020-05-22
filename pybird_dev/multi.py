import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from scipy.integrate import quad, dblquad, simps

class Multi(object):

    def __init__(self, 
        kin, Pin, 
        z, Dz, fz, xdata,
        compute='ps', kmin=0.001, kmax=0.25, smin=1., 
        number_multipoles=2, number_wedges=0,
        accboost=1.,
        DAz=None, Hz=None, Om_AP=None, z_AP=None,
        use_window=False, window_fourier_name=None, path_to_window=None, window_configspace_file=None, 
        binning=False, fibcol=False):

        
        print ("-- bird settings --")
        # Power spectrum or correlation function?
        self.cf = False
        if 'ps' in compute:
            print ("Computing power spectrum")
        elif 'cf' in compute: 
            print ("Computing correlation function")
            self.cf = True
        elif: raise Exception('Choose compute = \'ps\' (power spectrum), \'cf\' (correlation function)')

        # Multipoles or wedges?
        self.Nl = number_multipoles
        self.Nw = number_wedges
        if self.Nw is not 0: self.use_wedges = True
        else: self.use_wedges = False 

        # Mask?
        if self.use_window:
            print("Mask: on")
            try:
                for i, file in enumerate(window_configspace_file):
                    test = np.loadtxt( os.path.join(path_to_window, file) )
                if self.cf: self.window_fourier_name = None
            except Exception as e:
                print("You set use_window, but there is a problem with your file specifications!")
                print("Please check that you don't have a typo in your path, and that the file is present.")
                raise
        else:
            print("Mask: none")
            self.window_fourier_name = None
            self.path_to_window = None
            self.window_configspace_file = None

        self.z = z
        self.Nz = len(z)
        self.Dz = Dz
        self.fz = fz
        
        self.DAz = DAz
        self.Hz = Hz

        print ('Loading pybird classes')
        self.nonlinear = NonLinear(load=True, save=True, co=self.co)
        self.resum = Resum(co=self.co)

        self.projection = []
        for i in :
            self.projection.append( Projection(xdata, Om_AP=Om_AP[i], z_AP=z_AP[i], 
                window_fourier_name=self.window_fourier_name[i], path_to_window='./', window_configspace_file=self.window_configspace_file[i], 
                binning=binning, fibcol=fibcol, Nwedges=Nwedges, cf=cf, co=self.co) )

        self.birds = []

        for i, (z, D, f, DA, H) in enumerate(zip(self.z, self.Dz, self.fz, self.DAz, self.Hz)):

            if i is 0: 
                print('Computing main bird at redshift z=%.2f' % z)
                self.bird = Bird(kin, Pin, z=z, D=D, f=f, which='all', co=self.co)
                self.nonlinear.PsCf(self.bird)
                self.bird.setPsCfl()
                self.resum.Ps(self.bird, setIR=True, setPs=False)
                self.birds.append(self.bird)

            else:
                print('Rescaling main bird to get bird at redshift z=%.2f' % z)
                birdi = deepcopy(self.bird)
                birdi.setCosmo(z=z, D=D, f=f)
                
                Dp1 = D/self.bird.D
                Dp2 = Dp1**2
                
                birdi.P11l *= Dp2 
                birdi.Pctl *= Dp2
                birdi.Ploopl *= Dp2**2

                Dp2n = np.concatenate(( 2*[self.co.Na*[Dp2**(n+1)] for n in range(self.co.NIR)] ))

                birdi.IRPs11 = np.einsum('n,lnk->lnk', Dp2*Dp2n, birdi.IRPs11)
                birdi.IRPsct = np.einsum('n,lnk->lnk', Dp2*Dp2n, birdi.IRPsct)
                birdi.IRPsloop = np.einsum('n,lmnk->lmnk', Dp2**2*Dp2n, birdi.IRPsloop) 

                if self.cf: self.resum.PsCf(birdi, setIR=False, setPs=True, setCf=True)
                else: self.resum.Ps(birdi, setIR=False, setPs=True)

                self.birds.append(birdi)

        if self.cf: self.birds[0].setresumPs()
        else: self.birds[0].setresumPs()

        for i in range(Nz):
            print ('Projection on the sky')
            self.projection[i].AP(birds[i])
            if self.use_window: self.projection[i].Window(birds[i])
            if self.fibcol: self.projection[i].fibcolWindow(birds[i])
            if self.use_wedges: self.projection[i].Wedges(birds[i]) 
            if self.binning: self.projection[i].kbinning(birds[i])
            else: self.projection[i].kdata(birds[i])

    def setBias(self, bs):
        for i in range(self.Nz): self.birds[i].setreducePslb(bs[i])