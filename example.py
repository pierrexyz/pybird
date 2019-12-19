import pybird as pybird

### Example for Taylor grid

### To create outside the grid
nonlinear = pybird.NonLinear(load=True,save=True)
resum = pybird.Resum()


### To create at each grid step
kin, Plin = np.loadtxt('output/test/class_pk.dat', unpack = True)
Omega_m = (0.1284905+0.0233854)/0.690**2
z = 0.55
bird = pybird.Bird(kin, Plin, Omega_m, z, full=False)
nonlinear.PsCf(bird, window=None)
bird.setPsCfl()
resum.Ps(bird, full=False)
bird.subtractShotNoise()
### the 1-loop resummed power spectrum is stored in:
# bird.P11l = np.empty(shape=(co.Nl, co.N11, co.Nk))
# bird.Ploopl = np.empty(shape=(co.Nl, co.Nloop, co.Nk))
# bird.Pctl = np.empty(shape=(co.Nl, co.Nct, co.Nk))
### where:
# Nl = 2 : number of multipoles
# N11 = 3, Nloop = 12, Nct = 6
# Nk : number of k points

### the k points are:
k = pybird.common.k



### You can test if everything went well with the following method:
bs = np.array([2.3, 0.8, 0.2, 0.8, 0.4, -7., 0.])
bird.setreducePslb(bs)

l = 0
plt.plot(k, k*bird.fullPs[l])
