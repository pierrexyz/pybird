from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import numpy as np
import os
import sys
import Grid
import pybird
import copy
import time

time.sleep(4)

basedir = './'
OUTPATH = os.path.join(basedir, "output")
outpk = os.path.join(basedir, "Pk")


ncores = size
nrun = int(sys.argv[1])
runs = int(sys.argv[2])
lenrun = int(len(Grid.flattenedgrid) / runs)
thetarun = Grid.flattenedgrid[nrun * lenrun:(nrun+1) * lenrun]
Ntot = len(thetarun)
sizered = int(Ntot/ncores)

arrayred = thetarun[rank * sizered:(rank+1) * sizered]

freepar = Grid.freepar
print("lenrun, sizered", lenrun, sizered)
### To create outside the grid
nonlinear = pybird.NonLinear(load=True,save=True)
resum = pybird.Resum()
allPlin = []
allPloop = []
for i, theta in enumerate(arrayred):
    parameters = copy.deepcopy(Grid.parref)
    truetheta = Grid.valueref + theta * Grid.delta
    idx = nrun * lenrun + rank * sizered + i
    print("nrun, rank, i", nrun, rank, i)
    parameters["PathToOutput"] = os.path.join(OUTPATH, 'output' + str(nrun) + str(rank) + str(i))
    for k, var in enumerate(freepar):
        parameters[var] = truetheta[k]
    # parameters['h'] = 1/parameters['invh']
    kin, Plin, z, Omega_m = Grid.CompPterms(parameters)
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
    print(np.shape(np.swapaxes(bird.P11l, 0, 2)))
    print(np.shape(np.tile(k.reshape((50,1,1)), (1,3,2))))
    Plin = np.concatenate((np.tile(k.reshape((50,1,1)), (1,3,2)), np.swapaxes(bird.P11l, 0, 2)), axis=1)
    Plin = np.vstack((Plin[..., 0], Plin[..., 1]))
    Ploop1 = np.swapaxes(bird.Ploopl, 0, 2)
    Ploop2 = np.swapaxes(bird.Pctl, 0, 2)
    Ploopl0 = np.hstack((k.reshape((len(k), 1)), Ploop1[..., 0], Ploop2[..., 0]))
    Ploopl1 = np.hstack((k.reshape((len(k), 1)), Ploop1[..., 1], Ploop2[..., 1]))
    Ploop = np.vstack((Ploopl0, Ploopl1))
    idxcol = np.full([Plin.shape[0], 1], idx)
    allPlin.append(np.hstack([Plin, idxcol]))
    allPloop.append(np.hstack([Ploop, idxcol]))
    if (i == 0) or ((i+1) % 100 == 0):
        print("theta check: ", Grid.flattenedgrid[idx], theta, truetheta)
        # np.save(os.path.join(outpk, "temp", "Plin_run%s_rank%si%s.npy" % (str(nrun), str(rank), str(i))), np.array(allPlin))
        # np.save(os.path.join(outpk, "temp", "Ploop_run%s_rank%si%s.npy" % (str(nrun), str(rank), str(i))), np.array(allPloop))
    np.save(os.path.join(outpk, "Plin_run%s_rank%s.npy" % (str(nrun), str(rank))), np.array(allPlin))
    np.save(os.path.join(outpk, "Ploop_run%s_rank%s.npy" % (str(nrun), str(rank))), np.array(allPloop))
