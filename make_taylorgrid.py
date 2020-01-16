# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
rank = 0

import numpy as np
import os
import sys
import Grid
import pybird_fullresum as pybird
import copy
import time

#time.sleep(4)

basedir = './'
OUTPATH = os.path.join(basedir, "output")
outpk = os.path.join(basedir, "Pk")


ncores = 1#size
nrun = 1#int(sys.argv[1])
runs = 1#int(sys.argv[2])
lenrun = int(len(Grid.flattenedgrid) / runs)
thetarun = Grid.flattenedgrid[nrun * lenrun:(nrun+1) * lenrun]
Ntot = len(thetarun)
sizered = int(Ntot/ncores)

arrayred = Grid.flattenedgrid #thetarun[rank * sizered:(rank+1) * sizered]
sizearray = len(arrayred)

freepar = Grid.freepar
#print("lenrun, sizered", lenrun, sizered)

### To create outside the grid
nonlinear = pybird.NonLinear(load=True,save=True)
resum = pybird.Resum()
kbird = pybird.common.k
allk = np.concatenate([kbird, kbird]).reshape(-1,1)

allPlin = []
allPloop = []
for i, theta in enumerate(arrayred):
    parameters = copy.deepcopy(Grid.parref)
    truetheta = Grid.valueref + theta * Grid.delta
    #idx = nrun * lenrun + rank * sizered + i
    #print("nrun, rank, i", nrun, rank, i)
    idx = i
    print ("i on tot", i, sizearray)

    parameters["PathToOutput"] = os.path.join(OUTPATH, 'output' + str(nrun) + str(rank) + str(i))
    for k, var in enumerate(freepar):
        parameters[var] = truetheta[k]
    # parameters['h'] = 1/parameters['invh']
    kin, Plin, z, Omega_m = Grid.CompPterms(parameters)
    bird = pybird.Bird(kin, Plin, Omega_m, z, full=False)
    nonlinear.PsCf(bird)
    bird.setPsCfl()
    resum.Ps(bird, full=False)
    bird.subtractShotNoise()
    
    Plin, Ploop = bird.formatTaylor()
    idxcol = np.full([Plin.shape[0], 1], idx)
    allPlin.append(np.hstack([Plin, idxcol]))
    allPloop.append(np.hstack([Ploop, idxcol]))
    if (i == 0) or ((i+1) % 100 == 0):
        print("theta check: ", Grid.flattenedgrid[idx], theta, truetheta)
        # np.save(os.path.join(outpk, "temp", "Plin_run%s_rank%si%s.npy" % (str(nrun), str(rank), str(i))), np.array(allPlin))
        # np.save(os.path.join(outpk, "temp", "Ploop_run%s_rank%si%s.npy" % (str(nrun), str(rank), str(i))), np.array(allPloop))
    np.save(os.path.join(outpk, "Plin_run%s_rank%s.npy" % (str(nrun), str(rank))), np.array(allPlin))
    np.save(os.path.join(outpk, "Ploop_run%s_rank%s.npy" % (str(nrun), str(rank))), np.array(allPloop))
