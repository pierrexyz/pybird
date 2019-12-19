from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import numpy as np
import os
import sys
import copy
import Grid
import computederivs
from scipy import interpolate, integrate, special

time.sleep(3)

basedir = "../grouphome"
OUTPATH = os.path.join(basedir, "output")
outpk = os.path.join(basedir, "Pk")
gridpath = os.path.join(basedir, "GridsEFT", "newbird")
gridname = "z0p5-A_s-h-omega_cdm-omega_b-n_s"
#gridname = "z0p55-A_s-h-omega_cdm-omega_b-n_s-Sum_mnu-APnofcwinNGC"

linder = computederivs.load_pder(os.path.join(gridpath,  'DerPlin_%s.npy' % gridname))
loopder = computederivs.load_pder(os.path.join(gridpath, 'DerPloop_%s.npy' % gridname))
nmult = 2

ncores = size
nrun = int(sys.argv[1])
runs = int(sys.argv[2])

central = np.array([3.15, 0.319, 0.674, 0.022, 0.9649])
sigmas = np.array([0.08, 0.010, 0.009, 0.001, 0.039])
# sigmas = np.array([0.15, 0.015, 0.029, 0.000625, 0.058, 0.13833])
# sigmas = np.array([0.145, 0.0155, 0.0145, 0.0008, 0.0495, 0.11666])


bfit = {'b1': 2.4, 'b2': 1.4 / np.sqrt(2.), 'b3': 0., 'b4': 1.4 / np.sqrt(2.),
        'b5': 0., 'b6': -6., 'b7': 0., 'b8': 0.02,
        'b9': 0., 'b10': -2.8, 'b11': 0,
        'e1': 0, 'e2': 0}

# Algorithm to pick variables uniformly on a S_n: pick n x ~N(0, 1), and normalize
flattened = []
N = 100
dim = len(central)
rs = np.random.RandomState(seed=37)
"""for i in range(N):
    a = rs.normal(size=dim)
    x = a / np.linalg.norm(a)
    y = central + 2 * sigmas * x
    thisAs = 1e-10 * np.exp(y[0])
    thisOm = y[1]
    thish = y[2]
    thisomb = y[3]
    thisomc = thisOm * thish**2 - thisomb
    thisns = y[4]
    flattened.append(np.array([thisAs, thish, thisomc, thisomb, thisns]))
"""
flattened = np.load("thomas_cosmo.npy")

lenrun = int(len(flattened) / runs)
thetarun = flattened[nrun * lenrun:(nrun + 1) * lenrun]
Ntot = len(thetarun)
sizered = int(Ntot / ncores)
arrayred = thetarun[rank * sizered:(rank + 1) * sizered]

freepar = Grid.freepar
# print("lenrun, sizered", lenrun, sizered)

# nd = computederivs.nd
# km = computederivs.km
simname = "Challenge"
ZONE = "NGC"
allfP = []
allP = []
for i, theta in enumerate(arrayred):
    # print(theta)
    parameters = copy.deepcopy(Grid.parref)
    idx = nrun * lenrun + rank * sizered + i
    # print("nrun, rank, i", nrun, rank, i)
    parameters["PathToOutput"] = os.path.join(OUTPATH, 'output' + str(idx))
    for k, var in enumerate(freepar):
        parameters[var] = theta[k]

    dtheta = theta - Grid.valueref
    # print(dtheta)
    PlinTaylor = computederivs.get_PSTaylor(dtheta, linder)
    PloopTaylor = computederivs.get_PSTaylor(dtheta, loopder)
    kin, PSfake = computederivs.get_PSbias(PlinTaylor,PloopTaylor, bfit)
    allfP.append(PSfake)
    if (i == 0) or ((i + 1) % 100 == 0):
        print("theta check: ", Grid.flattenedgrid[idx], theta)
    np.save(os.path.join(outpk, "fP_run%s_rank%s.npy" % (str(nrun), str(rank))), np.array(allfP))
    Plintrue, Plooptrue = Grid.CompPterms(parameters)
    Plintrue = Plintrue.reshape((nmult, int(len(Plintrue) / nmult), Plintrue.shape[-1]))[:nmult, :, :]
    Plooptrue = Plooptrue.reshape((nmult, int(len(Plooptrue) / nmult), Plooptrue.shape[-1]))[:nmult, :, :19]
    ktrue = Plintrue[0, :, 0]
    kin, PStrue = computederivs.get_PSbias(Plintrue[:, :, :], Plooptrue[:, :, :19], bfit)
    allP.append(PStrue)
    np.save(os.path.join(outpk, "kin.npy"), kin)
    np.save(os.path.join(outpk, "P_run%s_rank%s.npy" % (str(nrun), str(rank))), np.array(allP))
