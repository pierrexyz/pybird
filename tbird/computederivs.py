import numpy as np
import Grid
import os
import time
import copy
import sys
from itertools import combinations
import findiff

shape = Grid.truegrid.shape
parameters = copy.deepcopy(Grid.parref)
try:
    basedir = "../homegroup" 
    griddir = os.path.join(basedir, "GridsEFT")
except:
    basedir = './'
    griddir = 'grids'
gridname = sys.argv[1]
# gridname = "z0p55-A_s-h-omega_cdm-omega_b-n_s-APfcwinSGC"
# psdatadir = os.path.join("input", "DataSims")
# covdatadir = os.path.join("input", "Covariance")
# dataname = "Challenge_A"

knl = 0.7
km = knl
nd = 4.5e-4
# Shape of the crd is now lenpar, gridsize, ... gridsize
# Shape of the PS is now gridsize, ... gridsize, nmult, nk, columns (including the k)
# Since I have the dx, I don't need the coordinates really


# Tested, good
def get_grids(mydir, name=gridname, nmult=2, nout=2):
    # Coordinates have shape (len(freepar), 2 * order_1 + 1, ..., 2 * order_n + 1)
    # order_i is the number of points away from the origin for parameter i
    # The len(freepar) sub-arrays are the outputs of a meshgrid, which I feed to findiff
    # Power spectra needs to be reshaped.
    crd = np.load(os.path.join(mydir, "Tablecoord_%s.npy" % name))  # Don't need this for uniform grid
    shapecrd = crd.shape
    padshape = [(1,1)] * (len(shapecrd) - 1) + [(0, 0)] * 3
    plin = np.load(os.path.join(mydir, "TablePlin_%s.npy" % name))
    plin = plin.reshape((*shapecrd[1:], nmult, plin.shape[-2] // nmult, plin.shape[-1]))  # This won't work with Python 2 :(
    plin = np.pad(plin, padshape, 'constant', constant_values=0)
    ploop = np.load(os.path.join(mydir, "TablePloop_%s.npy" % name))
    ploop = ploop.reshape((*shapecrd[1:], nmult, ploop.shape[-2] // nmult, ploop.shape[-1]))  # This won't work with Python 2 :(
    ploop = np.pad(ploop, padshape, 'constant', constant_values=0)
    # The output is not concatenated for multipoles since we remove the hexadecapole
    return plin[..., :nout, :, :], ploop[..., :nout, :, :]


# Tested, it works well
def get_pder_lin(pi, dx, filename):
    """ Calculates the derivative aroud the Grid.valueref points. Do this only once.
    gridshape is 2 * order + 1, times the number of free parameters
    pi is of shape gridshape, n multipoles, k length, P columns (zeroth being k's)"""
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    t0 = time.time()
    lenpar = len(Grid.valueref)
    idx = Grid.center
    t1 = time.time()
    
    p0 = pi[idx, idx, idx, idx, idx, :, :, :]
    print("Done p0 in %s sec" % str(t1 - t0))

    dpdx = np.array([findiff.FinDiff((i, dx[i], 1), acc=4)(pi)[idx, idx, idx, idx, idx, :, :, :] for i in range(lenpar)])
    t0 = time.time()
    print("Done dpdx in %s sec" % str(t1 - t0))

    # Second derivatives
    d2pdx2 = np.array([findiff.FinDiff((i, dx[i], 2), acc=2)(pi)[idx, idx, idx, idx, idx, :, :, :] for i in range(lenpar)])
    t1 = time.time()
    print("Done d2pdx2 in %s sec" % str(t1 - t0))
    
    d2pdxdy = np.array([[i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), acc=2)(pi)[idx, idx, idx, idx, idx, :, :, :]]
                        for (i, j) in combinations(range(lenpar), 2)])
    t0 = time.time()
    print("Done d2pdxdy in %s sec" % str(t1 - t0))
    
    # Third derivatives: we only need it for A_s, so I do this by hand
    d3pdx3 = np.array([findiff.FinDiff((i, dx[i], 3))(pi)[idx, idx, idx, idx, idx, :, :, :] for i in range(lenpar)])
    t1 = time.time()
    print("Done d3pdx3 in %s sec" % str(t1 - t0))
    
    t1 = time.time()
    d3pdx2dy = np.array([[i, j, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1))(pi)[idx, idx, idx, idx, idx, :, :, :]]
                          for (i, j) in combinations(range(lenpar), 2)])
    t0 = time.time()
    print("Done d3pdx2dy in %s sec" % str(t1 - t0))
    
    d3pdxdy2 = np.array([[i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2))(pi)[idx, idx, idx, idx, idx, :, :, :]]
                          for (i, j) in combinations(range(lenpar), 2)])
    t1 = time.time()
    print("Done d3pdxdy2 in %s sec" % str(t1 - t0))
    
    d3pdxdydz = np.array([[i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1))(pi)[idx, idx, idx, idx, idx, :, :, :]]
                          for (i, j, k) in combinations(range(lenpar), 3)])
    t0 = time.time()
    print("Done d3pdxdydz in %s sec" % str(t1 - t0))
    
    #allder = (p0, dpdx, d2pdx2, d2pdxdy, d3pdx3)
    #allder = (d3pdx2dy, d3pdxdy2)
    #allder = (d3pdxdydz, )
    allder = (p0, dpdx, d2pdx2, d2pdxdy, d3pdx3, d3pdx2dy, d3pdxdy2, d3pdxdydz)
    np.save(filename, allder)
    return allder


def get_pder_loop1(pi, dx, filename):
    t0 = time.time()
    lenpar = len(Grid.valueref)
    idx = Grid.center
    t1 = time.time()
    
    p0 = pi[idx, idx, idx, idx, idx, idx, :, :, :]
    print("Done p0 in %s sec" % str(t1 - t0))

    dpdx = np.array([findiff.FinDiff((i, dx[i], 1), acc=4)(pi)[idx, idx, idx, idx, idx, idx, :, :, :] for i in range(lenpar)])
    t0 = time.time()
    print("Done dpdx in %s sec" % str(t1 - t0))

    # Second derivatives
    d2pdx2 = np.array([findiff.FinDiff((i, dx[i], 2), acc=2)(pi)[idx, idx, idx, idx, idx, idx, :, :, :] for i in range(lenpar)])
    t1 = time.time()
    print("Done d2pdx2 in %s sec" % str(t1 - t0))
    
    d2pdxdy = np.array([[i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), acc=2)(pi)[idx, idx, idx, idx, idx, idx, :, :, :]]
                        for (i, j) in combinations(range(lenpar), 2)])
    t0 = time.time()
    print("Done d2pdxdy in %s sec" % str(t1 - t0))
    
    # Third derivatives: we only need it for A_s, so I do this by hand
    d3pdx3 = np.array([findiff.FinDiff((i, dx[i], 3))(pi)[idx, idx, idx, idx, idx, idx, :, :, :] for i in range(lenpar)])
    t1 = time.time()
    print("Done d3pdx3 in %s sec" % str(t1 - t0))
    allder = (p0, dpdx, d2pdx2, d2pdxdy, d3pdx3)
    np.save(filename, allder)
    return allder


def get_pder_loop2a(pi, dx, filename):
    """ Calculates the derivative aroud the Grid.valueref points. Do this only once.
    gridshape is 2 * order + 1, times the number of free parameters
    pi is of shape gridshape, n multipoles, k length, P columns (zeroth being k's)"""
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    lenpar = len(Grid.valueref)
    idx = Grid.center
    t1 = time.time()
    d3pdx2dy = np.array([[i, j, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1))(pi)[idx, idx, idx, idx, idx, idx, :, :, :]]
                          for (i, j) in combinations(range(lenpar), 2)])
    t0 = time.time()
    print("Done d3pdx2dy in %s sec" % str(t1 - t0))
    allder = (d3pdx2dy,)
    np.save(filename, allder)
    return allder

def get_pder_loop2b(pi, dx, filename):
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    lenpar = len(Grid.valueref)
    idx = Grid.center
    t0 = time.time()
    d3pdxdy2 = np.array([[i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2))(pi)[idx, idx, idx, idx, idx, idx, :, :, :]]
                          for (i, j) in combinations(range(lenpar), 2)])
    t1 = time.time()
    print("Done d3pdxdy2 in %s sec" % str(t1 - t0))
    allder = (d3pdxdy2,)
    np.save(filename, allder)
    return allder


def get_pder_loop3(pi, dx, filename):
    """ Calculates the derivative aroud the Grid.valueref points. Do this only once.
    gridshape is 2 * order + 1, times the number of free parameters
    pi is of shape gridshape, n multipoles, k length, P columns (zeroth being k's)"""
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    t0 = time.time()
    lenpar = len(Grid.valueref)
    idx = Grid.center
    t1 = time.time()
    d3pdxdydz = np.array([[i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1))(pi)[idx, idx, idx, idx, idx, idx, :, :, :]]
                          for (i, j, k) in combinations(range(lenpar), 3)])
    t0 = time.time()
    print("Done d3pdxdydz in %s sec" % str(t1 - t0))
    allder = (d3pdxdydz, )
    np.save(filename, allder)
    return allder


def load_pder(filename):
    allder = np.load(filename, allow_pickle=True, encoding='bytes')
    return allder


# It works well
def get_PSTaylor(dtheta, derivatives):
    # Shape of dtheta: number of free parameters
    # Shape of derivatives: tuple up to third derivative where each element has shape (num free par, multipoles, lenk, columns)
    t1 = np.einsum('p,pmkb->mkb', dtheta, derivatives[1])
    t2diag = np.einsum('p,pmkb->mkb', dtheta**2, derivatives[2])
    t2nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * d[2] for d in derivatives[3]], axis=0)
    t3diag = np.einsum('p,pmkb->mkb', dtheta**3, derivatives[4])
    t3semidiagx = np.sum([dtheta[d[0]]**2 * dtheta[d[1]] * d[2] for d in derivatives[5]], axis=0)
    t3semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]]**2 * d[2] for d in derivatives[6]], axis=0)
    t3nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[7]], axis=0)
    #t4diag = np.einsum('p,pmkb->mkb', dtheta**4, derivatives[7])
    #t4semidiag1 = np.sum([dtheta[d[0]]**3 * dtheta[d[1]] * d[2] for d in derivatives[8]], axis=0)
    #t4semidiag2 = np.sum([dtheta[d[0]]**2 * dtheta[d[1]]**2 * d[2] for d in derivatives[9]], axis=0)
    #t4semidiag3 = np.sum([dtheta[d[0]]**2 * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[10]], axis=0)
    # t4nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * dtheta[d[3]] * d[4] for d in derivatives[11]], axis=0)
    # t5diag = np.einsum('p,pmkb->mkb', dtheta**5, derivatives[12])
    allPS = (derivatives[0] + t1 + 0.5 * t2diag + t2nondiag # + t3nondiag)
             + t3diag / 6. + t3semidiagx / 2. + t3semidiagy / 2. + t3nondiag)
             # t4diag / 24.  + t4semidiag1 / 6. + t4semidiag2 / 4. + t4semidiag3 / 2. + t4nondiag)  # + t5diag / 120.)
    return allPS


# Nothing much to test, helper function
def get_bias(d_fit):
    vlin = np.array([0, 1, d_fit['b1'], d_fit['b1']**2])
    vloop = np.array([0, 1, d_fit['b1'], d_fit['b2'], d_fit['b3'], d_fit['b4'],
                      d_fit['b1']**2, d_fit['b1'] * d_fit['b2'], d_fit['b1'] * d_fit['b3'], d_fit['b1'] * d_fit['b4'],
                      d_fit['b2']**2, d_fit['b2'] * d_fit['b4'], d_fit['b4']**2,
                      d_fit['b1'] * d_fit['b5'] / knl**2, d_fit['b1'] * d_fit['b6'] / km**2,
                      d_fit['b1'] * d_fit['b7'] / km**2, d_fit['b5'] / knl**2, d_fit['b6'] / km**2, d_fit['b7'] / km**2])
    return vlin, vloop


# Tested
def get_PSbias(plin, ploop, dfit):
    """Given a dictionary of biases, gets the k's and the full PS.
    The shape of the PS is (nmultipoles, len(k))"""
    vlin, vloop = get_bias(dfit)
    kin = plin[0, :, 0]
    PS = np.einsum('c,mkc->mk', vlin, plin) + np.einsum('c,mkc->mk', vloop, ploop)
    PS[0] = PS[0] + dfit['b8'] / nd + dfit['b9'] / nd / km**2 * kin**2
    PS[1] = PS[1] + dfit['b10'] / nd / km**2 * kin**2
    return kin, PS


if __name__ == "__main__":
    print("Let's start!")
    t0 = time.time()
    plingrid, ploopgrid = get_grids(griddir)
    print("Got grids in %s seconds" % str(time.time() - t0))
    dx = Grid.delta
    #run = int(sys.argv[2])
    print("Calculate derivatives of linear PS")
    allderlin = get_pder_lin(plingrid, dx, os.path.join(griddir, "DerPlin_%s.npy" % gridname))
    print("Calculate derivatives of loop PS")
    allderlin = get_pder_lin(ploopgrid, dx, os.path.join(griddir, "DerPloop_%s.npy" % gridname))
    """
    if run == 0:
        print("Calculate derivatives of linear PS")
        allderlin = get_pder_lin(plingrid, dx, os.path.join(griddir, "DerPlin_%s.npy" % gridname))
    elif run == 1:
        print("Calculate derivatives of loop PS")
        derloop1 = get_pder_loop1(ploopgrid, dx, os.path.join(griddir, "DerPloop_%s_1.npy" % gridname))
    elif run == 2:
        print("Calculate derivatives of loop PS")
        derloop2a = get_pder_loop2a(ploopgrid, dx, os.path.join(griddir, "DerPloop_%s_2a.npy" % gridname))
    elif run == 3:
        print("Calculate derivatives of loop PS")
        derloop2b = get_pder_loop2b(ploopgrid, dx, os.path.join(griddir, "DerPloop_%s_2b.npy" % gridname))
    else:
        print("Calculate derivatives of loop PS")
        derloop3 = get_pder_loop3(ploopgrid, dx, os.path.join(griddir, "DerPloop_%s_3.npy" % gridname))
    """
