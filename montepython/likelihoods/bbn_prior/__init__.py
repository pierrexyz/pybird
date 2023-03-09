import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood

class bbn_prior(Likelihood):

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

    def loglkl(self, cosmo, data):
        return -0.5 * ((data.cosmo_arguments['omega_b'] - self.mean) / self.sigma)**2
