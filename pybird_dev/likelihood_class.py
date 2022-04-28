"""
.. module:: likelihood_class
   :synopsis: Definition of the major likelihoods
.. moduleauthor:: Julien Lesgourgues <lesgourg@cern.ch>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Contains the definition of the base likelihood class :class:`Likelihood`, with
basic functions, as well as more specific likelihood classes that may be reused
to implement new ones.

"""
import os
import numpy as np
import math
import warnings
import re
import scipy.constants as const
import scipy.integrate
import scipy.interpolate
import scipy.misc

import io_mp


class Likelihood(object):
    """
    General class that all likelihoods will inherit from.

    """

    def __init__(self, path, data, command_line):
        """
        It copies the content of self.path from the initialization routine of
        the :class:`Data <data.Data>` class, and defines a handful of useful
        methods, that every likelihood might need.

        If the nuisance parameters required to compute this likelihood are not
        defined (either fixed or varying), the code will stop.

        Parameters
        ----------
        data : class
            Initialized instance of :class:`Data <data.Data>`
        command_line : NameSpace
            NameSpace containing the command line arguments

        """

        self.name = self.__class__.__name__
        self.folder = os.path.abspath(os.path.join(
            data.path['MontePython'], 'likelihoods', self.name))
        if not data.log_flag:
            path = os.path.join(command_line.folder, 'log.param')

        # Define some default fields
        self.data_directory = ''

        # Store all the default fields stored, for the method read_file.
        self.default_values = ['data_directory']

        # Recover the values potentially read in the input.param file.
        if hasattr(data, self.name):
            exec("attributes = [e for e in dir(data.%s) if e.find('__') == -1]" % self.name)
            for elem in attributes:
                exec("setattr(self, elem, getattr(data.%s, elem))" % self.name)

        # Read values from the data file
        self.read_from_file(path, data, command_line)

        # Default state
        self.need_update = True

        # Check if the nuisance parameters are defined
        error_flag = False
        try:
            for nuisance in self.use_nuisance:
                if nuisance not in data.get_mcmc_parameters(['nuisance']):
                    error_flag = True
                    warnings.warn(
                        nuisance + " must be defined, either fixed or " +
                        "varying, for %s likelihood" % self.name)
            self.nuisance = self.use_nuisance
        except AttributeError:
            self.use_nuisance = []
            self.nuisance = []

        # If at least one is missing, raise an exception.
        if error_flag:
            raise io_mp.LikelihoodError(
                "Check your nuisance parameter list for your set of experiments")

        # Append to the log.param the value used (WARNING: so far no comparison
        # is done to ensure that the experiments share the same parameters)
        if data.log_flag:
            io_mp.log_likelihood_parameters(self, command_line)

    def loglkl(self, cosmo, data):
        """
        Placeholder to remind that this function needs to be defined for a
        new likelihood.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            'Must implement method loglkl() in your likelihood')

    def read_from_file(self, path, data, command_line):
        """
        Extract the information from the log.param concerning this likelihood.

        If the log.param is used, check that at least one item for each
        likelihood is recovered. Otherwise, it means the log.param does not
        contain information on the likelihood. This happens when the first run
        fails early, before calling the likelihoods, and the program did not
        log the information. This check might not be completely secure, but it
        is better than nothing.

        .. warning::

            This checks relies on the fact that a likelihood should always have
            at least **one** line of code written in the likelihood.data file.
            This should be always true, but in case a run fails with the error
            message described below, think about it.

        .. warning::

            As of version 2.0.2, you can specify likelihood options in the
            parameter file. They have complete priority over the ones specified
            in the `likelihood.data` file, and it will be reflected in the
            `log.param` file.

        """

        # Counting how many lines are read.
        counter = 0

        self.path = path
        self.dictionary = {}
        if os.path.isfile(path):
            data_file = open(path, 'r')
            for line in data_file:
                if line.find('#') == -1:
                    if line.find(self.name+'.') != -1:
                        # Recover the name and value from the .data file
                        regexp = re.match(
                            "%s.(.*)\s*=\s*(.*)" % self.name, line)
                        name, value = (
                            elem.strip() for elem in regexp.groups())
                        # If this name was already defined in the parameter
                        # file, be sure to take this value instead. Beware,
                        # there are a few parameters which are always
                        # predefined, such as data_directory, which should be
                        # ignored in this check.
                        is_ignored = False
                        if name not in self.default_values:
                            try:
                                value = getattr(self, name)
                                is_ignored = True
                            except AttributeError:
                                pass
                        if not is_ignored:
                            exec('self.'+name+' = '+value)
                        value = getattr(self, name)
                        counter += 1
                        self.dictionary[name] = value
            data_file.seek(0)
            data_file.close()

        # Checking that at least one line was read, exiting otherwise
        if counter == 0:
            raise io_mp.ConfigurationError(
                "No information on %s likelihood " % self.name +
                "was found in the %s file.\n" % path +
                "This can result from a failed initialization of a previous " +
                "run. To solve this, you can do a \n " +
                "]$ rm -rf %s \n " % command_line.folder +
                "Be sure there is noting in it before doing this !")

    def get_cl(self, cosmo, l_max=-1):
        """
        Return the :math:`C_{\ell}` from the cosmological code in
        :math:`\mu {\\rm K}^2`

        """
        # get C_l^XX from the cosmological code
        cl = cosmo.lensed_cl(int(l_max))

        # convert dimensionless C_l's to C_l in muK**2
        T = cosmo.T_cmb()
        for key in cl.iterkeys():
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            # phi cross-terms should only be multiplied with this factor once
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T*1.e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T*1.e6)

        return cl

    def get_unlensed_cl(self, cosmo, l_max=-1):
        """
        Return the :math:`C_{\ell}` from the cosmological code in
        :math:`\mu {\\rm K}^2`

        """
        # get C_l^XX from the cosmological code
        cl = cosmo.raw_cl(l_max)

        # convert dimensionless C_l's to C_l in muK**2
        T = cosmo.T_cmb()
        for key in cl.iterkeys():
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            # phi cross-terms should only be multiplied with this factor once
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T*1.e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T*1.e6)

        return cl

    def need_cosmo_arguments(self, data, dictionary):
        """
        Ensure that the arguments of dictionary are defined to the correct
        value in the cosmological code

        .. warning::

            So far there is no way to enforce a parameter where `smaller is
            better`. A bigger value will always overried any smaller one
            (`cl_max`, etc...)

        Parameters
        ----------
        data : dict
            Initialized instance of :class:`data`
        dictionary : dict
            Desired precision for some cosmological parameters

        """
        array_flag = False
        for key, value in dictionary.iteritems():
            try:
                data.cosmo_arguments[key]
                try:
                    float(data.cosmo_arguments[key])
                    num_flag = True
                except ValueError:
                    num_flag = False
                except TypeError:
                    num_flag = True
                    array_flag = True

            except KeyError:
                try:
                    float(value)
                    num_flag = True
                    data.cosmo_arguments[key] = 0
                except ValueError:
                    num_flag = False
                    data.cosmo_arguments[key] = ''
                except TypeError:
                    num_flag = True
                    array_flag = True
            if num_flag is False:
                if data.cosmo_arguments[key].find(value) == -1:
                    data.cosmo_arguments[key] += ' '+value+' '
            else:
                if array_flag is False:
                    if float(data.cosmo_arguments[key]) < value:
                        data.cosmo_arguments[key] = value
                else:
                    data.cosmo_arguments[key] = '%.2g' % value[0]
                    for i in range(1, len(value)):
                        data.cosmo_arguments[key] += ',%.2g' % (value[i])

    def read_contamination_spectra(self, data):

        for nuisance in self.use_nuisance:
            # read spectrum contamination (so far, assumes only temperature
            # contamination; will be trivial to generalize to polarization when
            # such templates will become relevant)
            setattr(self, "%s_contamination" % nuisance,
                    np.zeros(self.l_max+1, 'float64'))
            try:
                File = open(os.path.join(
                    self.data_directory, getattr(self, "%s_file" % nuisance)),
                    'r')
                for line in File:
                    l = int(float(line.split()[0]))
                    if ((l >= 2) and (l <= self.l_max)):
                        exec "self.%s_contamination[l]=float(line.split()[1])/(l*(l+1.)/2./math.pi)" % nuisance
            except:
                print 'Warning: you did not pass a file name containing '
                print 'a contamination spectrum regulated by the nuisance '
                print 'parameter '+nuisance

            # read renormalization factor
            # if it is not there, assume it is one, i.e. do not renormalize
            try:
                # do the following operation:
                # self.nuisance_contamination *= float(self.nuisance_scale)
                setattr(self, "%s_contamination" % nuisance,
                        getattr(self, "%s_contamination" % nuisance) *
                        float(getattr(self, "%s_scale" % nuisance)))
            except AttributeError:
                pass

            # read central value of nuisance parameter
            # if it is not there, assume one by default
            try:
                getattr(self, "%s_prior_center" % nuisance)
            except AttributeError:
                setattr(self, "%s_prior_center" % nuisance, 1.)

            # read variance of nuisance parameter
            # if it is not there, assume flat prior (encoded through
            # variance=0)
            try:
                getattr(self, "%s_prior_variance" % nuisance)
            except:
                setattr(self, "%s_prior_variance" % nuisance, 0.)

    def add_contamination_spectra(self, cl, data):

        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add contamination spectra multiplied by nuisance parameters
            for l in range(2, self.l_max):
                exec "cl['tt'][l] += nuisance_value*self.%s_contamination[l]" % nuisance

        return cl

    def add_nuisance_prior(self, lkl, data):

        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add prior on nuisance parameters
            if getattr(self, "%s_prior_variance" % nuisance) > 0:
                # convenience variables
                prior_center = getattr(self, "%s_prior_center" % nuisance)
                prior_variance = getattr(self, "%s_prior_variance" % nuisance)
                lkl += -0.5*((nuisance_value-prior_center)/prior_variance)**2

        return lkl

    def computeLikelihood(self, ctx):
        """
        Interface with CosmoHammer

        Parameters
        ----------
        ctx : Context
                Contains several dictionaries storing data and cosmological
                information

        """
        # Recover both instances from the context
        cosmo = ctx.get("cosmo")
        data = ctx.get("data")

        loglkl = self.loglkl(cosmo, data)

        return loglkl


###################################
#
# END OF GENERIC LIKELIHOOD CLASS
#
###################################



###################################
# PRIOR TYPE LIKELIHOOD
# --> H0,...
###################################
class Likelihood_prior(Likelihood):

    def loglkl(self):
        raise NotImplementedError('Must implement method loglkl() in your likelihood')


###################################
# NEWDAT TYPE LIKELIHOOD
# --> spt,boomerang,etc.
###################################
class Likelihood_newdat(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        # open .newdat file
        newdatfile = open(
            os.path.join(self.data_directory, self.file), 'r')

        # find beginning of window functions file names
        window_name = newdatfile.readline().strip('\n').replace(' ', '')

        # initialize list of fist and last band for each type
        band_num = np.zeros(6, 'int')
        band_min = np.zeros(6, 'int')
        band_max = np.zeros(6, 'int')

        # read number of bands for each of the six types TT, EE, BB, EB, TE, TB
        line = newdatfile.readline()
        for i in range(6):
            band_num[i] = int(line.split()[i])

        # read string equal to 'BAND_SELECTION' or not
        line = str(newdatfile.readline()).strip('\n').replace(' ', '')

        # if yes, read 6 lines containing 'min, max'
        if (line == 'BAND_SELECTION'):
            for i in range(6):
                line = newdatfile.readline()
                band_min[i] = int(line.split()[0])
                band_max[i] = int(line.split()[1])

        # if no, set min to 1 and max to band_num (=use all bands)
        else:
            band_min = [1 for i in range(6)]
            band_max = band_num

        # read line defining calibration uncertainty
        # contains: flag (=0 or 1), calib, calib_uncertainty
        line = newdatfile.readline()
        calib = float(line.split()[1])
        if (int(line.split()[0]) == 0):
            self.calib_uncertainty = 0
        else:
            self.calib_uncertainty = float(line.split()[2])

        # read line defining beam uncertainty
        # contains: flag (=0, 1 or 2), beam_width, beam_sigma
        line = newdatfile.readline()
        beam_type = int(line.split()[0])
        if (beam_type > 0):
            self.has_beam_uncertainty = True
        else:
            self.has_beam_uncertainty = False
        beam_width = float(line.split()[1])
        beam_sigma = float(line.split()[2])

        # read flag (= 0, 1 or 2) for lognormal distributions and xfactors
        line = newdatfile.readline()
        likelihood_type = int(line.split()[0])
        if (likelihood_type > 0):
            self.has_xfactors = True
        else:
            self.has_xfactors = False

        # declare array of quantitites describing each point of measurement
        # size yet unknown, it will be found later and stored as
        # self.num_points
        self.obs = np.array([], 'float64')
        self.var = np.array([], 'float64')
        self.beam_error = np.array([], 'float64')
        self.has_xfactor = np.array([], 'bool')
        self.xfactor = np.array([], 'float64')

        # temporary array to know which bands are actually used
        used_index = np.array([], 'int')

        index = -1

        # scan the lines describing each point of measurement
        for cltype in range(6):
            if (int(band_num[cltype]) != 0):
                # read name (but do not use it)
                newdatfile.readline()
                for band in range(int(band_num[cltype])):
                    # read one line corresponding to one measurement
                    line = newdatfile.readline()
                    index += 1

                    # if we wish to actually use this measurement
                    if ((band >= band_min[cltype]-1) and
                            (band <= band_max[cltype]-1)):

                        used_index = np.append(used_index, index)

                        self.obs = np.append(
                            self.obs, float(line.split()[1])*calib**2)

                        self.var = np.append(
                            self.var,
                            (0.5*(float(line.split()[2]) +
                                  float(line.split()[3]))*calib**2)**2)

                        self.xfactor = np.append(
                            self.xfactor, float(line.split()[4])*calib**2)

                        if ((likelihood_type == 0) or
                                ((likelihood_type == 2) and
                                (int(line.split()[7]) == 0))):
                            self.has_xfactor = np.append(
                                self.has_xfactor, [False])
                        if ((likelihood_type == 1) or
                                ((likelihood_type == 2) and
                                (int(line.split()[7]) == 1))):
                            self.has_xfactor = np.append(
                                self.has_xfactor, [True])

                        if (beam_type == 0):
                            self.beam_error = np.append(self.beam_error, 0.)
                        if (beam_type == 1):
                            l_mid = float(line.split()[5]) +\
                                0.5*(float(line.split()[5]) +
                                     float(line.split()[6]))
                            self.beam_error = np.append(
                                self.beam_error,
                                abs(math.exp(
                                    -l_mid*(l_mid+1)*1.526e-8*2.*beam_sigma *
                                    beam_width)-1.))
                        if (beam_type == 2):
                            if (likelihood_type == 2):
                                self.beam_error = np.append(
                                    self.beam_error, float(line.split()[8]))
                            else:
                                self.beam_error = np.append(
                                    self.beam_error, float(line.split()[7]))

                # now, skip and unused part of the file (with sub-correlation
                # matrices)
                for band in range(int(band_num[cltype])):
                    newdatfile.readline()

        # number of points that we will actually use
        self.num_points = np.shape(self.obs)[0]

        # total number of points, including unused ones
        full_num_points = index+1

        # read full correlation matrix
        full_covmat = np.zeros((full_num_points, full_num_points), 'float64')
        for point in range(full_num_points):
            full_covmat[point] = newdatfile.readline().split()

        # extract smaller correlation matrix for points actually used
        covmat = np.zeros((self.num_points, self.num_points), 'float64')
        for point in range(self.num_points):
            covmat[point] = full_covmat[used_index[point], used_index]

        # recalibrate this correlation matrix
        covmat *= calib**4

        # redefine the correlation matrix, the observed points and their
        # variance in case of lognormal likelihood
        if (self.has_xfactors):

            for i in range(self.num_points):

                for j in range(self.num_points):
                    if (self.has_xfactor[i]):
                        covmat[i, j] /= (self.obs[i]+self.xfactor[i])
                    if (self.has_xfactor[j]):
                        covmat[i, j] /= (self.obs[j]+self.xfactor[j])

            for i in range(self.num_points):
                if (self.has_xfactor[i]):
                    self.var[i] /= (self.obs[i]+self.xfactor[i])**2
                    self.obs[i] = math.log(self.obs[i]+self.xfactor[i])

        # invert correlation matrix
        self.inv_covmat = np.linalg.inv(covmat)

        # read window function files a first time, only for finding the
        # smallest and largest l's for each point
        self.win_min = np.zeros(self.num_points, 'int')
        self.win_max = np.zeros(self.num_points, 'int')
        for point in range(self.num_points):
            for line in open(os.path.join(
                    self.data_directory, 'windows', window_name) +
                    str(used_index[point]+1), 'r'):
                if any([float(line.split()[i]) != 0.
                        for i in range(1, len(line.split()))]):
                    if (self.win_min[point] == 0):
                        self.win_min[point] = int(line.split()[0])
                    self.win_max[point] = int(line.split()[0])

        # infer from format of window function files whether we will use
        # polarisation spectra or not
        num_col = len(line.split())
        if (num_col == 2):
            self.has_pol = False
        else:
            if (num_col == 5):
                self.has_pol = True
            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "Window function files are understood if they contain " +
                    "2 columns (l TT), or 5 columns (l TT TE EE BB)." +
                    "In this case the number of columns is %d" % num_col)

        # define array of window functions
        self.window = np.zeros(
            (self.num_points, max(self.win_max)+1, num_col-1), 'float64')

        # go again through window function file, this time reading window
        # functions; that are distributed as: l TT (TE EE BB) where the last
        # columns contaim W_l/l, not W_l we mutiply by l in order to store the
        # actual W_l
        for point in range(self.num_points):
            for line in open(os.path.join(
                    self.data_directory, 'windows', window_name) +
                    str(used_index[point]+1), 'r'):
                l = int(line.split()[0])
                if (((self.has_pol is False) and (len(line.split()) != 2))
                        or ((self.has_pol is True) and
                            (len(line.split()) != 5))):
                    raise io_mp.LikelihoodError(
                        "In likelihood %s. " % self.name +
                        "for a given experiment, all window functions should" +
                        " have the same number of columns, 2 or 5. " +
                        "This is not the case here.")
                if ((l >= self.win_min[point]) and (l <= self.win_max[point])):
                    self.window[point, l, :] = [
                        float(line.split()[i])
                        for i in range(1, len(line.split()))]
                    self.window[point, l, :] *= l

        # eventually, initialise quantitites used in the marginalization over
        # nuisance parameters
        if ((self.has_xfactors) and
                ((self.calib_uncertainty > 1.e-4) or
                 (self.has_beam_uncertainty))):
            self.halfsteps = 5
            self.margeweights = np.zeros(2*self.halfsteps+1, 'float64')
            for i in range(-self.halfsteps, self.halfsteps+1):
                self.margeweights[i+self.halfsteps] = np.exp(
                    -(float(i)*3./float(self.halfsteps))**2/2)
            self.margenorm = sum(self.margeweights)

        # store maximum value of l needed by window functions
        self.l_max = max(self.win_max)

        # impose that the cosmological code computes Cl's up to maximum l
        # needed by the window function
        self.need_cosmo_arguments(data, {'l_max_scalars': self.l_max})

        # deal with nuisance parameters
        try:
            self.use_nuisance
            self.nuisance = self.use_nuisance
        except:
            self.use_nuisance = []
            self.nuisance = []
        self.read_contamination_spectra(data)

        # end of initialisation

    def loglkl(self, cosmo, data):
        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # add contamination spectra multiplied by nuisance parameters
        cl = self.add_contamination_spectra(cl, data)

        # get likelihood
        lkl = self.compute_lkl(cl, cosmo, data)

        # add prior on nuisance parameters
        lkl = self.add_nuisance_prior(lkl, data)

        return lkl

    def compute_lkl(self, cl, cosmo, data):
        # checks that Cl's have been computed up to high enough l given window
        # function range. Normally this has been imposed before, so this test
        # could even be supressed.
        if (np.shape(cl['tt'])[0]-1 < self.l_max):
            raise io_mp.LikelihoodError(
                "%s computed Cls till l=" % data.cosmological_module_name +
                "%d " % (np.shape(cl['tt'])[0]-1) +
                "while window functions need %d." % self.l_max)

        # compute theoretical bandpowers, store them in theo[points]
        theo = np.zeros(self.num_points, 'float64')

        for point in range(self.num_points):

            # find bandpowers B_l by convolving C_l's with [(l+1/2)/2pi W_l]
            for l in range(self.win_min[point], self.win_max[point]):

                theo[point] += cl['tt'][l]*self.window[point, l, 0] *\
                    (l+0.5)/2./math.pi

                if (self.has_pol):
                    theo[point] += (
                        cl['te'][l]*self.window[point, l, 1] +
                        cl['ee'][l]*self.window[point, l, 2] +
                        cl['bb'][l]*self.window[point, l, 3]) *\
                        (l+0.5)/2./math.pi

        # allocate array for differencve between observed and theoretical
        # bandpowers
        difference = np.zeros(self.num_points, 'float64')

        # depending on the presence of lognormal likelihood, calibration
        # uncertainty and beam uncertainity, use several methods for
        # marginalising over nuisance parameters:

        # first method: numerical integration over calibration uncertainty:
        if (self.has_xfactors and
                ((self.calib_uncertainty > 1.e-4) or
                 self.has_beam_uncertainty)):

            chisq_tmp = np.zeros(2*self.halfsteps+1, 'float64')
            chisqcalib = np.zeros(2*self.halfsteps+1, 'float64')
            beam_error = np.zeros(self.num_points, 'float64')

            # loop over various beam errors
            for ibeam in range(2*self.halfsteps+1):

                # beam error
                for point in range(self.num_points):
                    if (self.has_beam_uncertainty):
                        beam_error[point] = 1.+self.beam_error[point] *\
                            (ibeam-self.halfsteps)*3/float(self.halfsteps)
                    else:
                        beam_error[point] = 1.

                # loop over various calibraion errors
                for icalib in range(2*self.halfsteps+1):

                    # calibration error
                    calib_error = 1+self.calib_uncertainty*(
                        icalib-self.halfsteps)*3/float(self.halfsteps)

                    # compute difference between observed and theoretical
                    # points, after correcting the later for errors
                    for point in range(self.num_points):

                        # for lognormal likelihood, use log(B_l+X_l)
                        if (self.has_xfactor[point]):
                            difference[point] = self.obs[point] -\
                                math.log(
                                    theo[point]*beam_error[point] *
                                    calib_error+self.xfactor[point])
                        # otherwise use B_l
                        else:
                            difference[point] = self.obs[point] -\
                                theo[point]*beam_error[point]*calib_error

                    # find chisq with those corrections
                    # chisq_tmp[icalib] = np.dot(np.transpose(difference),
                    # np.dot(self.inv_covmat, difference))
                    chisq_tmp[icalib] = np.dot(
                        difference, np.dot(self.inv_covmat, difference))

                minchisq = min(chisq_tmp)

            # find chisq marginalized over calibration uncertainty (if any)
                tot = 0
                for icalib in range(2*self.halfsteps+1):
                    tot += self.margeweights[icalib]*math.exp(
                        max(-30., -(chisq_tmp[icalib]-minchisq)/2.))

                chisqcalib[ibeam] = -2*math.log(tot/self.margenorm)+minchisq

            # find chisq marginalized over beam uncertainty (if any)
            if (self.has_beam_uncertainty):

                minchisq = min(chisqcalib)

                tot = 0
                for ibeam in range(2*self.halfsteps+1):
                    tot += self.margeweights[ibeam]*math.exp(
                        max(-30., -(chisqcalib[ibeam]-minchisq)/2.))

                chisq = -2*math.log(tot/self.margenorm)+minchisq

            else:
                chisq = chisqcalib[0]

        # second method: marginalize over nuisance parameters (if any)
        # analytically
        else:

            # for lognormal likelihood, theo[point] should contain log(B_l+X_l)
            if (self.has_xfactors):
                for point in range(self.num_points):
                    if (self.has_xfactor[point]):
                        theo[point] = math.log(theo[point]+self.xfactor[point])

            # find vector of difference between observed and theoretical
            # bandpowers
            difference = self.obs-theo

            # find chisq
            chisq = np.dot(
                np.transpose(difference), np.dot(self.inv_covmat, difference))

            # correct eventually for effect of analytic marginalization over
            # nuisance parameters
            if ((self.calib_uncertainty > 1.e-4) or self.has_beam_uncertainty):

                denom = 1.
                tmpi = np.dot(self.inv_covmat, theo)
                chi2op = np.dot(np.transpose(difference), tmp)
                chi2pp = np.dot(np.transpose(theo), tmp)

                # TODO beam is not defined here !
                if (self.has_beam_uncertainty):
                    for points in range(self.num_points):
                        beam[point] = self.beam_error[point]*theo[point]
                    tmp = np.dot(self.inv_covmat, beam)
                    chi2dd = np.dot(np.transpose(beam), tmp)
                    chi2pd = np.dot(np.transpose(theo), tmp)
                    chi2od = np.dot(np.transpose(difference), tmp)

                if (self.calib_uncertainty > 1.e-4):
                    wpp = 1/(chi2pp+1/self.calib_uncertainty**2)
                    chisq = chisq-wpp*chi2op**2
                    denom = denom/wpp*self.calib_uncertainty**2
                else:
                    wpp = 0

                if (self.has_beam_uncertainty):
                    wdd = 1/(chi2dd-wpp*chi2pd**2+1)
                    chisq = chisq-wdd*(chi2od-wpp*chi2op*chi2pd)**2
                    denom = denom/wdd

                chisq += math.log(denom)

        # finally, return ln(L)=-chi2/2

        self.lkl = -0.5 * chisq
        return self.lkl


###################################
# CLIK TYPE LIKELIHOOD
# --> clik_fake_planck,clik_wmap,etc.
###################################
class Likelihood_clik(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        try:
            import clik
        except ImportError:
            raise io_mp.MissingLibraryError(
                "You must first activate the binaries from the Clik " +
                "distribution. Please run : \n " +
                "]$ source /path/to/clik/bin/clik_profile.sh \n " +
                "and try again.")
        # for lensing, some routines change. Intializing a flag for easier
        # testing of this condition
        #if self.name == 'Planck_lensing':
        if 'lensing' in self.name and 'Planck' in self.name:
            self.lensing = True
        else:
            self.lensing = False

        try:
            if self.lensing:
                self.clik = clik.clik_lensing(self.path_clik)
                try:
                    self.l_max = max(self.clik.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    self.l_max = self.clik.get_lmax()
            else:
                self.clik = clik.clik(self.path_clik)
                self.l_max = max(self.clik.get_lmax())
        except clik.lkl.CError:
            raise io_mp.LikelihoodError(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated:\n%s\n"
                % (self.name,self.path_clik) +
                " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place.")
        except KeyError:
            raise io_mp.LikelihoodError(
                "In the %s.data file, the field 'clik' of the " % self.name +
                "path dictionary is expected to be defined. Please make sure"
                " it is the case in you configuration file")

        self.need_cosmo_arguments(
            data, {'l_max_scalars': self.l_max})

        self.nuisance = list(self.clik.extra_parameter_names)

        # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
        if (self.name == 'Planck_highl_lite') or (self.name == 'Planck_highl_TTTEEE_lite'):
            for i in range(len(self.nuisance)):
                if (self.nuisance[i] == 'A_Planck'):
                    self.nuisance[i] = 'A_planck'
            print "In %s, MontePython corrected nuisance parameter name A_Planck to A_planck" % self.name

        # testing if the nuisance parameters are defined. If there is at least
        # one non defined, raise an exception.
        exit_flag = False
        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        for nuisance in self.nuisance:
            if nuisance not in nuisance_parameter_names:
                exit_flag = True
                print '%20s\tmust be a fixed or varying nuisance parameter' % nuisance

        if exit_flag:
            raise io_mp.LikelihoodError(
                "The likelihood %s " % self.name +
                "expected some nuisance parameters that were not provided")

        # deal with nuisance parameters
        try:
            self.use_nuisance
        except:
            self.use_nuisance = []

        # Add in use_nuisance all the parameters that have non-flat prior
        for nuisance in self.nuisance:
            if hasattr(self, '%s_prior_center' % nuisance):
                self.use_nuisance.append(nuisance)

    def loglkl(self, cosmo, data):

        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])

        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # testing for lensing
        if self.lensing:
            try:
                length = len(self.clik.get_lmax())
                tot = np.zeros(
                    np.sum(self.clik.get_lmax()) + length +
                    len(self.clik.get_extra_parameter_names()))
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(2*self.l_max+length + len(self.clik.get_extra_parameter_names()))
        else:
            length = len(self.clik.get_has_cl())
            tot = np.zeros(
                np.sum(self.clik.get_lmax()) + length +
                len(self.clik.get_extra_parameter_names()))

        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if (self.clik.get_lmax()[i] > -1):
                    for j in range(self.clik.get_lmax()[i]+1):
                        if (i == 0):
                            tot[index+j] = cl['tt'][j]
                        if (i == 1):
                            tot[index+j] = cl['ee'][j]
                        if (i == 2):
                            tot[index+j] = cl['bb'][j]
                        if (i == 3):
                            tot[index+j] = cl['te'][j]
                        if (i == 4):
                            tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                        if (i == 5):
                            tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                    index += self.clik.get_lmax()[i]+1

        else:
            try:
                for i in range(length):
                    if (self.clik.get_lmax()[i] > -1):
                        for j in range(self.clik.get_lmax()[i]+1):
                            if (i == 0):
                                tot[index+j] = cl['pp'][j]
                            if (i == 1):
                                tot[index+j] = cl['tt'][j]
                            if (i == 2):
                                tot[index+j] = cl['ee'][j]
                            if (i == 3):
                                tot[index+j] = cl['bb'][j]
                            if (i == 4):
                                tot[index+j] = cl['te'][j]
                            if (i == 5):
                                tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                            if (i == 6):
                                tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                        index += self.clik.get_lmax()[i]+1

            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(self.l_max):
                        if (i == 0):
                            tot[index+j] = cl['pp'][j]
                        if (i == 1):
                            tot[index+j] = cl['tt'][j]
                    index += self.l_max+1

        # fill with nuisance parameters
        for nuisance in self.clik.get_extra_parameter_names():

            # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
            if (self.name == 'Planck_highl_lite') or (self.name == 'Planck_highl_TTTEEE_lite'):
                if nuisance == 'A_Planck':
                    nuisance = 'A_planck'

            if nuisance in nuisance_parameter_names:
                nuisance_value = data.mcmc_parameters[nuisance]['current'] *\
                    data.mcmc_parameters[nuisance]['scale']
            else:
                raise io_mp.LikelihoodError(
                    "the likelihood needs a parameter %s. " % nuisance +
                    "You must pass it through the input file " +
                    "(as a free nuisance parameter or a fixed parameter)")
            #print "found one nuisance with name",nuisance
            tot[index] = nuisance_value
            index += 1

        # compute likelihood
        #print "lkl:",self.clik(tot)
        lkl = self.clik(tot)[0]

        # add prior on nuisance parameters
        lkl = self.add_nuisance_prior(lkl, data)

        return lkl


###################################
# MOCK CMB TYPE LIKELIHOOD
# --> mock planck, cmbpol, etc.
###################################
class Likelihood_mock_cmb(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        ################
        # Noise spectrum
        ################

        try:
            self.noise_from_file
        except:
            self.noise_from_file = False

        if self.noise_from_file:

            try:
                self.noise_file
            except:
                raise io_mp.LikelihoodError("For reading noise from file, you must provide noise_file")

            self.noise_T = np.zeros(self.l_max+1, 'float64')
            self.noise_P = np.zeros(self.l_max+1, 'float64')
            if self.LensingExtraction:
                self.Nldd = np.zeros(self.l_max+1, 'float64')

            if os.path.exists(os.path.join(self.data_directory, self.noise_file)):
                noise = open(os.path.join(
                    self.data_directory, self.noise_file), 'r')
                line = noise.readline()
                while line.find('#') != -1:
                    line = noise.readline()

                for l in range(self.l_min, self.l_max+1):
                    ll = int(float(line.split()[0]))
                    if l != ll:
                        # if l_min is larger than the first l in the noise file we can skip lines
                        # until we are at the correct l. Otherwise raise error
                        while l > ll:
                            try:
                                line = noise.readline()
                                ll = int(float(line.split()[0]))
                            except:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the noise file")
                        if l < ll:
                            raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the noise file")
                    # read noise for C_l in muK**2
                    self.noise_T[l] = float(line.split()[1])
                    self.noise_P[l] = float(line.split()[2])
                    if self.LensingExtraction:
                        try:
                            # read noise for C_l^dd = l(l+1) C_l^pp
                            self.Nldd[l] = float(line.split()[3])/(l*(l+1)/2./math.pi)
                        except:
                            raise io_mp.LikelihoodError("For reading lensing noise from file, you must provide one more column")
                    line = noise.readline()
            else:
                raise io_mp.LikelihoodError("Could not find file ",self.noise_file)


        else:
            # convert arcmin to radians
            self.theta_fwhm *= np.array([math.pi/60/180])
            self.sigma_T *= np.array([math.pi/60/180])
            self.sigma_P *= np.array([math.pi/60/180])

            # compute noise in muK**2
            self.noise_T = np.zeros(self.l_max+1, 'float64')
            self.noise_P = np.zeros(self.l_max+1, 'float64')

            for l in range(self.l_min, self.l_max+1):
                self.noise_T[l] = 0
                self.noise_P[l] = 0
                for channel in range(self.num_channels):
                    self.noise_T[l] += self.sigma_T[channel]**-2 *\
                                       math.exp(
                                           -l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
                    self.noise_P[l] += self.sigma_P[channel]**-2 *\
                                       math.exp(
                                           -l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
                self.noise_T[l] = 1/self.noise_T[l]
                self.noise_P[l] = 1/self.noise_P[l]


        # trick to remove any information from polarisation for l<30
        try:
            self.no_small_l_pol
        except:
            self.no_small_l_pol = False

        if self.no_small_l_pol:
            for l in range(self.l_min,30):
                # plug a noise level of 100 muK**2, equivalent to no detection at all of polarisation
                self.noise_P[l] = 100.

        # trick to remove any information from temperature above l_max_TT
        try:
            self.l_max_TT
        except:
            self.l_max_TT = False

        if self.l_max_TT:
            for l in range(self.l_max_TT+1,l_max+1):
                # plug a noise level of 100 muK**2, equivalent to no detection at all of temperature
                self.noise_T[l] = 100.

        # impose that the cosmological code computes Cl's up to maximum l
        # needed by the window function
        self.need_cosmo_arguments(data, {'l_max_scalars': self.l_max})

        # if you want to print the noise spectra:
        #test = open('noise_T_P','w')
        #for l in range(self.l_min, self.l_max+1):
        #    test.write('%d  %e  %e\n'%(l,self.noise_T[l],self.noise_P[l]))

        ###########################################################################
        # implementation of default settings for flags describing the likelihood: #
        ###########################################################################

        # - ignore B modes by default:
        try:
            self.Bmodes
        except:
            self.Bmodes = False
        # - do not use delensing by default:
        try:
            self.delensing
        except:
            self.delensing = False
        # - do not include lensing extraction by default:
        try:
            self.LensingExtraction
        except:
            self.LensingExtraction = False
        # - neglect TD correlation by default:
        try:
            self.neglect_TD
        except:
            self.neglect_TD = True
        # - use lthe lensed TT, TE, EE by default:
        try:
            self.unlensed_clTTTEEE
        except:
            self.unlensed_clTTTEEE = False
        # - do not exclude TTEE by default:
        try:
            self.ExcludeTTTEEE
            if self.ExcludeTTTEEE and not self.LensingExtraction:
                raise io_mp.LikelihoodError("Mock CMB likelihoods where TTTEEE is not used have only been "
                                            "implemented for the deflection spectrum (i.e. not for B-modes), "
                                            "but you do not seem to have lensing extraction enabled")
        except:
            self.ExcludeTTTEEE = False

	#added by Siavash Yasini
        try:
            self.OnlyTT
            if self.OnlyTT and self.ExcludeTTTEEE: 
                raise io_mp.LikelihoodError("OnlyTT and ExcludeTTTEEE cannot be used simultaneously.")
        except:
            self.OnlyTT = False

        ##############################################
        # Delensing noise: implemented by  S. Clesse #
        ##############################################

        if self.delensing:

            try:
                self.delensing_file
            except:
                raise io_mp.LikelihoodError("For delensing, you must provide delensing_file")

            self.noise_delensing = np.zeros(self.l_max+1)
            if os.path.exists(os.path.join(self.data_directory, self.delensing_file)):
                delensing_file = open(os.path.join(
                    self.data_directory, self.delensing_file), 'r')
                line = delensing_file.readline()
                while line.find('#') != -1:
                    line = delensing_file.readline()

                for l in range(self.l_min, self.l_max+1):
                    ll = int(float(line.split()[0]))
                    if l != ll:
                        # if l_min is larger than the first l in the delensing file we can skip lines
                        # until we are at the correct l. Otherwise raise error
                        while l > ll:
                            try:
                                line = delensing_file.readline()
                                ll = int(float(line.split()[0]))
                            except:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                        if l < ll:
                            raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                    self.noise_delensing[ll] = float(line.split()[2])/(ll*(ll+1)/2./math.pi)
                    # change 3 to 4 in the above line for CMBxCIB delensing
                    line = delensing_file.readline()

            else:
                raise io_mp.LikelihoodError("Could not find file ",self.delensing_file)

        ###############################################################
        # Read data for TT, EE, TE, [eventually BB or phi-phi, phi-T] #
        ###############################################################

        # default:
        if not self.ExcludeTTTEEE:
            numCls = 3
        # default 0 if excluding TT EE
        else:
            numCls = 0

        # deal with BB:
        if self.Bmodes:
            self.index_B = numCls
            numCls += 1

        # deal with pp, pT (p = CMB lensing potential):
        if self.LensingExtraction:
            self.index_pp = numCls
            numCls += 1
            if not self.ExcludeTTTEEE:
                self.index_tp = numCls
                numCls += 1

            if not self.noise_from_file:
                # provide a file containing NlDD (noise for the extracted
                # deflection field spectrum) This option is temporary
                # because at some point this module will compute NlDD
                # itself, when logging the fiducial model spectrum.
                try:
                    self.temporary_Nldd_file
                except:
                    raise io_mp.LikelihoodError("For lensing extraction, you must provide a temporary_Nldd_file")

                # read the NlDD file
                self.Nldd = np.zeros(self.l_max+1, 'float64')

                if os.path.exists(os.path.join(self.data_directory, self.temporary_Nldd_file)):
                    fid_file = open(os.path.join(self.data_directory, self.temporary_Nldd_file), 'r')
                    line = fid_file.readline()
                    while line.find('#') != -1:
                        line = fid_file.readline()
                    while (line.find('\n') != -1 and len(line) == 1):
                        line = fid_file.readline()
                    for l in range(self.l_min, self.l_max+1):
                        ll = int(float(line.split()[0]))
                        if l != ll:
                            # if l_min is larger than the first l in the delensing file we can skip lines
                            # until we are at the correct l. Otherwise raise error
                            while l > ll:
                                try:
                                    line = fid_file.readline()
                                    ll = int(float(line.split()[0]))
                                except:
                                    raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                            if l < ll:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                        # this lines assumes that Nldd is stored in the
                        # 4th column (can be customised)
                        self.Nldd[ll] = float(line.split()[3])/(l*(l+1.)/2./math.pi)
                        line = fid_file.readline()
                else:
                    raise io_mp.LikelihoodError("Could not find file ",self.temporary_Nldd_file)

        # deal with fiducial model:
        # If the file exists, initialize the fiducial values
        self.Cl_fid = np.zeros((numCls, self.l_max+1), 'float64')
        self.fid_values_exist = False
        if os.path.exists(os.path.join(
                self.data_directory, self.fiducial_file)):
            self.fid_values_exist = True
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'r')
            line = fid_file.readline()
            while line.find('#') != -1:
                line = fid_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = fid_file.readline()
            for l in range(self.l_min, self.l_max+1):
                ll = int(line.split()[0])
                if not self.ExcludeTTTEEE:
                    self.Cl_fid[0, ll] = float(line.split()[1])
                    self.Cl_fid[1, ll] = float(line.split()[2])
                    self.Cl_fid[2, ll] = float(line.split()[3])
                # read BB:
                if self.Bmodes:
                    try:
                        self.Cl_fid[self.index_B, ll] = float(line.split()[self.index_B+1])
                    except:
                        raise io_mp.LikelihoodError(
                            "The fiducial model does not have enough columns.")
                # read DD, TD (D = deflection field):
                if self.LensingExtraction:
                    try:
                        self.Cl_fid[self.index_pp, ll] = float(line.split()[self.index_pp+1])
                        if not self.ExcludeTTTEEE:
                            self.Cl_fid[self.index_tp, ll] = float(line.split()[self.index_tp+1])
                    except:
                        raise io_mp.LikelihoodError(
                            "The fiducial model does not have enough columns.")

                line = fid_file.readline()

        # Else the file will be created in the loglkl() function.

        # Explicitly display the flags to be sure that likelihood does what you expect:
        print "Initialised likelihood_mock_cmb with following options:"
        if self.unlensed_clTTTEEE:
            print "  unlensed_clTTTEEE is True"
        else:
            print "  unlensed_clTTTEEE is False"
        if self.Bmodes:
            print "  Bmodes is True"
        else:
            print "  Bmodes is False"
        if self.delensing:
            print "  delensing is True"
        else:
            print "  delensing is False"
        if self.LensingExtraction:
            print "  LensingExtraction is True"
        else:
            print "  LensingExtraction is False"
        if self.neglect_TD:
            print "  neglect_TD is True"
        else:
            print "  neglect_TD is False"
        if self.ExcludeTTTEEE:
            print "  ExcludeTTTEEE is True"
        else:
            print "  ExcludeTTTEEE is False"
        if self.OnlyTT:
            print "  OnlyTT is True"
        else:
            print "  OnlyTT is False"
        print ""

        # end of initialisation
        return

    def loglkl(self, cosmo, data):

        # get Cl's from the cosmological code (returned in muK**2 units)

        # if we want unlensed Cl's
        if self.unlensed_clTTTEEE:
            cl = self.get_unlensed_cl(cosmo)
            # exception: for non-delensed B modes we need the lensed BB spectrum
            # (this case is usually not useful/relevant)
            if self.Bmodes and (not self.delensing):
                    cl_lensed = self.get_cl(cosmo)
                    for l in range(self.lmax+1):
                        cl[l]['bb']=cl_lensed[l]['bb']

        # if we want lensed Cl's
        else:
            cl = self.get_cl(cosmo)
            # exception: for delensed B modes we need the unlensed spectrum
            if self.Bmodes and self.delensing:
                cl_unlensed = self.get_unlensed_cl(cosmo)
                for l in range(self.lmax+1):
                        cl[l]['bb']=cl_unlensed[l]['bb']

        # get likelihood
        lkl = self.compute_lkl(cl, cosmo, data)

        return lkl

    def compute_lkl(self, cl, cosmo, data):

        # Write fiducial model spectra if needed (return an imaginary number in
        # that case)
        if self.fid_values_exist is False:
            # Store the values now.
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'w')
            fid_file.write('# Fiducial parameters')
            for key, value in data.mcmc_parameters.iteritems():
                fid_file.write(', %s = %.5g' % (
                    key, value['current']*value['scale']))
            fid_file.write('\n')
            for l in range(self.l_min, self.l_max+1):
                fid_file.write("%5d  " % l)
                if not self.ExcludeTTTEEE:
                    fid_file.write("%.8g  " % (cl['tt'][l]+self.noise_T[l]))
                    fid_file.write("%.8g  " % (cl['ee'][l]+self.noise_P[l]))
                    fid_file.write("%.8g  " % cl['te'][l])
                if self.Bmodes:
                    # next three lines added by S. Clesse for delensing
                    if self.delensing:
                        fid_file.write("%.8g  " % (cl['bb'][l]+self.noise_P[l]+self.noise_delensing[l]))
                    else:
                        fid_file.write("%.8g  " % (cl['bb'][l]+self.noise_P[l]))
                if self.LensingExtraction:
                    # we want to store clDD = l(l+1) clpp
                    # and ClTD = sqrt(l(l+1)) Cltp
                    fid_file.write("%.8g  " % (l*(l+1.)*cl['pp'][l] + self.Nldd[l]))
                    if not self.ExcludeTTTEEE:
                        fid_file.write("%.8g  " % (math.sqrt(l*(l+1.))*cl['tp'][l]))
                fid_file.write("\n")
            print '\n'
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        # compute likelihood

        chi2 = 0

        # count number of modes.
        # number of modes is different form number of spectra
        # modes = T,E,[B],[D=deflection]
        # spectra = TT,EE,TE,[BB],[DD,TD]
        # default:
        if not self.ExcludeTTTEEE:
	    if self.OnlyTT:
	        num_modes=1
	    else:
                num_modes=2
        # default 0 if excluding TT EE
        else:
            num_modes=0
        # add B mode:
        if self.Bmodes:
            num_modes += 1
        # add D mode:
        if self.LensingExtraction:
            num_modes += 1

        Cov_obs = np.zeros((num_modes, num_modes), 'float64')
        Cov_the = np.zeros((num_modes, num_modes), 'float64')
        Cov_mix = np.zeros((num_modes, num_modes), 'float64')

        for l in range(self.l_min, self.l_max+1):

            if self.Bmodes and self.LensingExtraction:
                raise io_mp.LikelihoodError("We have implemented a version of the likelihood with B modes, a version with lensing extraction, but not yet a version with both at the same time. You can implement it.")

            # case with B modes:
            elif self.Bmodes:
                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l], 0],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l], 0],
                    [0, 0, self.Cl_fid[3, l]]])
                # next 5 lines added by S. Clesse for delensing
                if self.delensing:
                    Cov_the = np.array([
                        [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0],
                        [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                        [0, 0, cl['bb'][l]+self.noise_P[l]+self.noise_delensing[l]]])
                else:
                    Cov_the = np.array([
                        [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0],
                        [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                        [0, 0, cl['bb'][l]+self.noise_P[l]]])

            # case with lensing
            # note that the likelihood is based on ClDD (deflection spectrum)
            # rather than Clpp (lensing potential spectrum)
            # But the Bolztmann code input is Clpp
            # So we make the conversion using ClDD = l*(l+1.)*Clpp
            # So we make the conversion using ClTD = sqrt(l*(l+1.))*Cltp

            # just DD, i.e. no TT or EE.
            elif self.LensingExtraction and self.ExcludeTTTEEE:
                cldd_fid = self.Cl_fid[self.index_pp, l]
                cldd = l*(l+1.)*cl['pp'][l]
                Cov_obs = np.array([[cldd_fid]])
                Cov_the = np.array([[cldd+self.Nldd[l]]])

            # Usual TTTEEE plus DD and TD
            elif self.LensingExtraction:
                cldd_fid = self.Cl_fid[self.index_pp, l]
                cldd = l*(l+1.)*cl['pp'][l]
                if self.neglect_TD:
                    cltd_fid = 0.
                    cltd = 0.
                else:
                    cltd_fid = self.Cl_fid[self.index_tp, l]
                    cltd = math.sqrt(l*(l+1.))*cl['tp'][l]

                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l], 0.*self.Cl_fid[self.index_tp, l]],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l], 0],
                    [cltd_fid, 0, cldd_fid]])
                Cov_the = np.array([
                    [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0.*math.sqrt(l*(l+1.))*cl['tp'][l]],
                    [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                    [cltd, 0, cldd+self.Nldd[l]]])
	  
	    # case with TT only (Added by Siavash Yasini)
            elif self.OnlyTT:
                Cov_obs = np.array([[self.Cl_fid[0, l]]])
                    
                Cov_the = np.array([[cl['tt'][l]+self.noise_T[l]]])
                    

            # case without B modes nor lensing:
            else:
                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l]],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l]]])
                Cov_the = np.array([
                    [cl['tt'][l]+self.noise_T[l], cl['te'][l]],
                    [cl['te'][l], cl['ee'][l]+self.noise_P[l]]])

            # get determinant of observational and theoretical covariance matrices
            det_obs = np.linalg.det(Cov_obs)
            det_the = np.linalg.det(Cov_the)

            # get determinant of mixed matrix (= sum of N theoretical
            # matrices with, in each of them, the nth column replaced
            # by that of the observational matrix)
            det_mix = 0.
            for i in range(num_modes):
                Cov_mix = np.copy(Cov_the)
                Cov_mix[:, i] = Cov_obs[:, i]
                det_mix += np.linalg.det(Cov_mix)

            chi2 += (2.*l+1.)*self.f_sky *\
                (det_mix/det_the + math.log(det_the/det_obs) - num_modes)

        return -chi2/2


###################################
# MPK TYPE LIKELIHOOD
# --> sdss, wigglez, etc.
###################################
class Likelihood_mpk(Likelihood):

    def __init__(self, path, data, command_line, common=False, common_dict={}):

        Likelihood.__init__(self, path, data, command_line)

        # require P(k) from class
        self.need_cosmo_arguments(data, {'output': 'mPk'})

        if common:
            self.add_common_knowledge(common_dict)

        try:
            self.use_halofit
        except:
            self.use_halofit = False

        if self.use_halofit:
            self.need_cosmo_arguments(data, {'non linear': 'halofit'})

        # sdssDR7 by T. Brinckmann
        # Based on Reid et al. 2010 arXiv:0907.1659 - Note: arXiv version not updated
        try:
            self.use_sdssDR7
        except:
            self.use_sdssDR7 = False

        # read values of k (in h/Mpc)
        self.k_size = self.max_mpk_kbands_use-self.min_mpk_kbands_use+1
        self.mu_size = 1
        self.k = np.zeros((self.k_size), 'float64')
        self.kh = np.zeros((self.k_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.kbands_file), 'r')
        for i in range(self.num_mpk_kbands_full):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            if i+2 > self.min_mpk_kbands_use and i < self.max_mpk_kbands_use:
                self.kh[i-self.min_mpk_kbands_use+1] = float(line.split()[0])
        datafile.close()

        khmax = self.kh[-1]

        # check if need hight value of k for giggleZ
        try:
            self.use_giggleZ
        except:
            self.use_giggleZ = False

        # Try a new model, with an additional nuisance parameter. Note
        # that the flag use_giggleZPP0 being True requires use_giggleZ
        # to be True as well. Note also that it is defined globally,
        # and not for every redshift bin.
        if self.use_giggleZ:
            try:
                self.use_giggleZPP0
            except:
                self.use_giggleZPP0 = False
        else:
            self.use_giggleZPP0 = False

        # If the flag use_giggleZPP0 is set to True, the nuisance parameters
        # P0_a, P0_b, P0_c and P0_d are expected.
        if self.use_giggleZPP0:
            if 'P0_a' not in data.get_mcmc_parameters(['nuisance']):
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "P0_a is not defined in the .param file, whereas this " +
                    "nuisance parameter is required when the flag " +
                    "'use_giggleZPP0' is set to true for WiggleZ")

        if self.use_giggleZ:
            datafile = open(os.path.join(self.data_directory,self.giggleZ_fidpk_file), 'r')

            line = datafile.readline()
            k = float(line.split()[0])
            line_number = 1
            while (k < self.kh[0]):
                line = datafile.readline()
                k = float(line.split()[0])
                line_number += 1
            ifid_discard = line_number-2
            while (k < khmax):
                line = datafile.readline()
                k = float(line.split()[0])
                line_number += 1
            datafile.close()
            self.k_fid_size = line_number-ifid_discard+1
            khmax = k

        if self.use_halofit:
            khmax *= 2

        # require k_max and z_max from the cosmological module
        if self.use_sdssDR7:
            self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
            self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 7.5*self.kmax})
        else:
            self.need_cosmo_arguments(
                data, {'P_k_max_h/Mpc': khmax, 'z_max_pk': self.redshift})

        # read information on different regions in the sky
        try:
            self.has_regions
        except:
            self.has_regions = False

        if (self.has_regions):
            self.num_regions = len(self.used_region)
            self.num_regions_used = 0
            for i in range(self.num_regions):
                if (self.used_region[i]):
                    self.num_regions_used += 1
            if (self.num_regions_used == 0):
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "Mpk: no regions begin used in this data set")
        else:
            self.num_regions = 1
            self.num_regions_used = 1
            self.used_region = [True]

        # read window functions
        self.n_size = self.max_mpk_points_use-self.min_mpk_points_use+1

        self.window = np.zeros(
            (self.num_regions, self.n_size, self.k_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.windows_file), 'r')
        for i_region in range(self.num_regions):
            for i in range(self.num_mpk_points_full):
                line = datafile.readline()
                while line.find('#') != -1:
                    line = datafile.readline()
                if (i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use):
                    for j in range(self.k_size):
                        self.window[i_region, i-self.min_mpk_points_use+1, j] = float(line.split()[j+self.min_mpk_kbands_use-1])
        datafile.close()

        # read measurements
        self.P_obs = np.zeros((self.num_regions, self.n_size), 'float64')
        self.P_err = np.zeros((self.num_regions, self.n_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.measurements_file), 'r')
        for i_region in range(self.num_regions):
            for i in range(self.num_mpk_points_full):
                line = datafile.readline()
                while line.find('#') != -1:
                    line = datafile.readline()
                if (i+2 > self.min_mpk_points_use and
                    i < self.max_mpk_points_use):
                    self.P_obs[i_region, i-self.min_mpk_points_use+1] = float(line.split()[3])
                    self.P_err[i_region, i-self.min_mpk_points_use+1] = float(line.split()[4])
        datafile.close()

        # read covariance matrices
        try:
            self.covmat_file
            self.use_covmat = True
        except:
            self.use_covmat = False

        try:
            self.use_invcov
        except:
            self.use_invcov = False

        self.invcov = np.zeros(
            (self.num_regions, self.n_size, self.n_size), 'float64')

        if self.use_covmat:
            cov = np.zeros((self.n_size, self.n_size), 'float64')
            invcov_tmp = np.zeros((self.n_size, self.n_size), 'float64')

            datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
            for i_region in range(self.num_regions):
                for i in range(self.num_mpk_points_full):
                    line = datafile.readline()
                    while line.find('#') != -1:
                        line = datafile.readline()
                    if (i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use):
                        for j in range(self.num_mpk_points_full):
                            if (j+2 > self.min_mpk_points_use and j < self.max_mpk_points_use):
                                cov[i-self.min_mpk_points_use+1,j-self.min_mpk_points_use+1] = float(line.split()[j])

                if self.use_invcov:
                    invcov_tmp = cov
                else:
                    invcov_tmp = np.linalg.inv(cov)
                for i in range(self.n_size):
                    for j in range(self.n_size):
                        self.invcov[i_region, i, j] = invcov_tmp[i, j]
            datafile.close()
        else:
            for i_region in range(self.num_regions):
                for j in range(self.n_size):
                    self.invcov[i_region, j, j] = \
                        1./(self.P_err[i_region, j]**2)

        # read fiducial model
        if self.use_giggleZ:
            self.P_fid = np.zeros((self.k_fid_size), 'float64')
            self.k_fid = np.zeros((self.k_fid_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.giggleZ_fidpk_file), 'r')
            for i in range(ifid_discard):
                line = datafile.readline()
            for i in range(self.k_fid_size):
                line = datafile.readline()
                self.k_fid[i] = float(line.split()[0])
                self.P_fid[i] = float(line.split()[1])
            datafile.close()

        # read integral constraint
        if self.use_sdssDR7:
            self.zerowindowfxn = np.zeros((self.k_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.zerowindowfxn_file), 'r')
            for i in range(self.k_size):
                line = datafile.readline()
                self.zerowindowfxn[i] = float(line.split()[0])
            datafile.close()
            self.zerowindowfxnsubtractdat = np.zeros((self.n_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.zerowindowfxnsubtractdat_file), 'r')
            line = datafile.readline()
            self.zerowindowfxnsubtractdatnorm = float(line.split()[0])
            for i in range(self.n_size):
                line = datafile.readline()
                self.zerowindowfxnsubtractdat[i] = float(line.split()[0])
            datafile.close()

        # initialize array of values for the nuisance parameters a1,a2
        if self.use_sdssDR7:
            nptsa1=self.nptsa1
            nptsa2=self.nptsa2
            a1maxval=self.a1maxval
            self.a1list=np.zeros(self.nptstot)
            self.a2list=np.zeros(self.nptstot)
            da1 = a1maxval/(nptsa1/2)
            da2 = self.a2maxpos(-a1maxval) / (nptsa2/2)
            count=0
            for i in range(-nptsa1/2, nptsa1/2+1):
                for j in range(-nptsa2/2, nptsa2/2+1):
                    a1val = da1*i
                    a2val = da2*j
                    if ((a2val >= 0.0 and a2val <= self.a2maxpos(a1val) and a2val >= self.a2minfinalpos(a1val)) or \
                        (a2val <= 0.0 and a2val <= self.a2maxfinalneg(a1val) and a2val >= self.a2minneg(a1val))):
                        if (self.testa1a2(a1val,a2val) == False):
                            raise io_mp.LikelihoodError(
                                'Error in likelihood %s ' % (self.name) +
                                'Nuisance parameter values not valid: %s %s' % (a1,a2) )
                        if(count >= self.nptstot):
                            raise io_mp.LikelihoodError(
                                'Error in likelihood %s ' % (self.name) +
                                'count > nptstot failure' )
                        self.a1list[count]=a1val
                        self.a2list[count]=a2val
                        count=count+1

        return

    # functions added for nuisance parameter space checks.
    def a2maxpos(self,a1val):
        a2max = -1.0
        if (a1val <= min(self.s1/self.k1,self.s2/self.k2)):
            a2max = min(self.s1/self.k1**2 - a1val/self.k1, self.s2/self.k2**2 - a1val/self.k2)
        return a2max

    def a2min1pos(self,a1val):
        a2min1 = 0.0
        if(a1val <= 0.0):
            a2min1 = max(-self.s1/self.k1**2 - a1val/self.k1, -self.s2/self.k2**2 - a1val/self.k2, 0.0)
        return a2min1

    def a2min2pos(self,a1val):
        a2min2 = 0.0
        if(abs(a1val) >= 2.0*self.s1/self.k1 and a1val <= 0.0):
            a2min2 = a1val**2/self.s1*0.25
        return a2min2

    def a2min3pos(self,a1val):
        a2min3 = 0.0
        if(abs(a1val) >= 2.0*self.s2/self.k2 and a1val <= 0.0):
            a2min3 = a1val**2/self.s2*0.25
        return a2min3

    def a2minfinalpos(self,a1val):
        a2minpos = max(self.a2min1pos(a1val),self.a2min2pos(a1val),self.a2min3pos(a1val))
        return a2minpos

    def a2minneg(self,a1val):
        if (a1val >= max(-self.s1/self.k1,-self.s2/self.k2)):
            a2min = max(-self.s1/self.k1**2 - a1val/self.k1, -self.s2/self.k2**2 - a1val/self.k2)
        else:
            a2min = 1.0
        return a2min

    def a2max1neg(self,a1val):
        if(a1val >= 0.0):
            a2max1 = min(self.s1/self.k1**2 - a1val/self.k1, self.s2/self.k2**2 - a1val/self.k2, 0.0)
        else:
            a2max1 = 0.0
        return a2max1

    def a2max2neg(self,a1val):
        a2max2 = 0.0
        if(abs(a1val) >= 2.0*self.s1/self.k1 and a1val >= 0.0):
            a2max2 = -a1val**2/self.s1*0.25
        return a2max2

    def a2max3neg(self,a1val):
        a2max3 = 0.0
        if(abs(a1val) >= 2.0*self.s2/self.k2 and a1val >= 0.0):
            a2max3 = -a1val**2/self.s2*0.25
        return a2max3

    def a2maxfinalneg(self,a1val):
        a2maxneg = min(self.a2max1neg(a1val),self.a2max2neg(a1val),self.a2max3neg(a1val))
        return a2maxneg

    def testa1a2(self,a1val, a2val):
        testresult = True
        # check if there's an extremum; either a1val or a2val has to be negative, not both
        if (a2val==0.):
             return testresult #not in the original code, but since a2val=0 returns True this way I avoid zerodivisionerror
        kext = -a1val/2.0/a2val
        diffval = abs(a1val*kext + a2val*kext**2)
        if(kext > 0.0 and kext <= self.k1 and diffval > self.s1):
            testresult = False
        if(kext > 0.0 and kext <= self.k2 and diffval > self.s2):
            testresult = False
        if (abs(a1val*self.k1 + a2val*self.k1**2) > self.s1):
            testresult = False
        if (abs(a1val*self.k2 + a2val*self.k2**2) > self.s2):
            testresult = False
        return testresult


    def add_common_knowledge(self, common_dictionary):
        """
        Add to a class the content of a shared dictionary of attributes

        The purpose of this method is to set some attributes globally for a Pk
        likelihood, that are shared amongst all the redshift bins (in
        WiggleZ.data for instance, a few flags and numbers are defined that
        will be transfered to wigglez_a, b, c and d

        """
        for key, value in common_dictionary.iteritems():
            # First, check if the parameter exists already
            try:
                exec("self.%s" % key)
                warnings.warn(
                    "parameter %s from likelihood %s will be replaced by " +
                    "the common knowledge routine" % (key, self.name))
            except:
                if type(value) != type('foo'):
                    exec("self.%s = %s" % (key, value))
                else:
                    exec("self.%s = '%s'" % (key, value))

    # compute likelihood
    def loglkl(self, cosmo, data):

        # reduced Hubble parameter
        h = cosmo.h()

        # WiggleZ and sdssDR7 specific
        if self.use_scaling:
            # angular diameter distance at this redshift, in Mpc
            d_angular = cosmo.angular_distance(self.redshift)

            # radial distance at this redshift, in Mpc, is simply 1/H (itself
            # in Mpc^-1). Hz is an array, with only one element.
            r, Hz = cosmo.z_of_r([self.redshift])
            d_radial = 1/Hz[0]

            # scaling factor = (d_angular**2 * d_radial)^(1/3) for the
            # fiducial cosmology used in the data files of the observations
            # divided by the same quantity for the cosmology we are comparing with.
            # The fiducial values are stored in the .data files for
            # each experiment, and are truly in Mpc. Beware for a potential
            # difference with CAMB conventions here.
            scaling = pow(
                (self.d_angular_fid/d_angular)**2 *
                (self.d_radial_fid/d_radial), 1./3.)
        else:
            scaling = 1
        # get rescaled values of k in 1/Mpc
        self.k = self.kh*h*scaling

        # get P(k) at right values of k, convert it to (Mpc/h)^3 and rescale it
        P_lin = np.zeros((self.k_size), 'float64')

        # If the flag use_giggleZ is set to True, the power spectrum retrieved
        # from Class will get rescaled by the fiducial power spectrum given by
        # the GiggleZ N-body simulations CITE
        if self.use_giggleZ:
            P = np.zeros((self.k_fid_size), 'float64')
            for i in range(self.k_fid_size):
                P[i] = cosmo.pk(self.k_fid[i]*h, self.redshift)
                power = 0
                # The following create a polynome in k, which coefficients are
                # stored in the .data files of the experiments.
                for j in range(6):
                    power += self.giggleZ_fidpoly[j]*self.k_fid[i]**j
                # rescale P by fiducial model and get it in (Mpc/h)**3
                P[i] *= pow(10, power)*(h/scaling)**3/self.P_fid[i]

            if self.use_giggleZPP0:
                # Shot noise parameter addition to GiggleZ model. It should
                # recover the proper nuisance parameter, depending on the name.
                # I.e., Wigglez_A should recover P0_a, etc...
                tag = self.name[-2:]  # circle over "_a", "_b", etc...
                P0_value = data.mcmc_parameters['P0'+tag]['current'] *\
                    data.mcmc_parameters['P0'+tag]['scale']
                P_lin = np.interp(self.kh,self.k_fid,P+P0_value)
            else:
                # get P_lin by interpolation. It is still in (Mpc/h)**3
                P_lin = np.interp(self.kh, self.k_fid, P)

        elif self.use_sdssDR7:
            kh = np.logspace(math.log(1e-3),math.log(1.0),num=(math.log(1.0)-math.log(1e-3))/0.01+1,base=math.exp(1.0)) # k in h/Mpc
            # Rescale the scaling factor by the fiducial value for h divided by the sampled value
            # h=0.701 was used for the N-body calibration simulations
            scaling = scaling * (0.701/h)
            k = kh*h # k in 1/Mpc

            # Define redshift bins and associated bao 2 sigma value [NEAR, MID, FAR]
            z = np.array([0.235, 0.342, 0.421])
            sigma2bao = np.array([86.9988, 85.1374, 84.5958])
            # Initialize arrays
            # Analytical growth factor for each redshift bin
            D_growth = np.zeros(len(z))
            # P(k) *with* wiggles, both linear and nonlinear
            Plin = np.zeros(len(k), 'float64')
            Pnl = np.zeros(len(k), 'float64')
            # P(k) *without* wiggles, both linear and nonlinear
            Psmooth = np.zeros(len(k), 'float64')
            Psmooth_nl = np.zeros(len(k), 'float64')
            # Damping function and smeared P(k)
            fdamp = np.zeros([len(k), len(z)], 'float64')
            Psmear = np.zeros([len(k), len(z)], 'float64')
            # Ratio of smoothened non-linear to linear P(k)
            nlratio = np.zeros([len(k), len(z)], 'float64')
            # Loop over each redshift bin
            for j in range(len(z)):
                # Compute growth factor at each redshift
                # This growth factor is normalized by the growth factor today
                D_growth[j] = cosmo.scale_independent_growth_factor(z[j])
                # Compute Pk *with* wiggles, both linear and nonlinear
                # Get P(k) at right values of k in Mpc**3, convert it to (Mpc/h)^3 and rescale it
                # Get values of P(k) in Mpc**3
                for i in range(len(k)):
                    Plin[i] = cosmo.pk_lin(k[i], z[j])
                    Pnl[i] = cosmo.pk(k[i], z[j])
                # Get rescaled values of P(k) in (Mpc/h)**3
                Plin *= h**3 #(h/scaling)**3
                Pnl *= h**3 #(h/scaling)**3
                # Compute Pk *without* wiggles, both linear and nonlinear
                Psmooth = self.remove_bao(kh,Plin)
                Psmooth_nl = self.remove_bao(kh,Pnl)
                # Apply Gaussian damping due to non-linearities
                fdamp[:,j] = np.exp(-0.5*sigma2bao[j]*kh**2)
                Psmear[:,j] = Plin*fdamp[:,j]+Psmooth*(1.0-fdamp[:,j])
                # Take ratio of smoothened non-linear to linear P(k)
                nlratio[:,j] = Psmooth_nl/Psmooth

            # Save fiducial model for non-linear corrections using the flat fiducial
            # Omega_m = 0.25, Omega_L = 0.75, h = 0.701
            # Re-run if changes are made to how non-linear corrections are done
            # e.g. the halofit implementation in CLASS
            # To re-run fiducial, set <experiment>.create_fid = True in .data file
            # Can leave option enabled, as it will only compute once at the start
            try:
                self.create_fid
            except:
                self.create_fid = False

            if self.create_fid == True:
                # Calculate relevant flat fiducial quantities
                fidnlratio, fidNEAR, fidMID, fidFAR = self.get_flat_fid(cosmo,data,kh,z,sigma2bao)
                try:
                    existing_fid = np.loadtxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat')
                    print 'sdss_lrgDR7: Checking fiducial deviations for near, mid and far bins:', np.sum(existing_fid[:,1] - fidNEAR),np.sum(existing_fid[:,2] - fidMID), np.sum(existing_fid[:,3] - fidFAR)
                    if np.sum(existing_fid[:,1] - fidNEAR) + np.sum(existing_fid[:,2] - fidMID) + np.sum(existing_fid[:,3] - fidFAR) < 10**-5:
                        self.create_fid = False
                except:
                    pass
                if self.create_fid == True:
                    print 'sdss_lrgDR7: Creating fiducial file with Omega_b = 0.25, Omega_L = 0.75, h = 0.701'
                    print '             Required for non-linear modeling'
                    # Save non-linear corrections from N-body sims for each redshift bin
                    arr=np.zeros((np.size(kh),7))
                    arr[:,0]=kh
                    arr[:,1]=fidNEAR
                    arr[:,2]=fidMID
                    arr[:,3]=fidFAR
                    # Save non-linear corrections from halofit for each redshift bin
                    arr[:,4:7]=fidnlratio
                    np.savetxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat',arr)
                    self.create_fid = False
                    print '             Fiducial created'

            # Load fiducial model
            fiducial = np.loadtxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat')
            fid = fiducial[:,1:4]
            fidnlratio = fiducial[:,4:7]

            # Put all factors together to obtain the P(k) for each redshift bin
            Pnear=np.interp(kh,kh,Psmear[:,0]*(nlratio[:,0]/fidnlratio[:,0])*fid[:,0]*D_growth[0]**(-2.))
            Pmid =np.interp(kh,kh,Psmear[:,1]*(nlratio[:,1]/fidnlratio[:,1])*fid[:,1]*D_growth[1]**(-2.))
            Pfar =np.interp(kh,kh,Psmear[:,2]*(nlratio[:,2]/fidnlratio[:,2])*fid[:,2]*D_growth[2]**(-2.))

            # Define and rescale k
            self.k=self.kh*h*scaling
            # Weighted mean of the P(k) for each redshift bin
            P_lin=(0.395*Pnear+0.355*Pmid+0.250*Pfar)
            P_lin=np.interp(self.k,kh*h,P_lin)*(1./scaling)**3 # remember self.k is scaled but self.kh isn't

        else:
            # get rescaled values of k in 1/Mpc
            self.k = self.kh*h*scaling
            # get values of P(k) in Mpc**3
            for i in range(self.k_size):
                P_lin[i] = cosmo.pk(self.k[i], self.redshift)
            # get rescaled values of P(k) in (Mpc/h)**3
            P_lin *= (h/scaling)**3

        # infer P_th from P_lin. It is still in (Mpc/h)**3. TODO why was it
        # called P_lin in the first place ? Couldn't we use now P_th all the
        # way ?
        P_th = P_lin

        if self.use_sdssDR7:
            chisq =np.zeros(self.nptstot)
            chisqmarg = np.zeros(self.nptstot)

            Pth = P_th
            Pth_k = P_th*(self.k/h) # self.k has the scaling included, so self.k/h != self.kh
            Pth_k2 = P_th*(self.k/h)**2

            WPth = np.dot(self.window[0,:], Pth)
            WPth_k = np.dot(self.window[0,:], Pth_k)
            WPth_k2 = np.dot(self.window[0,:], Pth_k2)

            sumzerow_Pth = np.sum(self.zerowindowfxn*Pth)/self.zerowindowfxnsubtractdatnorm
            sumzerow_Pth_k = np.sum(self.zerowindowfxn*Pth_k)/self.zerowindowfxnsubtractdatnorm
            sumzerow_Pth_k2 = np.sum(self.zerowindowfxn*Pth_k2)/self.zerowindowfxnsubtractdatnorm

            covdat = np.dot(self.invcov[0,:,:],self.P_obs[0,:])
            covth  = np.dot(self.invcov[0,:,:],WPth)
            covth_k  = np.dot(self.invcov[0,:,:],WPth_k)
            covth_k2  = np.dot(self.invcov[0,:,:],WPth_k2)
            covth_zerowin  = np.dot(self.invcov[0,:,:],self.zerowindowfxnsubtractdat)
            sumDD = np.sum(self.P_obs[0,:] * covdat)
            sumDT = np.sum(self.P_obs[0,:] * covth)
            sumDT_k = np.sum(self.P_obs[0,:] * covth_k)
            sumDT_k2 = np.sum(self.P_obs[0,:] * covth_k2)
            sumDT_zerowin = np.sum(self.P_obs[0,:] * covth_zerowin)

            sumTT = np.sum(WPth*covth)
            sumTT_k = np.sum(WPth*covth_k)
            sumTT_k2 = np.sum(WPth*covth_k2)
            sumTT_k_k = np.sum(WPth_k*covth_k)
            sumTT_k_k2 = np.sum(WPth_k*covth_k2)
            sumTT_k2_k2 = np.sum(WPth_k2*covth_k2)
            sumTT_zerowin = np.sum(WPth*covth_zerowin)
            sumTT_k_zerowin = np.sum(WPth_k*covth_zerowin)
            sumTT_k2_zerowin = np.sum(WPth_k2*covth_zerowin)
            sumTT_zerowin_zerowin = np.sum(self.zerowindowfxnsubtractdat*covth_zerowin)

            currminchisq = 1000.0

            # analytic marginalization over a1,a2
            for i in range(self.nptstot):
                a1val = self.a1list[i]
                a2val = self.a2list[i]
                zerowinsub = -(sumzerow_Pth + a1val*sumzerow_Pth_k + a2val*sumzerow_Pth_k2)
                sumDT_tot = sumDT + a1val*sumDT_k + a2val*sumDT_k2 + zerowinsub*sumDT_zerowin
                sumTT_tot = sumTT + a1val**2.0*sumTT_k_k + a2val**2.0*sumTT_k2_k2 + \
                    zerowinsub**2.0*sumTT_zerowin_zerowin + \
                    2.0*a1val*sumTT_k + 2.0*a2val*sumTT_k2 + 2.0*a1val*a2val*sumTT_k_k2 + \
                    2.0*zerowinsub*sumTT_zerowin + 2.0*zerowinsub*a1val*sumTT_k_zerowin + \
                    2.0*zerowinsub*a2val*sumTT_k2_zerowin
                minchisqtheoryamp = sumDT_tot/sumTT_tot
                chisq[i] = sumDD - 2.0*minchisqtheoryamp*sumDT_tot + minchisqtheoryamp**2.0*sumTT_tot
                chisqmarg[i] = sumDD - sumDT_tot**2.0/sumTT_tot + math.log(sumTT_tot) - \
                    2.0*math.log(1.0 + math.erf(sumDT_tot/2.0/math.sqrt(sumTT_tot)))
                if(i == 0 or chisq[i] < currminchisq):
                    myminchisqindx = i
                    currminchisq = chisq[i]
                    currminchisqmarg = chisqmarg[i]
                    minchisqtheoryampminnuis = minchisqtheoryamp
                if(i == int(self.nptstot/2)):
                    chisqnonuis = chisq[i]
                    minchisqtheoryampnonuis = minchisqtheoryamp
                    if(abs(a1val) > 0.001 or abs(a2val) > 0.001):
                         print 'sdss_lrgDR7: ahhhh! violation!!', a1val, a2val

            # numerically marginalize over a1,a2 now using values stored in chisq
            minchisq = np.min(chisqmarg)
            maxchisq = np.max(chisqmarg)

            LnLike = np.sum(np.exp(-(chisqmarg-minchisq)/2.0)/(self.nptstot*1.0))
            if(LnLike == 0):
                #LnLike = LogZero
                raise io_mp.LikelihoodError(
                    'Error in likelihood %s ' % (self.name) +
                    'LRG LnLike LogZero error.' )
            else:
                chisq = -2.*math.log(LnLike) + minchisq
            #print 'DR7 chi2/2=',chisq/2.

        #if we are not using DR7
        else:
            W_P_th = np.zeros((self.n_size), 'float64')

            # starting analytic marginalisation over bias

            # Define quantities living in all the regions possible. If only a few
            # regions are selected in the .data file, many elements from these
            # arrays will stay at 0.
            P_data_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            W_P_th_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            cov_dat_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            cov_th_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')

            normV = 0

            # Loop over all the available regions
            for i_region in range(self.num_regions):
                # In each region that was selected with the array of flags
                # self.used_region, define boundaries indices, and fill in the
                # corresponding windowed power spectrum. All the unused regions
                # will still be set to zero as from the initialization, which will
                # not contribute anything in the final sum.

                if self.used_region[i_region]:
                    imin = i_region*self.n_size
                    imax = (i_region+1)*self.n_size-1

                    W_P_th = np.dot(self.window[i_region, :], P_th)
                    #print W_P_th
                    for i in range(self.n_size):
                        P_data_large[imin+i] = self.P_obs[i_region, i]
                        W_P_th_large[imin+i] = W_P_th[i]
                        cov_dat_large[imin+i] = np.dot(
                            self.invcov[i_region, i, :],
                            self.P_obs[i_region, :])
                        cov_th_large[imin+i] = np.dot(
                            self.invcov[i_region, i, :],
                            W_P_th[:])

            # Explain what it is TODO
            normV += np.dot(W_P_th_large, cov_th_large)
            # Sort of bias TODO ?
            b_out = np.sum(W_P_th_large*cov_dat_large) / \
                np.sum(W_P_th_large*cov_th_large)

            # Explain this formula better, link to article ?
            chisq = np.dot(P_data_large, cov_dat_large) - \
                np.dot(W_P_th_large, cov_dat_large)**2/normV
            #print 'WiggleZ chi2=',chisq/2.

        return -chisq/2

    def remove_bao(self,k_in,pk_in):
        # De-wiggling routine by Mario Ballardini

        # This k range has to contain the BAO features:
        k_ref=[2.8e-2, 4.5e-1]

        # Get interpolating function for input P(k) in log-log space:
        _interp_pk = scipy.interpolate.interp1d( np.log(k_in), np.log(pk_in),
                                                 kind='quadratic', bounds_error=False )
        interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))

        # Spline all (log-log) points outside k_ref range:
        idxs = np.where(np.logical_or(k_in <= k_ref[0], k_in >= k_ref[1]))
        _pk_smooth = scipy.interpolate.UnivariateSpline( np.log(k_in[idxs]),
                                                         np.log(pk_in[idxs]), k=3, s=0 )
        pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

        # Find second derivative of each spline:
        fwiggle = scipy.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
        derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
        d2 = scipy.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0)

        # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
        # low-order spline through zeros to subtract smooth trend from wiggles fn.
        wzeros = d2.roots()
        wzeros = wzeros[np.where(np.logical_and(wzeros >= k_ref[0], wzeros <= k_ref[1]))]
        wzeros = np.concatenate((wzeros, [k_ref[1],]))
        wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)

        # Construct smooth no-BAO:
        idxs = np.where(np.logical_and(k_in > k_ref[0], k_in < k_ref[1]))
        pk_nobao = pk_smooth(k_in)
        pk_nobao[idxs] *= wtrend(k_in[idxs])

        # Construct interpolating functions:
        ipk = scipy.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                          bounds_error=False, fill_value=0. )

        pk_nobao = ipk(k_in)

        return pk_nobao

    def get_flat_fid(self,cosmo,data,kh,z,sigma2bao):
        # SDSS DR7 LRG specific function
        # Compute fiducial properties for a flat fiducial
        # with Omega_m = 0.25, Omega_L = 0.75, h = 0.701
        param_backup = data.cosmo_arguments
        data.cosmo_arguments = {'P_k_max_h/Mpc': 1.5, 'ln10^{10}A_s': 3.0, 'N_ur': 3.04, 'h': 0.701,
                                'omega_b': 0.035*0.701**2, 'non linear': ' halofit ', 'YHe': 0.24, 'k_pivot': 0.05,
                                'n_s': 0.96, 'tau_reio': 0.084, 'z_max_pk': 0.5, 'output': ' mPk ',
                                'omega_cdm': 0.215*0.701**2, 'T_cmb': 2.726}
        cosmo.empty()
        cosmo.set(data.cosmo_arguments)
        cosmo.compute(['lensing'])
        h = data.cosmo_arguments['h']
        k = kh*h
        # P(k) *with* wiggles, both linear and nonlinear
        Plin = np.zeros(len(k), 'float64')
        Pnl = np.zeros(len(k), 'float64')
        # P(k) *without* wiggles, both linear and nonlinear
        Psmooth = np.zeros(len(k), 'float64')
        Psmooth_nl = np.zeros(len(k), 'float64')
        # Damping function and smeared P(k)
        fdamp = np.zeros([len(k), len(z)], 'float64')
        Psmear = np.zeros([len(k), len(z)], 'float64')
        # Ratio of smoothened non-linear to linear P(k)
        fidnlratio = np.zeros([len(k), len(z)], 'float64')
        # Loop over each redshift bin
        for j in range(len(z)):
            # Compute Pk *with* wiggles, both linear and nonlinear
            # Get P(k) at right values of k in Mpc**3, convert it to (Mpc/h)^3 and rescale it
            # Get values of P(k) in Mpc**3
            for i in range(len(k)):
                Plin[i] = cosmo.pk_lin(k[i], z[j])
                Pnl[i] = cosmo.pk(k[i], z[j])
            # Get rescaled values of P(k) in (Mpc/h)**3
            Plin *= h**3 #(h/scaling)**3
            Pnl *= h**3 #(h/scaling)**3
            # Compute Pk *without* wiggles, both linear and nonlinear
            Psmooth = self.remove_bao(kh,Plin)
            Psmooth_nl = self.remove_bao(kh,Pnl)
            # Apply Gaussian damping due to non-linearities
            fdamp[:,j] = np.exp(-0.5*sigma2bao[j]*kh**2)
            Psmear[:,j] = Plin*fdamp[:,j]+Psmooth*(1.0-fdamp[:,j])
            # Take ratio of smoothened non-linear to linear P(k)
            fidnlratio[:,j] = Psmooth_nl/Psmooth

        # Polynomials to shape small scale behavior from N-body sims
        kdata=kh
        fidpolyNEAR=np.zeros(np.size(kdata))
        fidpolyNEAR[kdata<=0.194055] = (1.0 - 0.680886*kdata[kdata<=0.194055] + 6.48151*kdata[kdata<=0.194055]**2)
        fidpolyNEAR[kdata>0.194055] = (1.0 - 2.13627*kdata[kdata>0.194055] + 21.0537*kdata[kdata>0.194055]**2 - 50.1167*kdata[kdata>0.194055]**3 + 36.8155*kdata[kdata>0.194055]**4)*1.04482
        fidpolyMID=np.zeros(np.size(kdata))
        fidpolyMID[kdata<=0.19431] = (1.0 - 0.530799*kdata[kdata<=0.19431] + 6.31822*kdata[kdata<=0.19431]**2)
        fidpolyMID[kdata>0.19431] = (1.0 - 1.97873*kdata[kdata>0.19431] + 20.8551*kdata[kdata>0.19431]**2 - 50.0376*kdata[kdata>0.19431]**3 + 36.4056*kdata[kdata>0.19431]**4)*1.04384
        fidpolyFAR=np.zeros(np.size(kdata))
        fidpolyFAR[kdata<=0.19148] = (1.0 - 0.475028*kdata[kdata<=0.19148] + 6.69004*kdata[kdata<=0.19148]**2)
        fidpolyFAR[kdata>0.19148] = (1.0 - 1.84891*kdata[kdata>0.19148] + 21.3479*kdata[kdata>0.19148]**2 - 52.4846*kdata[kdata>0.19148]**3 + 38.9541*kdata[kdata>0.19148]**4)*1.03753

        fidNEAR=np.interp(kh,kdata,fidpolyNEAR)
        fidMID=np.interp(kh,kdata,fidpolyMID)
        fidFAR=np.interp(kh,kdata,fidpolyFAR)

        cosmo.empty()
        data.cosmo_arguments = param_backup
        cosmo.set(data.cosmo_arguments)
        cosmo.compute(['lensing'])

        return fidnlratio, fidNEAR, fidMID, fidFAR

class Likelihood_sn(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # try and import pandas
        try:
            import pandas
        except ImportError:
            raise io_mp.MissingLibraryError(
                "This likelihood has a lot of IO manipulation. You have "
                "to install the 'pandas' library to use it. Please type:\n"
                "`(sudo) pip install pandas --user`")

        # check that every conflicting experiments is not present in the list
        # of tested experiments, in which case, complain
        if hasattr(self, 'conflicting_experiments'):
            for conflict in self.conflicting_experiments:
                if conflict in data.experiments:
                    raise io_mp.LikelihoodError(
                        'conflicting %s measurements, you can ' % conflict +
                        ' have either %s or %s ' % (self.name, conflict) +
                        'as an experiment, not both')

        # Read the configuration file, supposed to be called self.settings.
        # Note that we unfortunately can not
        # immediatly execute the file, as it is not formatted as strings.
        assert hasattr(self, 'settings') is True, (
            "You need to provide a settings file")
        self.read_configuration_file()

    def read_configuration_file(self):
        """
        Extract Python variables from the configuration file

        This routine performs the equivalent to the program "inih" used in the
        original c++ library.
        """
        settings_path = os.path.join(self.data_directory, self.settings)
        with open(settings_path, 'r') as config:
            for line in config:
                # Dismiss empty lines and commented lines
                if line and line.find('#') == -1 and line not in ['\n', '\r\n']:
                    lhs, rhs = [elem.strip() for elem in line.split('=')]
                    # lhs will always be a string, so set the attribute to this
                    # likelihood. The right hand side requires more work.
                    # First case, if set to T or F for True or False
                    if str(rhs) in ['T', 'F']:
                        rhs = True if str(rhs) == 'T' else False
                    # It can also be a path, starting with 'data/'. We remove
                    # this leading folder path
                    elif str(rhs).find('data/') != -1:
                        rhs = rhs.replace('data/', '')
                    else:
                        # Try  to convert it to a float
                        try:
                            rhs = float(rhs)
                        # If it fails, it is a string
                        except ValueError:
                            rhs = str(rhs)
                    # Set finally rhs to be a parameter of the class
                    setattr(self, lhs, rhs)

    def read_matrix(self, path):
        """
        extract the matrix from the path

        This routine uses the blazing fast pandas library (0.10 seconds to load
        a 740x740 matrix). If not installed, it uses a custom routine that is
        twice as slow (but still 4 times faster than the straightforward
        numpy.loadtxt method)

        .. note::

            the length of the matrix is stored on the first line... then it has
            to be unwrapped. The pandas routine read_table understands this
            immediatly, though.

        """
        from pandas import read_table
        path = os.path.join(self.data_directory, path)
        # The first line should contain the length.
        with open(path, 'r') as text:
            length = int(text.readline())

        # Note that this function does not require to skiprows, as it
        # understands the convention of writing the length in the first
        # line
        matrix = read_table(path).as_matrix().reshape((length, length))

        return matrix

    def read_light_curve_parameters(self):
        """
        Read the file jla_lcparams.txt containing the SN data

        .. note::

            the length of the resulting array should be equal to the length of
            the covariance matrices stored in C00, etc...

        """
        from pandas import read_table
        path = os.path.join(self.data_directory, self.data_file)

        # Recover the names of the columns. The names '3rdvar' and 'd3rdvar'
        # will be changed, because 3rdvar is not a valid variable name
        with open(path, 'r') as text:
            clean_first_line = text.readline()[1:].strip()
            names = [e.strip().replace('3rd', 'third')
                     for e in clean_first_line.split()]

        lc_parameters = read_table(
            path, sep=' ', names=names, header=0, index_col=False)
        return lc_parameters


class Likelihood_clocks(Likelihood):
    """Base implementation of H(z) measurements"""

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Read the content of the data file, containing z, Hz and error
        total = np.loadtxt(
            os.path.join(self.data_directory, self.data_file))

        # Store the columns separately
        self.z = total[:, 0]
        self.Hz = total[:, 1]
        self.err = total[:, 2]

    def loglkl(self, cosmo, data):

        # Store the speed of light in km/s
        c_light_km_per_sec = const.c/1000.
        chi2 = 0

        # Loop over the redshifts
        for index, z in enumerate(self.z):
            # Query the cosmo module for the Hubble rate (in 1/Mpc), and
            # convert it to km/s/Mpc
            H_cosmo = cosmo.Hubble(z)*c_light_km_per_sec
            # Add to the tota chi2
            chi2 += (self.Hz[index]-H_cosmo)**2/self.err[index]**2

        return -0.5 * chi2

###################################
# ISW-Likelihood
# by B. Stoelzner
###################################
class Likelihood_isw(Likelihood):
    def __init__(self, path, data, command_line):
        # Initialize
        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(data, {'output': 'mPk','P_k_max_h/Mpc' : 300,'z_max_pk' : 5.1})

        # Read l,C_l, and the covariance matrix of the autocorrelation of the survey and the crosscorrelation of the survey with the CMB
        self.l_cross,cl_cross=np.loadtxt(os.path.join(self.data_directory,self.cl_cross_file),unpack=True,usecols=(0,1))
        self.l_auto,cl_auto=np.loadtxt(os.path.join(self.data_directory,self.cl_auto_file),unpack=True,usecols=(0,1))
        cov_cross=np.loadtxt(os.path.join(self.data_directory,self.cov_cross_file))
        cov_auto=np.loadtxt(os.path.join(self.data_directory,self.cov_auto_file))

        # Extract data in the specified range in l.
        self.l_cross=self.l_cross[self.l_min_cross:self.l_max_cross+1]
        cl_cross=cl_cross[self.l_min_cross:self.l_max_cross+1]
        self.l_auto=self.l_auto[self.l_min_auto:self.l_max_auto+1]
        cl_auto=cl_auto[self.l_min_auto:self.l_max_auto+1]
        cov_cross=cov_cross[self.l_min_cross:self.l_max_cross+1,self.l_min_cross:self.l_max_cross+1]
        cov_auto=cov_auto[self.l_min_auto:self.l_max_auto+1,self.l_min_auto:self.l_max_auto+1]

        # Create logarithically spaced bins in l.
        self.bins_cross=np.ceil(np.logspace(np.log10(self.l_min_cross),np.log10(self.l_max_cross),self.n_bins_cross+1))
        self.bins_auto=np.ceil(np.logspace(np.log10(self.l_min_auto),np.log10(self.l_max_auto),self.n_bins_auto+1))

        # Bin l,C_l, and covariance matrix in the previously defined bins
        self.l_binned_cross,self.cl_binned_cross,self.cov_binned_cross=self.bin_cl(self.l_cross,cl_cross,self.bins_cross,cov_cross)
        self.l_binned_auto,self.cl_binned_auto,self.cov_binned_auto=self.bin_cl(self.l_auto,cl_auto,self.bins_auto,cov_auto)

        # Read the redshift distribution of objects in the survey, perform an interpolation of dN/dz(z), and calculate the normalization in this redshift bin
        zz,dndz=np.loadtxt(os.path.join(self.data_directory,self.dndz_file),unpack=True,usecols=(0,1))
        self.dndz=scipy.interpolate.interp1d(zz,dndz,kind='cubic')
        self.norm=scipy.integrate.quad(self.dndz,self.z_min,self.z_max)[0]

    def bin_cl(self,l,cl,bins,cov=None):
        # This function bins l,C_l, and the covariance matrix in given bins in l
        B=[]
        for i in range(1,len(bins)):
            if i!=len(bins)-1:
                a=np.where((l<bins[i])&(l>=bins[i-1]))[0]
            else:
                a=np.where((l<=bins[i])&(l>=bins[i-1]))[0]
            c=np.zeros(len(l))
            c[a]=1./len(a)
            B.append(c)
        l_binned=np.dot(B,l)
        cl_binned=np.dot(B,cl)
        if cov is not None:
            cov_binned=np.dot(B,np.dot(cov,np.transpose(B)))
            return l_binned,cl_binned,cov_binned
        else:
            return l_binned,cl_binned

    def integrand_cross(self,z,cosmo,l):
        # This function will be integrated to calculate the exspected crosscorrelation between the survey and the CMB
        c= const.c/1000.
        H0=cosmo.h()*100
        Om=cosmo.Omega0_m()
        k=lambda z:(l+0.5)/(cosmo.angular_distance(z)*(1+z))
        return (3*Om*H0**2)/((c**2)*(l+0.5)**2)*self.dndz(z)*cosmo.Hubble(z)*cosmo.scale_independent_growth_factor(z)*scipy.misc.derivative(lambda z:cosmo.scale_independent_growth_factor(z)*(1+z),x0=z,dx=1e-4)*cosmo.pk(k(z),0)/self.norm

    def integrand_auto(self,z,cosmo,l):
        # This function will be integrated to calculate the expected autocorrelation of the survey
        c= const.c/1000.
        H0=cosmo.h()*100
        k=lambda z:(l+0.5)/(cosmo.angular_distance(z)*(1+z))
        return (self.dndz(z))**2*(cosmo.scale_independent_growth_factor(z))**2*cosmo.pk(k(z),0)*cosmo.Hubble(z)/(cosmo.angular_distance(z)*(1+z))**2/self.norm**2

    def compute_loglkl(self, cosmo, data,b):
        # Retrieve sampled parameter
        A=data.mcmc_parameters['A_ISW']['current']*data.mcmc_parameters['A_ISW']['scale']

        # Calculate the expected auto- and crosscorrelation by integrating over the redshift.
        cl_binned_cross_theory=np.array([(scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.bins_cross[ll]))[0]+scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.bins_cross[ll+1]))[0]+scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.l_binned_cross[ll]))[0])/3 for ll in range(self.n_bins_cross)])
        cl_binned_auto_theory=np.array([scipy.integrate.quad(self.integrand_auto,self.z_min,self.z_max,args=(cosmo,ll),epsrel=1e-8)[0] for ll in self.l_binned_auto])

        # Calculate the chi-square of auto- and crosscorrelation
        chi2_cross=np.asscalar(np.dot(self.cl_binned_cross-A*b*cl_binned_cross_theory,np.dot(np.linalg.inv(self.cov_binned_cross),self.cl_binned_cross-A*b*cl_binned_cross_theory)))
        chi2_auto=np.asscalar(np.dot(self.cl_binned_auto-b**2*cl_binned_auto_theory,np.dot(np.linalg.inv(self.cov_binned_auto),self.cl_binned_auto-b**2*cl_binned_auto_theory)))
        return -0.5*(chi2_cross+chi2_auto)


############################################################################################################################################
#
#
#
############################################################################################################################################


import scipy.constants as conts
import yaml
from copy import deepcopy
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from scipy.fftpack import dst
from scipy import stats
def pvalue(minchi2, dof): return 1. - stats.chi2.cdf(minchi2, dof)
try:
    import pybird as pb
except ImportError:
    raise Exception('Cannot find pybird library')

class Likelihood_bird(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.config = yaml.full_load(open(os.path.join(self.data_directory, self.configfile), 'r'))

        options = ["nonmarg", "get_chi2_from_marg", "with_derived_bias", "get_fit", "get_fake",
        "with_window", "with_fibercol", "with_redshift_bin", "with_stoch", "with_exact_time", "with_tidal_alignments", "with_nnlo_counterterm", 
        "with_quintessence", "with_cf_sys", "xmaxspacing", "with_rs_marg"]

        for keys in options:
            if not keys in self.config: self.config[keys] = False
            print (keys, ':', self.config[keys])

        # loading EFT parameters to vary and priors
        self.eft_parameters_list = [param for param in self.config["eft_prior"]]
        self.nonmarg_gauss_eft_parameters_list = [param for param, prior in self.config["eft_prior"].items() if prior["type"] == 'gauss']
        self.nonmarg_gauss_eft_parameters_prior_mean = np.array([self.config["eft_prior"][param]["mean"] for param in self.nonmarg_gauss_eft_parameters_list]).T
        self.nonmarg_gauss_eft_parameters_prior_sigma = np.array([self.config["eft_prior"][param]["range"] for param in self.nonmarg_gauss_eft_parameters_list]).T
        self.marg_gauss_eft_parameters_list = [param for param, prior in self.config["eft_prior"].items() if prior["type"] == 'marg_gauss']
        self.marg_gauss_eft_parameters_prior_mean = np.array([self.config["eft_prior"][param]["mean"] for param in self.marg_gauss_eft_parameters_list]).T
        self.marg_gauss_eft_parameters_prior_sigma = np.array([self.config["eft_prior"][param]["range"] for param in self.marg_gauss_eft_parameters_list]).T
        self.marg_gauss_eft_parameters_prior_matrix = np.array([np.diagflat(1. / sigma**2) for sigma in self.marg_gauss_eft_parameters_prior_sigma])

        if self.config["get_fit"] and "fit_filename" not in self.config: 
            self.config["fit_filename"] = "./fit_bird"
            print("fit saved to (specified in \'fit_filename\' otherwise default): %s" % self.config["fit_filename"])

        # Loading data
        self.x, self.xmask, self.ydata, self.chi2data, self.invcov, self.invcovdata = [], [], [], [], [], []
        if self.config["with_redshift_bin"] and self.config["skycut"] > 1: self.config["zz"], self.config["nz"] = [], []
        
        self.xmax = 0.
        for i in range(self.config["skycut"]):

            if self.config.get("xmax") is None:
                xmax0, xmax1 = self.config["xmax0"][i], self.config["xmax1"][i]
                xmax = max(xmax0, xmax1)
            else: xmax, xmax0, xmax1 = self.config["xmax"][i], None, None

            if self.xmax < xmax: self.xmax = xmax # to increase the kmax to provide to pybird

            if self.config.get("xmin") is None:
                xmin0, xmin1 = self.config["xmin0"][i], self.config["xmin1"][i]
                xmin = min(xmin0, xmin1)
            else: xmin, xmin0, xmin1 = self.config["xmin"][i], None, None

            if self.config["with_bao"]: baoH, baoD = self.config["baoH"][i], self.config["baoD"][i]
            else: baoH, baoD = None, None

            if "Pk" in self.config["output"]:
                xi, xmaski, ydatai, chi2datai, invcovi, invcovdatai = self.__load_data_ps(
                    self.config["multipole"], self.config["wedge"],
                    self.data_directory, self.config["spectrum_file"][i], self.config["covmat_file"][i],
                    xmin=self.config["xmin"][i], xmax=xmax, xmax0=xmax0, xmax1=xmax1, xmaxspacing=self.config["xmaxspacing"], with_bao=self.config["with_bao"], baoH=baoH, baoD=baoD)
            else:
                xi, xmaski, ydatai, chi2datai, invcovi, invcovdatai = self.__load_data_cf(
                    self.config["multipole"], self.config["wedge"],
                    self.data_directory, self.config["spectrum_file"][i], self.config["covmat_file"][i],
                    xmax=self.config["xmax"][i], xmin=xmin, xmin0=xmin0, xmin1=xmin1, with_bao=self.config["with_bao"], baoH=baoH, baoD=baoD)

            if self.config["with_redshift_bin"]:  # BOSS
                try:
                    if "None" in self.config["density"][i]: zz, nz = [0.32], None
                    else:
                        try: z, _, _, nz = np.loadtxt(os.path.join(self.data_directory, self.config["density"][i]), unpack=True)
                        except: z, nz = np.loadtxt(os.path.join(self.data_directory, self.config["density"][i]), unpack=True)
                        nz /= np.trapz(nz, x=z)
                        zz = np.linspace(z[0], z[-1], 50)
                        nz = interp1d(z, nz, kind='cubic')(zz)
                        nz /= np.trapz(nz, x=zz)
                    if self.config["skycut"] > 1:
                        self.config["zz"].append(zz)
                        self.config["nz"].append(nz)
                    else:
                        self.config["zz"], self.config["nz"] = zz, nz
                except:
                    raise Exception('galaxy count distribution: %s not found!' % self.config["density"][i])

            # self.Nx.append(Nxi)
            self.x.append(xi)
            self.xmask.append(xmaski)
            self.ydata.append(ydatai)
            self.chi2data.append(chi2datai)
            self.invcov.append(invcovi)
            self.invcovdata.append(invcovdatai)

        # formatting configuration for pybird
        self.config["xdata"] = self.x
        if self.config["with_window"]:
            if self.config["skycut"] > 1:
                self.config["windowPk"] = [os.path.join(
                    self.data_directory, self.config["windowPk"][i]) for i in range(self.config["skycut"])]
                self.config["windowCf"] = [os.path.join(
                    self.data_directory, self.config["windowCf"][i]) for i in range(self.config["skycut"])]
            else:
                self.config["windowPk"] = os.path.join(self.data_directory, self.config["windowPk"][i])
                self.config["windowCf"] = os.path.join(self.data_directory, self.config["windowCf"][i])
        if "Pk" in self.config["output"]:
            self.config["kmax"] = self.xmax + 0.05

        print ("output: %s" % self.config["output"])
        print ("skycut: %s" % self.config["skycut"])
        print ("multipole: %s" % self.config["multipole"])
        print ("wedge: %s" % self.config["wedge"])

        # BBN prior?
        if self.config["with_bbn"] and self.config["omega_b_BBNcenter"] is not None and self.config["omega_b_BBNsigma"] is not None:
            print ('BBN prior on omega_b: on')
        else:
            self.config["with_bbn"] = False
            print ('BBN prior on omega_b: none')

        # setting pybird correlator configuration
        self.correlator = pb.Correlator()
        self.correlator.set(self.config)
        self.first_evaluation = True

        # setting classy for pybird
        log10kmax_classy = 0 
        if self.config["with_rs_marg"] or self.config["with_nnlo_counterterm"]: log10kmax_classy = 1 # slower, but useful for the wiggle-no-wiggle split
        self.need_cosmo_arguments(data, {'output': 'mPk', 'z_max_pk': max(self.config["z"]), 'P_k_max_h/Mpc': 10.**log10kmax_classy})
        self.kin = np.logspace(-5, log10kmax_classy, 200)

    def loglkl(self, cosmo, data):

        if self.config["with_derived_bias"]: data.derived_lkl = {}
        
        if self.first_evaluation: # if we run with zero varying cosmological parameter, we evaluate the model only once
            data.update_cosmo_arguments() 
            data.need_cosmo_update = True
        
        if data.need_cosmo_update: 
            self.correlator.compute(self.__set_cosmo(cosmo, data))
        
        free_eft_parameters_list = self.use_nuisance
        if self.config["with_rs_marg"]: free_eft_parameters_list = self.use_nuisance[1:]
        free_eft_parameters_list = np.array(free_eft_parameters_list).reshape(self.config["skycut"], -1)
        bdict = [] # list of dictionaries of free EFT parameters per sky
        for free_eft_parameters_list_per_sky in free_eft_parameters_list:
            bdict.append( {p: data.mcmc_parameters[k]['current'] * data.mcmc_parameters[k]['scale'] 
                for k in free_eft_parameters_list_per_sky for p in self.eft_parameters_list if p in k} )
        for i in range(self.config["skycut"]): bdict[i].update({p: 0. for p in self.eft_parameters_list if p not in bdict[i]})

        if not self.config["nonmarg"]:
            chi2 = 0.
            correlator = self.correlator.get(bdict) 
            marg_correlator = self.correlator.getmarg(bdict, self.marg_gauss_eft_parameters_list)
            for i in range(self.config["skycut"]):
                if self.config["skycut"] == 1: modelX = correlator
                elif self.config["skycut"] > 1: modelX = correlator[i]
                chi2_i, bg_i = self.__get_chi2(modelX, cosmo, data, marg=True, marg_correlator=marg_correlator, i=i)
                chi2 += chi2_i
                if self.config["get_chi2_from_marg"]: bdict[i].update({p: b for p, b in zip(self.marg_gauss_eft_parameters_list, bg_i)})

        if self.config["get_chi2_from_marg"] or self.config["nonmarg"]: 
            chi2 = 0.
            nonmarg_correlator = self.correlator.get(bdict)
            for i in range(self.config["skycut"]):
                if self.config["skycut"] == 1: modelX = nonmarg_correlator
                elif self.config["skycut"] > 1: modelX = nonmarg_correlator[i]
                chi2_i, _ = self.__get_chi2(modelX, cosmo, data, marg=False, i=i)
                chi2 += chi2_i
                chi2 += self.__set_prior(np.array([bdict[i][param] for param in self.marg_gauss_eft_parameters_list]), 
                    self.marg_gauss_eft_parameters_prior_mean[i], self.marg_gauss_eft_parameters_prior_sigma[i])

            if self.first_evaluation:
                if self.config["get_fake"]: self.get_fake(cosmo, data, bdict, nonmarg_correlator)
                if self.config["get_fit"]: self.get_fit(cosmo, data, bdict, nonmarg_correlator) # this doesn't work yet with (varying-kmax) wedges

        prior = 0.
        if len(self.nonmarg_gauss_eft_parameters_list) > 0: 
            for i in range(self.config["skycut"]): 
                prior += self.__set_prior(np.array([bdict[i][param] for param in self.nonmarg_gauss_eft_parameters_list]), 
                    self.nonmarg_gauss_eft_parameters_prior_mean[i], self.nonmarg_gauss_eft_parameters_prior_sigma[i])
        if self.config["with_tidal_alignments"]:
            for i in range(self.config["skycut"]): 
                prior += ( (bdict[i]["bq"]+0.05)/0.05 )**2

        if self.config["with_bbn"]: 
            prior += ((data.cosmo_arguments['omega_b'] - self.config["omega_b_BBNcenter"]) / self.config["omega_b_BBNsigma"])**2

        if self.first_evaluation: self.first_evaluation = False
        
        lkl = - 0.5 * ( chi2 + prior )
        return lkl

    def __set_prior(self, x, mean, sigma):
        return np.sum(((x-mean)/sigma)**2)

    def __get_chi2(self, modelX, cosmo, data, marg=True, marg_correlator=None, i=0):

        modelX = modelX.reshape(-1)

        if self.config["with_bao"] and self.config["baoH"][i] > 0 and self.config["baoD"][i] > 0:  # BAO
            DM_at_z = cosmo.angular_distance(self.config["zbao"][i]) * (1. + self.config["zbao"][i])
            H_at_z = cosmo.Hubble(self.config["zbao"][i]) * conts.c / 1000.0
            rd = cosmo.rs_drag() * self.config["rs_rescale"][i]
            theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.config["rd_fid_in_Mpc"][i]
            theo_H_rd_by_rdfid = H_at_z * rd / self.config["rd_fid_in_Mpc"][i]
            modelX = np.concatenate((modelX, [theo_H_rd_by_rdfid, theo_DM_rdfid_by_rd_in_Mpc]))

        modelX = modelX[self.xmask[i]]

        if marg:
            if self.config["skycut"] is 1: Pi = self.__get_Pi_for_marg(marg_correlator, self.xmask[i])
            elif self.config["skycut"] > 1: Pi = self.__get_Pi_for_marg(marg_correlator[i], self.xmask[i])
            # chi2, bg = self.__get_chi2_marg(modelX, Pi, self.invcov[i], self.ydata[i], self.priormat[i], data, isky=i)
            chi2, bg = self.__get_chi2_marg(modelX, Pi, self.invcov[i], self.invcovdata[i], self.chi2data[i], 
                self.marg_gauss_eft_parameters_prior_matrix[i], self.marg_gauss_eft_parameters_prior_mean[i], data, isky=i)
        else: 
            chi2 = self.__get_chi2_non_marg(modelX, self.invcov[i], self.ydata[i])
            bg = None

        return chi2, bg # chi^2, b_gaussian 

    def __get_chi2_non_marg(self, modelX, invcov, ydata):
        chi2 = np.einsum('k,p,kp->', modelX-ydata, modelX-ydata, invcov)
        return chi2

    def __get_chi2_marg(self, modelX, Pi, invcov, invcovdata, chi2data, priormat, priormean, data, isky=0):
        Covbi = np.dot(Pi, np.dot(invcov, Pi.T)) + priormat
        Cinvbi = np.linalg.inv(Covbi)
        vectorbi = np.dot(modelX, np.dot(invcov, Pi.T)) - np.dot(invcovdata, Pi.T)
        chi2nomar = np.dot(modelX, np.dot(invcov, modelX)) - 2. * np.dot(invcovdata, modelX) + chi2data
        vectorbi -= np.dot(priormean, priormat)
        chi2nomar += np.dot(priormean, np.dot(priormat, priormean))
        chi2mar = - np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.abs(np.linalg.det(Covbi)))
        chi2tot = chi2mar + chi2nomar - priormat.shape[0] * np.log(2. * np.pi)
        bg = - np.dot(Cinvbi, vectorbi)

        if self.config["with_derived_bias"]:
            for i, elem in enumerate(data.get_mcmc_parameters(['derived_lkl'])):
                if i >= isky * len(bg) and i < (isky + 1) * len(bg):
                    data.derived_lkl[elem] = bg[i - isky * len(bg)]

        return chi2tot, bg

    # def __get_chi2_marg(self, Png, Pg, invcov, ydata, priormat, montepython_data, isky=0): # slower because of the einsum

    #     F2 = np.einsum('ak,bp,kp->ab', Pg, Pg, invcov) + priormat
    #     invF2 = np.linalg.inv(F2)
    #     F1 = np.einsum('ak,p,kp->a', Pg, Png-ydata, invcov)
    #     F0 = self.__get_chi2_non_marg(Png, invcov, ydata) 
    #     chi2 = F0 - np.einsum('a,b,ab->', F1, F1, invF2) + np.log(np.linalg.det(F2)) 
    #     bg = - np.einsum('a,ab->b', F1, invF2) 

    #     if self.config["with_derived_bias"]:
    #         for i, elem in enumerate(montepython_data.get_mcmc_parameters(['derived_lkl'])):
    #             if i >= isky * len(bg) and i < (isky + 1) * len(bg):
    #                 montepython_data.derived_lkl[elem] = bg[i - isky * len(bg)]

    #     return chi2, bg

    def __get_Pi_for_marg(self, marg_correlator, xmask):
        Pi = marg_correlator
        if self.config["with_bao"]:  # BAO
            newPi = np.zeros(shape=(Pi.shape[0], Pi.shape[1] + 2))
            newPi[:Pi.shape[0], :Pi.shape[1]] = Pi
            Pi = 1. * newPi
        Pi = Pi[:, xmask]
        return Pi

    def __set_cosmo(self, M, data):

        zfid = self.config["z"][0]

        cosmo = {}

        cosmo["k11"] = self.kin  # k in h/Mpc
        cosmo["P11"] = np.array([M.pk_lin(k * M.h(), zfid) * M.h()**3 for k in self.kin])  # P(k) in (Mpc/h)**3

        if self.config["skycut"] == 1:
            # if self.config["multipole"] is not 0:
            cosmo["f"] = M.scale_independent_growth_factor_f(zfid)
            if self.config["with_exact_time"] or self.config["with_quintessence"]:
                cosmo["z"] = self.config["z"][0]
                cosmo["Omega0_m"] = M.Omega0_m()
                try: cosmo["w0_fld"] = data.cosmo_arguments['w0_fld']
                except: pass

            if self.config["with_AP"]:
                cosmo["DA"] = M.angular_distance(zfid) * M.Hubble(0.)
                cosmo["H"] = M.Hubble(zfid) / M.Hubble(0.)

        elif self.config["skycut"] > 1:
            # if self.config["multipole"] is not 0:
            cosmo["f"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["z"]])
            cosmo["D"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["z"]])

            if self.config["with_AP"]:
                cosmo["DA"] = np.array([M.angular_distance(z) * M.Hubble(0.) for z in self.config["z"]])
                cosmo["H"] = np.array([M.Hubble(z) / M.Hubble(0.) for z in self.config["z"]])

        if self.config["with_redshift_bin"]:
            def comoving_distance(z): return M.angular_distance(z) * (1+z) * M.h()
            if self.config["skycut"] == 1:
                cosmo["D"] = M.scale_independent_growth_factor(self.config["z"][0])
                cosmo["Dz"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["zz"]])
                cosmo["fz"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["zz"]])
                cosmo["rz"] = np.array([comoving_distance(z) for z in self.config["zz"]])

            elif self.config["skycut"] > 1:
                cosmo["Dz"] = np.array([ [M.scale_independent_growth_factor(z) for z in zz] for zz in self.config["zz"] ])
                cosmo["fz"] = np.array([ [M.scale_independent_growth_factor_f(z) for z in zz] for zz in self.config["zz"] ])
                cosmo["rz"] = np.array([ [comoving_distance(z) for z in zz] for zz in self.config["zz"] ])

        if self.config["with_quintessence"]: 
            # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum. 
            # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
            # This does not work for multi skycuts nor for redshift bins.
            zm = 5. # z in matter domination
            def scale_factor(z): return 1/(1.+z)
            Omega0_m = cosmo["Omega0_m"]
            w = cosmo["w0_fld"]
            GF = pb.GreenFunction(Omega0_m, w=w, quintessence=True)
            Dq = GF.D(scale_factor(zfid)) / GF.D(scale_factor(zm))
            Dm = M.scale_independent_growth_factor(zfid) / M.scale_independent_growth_factor(zm)
            cosmo["P11"] *= Dq**2 / Dm**2 * ( 1 + (1+w)/(1.-3*w) * (1-Omega0_m)/Omega0_m * (1+zm)**(3*w) )**2 # 1611.07966 eq. (4.15)
            cosmo["f"] = GF.fplus(1/(1.+cosmo["z"]))

        # wiggle-no-wiggle split # algo: 1003.3999; details: 2004.10607
        def get_smooth_wiggle_resc(kk, pk, alpha_rs=1.): # k [h/Mpc], pk [(Mpc/h)**3]
            kp = np.linspace(1.e-7, 7, 2**16)   # 1/Mpc
            ilogpk = interp1d(np.log(kk * M.h()), np.log(pk / M.h()**3), fill_value="extrapolate") # Mpc**3
            lnkpk = np.log(kp) + ilogpk(np.log(kp))
            harmonics = dst(lnkpk, type=2, norm='ortho')
            odd, even = harmonics[::2], harmonics[1::2]
            nn = np.arange(0, odd.shape[0], 1)
            nobao = np.delete(nn, np.arange(120, 240,1))
            smooth_odd = interp1d(nn, odd, kind='cubic')(nobao)
            smooth_even = interp1d(nn, even, kind='cubic')(nobao)
            smooth_odd = interp1d(nobao, smooth_odd, kind='cubic')(nn)
            smooth_even = interp1d(nobao, smooth_even, kind='cubic')(nn)
            smooth_harmonics =  np.array([[o, e] for (o, e) in zip(smooth_odd, smooth_even)]).reshape(-1)
            smooth_lnkpk = dst(smooth_harmonics, type=3, norm='ortho')
            smooth_pk = np.exp(smooth_lnkpk) / kp
            wiggle_pk = np.exp(ilogpk(np.log(kp))) - smooth_pk
            spk = interp1d(kp, smooth_pk, bounds_error=False)(kk * M.h()) * M.h()**3 # (Mpc/h)**3
            wpk_resc = interp1d(kp, wiggle_pk, bounds_error=False)(alpha_rs * kk * M.h()) * M.h()**3 # (Mpc/h)**3 # wiggle rescaling
            kmask = np.where(kk < 1.02)[0]
            return kk[kmask], spk[kmask], spk[kmask]+wpk_resc[kmask]

        if self.config["with_nnlo_counterterm"]: 
            cosmo["k11"], cosmo["Psmooth"], cosmo["P11"] = get_smooth_wiggle_resc(cosmo["k11"], cosmo["P11"])

        if self.config["with_rs_marg"]:
            # wiggle rescaling parameter
            alpha_rs = data.mcmc_parameters['alpha_rs']['current'] * data.mcmc_parameters['alpha_rs']['scale']
            cosmo["k11"], _, cosmo["P11"] = get_smooth_wiggle_resc(cosmo["k11"], cosmo["P11"], alpha_rs=alpha_rs)

        return cosmo

    def __load_data_ps(self, multipole, wedge, data_directory, spectrum_file, covmat_file, xmin, xmax=None, xmax0=None, xmax1=None, xmaxspacing='default', with_bao=False, baoH=None, baoD=None):

        # cov = None
        # try:
        xdata, ydata = self.__load_spectrum(data_directory, spectrum_file)  # read values of k (in h/Mpc)
        # except: xdata, ydata, cov = self.__load_gaussian_spectrum(data_directory, spectrum_file) # with gaussian case: column 1: k[h/Mpc]  column 2-N+2: signal  column N+3-2N+2: error
        cov = np.loadtxt(os.path.join(data_directory, covmat_file))
        
        if wedge is not 0:
            x = xdata.reshape(wedge, -1)[0]
            Nx = len(x)

            if xmax0 is not None and xmax1 is not None:
                xmax = max(xmax0, xmax1)
            elif xmax is not None:
                xmax0 = xmax
                xmax1 = xmax

            xmask0 = np.argwhere((x <= xmax0) & (x >= xmin))[:, 0]
            xmask = xmask0

            if 'linear' in xmaxspacing:
                dxmax = (xmax1 - xmax0) / (wedge - 1.)
                for i in range(wedge - 1):
                    xmaski = np.argwhere((x <= xmax0 + (i + 1) * dxmax) & (x >= xmin))[:, 0] + (i + 1) * Nx
                    xmask = np.concatenate((xmask, xmaski))
            elif 'formula0' in xmaxspacing:
                def get_xmax(k0, k1, N=wedge):
                    a = ((k0 - k1) * (-1 + 2 * N)**2) / (16. * (-1 + N) * N**3)
                    b = -(k0 - k1 + 4 * k1 * N - 4 * k1 * N**2) / (4. * (-1 + N) * N)
                    mu = (np.arange(0, N, 1) + 0.5) / N
                    return a / mu**2 + b
                xmaxs = get_xmax(xmax0, xmax1)
                for i, xmaxi in enumerate(xmaxs[1:]):
                    xmaski = np.argwhere((x <= xmaxi) & (x >= xmin))[:, 0] + (i + 1) * Nx
                    xmask = np.concatenate((xmask, xmaski))
            elif 'optimal' in xmaxspacing: # optimal
                cov_resh = cov.reshape((3, cov.shape[0] // 3, 3, cov.shape[1] // 3))
                mat = np.array([[1., -3./7., 11./56.], [1., -3/8., 15/128.], [1., 3/8., -15./128.]]) # PA + w1 + w2
                ydata = np.einsum('al,lk->ak', mat, ydata.reshape(3,-1)).reshape(-1) # rotate the data to optimal basis for minimal theoretical error
                cov = np.einsum('al,bm,lkmj->akbj', mat, mat, cov_resh).reshape(cov.shape) # rotate the covariance to optimal basis for minimal theoretical error
                err = np.sqrt(np.diag(cov)).reshape(3, -1)
                masksigma = np.argwhere((x <= 0.18) & (x >= 0.12))[:, 0]
                ratiosigmaB = np.mean(err[1, masksigma]) / np.mean(err[0, masksigma])
                ratiosigmaC = np.mean(err[2, masksigma]) / np.mean(err[0, masksigma])
                ratioknl2 = 8.
                D0 = 1. + (1/5.  + 1/7.  ) * ratioknl2**2
                D2 = 0. + (4/7.  + 10/21.) * ratioknl2**2
                D4 = 0. + (8/35. + 24/77.) * ratioknl2**2
                D = np.array([D0, D2, D4])
                kmaxA = xmax
                kmaxB = kmaxA * ratiosigmaB**0.25 * np.einsum('a,a->', mat[1], D)**(-0.25)
                kmaxC = kmaxA * ratiosigmaC**0.25 * np.einsum('a,a->', mat[2], D)**(-0.25)
                with np.printoptions(precision=2, suppress=True): 
                    print ('kmax: ', np.array([kmaxA, kmaxB, kmaxC]))
                    print ('errB/errA, errC/errA', np.array([ratiosigmaB, ratiosigmaC]))
                def getmask(kmax, i): return np.argwhere((x <= kmax) & (x >= xmin))[:, 0] + i * Nx
                kmask = []
                for i, kmax in enumerate([kmaxA, kmaxB, kmaxC]): kmask.append(getmask(kmax, i))
                xmask = np.concatenate((kmask))
            elif 'great' in xmaxspacing: # great
                cov_resh = cov.reshape((3, cov.shape[0] // 3, 3, cov.shape[1] // 3))
                mat = np.array([[1., -3./7., 11./56.], [1., -3/8., 15/128.], [1., 3/8., -15./128.]]) # PA + w1 + w2
                ydata = np.einsum('al,lk->ak', mat, ydata.reshape(3,-1)).reshape(-1) # rotate the data to optimal basis for minimal theoretical error
                cov = np.einsum('al,bm,lkmj->akbj', mat, mat, cov_resh).reshape(cov.shape) # rotate the covariance to optimal basis for minimal theoretical error
                err = np.sqrt(np.diag(cov)).reshape(3, -1)
                masksigmaB = np.argwhere((x <= 0.3) & (x >= 0.2))[:, 0]
                ratiosigmaB = np.mean(err[1, masksigmaB]) / np.mean(err[0, masksigmaB])
                masksigmaC = np.argwhere((x <= 0.2) & (x >= 0.1))[:, 0]
                ratiosigmaC = np.mean(err[2, masksigmaC]) / np.mean(err[0, masksigmaC])
                ratioknl2 = 8.
                kmaxA = xmax
                kmaxB = kmaxA * ratiosigmaB**0.25 * (1 + 3 * ratioknl2 * 0.0834097)**(-0.25)
                kmaxC = kmaxA * ratiosigmaC**0.25 * (1 + 3 * ratioknl2 * 0.773912)**(-0.25)
                with np.printoptions(precision=2, suppress=True): 
                    print ('kmax: ', np.array([kmaxA, kmaxB, kmaxC]))
                    print ('errB/errA, errC/errA', np.array([ratiosigmaB, ratiosigmaC]))
                def getmask(kmax, i): 
                    return np.argwhere((x <= kmax) & (x >= xmin))[:, 0] + i * Nx
                kmask = []
                for i, kmax in enumerate([kmaxA, kmaxB, kmaxC]): kmask.append(getmask(kmax, i))
                xmask = np.concatenate((kmask))

        elif multipole != 0:
            x = xdata.reshape(3, -1)[0]
            Nx = len(x)
            xmask0 = np.argwhere((x <= xmax) & (x >= xmin))[:, 0]
            xmask = xmask0
            for i in range(multipole - 1):
                xmaski = np.argwhere((x <= xmax) & (x >= xmin))[:, 0] + (i + 1) * Nx
                xmask = np.concatenate((xmask, xmaski))

        elif multipole == 0:
            x = xdata
            xmask = np.argwhere((x <= xmax) & (x >= xmin))[:, 0]
            xmask0 = xmask

        xdata = x[xmask0]
        ydata = ydata[xmask]

        # BAO
        if with_bao and baoH > 0 and baoD > 0:
            ydata = np.concatenate((ydata, [baoH, baoD]))
            xmask = np.concatenate((xmask, [-2, -1]))
            print ("BAO recon: on")
        else:
            print ("BAO recon: none")

        covred = cov[xmask.reshape((len(xmask), 1)), xmask]
        invcov = np.linalg.inv(covred)

        chi2data = np.dot(ydata, np.dot(invcov, ydata))
        invcovdata = np.dot(ydata, invcov)

        return x, xmask, ydata, chi2data, invcov, invcovdata

    def __load_data_cf(self, multipole, wedge, data_directory, spectrum_file, covmat_file, xmax, xmin=None, xmin0=None, xmin1=None, xminspacing='default', with_bao=False, baoH=None, baoD=None):

        # cov = None
        # try:
        xdata, ydata = self.__load_spectrum(data_directory, spectrum_file)  # read values of k (in h/Mpc)
        # except: xdata, ydata, cov = self.__load_gaussian_spectrum(data_directory, spectrum_file) # with gaussian case: column 1: k[h/Mpc]  column 2-N+2: signal  column N+3-2N+2: error

        if wedge is not 0:
            x = xdata.reshape(wedge, -1)[0]
            Nx = len(x)

            if xmin0 is not None and xmin1 is not None:
                xmin = min(xmin0, xmin1)
            elif xmin is not None:
                xmin0 = xmin
                xmin1 = xmin

            xmask0 = np.argwhere((x <= xmax) & (x >= xmin0))[:, 0]
            xmask = xmask0

            if 'linear' in xminspacing:
                dxmax = (xmin1 - xmin0) / (wedge - 1.)
                for i in range(wedge - 1):
                    xmaski = np.argwhere((x >= xmin0 + (i + 1) * dxmax) & (x <= xmax))[:, 0] + (i + 1) * Nx
                    xmask = np.concatenate((xmask, xmaski))

            else:
                def get_xmin(s0, s1, N=wedge):
                    a = ((s1 - s0) * (-1 + 2 * N)**2) / (16. * (-1 + N) * N**3)
                    b = -(s1 - s0 + 4 * s0 * N - 4 * s0 * N**2) / (4. * (-1 + N) * N)
                    mu = (np.arange(0, N, 1) + 0.5) / N
                    return a / mu**2 + b
                xmins = get_xmin(xmin1, xmin0)
                for i, xmini in enumerate(xmins[1:]):
                    xmaski = np.argwhere((x >= xmini) & (x <= xmax))[:, 0] + (i + 1) * Nx
                    xmask = np.concatenate((xmask, xmaski))

                print (xmins)

        elif multipole is not 0:
            x = xdata.reshape(3, -1)[0]
            Nx = len(x)
            xmask0 = np.argwhere((x <= xmax) & (x >= xmin))[:, 0]
            xmask = xmask0
            for i in range(multipole - 1):
                xmaski = np.argwhere((x <= xmax) & (x >= xmin))[:, 0] + (i + 1) * Nx
                xmask = np.concatenate((xmask, xmaski))

        xdata = x[xmask0]
        ydata = ydata[xmask]

        # BAO
        if with_bao and baoH > 0 and baoD > 0:
            ydata = np.concatenate((ydata, [baoH, baoD]))
            xmask = np.concatenate((xmask, [-2, -1]))
            print ("BAO recon: on")
        else:
            print ("BAO recon: none")

        # if cov is None:
        cov = np.loadtxt(os.path.join(data_directory, covmat_file))
        covred = cov[xmask.reshape((len(xmask), 1)), xmask]
        invcov = np.linalg.inv(covred)

        chi2data = np.dot(ydata, np.dot(invcov, ydata))
        invcovdata = np.dot(ydata, invcov)

        return x, xmask, ydata, chi2data, invcov, invcovdata

    def __load_spectrum(self, data_directory, spectrum_file):
        fname = os.path.join(data_directory, spectrum_file)
        kPS, PSdata = np.loadtxt(fname, usecols=(0,1), unpack=True)
        return kPS, PSdata

    def __load_gaussian_spectrum(self, data_directory, spectrum_file):
        """
        Helper function to read in the full data vector with gaussian error:
        column 1: k[h/Mpc]  column 2-N+2: signal  column N+3-2N+2: error
        """
        if self.config["wedge"] == 0: Nd = self.config["multipole"]
        else: Nd = self.config["wedge"]
        raw = np.loadtxt(os.path.join(data_directory, spectrum_file)).T
        k = raw[0]
        allk = np.concatenate([k for i in range(Nd)])
        allPS = np.concatenate([raw[1 + i] for i in range(Nd)])
        diag = np.concatenate([raw[1 + Nd + i] for i in range(Nd)])
        cov = np.diagflat(diag**2)
        # kPS = np.vstack([allkpt, allwpt]).T
        return allk, allPS, cov

    def get_fake(self, cosmo, data, bdict, nonmarg_correlator):
        best_fit_string = "best fit: "
        for key, value in cosmo.get_current_derived_parameters(["Omega_m", "h", "A_s", "n_s", "sigma8"]).items(): 
            best_fit_string += "%s: %.4e, " % (key, value)
        best_fit_string += "\n"
        if self.config["skycut"] == 1: 
            theo_correlator = nonmarg_correlator.reshape(-1)
            for key, value in bdict[0].items(): best_fit_string += "%s: %.4f, " % (key, value)
            best_fit_string += "\n"
            np.savetxt(
                '%s.dat' % self.config["fake_filename"], 
                np.vstack([ np.concatenate([self.x[0] for l in range(3)]), np.pad(nonmarg_correlator.reshape(-1), (0, (3-self.config["multipole"])*len(self.x[0])), mode='constant', constant_values=0.) ]).T,  
                header=best_fit_string + "\n k [h/Mpc], Pfake_l",
                fmt="%.4f %.6e",
            )
        elif self.config["skycut"] > 1: 
            for i in range(self.config["skycut"]):
                best_fit_string_i = deepcopy(best_fit_string)
                for key, value in bdict[i].items(): best_fit_string_i += "%s: %.4f, " % (key, value)
                best_fit_string_i += "\n"
                np.savetxt(
                    '%s.dat' % self.config["fake_filename"][i], 
                    np.vstack([ np.concatenate([self.x[i] for l in range(3)]), np.pad(nonmarg_correlator[i].reshape(-1), (0, (3-self.config["multipole"])*len(self.x[i])), mode='constant', constant_values=0.) ]).T, 
                    header=best_fit_string_i + "\n k [h/Mpc], Pfake_l",
                    fmt="%.4f %.6e",
                )

    def get_fit(self, cosmo, data, bdict, nonmarg_correlator):
        ndatapoints = len(np.concatenate([self.ydata[i].reshape(-1) for i in range(self.config["skycut"])]))
        nparams = len(data.get_mcmc_parameters(['varying'])) + len(data.get_mcmc_parameters(['derived_lkl']))
        dof = ndatapoints - nparams
        best_fit_string = "Note: in dev... these numbers might not be correct depending on the options used... please refer to the specific implementation. "
        best_fit_string += "chi2 = %.3f, dof = %.0f-%.0f, chi2/dof = %.3f, pvalue = %.4f \n" % (chi2, ndatapoints, nparams, chi2/dof, pvalue(chi2, dof))
        best_fit_string += "best fit: "
        for key, value in cosmo.get_current_derived_parameters(["Omega_m", "h", "A_s", "n_s", "sigma8"]).items(): 
            best_fit_string += "%s: %.4e, " % (key, value)
        best_fit_string += "\n"
        if self.config["skycut"] == 1: 
            theo_correlator = nonmarg_correlator.reshape(-1)[self.xmask[0]]
            data_correlator = self.ydata[0].reshape(-1)
            data_error = np.sqrt(np.diag(np.linalg.inv(self.invcov[0])))
            for key, value in bdict[0].items(): best_fit_string += "%s: %.4f, " % (key, value)
            best_fit_string += "\n"
            kconc = np.concatenate([self.x[0] for l in range(self.config["multipole"])])[self.xmask[0]]
            np.savetxt(
                '%s.dat' % self.config["fit_filename"], 
                np.vstack([ kconc, theo_correlator, data_correlator, data_error ]).T, 
                header=best_fit_string + "\n k [h/Mpc], Pfit_l, Pdata_l, sigmaPdata_l",
                fmt="%.4f %.6e %.6e %.6e",
            )
        elif self.config["skycut"] > 1: 
            for i in range(self.config["skycut"]):
                theo_correlator = nonmarg_correlator[i].reshape(-1)[self.xmask[i]]
                data_correlator = self.ydata[i].reshape(-1)
                data_error = np.sqrt(np.diag(np.linalg.inv(self.invcov[i])))
                best_fit_string_i = deepcopy(best_fit_string)
                for key, value in bdict[i].items(): best_fit_string_i += "%s: %.4f, " % (key, value)
                best_fit_string_i += "\n"
                kconc = np.concatenate([self.x[i] for l in range(self.config["multipole"])])[self.xmask[i]]
                np.savetxt(
                    '%s.dat' % self.config["fit_filename"][i], 
                    np.vstack([kconc, theo_correlator, data_correlator, data_error ]).T, 
                    header=best_fit_string_i + "\n k [h/Mpc], Pfit_l, Pdata_l, sigmaPdata_l",
                    fmt="%.4f %.6e %.6e %.6e",
                )




# #################################################
# #################################################

# class Likelihood_cfps(Likelihood):

#     def __init__(self, path, data, command_line):

#         Likelihood.__init__(self, path, data, command_line)

#         # try:
#         #     with open(self.configfile, 'r') as f:
#         #         self.config = yaml.full_load(f)
#         # except OSError:
#         #     newpath = os.path.join(self.data_directory, self.configfile)
#         #     print("WARNING: We are loading the configuration in %s. Please make sure this is what you want!", newpath)
#         #     with open(newpath, 'r') as f:
#         #         self.config = yaml.full_load(f)

#         self.config = yaml.full_load(open(os.path.join(self.data_directory, self.configfile), 'r'))
#         self.config_ps = yaml.full_load(open(os.path.join(self.data_directory, self.configfile_ps), 'r'))

#         try: self.config["with_derived_bias"]
#         except: self.config["with_derived_bias"] = False
#         try: self.config["with_quintessence"]
#         except: self.config["with_quintessence"] = False
#         try: self.config["get_chi2_from_marg"]
#         except: self.config["get_chi2_from_marg"] = False
#         try: self.config["split"]
#         except: self.config["split"] = 0

#         print ("skycut: %s" % self.config["skycut"])

#         self.s = []
#         self.k = []
#         self.xmask = []
#         self.ydata = []
#         self.chi2data = []
#         self.invcov = []
#         self.invcovdata = []
#         self.priormat = []

#         for i in range(self.config["skycut"]):

#             if self.config["with_bao"]:
#                 baoH = self.config["baoH"][i]
#                 baoD = self.config["baoD"][i]
#             else:
#                 baoH = None
#                 baoD = None

#             si, ki, xmaski, ydatai, chi2datai, invcovi, invcovdatai = self.__load_data(
#                 self.config["multipole"], self.data_directory, self.config["spectrum_file"][i], self.config_ps["spectrum_file"][i], self.config["covmat_file"][i],
#                 smin=self.config["xmin"][i], smax=self.config["xmax"][i], kmin=self.config_ps["xmin"][i], kmax=self.config_ps["xmax"][i], 
#                 with_bao=self.config["with_bao"], baoH=baoH, baoD=baoD)

#             priormati = self.__set_prior(self.config_ps["multipole"])

#             self.s.append(si)
#             self.k.append(ki)
#             self.xmask.append(xmaski)
#             self.ydata.append(ydatai)
#             self.chi2data.append(chi2datai)
#             self.invcov.append(invcovi)
#             self.invcovdata.append(invcovdatai)
#             self.priormat.append(priormati)

#         # formatting configuration for pybird
#         self.config["xdata"] = self.s
#         self.config_ps["xdata"] = self.k

#         if self.config_ps["with_window"]:
#             if self.config_ps["skycut"] > 1:
#                 self.config_ps["windowPk"] = [os.path.join(self.data_directory, self.config_ps["windowPk"][i]) for i in range(self.config_ps["skycut"])]
#                 self.config_ps["windowCf"] = [os.path.join(self.data_directory, self.config_ps["windowCf"][i]) for i in range(self.config_ps["skycut"])]
#             else:
#                 self.config_ps["windowPk"] = os.path.join(self.data_directory, self.config_ps["windowPk"][i])
#                 self.config_ps["windowCf"] = os.path.join(self.data_directory, self.config_ps["windowCf"][i])
#         if "Pk" in self.config_ps["output"]: self.config_ps["kmax"] = self.config_ps["xmax"][0] + 0.05

#         # BBN prior?
#         if self.config["with_bbn"] and self.config["omega_b_BBNcenter"] is not None and self.config["omega_b_BBNsigma"] is not None:
#             print ('BBN prior on omega_b: on')
#         else:
#             self.config["with_bbn"] = False
#             print ('BBN prior on omega_b: none')

#         # setting pybird correlator configuration
#         self.correlator = pb.Correlator()
#         self.correlator.set(self.config)
#         self.correlator_ps = pb.Correlator()
#         self.correlator_ps.set(self.config_ps)

#         # setting classy for pybird
#         self.need_cosmo_arguments(data, {'output': 'mPk', 'z_max_pk': max(self.config["z"]), 'P_k_max_h/Mpc': 1.})
#         self.kin = np.logspace(-5, 0, 200)

#     def bias_array_to_dict(self, bs, with_stoch=False):
#         if with_stoch: bdict = {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3], "cct": bs[4], "cr1": bs[5], "ce0": bs[7], "ce1": bs[8], "ce2": bs[9]}
#         else: bdict = {"b1": bs[0], "b2": bs[1], "b3": bs[2], "b4": bs[3], "cct": bs[4], "cr1": bs[5]}
#         return bdict

#     def bias_custom_to_all(self, bs):
#         # b1, b2, b3, b4, cct, cr1, cr2, ce0, ce1, ce2, bq (opt), bnlo (opt) # bnlo is actually b_nnlo * k^4 P11
#         ball = [bs[0], bs[1] / np.sqrt(2.), 0., bs[1] / np.sqrt(2.), 0., 0., 0., 0., 0., 0., 0., 0.]
#         return ball

#     def bias_nonmarg_to_all(self, bs, bg, with_stoch=False): 
#         if with_stoch: biaslist = [bs[0], bs[1] / np.sqrt(2.), bg[0], bs[1] / np.sqrt(2.), bg[1], bg[2], 0., bg[4], 0., bg[3]]
#         else: biaslist = [bs[0], bs[1] / np.sqrt(2.), bg[0], bs[1] / np.sqrt(2.), bg[1], bg[2], 0., 0., 0., 0.]
#         return biaslist

#     def loglkl(self, cosmo, data):

#         if data.need_cosmo_update: 
#             self.correlator.compute(self.__set_cosmo(cosmo, data))
#             self.correlator_ps.compute(self.__set_cosmo(cosmo, data))
#         else: pass

#         bng = np.array([data.mcmc_parameters[k]['current'] * data.mcmc_parameters[k]['scale'] for k in self.use_nuisance])
        
#         if self.config["split"] == 1:
#             bng = np.swapaxes( bng.reshape(self.config["skycut"], 2, -1), axis1=0, axis2=1 )
#             bdict = np.array([self.bias_array_to_dict(self.bias_custom_to_all(bs), with_stoch=False) for bs in bng[0]])
#             bdict_ps = np.array([self.bias_array_to_dict(self.bias_custom_to_all(bs), with_stoch=True) for bs in bng[1]])

#             correlator = self.correlator.get(bdict)
#             correlator_ps = self.correlator_ps.get(bdict_ps)
#             marg_correlator = self.correlator.getmarg(bdict, model=self.config["model"])
#             marg_correlator_ps = self.correlator_ps.getmarg(bdict_ps, model=self.config_ps["model"])
#         else:
#             bng = bng.reshape(self.config["skycut"], -1)
#             bdict = np.array([self.bias_array_to_dict(self.bias_custom_to_all(bs), with_stoch=False) for bs in bng])
#             bdict_ps = np.array([self.bias_array_to_dict(self.bias_custom_to_all(bs), with_stoch=True) for bs in bng])

#             correlator = self.correlator.get(bdict)
#             correlator_ps = self.correlator_ps.get(bdict_ps)
#             marg_correlator = self.correlator.getmarg(bdict, model=self.config["model"])
#             marg_correlator_ps = self.correlator_ps.getmarg(bdict, model=self.config_ps["model"])


#         chi2 = 0.
#         if self.config["get_chi2_from_marg"]: bg = []

#         for i in range(self.config["skycut"]):
#             if self.config["skycut"] is 1: 
#                 modelcf = correlator 
#                 modelps = correlator_ps
#             elif self.config["skycut"] > 1: 
#                 modelcf = correlator[i]
#                 modelps = correlator_ps[i]
#             modelX = np.concatenate(( modelcf.reshape(-1), modelps.reshape(-1) ))
            
#             if self.config["with_bao"] and self.config["baoH"][i] > 0 and self.config["baoD"][i] > 0:  # BAO
#                 DM_at_z = cosmo.angular_distance(self.config["zbao"][i]) * (1. + self.config["zbao"][i])
#                 H_at_z = cosmo.Hubble(self.config["zbao"][i]) * conts.c / 1000.0
#                 rd = cosmo.rs_drag() * self.config["rs_rescale"][i]
#                 theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.config["rd_fid_in_Mpc"][i]
#                 theo_H_rd_by_rdfid = H_at_z * rd / self.config["rd_fid_in_Mpc"][i]
#                 modelX = np.concatenate((modelX, [theo_H_rd_by_rdfid, theo_DM_rdfid_by_rd_in_Mpc]))
            
#             modelX = modelX[self.xmask[i]]
            
#             if self.config["skycut"] is 1: Pi = self.__get_Pi_for_marg(marg_correlator, marg_correlator_ps, self.xmask[i])
#             elif self.config["skycut"] > 1: Pi = self.__get_Pi_for_marg(marg_correlator[i], marg_correlator_ps[i], self.xmask[i])
            
#             c2, bgi = self.__get_chi2(modelX, Pi, self.invcov[i], self.invcovdata[i], self.chi2data[i], self.priormat[i], data, isky=i)
#             chi2 += c2
            
#             if self.config["get_chi2_from_marg"]: bg.append(bgi)

#         if self.config["get_chi2_from_marg"]: 
#             chi2 = 0.
#             if self.config["split"] == 1:
#                 nonmarg_bdict = np.array([ self.bias_array_to_dict(self.bias_nonmarg_to_all(bs, bgi, with_stoch=False), with_stoch=False) for (bs, bgi) in zip(bng[0], bg) ])
#                 nonmarg_bdict_ps = np.array([ self.bias_array_to_dict(self.bias_nonmarg_to_all(bs, bgi, with_stoch=True), with_stoch=True) for (bs, bgi) in zip(bng[1], bg) ])
#             else:
#                 nonmarg_bdict = np.array([ self.bias_array_to_dict(self.bias_nonmarg_to_all(bs, bgi, with_stoch=False), with_stoch=False) for (bs, bgi) in zip(bng, bg) ])
#                 nonmarg_bdict_ps = np.array([ self.bias_array_to_dict(self.bias_nonmarg_to_all(bs, bgi, with_stoch=True), with_stoch=True) for (bs, bgi) in zip(bng, bg) ])
#             nonmarg_correlator = self.correlator.get(nonmarg_bdict)
#             nonmarg_correlator_ps = self.correlator_ps.get(nonmarg_bdict_ps)
#             for i in range(self.config["skycut"]):
#                 if self.config["skycut"] is 1: 
#                     modelcf = nonmarg_correlator 
#                     modelps = nonmarg_correlator_ps
#                 elif self.config["skycut"] > 1: 
#                     modelcf = nonmarg_correlator[i]
#                     modelps = nonmarg_correlator_ps[i]
#                 modelX = np.concatenate(( modelcf.reshape(-1), modelps.reshape(-1) ))
#                 if self.config["with_bao"]:  # BAO
#                     DM_at_z = cosmo.angular_distance(self.config["zbao"][i]) * (1. + self.config["zbao"][i])
#                     H_at_z = cosmo.Hubble(self.config["zbao"][i]) * conts.c / 1000.0
#                     rd = cosmo.rs_drag() * self.config["rs_rescale"][i]
#                     theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.config["rd_fid_in_Mpc"][i]
#                     theo_H_rd_by_rdfid = H_at_z * rd / self.config["rd_fid_in_Mpc"][i]
#                     modelX = np.concatenate((modelX, [theo_H_rd_by_rdfid, theo_DM_rdfid_by_rd_in_Mpc]))
#                 modelX = modelX[self.xmask[i]]
#                 chi2 += self.__get_chi2_non_marg(modelX, self.invcov[i], self.ydata[i])

#         prior = 0.
#         if self.config["with_bbn"]: prior += -0.5 * ((data.cosmo_arguments['omega_b'] - self.config["omega_b_BBNcenter"]) / self.config["omega_b_BBNsigma"])**2
#         lkl = - 0.5 * chi2 + prior
#         return lkl

#     def __get_chi2_non_marg(self, modelX, invcov, ydata):
#         chi2 = np.dot(modelX-ydata, np.dot(invcov, modelX-ydata))
#         return chi2

#     def __get_chi2(self, modelX, Pi, invcov, invcovdata, chi2data, priormat, data, isky=0):

#         Covbi = np.dot(Pi, np.dot(invcov, Pi.T)) + priormat
#         Cinvbi = np.linalg.inv(Covbi)
#         vectorbi = np.dot(modelX, np.dot(invcov, Pi.T)) - np.dot(invcovdata, Pi.T)
#         chi2nomar = np.dot(modelX, np.dot(invcov, modelX)) - 2. * np.dot(invcovdata, modelX) + chi2data
#         chi2mar = - np.dot(vectorbi, np.dot(Cinvbi, vectorbi)) + np.log(np.abs(np.linalg.det(Covbi)))
#         chi2tot = chi2mar + chi2nomar - priormat.shape[0] * np.log(2. * np.pi)

#         if self.config["get_chi2_from_marg"]: 
#             bg = - np.dot(Cinvbi, vectorbi)
#             return chi2tot, bg
#         else: return chi2tot, None

#     def __get_Pi_for_marg(self, marg_cf, marg_ps, xmask):
#         if self.config["split"] == 0:
#             Pi = np.zeros(shape=(marg_ps.shape[0], marg_cf.shape[1]+marg_ps.shape[1]))
#             Pi[:marg_cf.shape[0], :marg_cf.shape[1]] = marg_cf
#             Pi[:, marg_cf.shape[1]:] = marg_ps
#         elif self.config["split"] == 1:
#             Pi = block_diag(marg_cf, marg_ps)
#         elif self.config["split"] == 2:
#             Pi = np.zeros(shape=(marg_ps.shape[0]+2, marg_cf.shape[1]+marg_ps.shape[1]))
#             Pi[:marg_cf.shape[0], :marg_cf.shape[1]] = marg_cf
#             Pi[0, marg_cf.shape[1]:] = marg_ps[0]
#             Pi[marg_cf.shape[0]:, marg_cf.shape[1]:] = marg_ps[1:]

#         if self.config["with_bao"]:  # BAO
#             newPi = np.zeros(shape=(Pi.shape[0], Pi.shape[1] + 2))
#             newPi[:Pi.shape[0], :Pi.shape[1]] = Pi
#             Pi = 1. * newPi
#         Pi = Pi[:, xmask]
#         return Pi

#     def __set_prior(self, multipole):

#         if multipole is 2:
#             if self.config["split"] == 0:
#                 priors = np.array([2., 2., 8., 2., 2.])
#                 b3, cct, cr1, ce2, sn = priors
#                 print ('EFT priors: b3: %s, cct: %s, cr1(+cr2): %s, ce2: %s, shotnoise: %s (default)' % (b3, cct, cr1, ce2, sn))
#             elif self.config["split"] == 1:
#                 priors = np.array([2., 2., 8., 2., 2., 8., 2., 2.])
#                 b3_cf, cct_cf, cr1_cf, b3, cct, cr1, ce2, sn = priors
#                 print ('EFT priors: b3_cf: %s, cct_cf: %s, cr1_cf(+cr2_cf): %s, b3_ps: %s, cct_ps: %s, cr1_ps(+cr2_ps): %s, ce2_ps: %s, shotnoise_ps: %s (default)' % (b3_cf, cct_cf, cr1_cf, b3, cct, cr1, ce2, sn))
#             elif self.config["split"] == 2:
#                 priors = np.array([2., 2., 8., 2., 8., 2., 2.])
#                 b3, cct_cf, cr1_cf, cct, cr1, ce2, sn = priors
#                 print ('EFT priors: b3: %s, cct_cf: %s, cr1_cf(+cr2_cf): %s, cct_ps: %s, cr1_ps(+cr2_ps): %s, ce2_ps: %s, shotnoise_ps: %s (default)' % (b3, cct_cf, cr1_cf, cct, cr1, ce2, sn))
#         priormat = np.diagflat(1. / priors**2)

#         return priormat

#     def __set_cosmo(self, M, data):

#         zfid = self.config["z"][0]

#         cosmo = {}

#         cosmo["k11"] = self.kin  # k in h/Mpc
#         cosmo["P11"] = np.array([M.pk_lin(k * M.h(), zfid) * M.h()**3 for k in self.kin])  # P(k) in (Mpc/h)**3

#         if self.config["skycut"] == 1:
#             # if self.config["multipole"] is not 0:
#             cosmo["f"] = M.scale_independent_growth_factor_f(zfid)
#             if self.config["with_exact_time"] or self.config["with_quintessence"]:
#                 cosmo["z"] = self.config["z"][0]
#                 cosmo["Omega0_m"] = M.Omega0_m()
#                 try: cosmo["w0_fld"] = data.cosmo_arguments['w0_fld']
#                 except: pass

#             if self.config["with_AP"]:
#                 cosmo["DA"] = M.angular_distance(zfid) * M.Hubble(0.)
#                 cosmo["H"] = M.Hubble(zfid) / M.Hubble(0.)

#         elif self.config["skycut"] > 1:
#             # if self.config["multipole"] is not 0:
#             cosmo["f"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["z"]])
#             cosmo["D"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["z"]])

#             if self.config["with_AP"]:
#                 cosmo["DA"] = np.array([M.angular_distance(z) * M.Hubble(0.) for z in self.config["z"]])
#                 cosmo["H"] = np.array([M.Hubble(z) / M.Hubble(0.) for z in self.config["z"]])

#         if self.config["with_redshift_bin"]:
#             def comoving_distance(z): return M.angular_distance(z) * (1+z) * M.h()
#             if self.config["skycut"] == 1:
#                 cosmo["D"] = M.scale_independent_growth_factor(self.config["z"])
#                 cosmo["Dz"] = np.array([M.scale_independent_growth_factor(z) for z in self.config["zz"]])
#                 cosmo["fz"] = np.array([M.scale_independent_growth_factor_f(z) for z in self.config["zz"]])
#                 cosmo["rz"] = np.array([comoving_distance(z) for z in self.config["zz"]])

#             elif self.config["skycut"] > 1:
#                 cosmo["Dz"] = np.array([ [M.scale_independent_growth_factor(z) for z in zz] for zz in self.config["zz"] ])
#                 cosmo["fz"] = np.array([ [M.scale_independent_growth_factor_f(z) for z in zz] for zz in self.config["zz"] ])
#                 cosmo["rz"] = np.array([ [comoving_distance(z) for z in zz] for zz in self.config["zz"] ])

#         if self.config["with_quintessence"]: 
#             # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum. 
#             # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
#             # This does not work for multi skycuts nor for redshift bins.
#             zm = 5. # z in matter domination
#             def scale_factor(z): return 1/(1.+z)
#             Omega0_m = cosmo["Omega0_m"]
#             w = cosmo["w0_fld"]
#             GF = pb.GreenFunction(Omega0_m, w=w, quintessence=True)
#             Dq = GF.D(scale_factor(zfid)) / GF.D(scale_factor(zm))
#             Dm = M.scale_independent_growth_factor(zfid) / M.scale_independent_growth_factor(zm)
#             cosmo["P11"] *= Dq**2 / Dm**2 * ( 1 + (1+w)/(1.-3*w) * (1-Omega0_m)/Omega0_m * (1+zm)**(3*w) )**2 # 1611.07966 eq. (4.15)
#             cosmo["f"] = GF.fplus(1/(1.+cosmo["z"]))

#         return cosmo

#     def __load_data(self, multipole, data_directory, cf_spectrum_file, ps_spectrum_file, covmat_file, 
#                     smin=None, smax=None, kmin=None, kmax=None, 
#                     with_bao=False, baoH=None, baoD=None): 

#         sdata, cfdata = self.__load_spectrum(data_directory, cf_spectrum_file) # cf
#         s = sdata.reshape(3, -1)[0]
#         Ns = len(s)
#         smask0 = np.argwhere((s <= smax) & (s >= smin))[:, 0]
#         smask = smask0
#         for i in range(multipole - 1):
#             smaski = np.argwhere((s <= smax) & (s >= smin))[:, 0] + (i + 1) * Ns
#             smask = np.concatenate((smask, smaski))
#         sdata = s[smask0]
#         cfdata = cfdata[smask]

#         kdata, psdata = self.__load_spectrum(data_directory, ps_spectrum_file) # ps
#         k = kdata.reshape(3, -1)[0]
#         Nk = len(k)
#         kmask0 = np.argwhere((k <= kmax) & (k >= kmin))[:, 0]
#         kmask = kmask0
#         for i in range(multipole - 1):
#             kmaski = np.argwhere((k <= kmax) & (k >= kmin))[:, 0] + (i + 1) * Nk
#             kmask = np.concatenate((kmask, kmaski))
#         kdata = k[kmask0]
#         psdata = psdata[kmask]

#         xmask = np.concatenate((smask, kmask + 2 * Ns))
#         ydata = np.concatenate((cfdata, psdata))

#         if with_bao and baoH > 0 and baoD > 0: # BAO
#             ydata = np.concatenate((ydata, [baoH, baoD]))
#             xmask = np.concatenate((xmask, [-2, -1]))
#             print ("BAO recon: on")
#         else:
#             print ("BAO recon: none")

#         cov = np.loadtxt(os.path.join(data_directory, covmat_file))
#         covred = cov[xmask.reshape((len(xmask), 1)), xmask]

#         invcov = np.linalg.inv(covred)

#         chi2data = np.dot(ydata, np.dot(invcov, ydata))
#         invcovdata = np.dot(ydata, invcov)

#         return s, k, xmask, ydata, chi2data, invcov, invcovdata

#     def __load_spectrum(self, data_directory, spectrum_file):
#         fname = os.path.join(data_directory, spectrum_file)
#         try: 
#             kPS, PSdata, _ = np.loadtxt(fname, unpack=True)
#         except:
#             try:
#                 kPS, PSdata = np.loadtxt(fname, unpack=True)
#             except:
#                 kPS, l0, l2, l4 = np.loadtxt(fname, unpack=True)
#                 kPS = np.concatenate([kPS, kPS, kPS])
#                 PSdata = np.concatenate([l0, l2, l4])
#         return kPS, PSdata


