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
from numpy.lib.function_base import interp
import scipy.constants as const
import scipy.integrate
import scipy.interpolate
import scipy.misc
import six
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
        #for key in cl.iterkeys():
        for key in six.iterkeys(cl):
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
        #for key in cl.iterkeys():
        for key in six.iterkeys(cl):        
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
        #for key, value in dictionary.iteritems():
        for key, value in six.iteritems(dictionary):
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
                        exec ("self.%s_contamination[l]=float(line.split()[1])/(l*(l+1.)/2./math.pi)" % nuisance)
            except:
                print ('Warning: you did not pass a file name containing ')
                print ('a contamination spectrum regulated by the nuisance ')
                print ('parameter '+nuisance)

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
                exec ("cl['tt'][l] += nuisance_value*self.%s_contamination[l]" % nuisance)

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


############################################################################################################################################
#
#
#
############################################################################################################################################
import yaml
sys.path.append('/Users/Pierre/Documents/github/pybird/pybird_dev/')
import pybird as pb

class Likelihood_bird(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.c = yaml.full_load(open(os.path.join(self.data_directory, self.config_file), 'r'))
        self.L = pb.Likelihood(self.c)

        self.need_cosmo_arguments(data, self.L.class_settings)

    def loglkl(self, cosmo, data):

        # if we run with zero varying cosmological parameter, we evaluate the model only once
        if self.first_evaluation: 
            data.update_cosmo_arguments() 
            data.need_cosmo_update = True
            self.first_evaluation = False 
        
        free_b_name = self.use_nuisance
        free_b = [data.mcmc_parameters[fbn]['current'] * data.mcmc_parameters[fbn]['scale'] for fbn in free_b_name]

        return self.L.loglkl(free_b, free_b_name, cosmo, data.need_cosmo_update) 






