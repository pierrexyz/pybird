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
        "with_quintessence", "with_cf_sys", "multipole_rotation", "with_rs_marg"]

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
                    xmin=self.config["xmin"][i], xmax=xmax, xmax0=xmax0, xmax1=xmax1, multipole_rotation=self.config["multipole_rotation"], with_bao=self.config["with_bao"], baoH=baoH, baoD=baoD)
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
                if self.config["get_chi2_from_marg"]: 
                    bdict[i].update({p: b for p, b in zip(self.marg_gauss_eft_parameters_list, bg_i)})

        if self.config["get_chi2_from_marg"] or self.config["nonmarg"]: 
            chi2 = 0.
            nonmarg_correlator = self.correlator.get(bdict)
            for i in range(self.config["skycut"]):
                if self.config["skycut"] == 1: modelX = nonmarg_correlator
                elif self.config["skycut"] > 1: modelX = nonmarg_correlator[i]
                chi2_i, _ = self.__get_chi2(modelX, cosmo, data, marg=False, i=i)
                # print ('sky %s, chi2 = %s' % (i+1, chi2_i))
                chi2 += chi2_i
                chi2 += self.__set_prior(np.array([bdict[i][param] for param in self.marg_gauss_eft_parameters_list]), 
                    self.marg_gauss_eft_parameters_prior_mean[i], self.marg_gauss_eft_parameters_prior_sigma[i])

                # print ('chi2_prior = %s' % self.__set_prior(np.array([bdict[i][param] for param in self.marg_gauss_eft_parameters_list]), self.marg_gauss_eft_parameters_prior_mean[i], self.marg_gauss_eft_parameters_prior_sigma[i]))

            if self.first_evaluation:
                if self.config["get_fake"]: self.get_fake(cosmo, data, bdict, nonmarg_correlator)
                if self.config["get_fit"]: self.get_fit(cosmo, data, bdict, nonmarg_correlator) # this doesn't work yet with (varying-kmax) wedges

        prior = 0.
        if len(self.nonmarg_gauss_eft_parameters_list) > 0: 
            for i in range(self.config["skycut"]): 
                prior += self.__set_prior(np.array([bdict[i][param] for param in self.nonmarg_gauss_eft_parameters_list]), 
                    self.nonmarg_gauss_eft_parameters_prior_mean[i], self.nonmarg_gauss_eft_parameters_prior_sigma[i])
                #print ('sky %s, chi2_prior_nonmarg = %s' % (i+1, self.__set_prior(np.array([bdict[i][param] for param in self.nonmarg_gauss_eft_parameters_list]), self.nonmarg_gauss_eft_parameters_prior_mean[i], self.nonmarg_gauss_eft_parameters_prior_sigma[i])))
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

        # from scipy.signal import find_peaks
        # def get_smooth_wiggle_resc(kh, pk, alpha_rs=1., interpkind='quadratic'):
        #     '''
        #     Returns the no-wiggled P(k) from finding the inflection point of the BAO and interpolating
        #     '''
        #     logkh = np.log10(kh)
        #     logpk = np.log10(pk)
        #     gradient = np.gradient(logpk)

        #     locmax,extra = find_peaks(gradient)
        #     locmin,extra = find_peaks(-gradient)

        #     glomax = np.argmax(logpk)
        #     locmax = locmax[locmax>glomax] #needed if not it detects maxima before the global maxima
        #     locmin = locmin[locmin>glomax] #needed if not it detects minima before the global maxima

        #     commonend  = np.max(np.hstack([locmax,locmin]))

        #     "common left part"
        #     logkhl = logkh[:glomax]
        #     logpkl = logpk[:glomax]
        #     "common right part"
        #     logkhr = logkh[commonend+1:]
        #     logpkr = logpk[commonend+1:]

        #     "build max curve"
        #     logkhma = np.hstack((logkhl,logkh[locmax],logkhr))
        #     logpkma = np.hstack((logpkl,logpk[locmax],logpkr))

        #     "build min curve"
        #     logkhmi = np.hstack((logkhl,logkh[locmin],logkhr))
        #     logpkmi = np.hstack((logpkl,logpk[locmin],logpkr))

        #     maxfun =interp1d(logkhma,logpkma,kind=interpkind)
        #     minfun =interp1d(logkhmi,logpkmi,kind=interpkind)

        #     spk = (10**maxfun(logkh)+10**minfun(logkh))*0.5
        #     wpk_resc = interp1d(kh, pk-spk, bounds_error=False)(alpha_rs * kh)
        #     kmask = np.where(kh < 1.02)[0]
        #     return kh[kmask], spk[kmask], spk[kmask]+wpk_resc[kmask]

        if self.config["with_nnlo_counterterm"]: 
            cosmo["k11"], cosmo["Psmooth"], cosmo["P11"] = get_smooth_wiggle_resc(cosmo["k11"], cosmo["P11"])

        if self.config["with_rs_marg"]:
            # wiggle rescaling parameter
            alpha_rs = data.mcmc_parameters['alpha_rs']['current'] * data.mcmc_parameters['alpha_rs']['scale']
            cosmo["k11"], _, cosmo["P11"] = get_smooth_wiggle_resc(cosmo["k11"], cosmo["P11"], alpha_rs=alpha_rs)

        return cosmo

    def __load_data_ps(self, multipole, wedge, data_directory, spectrum_file, covmat_file, xmin, xmax=None, xmax0=None, xmax1=None, multipole_rotation='default', with_bao=False, baoH=None, baoD=None):

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
            
            def getmask(kmax, i): return np.argwhere((x <= kmax) & (x >= xmin))[:, 0] + i * Nx

            if 'PA_w1_w2_1loop' in multipole_rotation: # optimal
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
                xmask = np.array([getmask(kmax, i) for i, kmax in enumerate([kmaxA, kmaxB, kmaxC])])
            elif 'PA_w1_w2_1loop+' in multipole_rotation: # great
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
                xmask = np.array([getmask(kmax, i) for i, kmax in enumerate([kmaxA, kmaxB, kmaxC])])
            elif 'Q0_l0_l2' in multipole_rotation: 
                cov_resh = cov.reshape((3, cov.shape[0] // 3, 3, cov.shape[1] // 3))
                mat = np.array([[1., -1./2., 3./8.], [1., 0., 0.], [0., 1., 0.]]) # Q0 + P0 + P2
                ydata = np.einsum('al,lk->ak', mat, ydata.reshape(3,-1)).reshape(-1) # rotate the data to optimal basis for minimal theoretical error
                cov = np.einsum('al,bm,lkmj->akbj', mat, mat, cov_resh).reshape(cov.shape) # rotate the covariance to optimal basis for minimal theoretical error
                err = np.sqrt(np.diag(cov)).reshape(3, -1)
                kmaxs = np.array([0.40, 0.20, 0.20]) # PZ: hard-coded !!!
                with np.printoptions(precision=2, suppress=True): print ('kmax: ', kmaxs)
                xmask = np.array([getmask(kmax, i) for i, kmax in enumerate(kmaxs)])

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
        # hartlap = (2048. - len(xmask) - 2.) / (2048. - 1.) 
        hartlap = 1.
        invcov = hartlap * np.linalg.inv(covred)

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




