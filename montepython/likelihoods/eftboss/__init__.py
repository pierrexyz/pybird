from montepython.likelihood_class import Likelihood

import yaml
import os, sys

class eftboss(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        self.c = yaml.full_load(open(os.path.join(self.data_directory, self.config_file), 'r'))
        if 'with_emu' in self.c:
            if self.c['with_emu']:
                os.environ["JAX_PLATFORMS"] = "cpu"
                from pybird.config import set_jax_enabled
                set_jax_enabled(True)
        from pybird.likelihood import Likelihood as Likelihood_bird
        self.L = Likelihood_bird(self.c)
        self.need_cosmo_arguments(data, self.L.class_settings)
        self.first_evaluation = True

    def loglkl(self, cosmo, data):

        # if we run with zero varying cosmological parameter, we evaluate the model only once
        if self.first_evaluation:
            data.update_cosmo_arguments()
            data.need_cosmo_update = True
            self.first_evaluation = False

        free_b_name = self.use_nuisance
        free_b = [data.mcmc_parameters[fbn]['current'] * data.mcmc_parameters[fbn]['scale'] for fbn in free_b_name]

        return self.L.loglkl(free_b, free_b_name, cosmo_engine=cosmo, cosmo_module='class', need_cosmo_update=data.need_cosmo_update)
