from pybird.module import *
from pybird.common import co
import pybird.integrated_model_jax as integrated_model

class Emulator():
    """A class to emulate EFT of LSS calculations using neural networks.
    
    The Emulator class provides fast predictions for various components of the EFT
    calculations by using pre-trained neural network models. It handles the emulation
    of loop corrections, counterterms, and IR resummation components.
    
    Attributes:
        emu_path (str): Path to directory containing emulator model files.
        knots_path (str): Path to file containing emulator knot points.
        kmax_emu (float): Maximum k value for which the emulator is valid.
        co (Common): Common parameters shared across calculations.
        knots (ndarray): Array of knot points used in the emulation.
        logknots (ndarray): Log of the knot points.
        
        emu_ploopl_mono: Emulator model for monopole loop corrections.
        emu_ploopl_quad: Emulator model for quadrupole loop corrections.
        emu_ploopl_hex: Emulator model for hexadecapole loop corrections.
        emu_IRPs11: Emulator model for IR resummation P11 terms.
        emu_IRPsct: Emulator model for IR resummation counterterms.
        emu_IRPsloop_mono: Emulator model for monopole IR resummation loop terms.
        emu_IRPsloop_quad: Emulator model for quadrupole IR resummation loop terms.
        emu_IRPsloop_hex: Emulator model for hexadecapole IR resummation loop terms.
    
    Methods:
        load_models(): Load all emulator models from the specified path.
        make_params(): Prepare input parameters for emulator predictions.
        setPsCfl(): Set power spectrum and correlation function using emulator predictions.
        PsCf_resum(): Apply IR resummation to emulated results.
    """

    def __init__(self, emu_path, knots_path, load_models=True, co=co):
        """Initialize the Emulator for fast EFT calculations.
        
        Parameters
        ----------
        emu_path : str
            Path to directory containing trained emulator model files
        knots_path : str
            Path to file containing emulator knot points
        load_models : bool, optional
            Whether to load models immediately upon initialization, by default True
        co : Common, optional
            Common parameters object, by default co
            
        Notes
        -----
        The emulator provides 1000x speedup over full EFT calculations by using
        pre-trained neural networks. Models are valid up to kmax_emu = 0.4 h/Mpc.
        """
        
        self.emu_path = emu_path
        self.knots_path = knots_path
        self.kmax_emu = 0.4

        if co.kmax > self.kmax_emu:
            print(f"Warning: Asked kmax larger than kmax_emu. To avoid issues with the emulator, resetting the kmax to {self.kmax_emu}...")
            co.kmax = self.kmax_emu
        else:
            self.kmax_emu = co.kmax

        with h5py.File(self.emu_path + "/k_emu.h5", "r") as f: k_emu = array(f["k_emu"]) #load the fixed emulator k-array
        k_emu = k_emu[k_emu <= self.kmax_emu]
        co.k, co.Nk, co.Nloop = k_emu,  k_emu.shape[0], 35 # change internal k in the common class
        self.co = co

        self.knots = load(self.knots_path) 
        self.logknots = log(self.knots)  # Sorting and removing duplicates

        if load_models:
            self.load_models()


    def load_models(self):
        """Load all pre-trained emulator models from disk.
        
        Notes
        -----
        This method loads neural network models for:
        - Loop corrections (monopole, quadrupole, hexadecapole)
        - IR resummation terms (P11, counterterms, loop terms)
        
        Models are stored as HDF5 files and compiled for JAX execution.
        """
        self.emu_ploopl_mono = integrated_model.IntegratedModel(None, None, None)
        self.emu_ploopl_mono.restore(self.emu_path + '/ploopl_mono_80knots_jax_model.h5')

        self.emu_ploopl_quad = integrated_model.IntegratedModel(None, None, None)
        self.emu_ploopl_quad.restore(self.emu_path + '/ploopl_quad_80knots_jax_model.h5')

        self.emu_ploopl_hex = integrated_model.IntegratedModel(None, None, None)
        self.emu_ploopl_hex.restore(self.emu_path + '/ploopl_hex_80knots_jax_model.h5')
        
        self.emu_IRPs11 = integrated_model.IntegratedModel(None, None, None)
        # self.emu_IRPs11.restore(self.emu_path + '/irps11_80knots_jax_model.h5')
        self.emu_IRPs11.restore('/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/saved_models/irps11_80knots_irps11_lognormz_irresum_fixed_17_06_2025_1pgc_256pca_jax_model.h5')

        self.emu_IRPsct = integrated_model.IntegratedModel(None, None, None)
        # self.emu_IRPsct.restore(self.emu_path + '/irpsct_80knots_jax_model.h5')
        self.emu_IRPsct.restore('/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/saved_models/irpsct_80knots_full_ct_lognormz_irresum_fixed_17_06_25_jax_model.h5')

        self.emu_IRPsloop_mono = integrated_model.IntegratedModel(None, None, None)
        # self.emu_IRPsloop_mono.restore(self.emu_path + '/irpsloop_mono_80knots_jax_model.h5')
        self.emu_IRPsloop_mono.restore('/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/saved_models/irpsloop_mono_80knots_full_mono_lognormz_irresum_fixed_18_06_25_ircutoff_rework_moredata_jax_model.h5')

        self.emu_IRPsloop_quad = integrated_model.IntegratedModel(None, None, None)
        # self.emu_IRPsloop_quad.restore(self.emu_path + '/irpsloop_quad_80knots_jax_model.h5')
        self.emu_IRPsloop_quad.restore('/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/saved_models/irpsloop_quad_80knots_full_quad_lognormz_irresumfixed_20_06_25_ircutoff_more_data_jax_model.h5')

        self.emu_IRPsloop_hex = integrated_model.IntegratedModel(None, None, None)
        # self.emu_IRPsloop_hex.restore(self.emu_path + '/irpsloop_hex_80knots_jax_model.h5')
        self.emu_IRPsloop_hex.restore('/cluster/work/refregier/alexree/local_packages/train_pybird_emulators/src/train_pybird_emulators/data/saved_models/irpsloop_hex_80knots_full_hex_lognormz_irresumfixed_19_06_25_ircutoff1_more_data_jax_model.h5')


    def make_params(self, kk, pk, f=1.0, time=False, ir=False, pca=False):
        """Prepare input parameters for emulator predictions.
        
        Parameters
        ----------
        kk : ndarray
            k-array in h/Mpc
        pk : ndarray
            Linear power spectrum in (Mpc/h)^3
        f : float, optional
            Growth rate for RSD effects, by default 1.0
        time : bool, optional
            Whether to include time dependence, by default False
        ir : bool, optional
            Whether this is for IR resummation, by default False
        pca : bool, optional
            Whether to use PCA compression, by default False
            
        Returns
        -------
        ndarray
            Formatted input parameters for the emulator models
        """

        pk_max = max(pk)
        self.pk_max = array(pk_max)

        pk_max_reshaped = self.pk_max.reshape(-1, 1)

        # Calculate ilogpk using JAX-compatible interp1d
        ilogpk = interp1d(log(kk), log(pk / pk_max_reshaped), axis=-1, kind='linear')
        logpk = ilogpk(self.logknots)

        if ir: 
            pk_max_array = array([[pk_max]]) # Make it two-dimensional
            f_array = array([[f]]) # Make it two-dimensional
            self.params = concatenate([logpk, pk_max_array, f_array], axis=1)
        
        else: 
            self.params = logpk

        return logpk

    def setPsCfl(self, bird, kk, pk, time, make_params=True):
        """Set power spectrum and correlation function using emulator predictions.
        
        Uses neural network emulators to predict loop corrections and sets them
        in the provided Bird object.

        Parameters
        ----------
        bird : Bird
            Bird object to store the emulated results
        kk : ndarray
            k-array in h/Mpc
        pk : ndarray
            Linear power spectrum in (Mpc/h)^3
        time : bool
            Whether to include time dependence
        make_params : bool, optional
            Whether to prepare input parameters, by default True
            
        Notes
        -----
        This method emulates the computationally expensive loop calculations,
        providing predictions for P22, P13, and correlation function multipoles.
        """
        if make_params: self.make_params(kk, pk, time=time,ir=False)

        predictions = hstack([
            self.emu_ploopl_mono.predict(self.params),
            self.emu_ploopl_quad.predict(self.params),
            self.emu_ploopl_hex.predict(self.params)
        ])

        shape = (3, 35, -1)
        scaling_factor = self.pk_max ** 2

        bird.Ploopl = predictions.reshape(shape)[:self.co.Nl, :, :self.co.Nk] * scaling_factor

    def PsCf_resum(self, bird, kk, pk, f, time, make_params=True,ir=True, pca=False):
        if make_params: self.make_params(kk, pk, f=f, time=time, ir=ir, pca=pca)

        shape_3 = (3, 3, -1)
        shape_6 = (3, 6, -1)
        shape_35 = (3, 35, -1)
        scaling_factor = self.pk_max

        scaling_factor_squared = scaling_factor ** 2

        # Ensuring correct reshaping and multiplication
        fullIRPs11 = self.emu_IRPs11.predict(self.params).reshape(shape_3) * scaling_factor
        fullIRPsct = self.emu_IRPsct.predict(self.params).reshape(shape_6) * scaling_factor
        fullIRPsloop = hstack([
            self.emu_IRPsloop_mono.predict(self.params),
            self.emu_IRPsloop_quad.predict(self.params),
            self.emu_IRPsloop_hex.predict(self.params)
        ]).reshape(shape_35) * scaling_factor_squared

        bird.Pctl = fullIRPsct[:self.co.Nl, :, :self.co.Nk]
        bird.Ploopl = fullIRPsloop[:self.co.Nl, :, :self.co.Nk] 
        bird.P11l += fullIRPs11[:self.co.Nl, :, :self.co.Nk]

