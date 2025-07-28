from pybird.module import *
import pickle
import flax 
from flax import linen as nn
from jax.nn import sigmoid
from flax import traverse_util
import jax.random as random


class Network(nn.Module):
    weights: list  # List of (weight, bias) tuples for each layer
    hyper_params: list  # List of (alpha, beta) for each custom activation layer

    @nn.compact
    def __call__(self, x):
        # Loop over layers except the last one
        for (w, b), (a, b_hyper) in zip(self.weights[:-1], self.hyper_params):
            x = CustomActivation_jax(a=a, b=b_hyper)(dot(x, w) + b)

        # Final layer (no activation)
        final_w, final_b = self.weights[-1]
        x = dot(x, final_w) + final_b
        return x


class CustomActivation_jax(nn.Module):
    a: float  # alpha hyperparameter
    b: float  # beta hyperparameter

    @nn.compact
    def __call__(self, x):
        return multiply(add(self.b, multiply(sigmoid(multiply(self.a, x)), subtract(1., self.b))), x)


def insert_zero_columns(prediction, zero_columns_indices):
    # Total number of columns in the final output
    num_columns = prediction.shape[1] + len(zero_columns_indices)

    # Initialize the final_prediction array
    final_prediction = zeros((prediction.shape[0], num_columns))

    # Indices in final_prediction where values from prediction will be inserted
    non_zero_indices = delete(arange(num_columns), zero_columns_indices)

    # Insert values from prediction into the correct positions in final_prediction
    final_prediction = final_prediction.at[:, non_zero_indices].set(prediction)

    return final_prediction

class IntegratedModel:
    """A class to integrate and manage Flax models in JAX with preprocessing.
    
    The IntegratedModel class combines a Flax model with input/output scaling,
    PCA transformations, and other preprocessing steps. It handles model predictions
    with proper scaling and transformation, and can restore models from saved files.
    
    Attributes:
        model (flax.linen.Module): The Flax model for predictions.
        input_scaler (object): Scaler for input data preprocessing.
        output_scaler (object): Scaler for output data postprocessing.
        offset (float): Offset value for output adjustment.
        zero_columns (list): Indices of columns to be set to zero in output.
        rescaling_factor (float): Factor for rescaling output.
        temp_file (str): Path to temporary file for model storage.
        train_losses (list): History of training losses.
        val_losses (list): History of validation losses.
        log_preprocess (bool): Whether to apply log preprocessing.
        pca (object): PCA transformation object if used.
        pca_scaler (object): Scaler for PCA transformed data.
        verbose (bool): Whether to print verbose information.
        
        scaler_mean_in (ndarray): Mean values for input scaling.
        scaler_scale_in (ndarray): Scale values for input scaling.
        scaler_mean_out (ndarray): Mean values for output scaling.
        scaler_scale_out (ndarray): Scale values for output scaling.
        pca_components (ndarray): PCA components if used.
        pca_mean (ndarray): PCA mean values if used.
        pca_scaler_mean (ndarray): Mean values for PCA scaling.
        pca_scaler_scale (ndarray): Scale values for PCA scaling.
    
    Methods:
        predict(): Make predictions with proper preprocessing and postprocessing.
        restore(): Restore model parameters and scalers from a saved file.
    """
    def __init__(self, keras_model, input_scaler, output_scaler, temp_file=None, offset=None, log_preprocess=False, zero_columns=None, rescaling_factor=None, pca=None, pca_scaler=None, verbose=False):
        self.model = keras_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.offset = offset
        self.zero_columns = zero_columns
        self.rescaling_factor = rescaling_factor
        self.temp_file = temp_file
        self.train_losses = []
        self.val_losses = []
        self.log_preprocess = log_preprocess
        self.pca = pca
        self.pca_scaler = pca_scaler
        self.verbose = verbose


    def predict(self, data): #add udoing the log preprocess
        scaled_data = (data - self.scaler_mean_in)/self.scaler_scale_in
        scaled_data = array(scaled_data, dtype=float32)


        prediction = self.jax_model.apply(self.jax_params, scaled_data)
        prediction = prediction*self.scaler_scale_out + self.scaler_mean_out

        if self.pca:
            prediction = dot(prediction, self.pca_components) + self.pca_mean
            prediction = prediction*self.pca_scaler_scale + self.pca_scaler_mean

        if self.zero_columns is not None:
            final_prediction = insert_zero_columns(prediction, self.zero_columns)

        else:
            final_prediction = prediction


        return final_prediction


    def restore(self, h5_filename):
        """
        Load pre-saved IntegratedModel attributes' from an h5 file.
        Parameters:
            h5_filename (str): filename of the .h5 file where model was saved
        """
        def get_numeric_key(key):
            # Extract numeric part from the key
            return int(key.split('_')[1])

        with h5py.File(h5_filename, 'r') as h5f:
            # Load weights
            weights_dict = {}
            weights_group = h5f['weights']
            for key in sorted(weights_group, key=get_numeric_key):
                layer_group = weights_group[key]
                weights_dict[key] = [layer_group[str(i)][()] for i in range(len(layer_group))]

            #load hyperparameters
            hyperparameters_group = h5f['hyperparameters']
            for key in sorted(hyperparameters_group, key=get_numeric_key):
                weights_dict[key] = hyperparameters_group[key][()]

            # Load attributes
            attr_group = h5f['attributes']
            input_scaler_mean = attr_group['input_scaler_mean'][()]
            input_scaler_scale = attr_group['input_scaler_scale'][()]
            output_scaler_mean = attr_group['output_scaler_mean'][()]
            output_scaler_scale = attr_group['output_scaler_scale'][()]
            offset = attr_group['offset'][()]
            log_preprocess = attr_group['log_preprocess'][()]
            zero_columns = attr_group['zero_columns'][()] if 'zero_columns' in attr_group.keys() else None
            rescaling_factor = attr_group['rescaling_factor'][()] if 'rescaling_factor' in attr_group.keys() else None
            pca_components = attr_group['pca_components'][()] if 'pca_components' in attr_group.keys() else None
            pca_mean = attr_group['pca_mean'][()] if 'pca_mean' in attr_group.keys() else None
            pca_scaler_mean = attr_group['pca_scaler_mean'][()] if 'pca_scaler_mean' in attr_group.keys() else None
            pca_scaler_scale = attr_group['pca_scaler_scale'][()] if 'pca_scaler_scale' in attr_group.keys() else None

            # Set attributes
            self.scaler_mean_in = input_scaler_mean
            self.scaler_scale_in = input_scaler_scale
            self.scaler_mean_out = output_scaler_mean
            self.scaler_scale_out = output_scaler_scale
            self.offset = offset
            self.log_preprocess = log_preprocess
            self.zero_columns = zero_columns
            self.rescaling_factor = rescaling_factor
            self.num_zero_columns = len(self.zero_columns) if self.zero_columns is not None else 0

            if pca_components is not None:
                self.pca = True
                self.pca_components = pca_components
                self.pca_mean = pca_mean
                self.pca_scaler_mean = pca_scaler_mean
                self.pca_scaler_scale = pca_scaler_scale


            # Restore the model parameters
            weights_keys = sorted((key for key in weights_dict if key.startswith("weights_")),
                                key=lambda k: int(k.split('_')[1]))

            hyperparameter_keys = sorted((key for key in weights_dict if key.startswith("hyperparameters_")),
                                        key=lambda k: int(k.split('_')[1]))

            self.jax_model = Network(hyper_params=[weights_dict[key] for key in hyperparameter_keys], 
                                    weights=[weights_dict[key] for key in weights_keys])

            # Initialize the model with dummy data
            input_shape = weights_dict["weights_0"][0].shape[0]
            rng = random.PRNGKey(0)
            dummy_input = ones((1, input_shape))
            params = self.jax_model.init(rng, dummy_input)

            # flattened_params = traverse_util.tree_flatten(params)
            flattened_params = flax.serialization.to_state_dict(params)

            for i, layer_path in enumerate(flattened_params.keys()):
                if "CustomActivation" in layer_path: 
                    flattened_params[layer_path] = weights_dict[f"hyperparameters_{i}"]
                else:
                    flattened_params[layer_path] = weights_dict[f"weights_{i}"]

            # self.jax_params = traverse_util.tree_unflatten(flattened_params)
            self.jax_params = flax.serialization.from_state_dict(params, flattened_params)
            if self.verbose:
                print("restore successful")

