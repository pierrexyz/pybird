# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from unittest.mock import MagicMock

# Add the project root and pybird package to Python path
sys.path.insert(0, os.path.abspath('../..'))  # pybird_emu root
sys.path.insert(0, os.path.abspath('..'))     # pybird package directory

# Standard autodoc mock imports - simple and reliable
autodoc_mock_imports = [
    # NumPy and all its submodules
    'numpy', 'numpy.fft', 'numpy.linalg', 'numpy.random',

    # SciPy and all its submodules
    'scipy', 'scipy.optimize', 'scipy.interpolate', 'scipy.special',
    'scipy.integrate', 'scipy.linalg', 'scipy.fftpack', 'scipy.constants',
    'scipy.stats', 'scipy.ndimage', 'scipy.signal',

    # External dependencies
    'h5py', 'yaml', 'classy', 'pandas',

    # JAX ecosystem
    'jax', 'jax.numpy', 'jax.nn', 'jax.random', 'jax.scipy', 'jax.scipy.linalg', 'jax.scipy.special',
    'flax', 'flax.linen', 'flax.training', 'flax.training.train_state', 'flax.linen.nn',
    'optax',

    # MCMC and sampling
    'iminuit', 'emcee', 'zeus', 'blackjax', 'numpyro',

    # Plotting and visualization
    'matplotlib', 'matplotlib.pyplot', 'getdist',

    # FFTLog dependencies
    'fftlog', 'fftlog.fftlog', 'fftlog.sbt', 'fftlog.utils',

    # Additional common dependencies
    'tqdm', 'pickle', 'joblib', 'multiprocessing'
]

# Configuration
project = 'PyBird'
copyright = '2025, Pierre Zhang, Guido D\'Amico and Alexander Reeves'
author = 'Pierre Zhang, Guido D\'Amico and Alexander Reeves'
version = '0.2.1'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]

# Try to add nbsphinx, but make it optional
# Disable notebook rendering to avoid Pandoc issues in GitHub Actions
NBSphinx_AVAILABLE = False
print("Notebook rendering disabled for GitHub Actions compatibility")

# Show both class docstring and __init__ docstring
autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Standard mocking - no custom handling needed

# Theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None  # Add your logo path if you have one
html_favicon = None  # Add your favicon if you have one

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = []

# Default role for `text`
default_role = 'py:obj'

# Show type hints in description not signature
napoleon_use_param = True
napoleon_use_rtype = True

# -- NBSphinx configuration ---------------------------------------------------
# https://nbsphinx.readthedocs.io/en/latest/configuration.html

if NBSphinx_AVAILABLE:
    # Configure nbsphinx for Jupyter notebook rendering
    nbsphinx_execute = 'never'
    nbsphinx_allow_errors = True  # Allow notebooks with errors to still render
    nbsphinx_prolog = """
    .. raw:: html

        <div class="admonition note">
            <p class="first admonition-title">Note</p>
            <p class="last">
                This page was generated from a Jupyter notebook. 
                You can view the source notebook in the 
                <a href="https://github.com/pierrexyz/pybird/tree/main/demo">PyBird demo directory</a>.
            </p>
        </div>

    """

    # Configure timeout for notebook execution (if needed)
    nbsphinx_timeout = 600  # 10 minutes timeout

    # Configure kernel for notebook execution (if needed)
    nbsphinx_kernel_name = 'python3'

    # Add source links to notebooks
    nbsphinx_epilog = """
    .. raw:: html

        <div class="admonition tip">
            <p class="first admonition-title">Source Code</p>
            <p class="last">
                The source notebook for this page can be found in the 
                <a href="https://github.com/pierrexyz/pybird/tree/main/demo">PyBird demo directory</a>.
            </p>
        </div>

    """
else:
    # If nbsphinx is not available, exclude notebook files
    exclude_patterns.extend(['notebooks/*.ipynb'])
