Installation Guide
==================

PyBird supports flexible installation modes to accommodate different computational needs and environments.

Installation from Repository
---------------------------------------------------

**For current development and testing**, install directly from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/pierrexyz/pybird.git
   cd pybird

Choose your installation mode:

Core PyBird
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

**Includes**: NumPy, SciPy, H5PY, pyyaml, fftlog, emcee, iminuit + documentation tools + development tools  


PyBird-JAX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[jax]"

**Includes**: Everything in Core + JAX ecosystem (jax, jaxlib, flax, optax, numpyro, blackjax, nautilus-sampler)

PyPI Installation
----------------------------------

PyBird is available on PyPI with both core and JAX installation modes:

Core Installation (NumPy Backend)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For basic functionality with NumPy backend:

.. code-block:: bash

   pip install pybird-lss

This installs essential dependencies:

* numpy, scipy, h5py
* emcee, iminuit  
* pyyaml
* fftlog-lss
* documentation and development tools

JAX Installation (Accelerated Computing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For JAX-accelerated functionality:

.. code-block:: bash

   pip install "pybird-lss[jax]"

This includes everything in Core plus the JAX ecosystem:

* jax, jaxlib
* flax, optax
* numpyro, blackjax
* nautilus-sampler
* interpax


Troubleshooting
---------------

JAX Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

If JAX installation fails, try platform-specific installations:

.. code-block:: bash

   # CPU-only JAX (most compatible)
   pip install -e ".[jax]" --extra-index-url https://storage.googleapis.com/jax-releases/jax_cpu_releases.html

   # CUDA support (NVIDIA GPUs)
   pip install -e ".[jax]" --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Testing Your Installation
--------------------------

Verify your PyBird installation works correctly by running the test suite. These may take a few minutes but we recommend running tests after an install as these will also automatically build cached loop matrices for the default configuration that can be used in future computations.

.. code-block:: bash

   cd tests
   
   # Run all tests (recommended)
   python run_tests.py 
      
   # Test specific components
   python run_tests.py --class correlator
   python run_tests.py --class bird
   python run_tests.py --class likelihood
   python run_tests.py --class emulator
   python run_tests.py --class utils


**Expected Output:**

.. code-block:: text

   🎉 All tests passed!

The tests automatically work with both NumPy and JAX backends (i.e. it will skip the JAX tests if JAX is not installed), verifying that your installation mode is functioning correctly.


Dependencies
------------

Core PyBird
~~~~~~~~~~~~~~~~~~~~~~
* `NumPy <https://numpy.org/>`_ - Numerical computing library  
* `SciPy <https://scipy.org/>`_ - Scientific computing library  
* `h5py <https://www.h5py.org/>`_ - HDF5 file format support  
* `PyYAML <https://pyyaml.org/>`_ - YAML configuration file support  
* `fftlog-lss <https://pypi.org/project/fftlog-lss/>`_ - FFTLog routines

Optional but recommended
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `iminuit <https://iminuit.readthedocs.io/en/stable/>`_ - For minimization  
* `emcee <https://emcee.readthedocs.io/en/stable/>`_ - For MCMC sampling  

JAX 
~~~~~~~~~~~
* `JAX <https://github.com/google/jax>`_ - Accelerated computing: jit, vmap, AD, and NN-embedding  
* `jaxlib <https://pypi.org/project/jaxlib/>`_ - Companion to JAX providing XLA and GPU/TPU support  
* `Flax <https://github.com/google/flax>`_ - Neural network library for JAX  
* `Optax <https://github.com/deepmind/optax>`_ - Gradient processing and optimization library for JAX  
* `NumPyro <https://num.pyro.ai/en/stable/>`_ - Probabilistic programming with NumPy and JAX  
* `BlackJAX <https://blackjax-devs.github.io/blackjax/>`_ - Sampling algorithms for JAX  
* `nautilus-sampler <https://github.com/Intelligent-Systems-Phystech/nautilus-sampler>`_ - Nested sampling for high-dimensional inference  

Boltzmann codes
~~~~~~~~~~~~~~~~~~~~~~~~
Cosmological Boltzmann solver for background evolution and linear perturbations compatible with PyBird

* `CLASS <http://class-code.net/>`_  
* `CosmoPower-JAX <https://github.com/dpiras/cosmopower-jax>`_ [JAX-compatible]  
* `Symbolic-Pk <https://github.com/DeaglanBartlett/symbolic_pofk>`_ [currently embedded in PyBird `here <https://github.com/pierrexyz/pybird/tree/master/pybird/symbolic.py>`_ in a JAX-compatible version]

Running with MontePython
------------------------

To run with `MontePython 3 <https://github.com/brinckmann/montepython_public>`_, once PyBird is installed as above,  

* Copy the likelihood folder `montepython/likelihoods/eftboss <https://github.com/pierrexyz/pybird/tree/master/montepython/likelihoods/eftboss>`_ to your working MontePython repository: montepython_public/montepython/likelihoods/  
* Copy the data folder `data/eftboss <https://github.com/pierrexyz/pybird/tree/master/data/eftboss>`_ to your working MontePython data folder: montepython_public/data/  
* Run the likelihood of BOSS DR12 with the input param file `montepython/eftboss.param <https://github.com/pierrexyz/pybird/tree/master/montepython/eftboss.param>`_  

* Posterior covariances for Metropolis-Hasting Gaussian proposal (in MontePython format) can be found `here <montepython/chains>`_.  

