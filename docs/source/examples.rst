Demo Notebooks
==============

**New to PyBird?** Start with the **Correlator Tutorial** - it showcases the basic usage and key features of PyBird.

All demo notebooks are located in the ``demo/`` directory. You can run them locally or `view them on GitHub <https://github.com/pierrexyz/pybird/tree/master/demo>`_.

**Correlator Tutorial** (Start Here!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main PyBird tutorial covering:

* Basic correlator setup and configuration
* Computing power spectra and correlation functions
* JAX acceleration and neural network emulator

A quick note on the backend control:

Regardless of installation mode, you control which backend PyBird uses:

.. code-block:: python

   from pybird.config import set_jax_enabled
   
   # Use NumPy backend (works with both installation modes)
   set_jax_enabled(False)
   
   # Use JAX backend (works if JAX dependencies are available)
   set_jax_enabled(True)


**View:** `correlator.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/correlator.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/correlator.ipynb>`_

**Likelihood Tutorial**
~~~~~~~~~~~~~~~~~~~~~~~

Data analysis with PyBird:

* Loading BOSS data and configuring likelihoods
* Parameter estimation and χ² evaluation
* Survey effects and observational systematics

**View:** `likelihood.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/likelihood.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/likelihood.ipynb>`_

**Complete Parameter Inference**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level PyBird workflow:

* Parameter inference with the Run class
* MCMC sampling and optimization
* GetDist integration and posterior analysis

**View:** `run.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/run.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/run.ipynb>`_

**BOSS DR12 Analysis**
~~~~~~~~~~~~~~~~~~~~~~

Full inference pipeline with BOSS DR12 LRG data:

* Loading and configuring BOSS likelihood
* Running MCMC with real data
* Posterior analysis and visualization

**View:** `run_boss.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/run_boss.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/run_boss.ipynb>`_

**Abacus Simulations**
~~~~~~~~~~~~~~~~~~~~~~

Running PyBird with Abacus mock data:

* Configuring Abacus likelihood
* ELG, LRG, and QSO tracers
* Validation against simulations

**View:** `run_abacus.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/run_abacus.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/run_abacus.ipynb>`_

**JAX Benchmarking**
~~~~~~~~~~~~~~~~~~~~

Speed optimization and JAX features:

* Performance comparison: Standard vs JAX vs Emulator
* Up to 1000x speedup demonstration
* Vectorized batch processing

**View:** `jaxbird_benchmarking.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/jaxbird_benchmarking.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/jaxbird_benchmarking.ipynb>`_

**Fake Data Generation**
~~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic datasets:

* Creating realistic mock catalogs
* Survey geometry simulation
* Statistical validation

**View:** `fake.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/fake.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/fake.ipynb>`_

**Advanced Inference**
~~~~~~~~~~~~~~~~~~~~~~

Low-level inference control:

* Custom sampler configuration
* Fisher matrix calculations
* Taylor expansion methods

**View:** `inference.ipynb on GitHub <https://github.com/pierrexyz/pybird/blob/master/demo/inference.ipynb>`_ | `NBViewer <https://nbviewer.org/github/pierrexyz/pybird/blob/master/demo/inference.ipynb>`_

Running Locally
---------------

**Quick Setup:**

.. code-block:: bash

   # Clone and install
   git clone https://github.com/pierrexyz/pybird.git
   cd pybird
   pip install jupyter matplotlib getdist
   
   # Launch notebooks
   jupyter notebook demo/

Learning Path
-------------

1. **Start:** `correlator.ipynb` - Learn PyBird basics and JAX features
2. **Data Analysis:** `likelihood.ipynb` - Understand data fitting  
3. **Complete Workflow:** `run.ipynb` - Master parameter inference
4. **Real Data:** `run_boss.ipynb` - BOSS DR12 analysis
5. **Simulations:** `run_abacus.ipynb` - Abacus mock data
6. **Custom Analysis:** `inference.ipynb` + `fake.ipynb` - Build custom solutions
7. **Performance (optional):** `jaxbird_benchmarking.ipynb` - Test the speed of PyBird on your local machine 

Getting Help
------------

* `GitHub Issues <https://github.com/pierrexyz/pybird/issues>`_
* Email the developers: `Pierre Zhang <pierrexyz@protonmail.com>`_, `Guido D'Amico <damico.guido@gmail.com>`_ or `Alexander Reeves <alexcharlesreeves@gmail.com>`_
* Check the installation guide: :doc:`installation`