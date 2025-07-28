PyBird Documentation
===================

*PyBird: Python for Biased Tracers in Redshift Space*

PyBird is a Python package for computing power spectra and correlation functions for biased tracers in redshift space using the Effective Field Theory of Large Scale Structure (EFTofLSS).

What is PyBird?
---------------

PyBird provides fast and accurate predictions for:

* **One-loop EFT predictions** for two-point functions of dark matter or biased tracers
* **Real and redshift space** calculations
* **Fourier space (power spectrum)** and **configuration space (correlation function)** outputs
* **JAX acceleration** for high-performance computing with neural network emulators
* **Additional modeling** including geometrical distortion, survey effects, and exact-time dependence

Getting Started
---------------

**New to PyBird?** Start with the **Correlator Tutorial** notebook which showcases the basic usage and features of PyBird.

.. toctree::
   :maxdepth: 1
   :caption: Demo Notebooks

   examples

The demo notebooks provide hands-on tutorials for all PyBird capabilities. They are the best way to learn PyBird!

Quick Setup
-----------

.. toctree::
   :maxdepth: 1
   :caption: Setup Guide

   installation

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/modules

Key Features
------------

* **Fast correlator computation** with optimized algorithms
* **Neural network emulator** for 1000x speedup in parameter inference
* **JAX acceleration** with JIT compilation, vectorization, and automatic differentiation
* **MontePython integration** for cosmological parameter inference
* **BOSS and eBOSS likelihoods** for real data analysis
* **Comprehensive API** for custom analyses

Developers
----------

* Pierre Zhang [main author]
* Guido D'Amico [main author]
* Alexander Reeves

Additional Resources
--------------------

* `GitHub Repository <https://github.com/pierrexyz/pybird>`_
* `arXiv Paper <https://arxiv.org/abs/2003.07956>`_
* `MontePython Integration <https://github.com/brinckmann/montepython_public>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Special Thanks For Contributions
--------------------------------

* Marco Bonici
* Thomas Colas
* Yan Lai
* Arnaud de Mattia
* Théo Simon
* Luis Ureña
* Henry Zheng
* Zhiyu Lu