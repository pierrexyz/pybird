.. PyBird documentation master file, created by
   sphinx-quickstart on Wed Mar 18 14:19:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyBird
==================================

PyBird is a code written in Python 3, designed for evaluating the multipoles of the power spectrum of biased tracers in redshift space. In general, PyBird can evaluate the power spectrum of matter or biased tracers in real or redshift space. The equations on which PyBird is based can be found in arXiv:1610.09321. or arXiv:1909.05271. The main technology used by the code is the FFTLog, used to evaluate the one-loop power spectrum and the IR resummation, see Sec.~4.1 in arXiv:2003.XXXXX for details.

PyBird is designed for a fast evaluation of the power spectra, and can be easily inserted in a data analysis pipeline. In fact, it is a standalone tool whose input is the linear matter power spectrum which can be obtained from any Boltzmann code, such as CAMB or CLASS. The Pybird output can be used in a likelihood code which can be part of the routine of a standard MCMC sampler. The design is modular and concise, such that parts of the code can be easily adapted to other case uses (e.g., power spectrum at two loops or bispectrum).

PyBird can be used in different ways. The code can evaluate the power spectrum either given one set of EFT parameters, or independently of the EFT parameters. If the former option is faster, the latter is useful for subsampling or partial marginalization over the EFT parameters, or to Taylor expand around a fiducial cosmology for efficient parameter exploration, see e.g. arXiv:1909.07951. PyBird runs in less than a second on a laptop.

.. toctree::
   :maxdepth: 2
   :caption: Class documentation
   
   classes/common
   classes/bird
   classes/fftlog
   classes/nonlinear
   classes/resum
   classes/projection
   

