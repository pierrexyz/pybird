[![](https://img.shields.io/badge/arXiv-2003.07956%20-red.svg)](https://arxiv.org/abs/2003.07956)
[![](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/pierrexyz/pybird/blob/master/LICENSE)
[![](https://readthedocs.org/projects/pybird/badge/?version=latest)](https://pybird.readthedocs.io/en/latest/?badge=latest)

# PyBird: Python code for Biased tracers in redshift space
* Written by [Pierre Zhang](mailto:pierrexyz@protonmail.com) and [Guido D'Amico](mailto:damico.guido@gmail.com)
* License: MIT

## General info
PyBird is a code written in Python 3, designed for evaluating the multipoles of the power spectrum of biased tracers in redshift space.
In general, PyBird can evaluate the power spectrum of matter or biased tracers in real or redshift space.
The equations on which PyBird is based can be found in [arXiv:1610.09321](https://arxiv.org/abs/1610.09321) or [arXiv:1909.05271](https://arxiv.org/abs/1909.05271). 
The main technology used by the code is the [FFTLog](https://jila.colorado.edu/~ajsh/FFTLog/index.html), used to evaluate the one-loop power spectrum and the IR resummation, see Sec.~4.1 in [arXiv:2003.07956](https://arxiv.org/abs/2003.07956) for details. 

PyBird is designed for a fast evaluation of the power spectra, and can be easily inserted in a data analysis pipeline.
In fact, it is a standalone tool whose input is the linear matter power spectrum which can be obtained from any Boltzmann code, such as [CAMB](https://camb.info/) or [CLASS](http://class-code.net/).
The Pybird output can be used in a likelihood code which can be part of the routine of a standard MCMC sampler.
The design is modular and concise, such that parts of the code can be easily adapted to other case uses (e.g., power spectrum at two loops or bispectrum).

PyBird can be used in different ways.
The code can evaluate the power spectrum either given one set of EFT parameters, or independently of the EFT parameters.
If the former option is faster, the latter is useful for subsampling or partial marginalization over the EFT parameters, or to Taylor expand around a fiducial cosmology for efficient parameter exploration, see e.g. [arXiv:1909.07951](https://arxiv.org/abs/1909.07951). PyBird runs in less than a second on a laptop. 

## Dependencies
PyBird depends on the numerical libraries [NumPy](https://numpy.org/) and [SciPy](http://scipy.org/).

## Installation
PyBird is pip-installable.
Just clone the repo, and install it as a Python package in development mode so your changes will be immediately available:
```
git clone https://github.com/pierrexyz/pybird.git
pip install --editable pybird --upgrade
```
That's it, now you can simply `from pybird import pybird` from wherever in your projects.

## Getting Started
If you are a [MontePython 3] user, the code can be installed 'with less than a cup of coffee'.
* Download and install pybird as above
* Link the files from the folder pybird/montepython_tree/ to your_montepython/
* Try to run the likelihood of BOSS DR12 CMASS NGC with the param file provided in pybird/montepython_tree/input/eft_highzNGC.param

That's it, you are all set!

Alternatively, if you'd like to have a glimpse at the code, [here](https://github.com/pierrexyz/pybird/blob/master/run_pybird.ipynb) you can find a simple Jupyter notebook to start with.

## Documentation
Read the docs at [https://pybird.readthedocs.io](https://pybird.readthedocs.io).

## Architecture
PyBird consists of the following classes:
* Bird: Main class which contains the power spectrum and correlation function, given a cosmology and a set of EFT parameters.
* Nonlinear: given a Bird() object, computes the one-loop power spectrum and one-loop correlation function.
* Resum: given a Bird() object, performs the IR-resummation of the power spectrum.
* Projection: given a Bird() object, applies geometrical effects on the power spectrum: Alcock-Paczynski effect, window functions, fiber collisions, binning.
* Common: containing shared objects among the other classes, such as k-array, multipole decomposition, etc.

## Extras
You can also find in this repositery:
* run_pybird.ipynb: a Jupyter notebook containing examples to start with
* montepython_tree : an explicit integration in the MCMC sampler [MontePython 3](https://github.com/brinckmann/montepython_public)
* cbird.nb : a Mathematica notebook on which PyBird algebra is based
* tbird : Taylor approximation around a fiducial cosmology of the bird power spectrum. See [arXiv:1909.07951](https://arxiv.org/abs/1909.07951). Some documentation can be found in [cbird](https://github.com/pierrexyz/cbird/) (not supported anymore)

## Citation
When using PyBird in a publication, please acknowledge the code by citing the following paper: 
* [arXiv:2003.07956](https://arxiv.org/abs/2003.07956): "Limits on wCDM from the EFTofLSS with the PyBird code" by Guido D’Amico, Leonardo Senatore, Pierre Zhang

The BibTeX entry for it is:

```
@article{DAmico:2020kxu,
      author         = "D'Amico, Guido and Senatore, Leonardo and Zhang, Pierre",
      title          = "{Limits on $w$CDM from the EFTofLSS with the PyBird
                        code}",
      year           = "2020",
      eprint         = "2003.07956",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      SLACcitation   = "%%CITATION = ARXIV:2003.07956;%%"
}
```
