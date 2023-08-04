

# PyBird
**The Python code for Biased tracers in redshift space**  

- EFT predictions for correlators of biased tracers in redshift space  
- Likelihoods of galaxy-clustering data with EFT predictions  

[![](https://img.shields.io/badge/arXiv-2003.07956%20-red.svg)](https://arxiv.org/abs/2003.07956)
[![](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/pierrexyz/pybird/blob/master/LICENSE)
[![](https://readthedocs.org/projects/pybird/badge/?version=latest)](https://pybird.readthedocs.io/en/latest/?badge=latest)


## General info
#### Fast correlator computation
- One-loop EFT predictions for two-point (2pt) functions:  
      - dark matter or biased tracers  
      - real or redshift space  
      - Fourier (power spectrum) or configuration space (correlation function)  
- Additional modeling:  
      - geometrical (AP) distortion  
      - survey mask  
      - binning  
      - exact-time dependence  
      - wedges / ~~P~~-statistics  
      - and more...  

#### Likelihoods with EFT predictions
Currently available: 
> [BOSS DR12 LRG 2pt full-shape + rec. bao](montepython/likelihoods/eftboss)  
> [eBOSS DR16 QSO 2pt full-shape](montepython/likelihoods/efteboss)

Soon available: 
> [BOSS DR12 LRG 3pt full-shape]

## Dependencies
PyBird depends on the numerical libraries [NumPy](https://numpy.org/) and [SciPy](http://scipy.org/).  

The following packages are not strictly neccessary but recommended to run the cookbooks:
* PyBird has extra compatibility with **[CLASS](https://lesgourg.github.io/class_public/class.html)**.  
* PyBird likelihoods are integrated within **[MontePython 3](https://github.com/brinckmann/montepython_public)**. 
* PyBird likelihoods are showcased with **[iminuit](https://iminuit.readthedocs.io/)** and **[emcee](https://emcee.readthedocs.io/)**. 

## Installation
Clone the repo, and install it as a Python package using `pip`:
```
git clone https://github.com/pierrexyz/pybird.git
cd pybird
pip install .
```
That's it, now you can simply `import pybird` from wherever in your projects.

## Getting Started -- likelihood
If you are a **[MontePython 3](https://github.com/brinckmann/montepython_public)** user, likelihoods can be installed 'with less than a cup of coffee'.
* Clone and install PyBird as above
* Copy the likelihood folder [montepython/likelihoods/eftboss](montepython/likelihoods/eftboss) to your working MontePython repository: montepython_public/montepython/likelihoods/ 
* Copy the data folder [data/eftboss](data/eftboss) to your working MontePython data folder: montepython_public/data/
* Try to run the likelihood of BOSS DR12 with the input param file [montepython/eftboss.param](montepython/eftboss.param)

*** Note (23/03/08): MontePython v3.5 seems to have some incompatibilities with the PyBird likelihood related to the function `data.need_cosmo_arguments()`. To resolve it, see this [pull-request](https://github.com/brinckmann/montepython_public/pull/276). 

That's it, you are all set!

* If any doubt, benchmark $\Lambda$CDM posteriors are shown [here](notebooks/potatoes.ipynb).
* Posterior covariances for Metropolis-Hasting Gaussian proposal (in MontePython format) can be found [here](montepython/chains). 

## Cookbooks
Alternatively, if you are curious, here are three cookbooks that should answer the following questions: 
* [Correlator](notebooks/correlator.ipynb): How to ask PyBird to compute EFT predictions? 
* [Likelihood](notebooks/likelihood.ipynb): How does the PyBird likelihood work? 
* [Data](notebooks/datastruct.ipynb): What are the data read by PyBird likelihood?
* [cbird](notebooks/cbird.nb): What is the algebra of the EFT predictions PyBird is based on?

## Documentation
Read the docs at [https://pybird.readthedocs.io](https://pybird.readthedocs.io).

## Attribution
* Written by [Pierre Zhang](mailto:pierrexyz@protonmail.com) and [Guido D'Amico](mailto:damico.guido@gmail.com)
* License: MIT
* Special thanks to: Arnaud de Mattia, Thomas Colas, Théo Simon, Luis Ureña

When using PyBird in a publication, please acknowledge the code by citing the following paper:  
> G. D’Amico, L. Senatore and P. Zhang, "Limits on wCDM from the EFTofLSS with the PyBird code", JCAP 01 (2021) 006, [2003.07956](https://arxiv.org/abs/2003.07956)

The BibTeX entry for it is:
```
@article{DAmico:2020kxu,
    author = "D'Amico, Guido and Senatore, Leonardo and Zhang, Pierre",
    title = "{Limits on $w$CDM from the EFTofLSS with the PyBird code}",
    eprint = "2003.07956",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2021/01/006",
    journal = "JCAP",
    volume = "01",
    pages = "006",
    year = "2021"
}
```

We would be grateful if you also cite the theory papers when relevant:  
> The Effective-Field Theory of Large-Scale Structure: [1004.2488](https://arxiv.org/abs/1004.2488), [1206.2926](https://arxiv.org/abs/1206.2926)  

> One-loop power spectrum of biased tracers in redshift space: [1610.09321](https://arxiv.org/abs/1610.09321)  

> Exact-time dependence: [2005.04805](https://arxiv.org/abs/2005.04805), [2111.05739](https://arxiv.org/abs/2111.05739)

> Wedges / ~~P~~-statistics: [2110.00016](https://arxiv.org/abs/2110.00016)

When using the likelihoods, here are some relevant references:  
> BOSS DR12 data: [1607.03155](https://arxiv.org/abs/1607.03155), catalogs: [1509.06529](https://arxiv.org/abs/1509.06529), patchy mocks (for covariance estimation): [1509.06400](https://arxiv.org/abs/1509.06400)

> BOSS DR12 LRG power spectrum measurements: from [2206.08327](https://arxiv.org/abs/2206.08327), using **[Rustico](https://github.com/hectorgil/Rustico)**

> BOSS DR12 LRG correlation function measurements: from [2110.07539](https://arxiv.org/abs/2110.07539), using **[FCFC](https://github.com/cheng-zhao/FCFC)**

> BOSS DR12 LRG rec. bao parameters: from [2003.07956](https://arxiv.org/abs/2003.07956), based on post-reconstructed measurements from [1509.06373](https://arxiv.org/abs/1509.06373)

> BOSS DR12 survey mask measurements: following [1810.05051](https://arxiv.org/abs/1810.05051) with integral constraints and consistent normalization following [1904.08851](https://arxiv.org/abs/1904.08851), from **[fkpwin](https://github.com/pierrexyz/fkpwin)**, using **[nbodykit](https://nbodykit.readthedocs.io/)**

> BOSS EFT likelihood: besides the PyBird paper, see also: [1909.05271](https://arxiv.org/abs/1909.05271), [1909.07951](https://arxiv.org/abs/1909.07951), [2110.07539](https://arxiv.org/abs/2110.07539)

> eBOSS DR16 data: [2007.08991](https://arxiv.org/abs/2007.08991), catalogs: [2007.09000](https://arxiv.org/abs/2007.09000), EZmocks (for covariance estimation): [1409.1124](https://arxiv.org/abs/1409.1124)

> eBOSS DR16 QSO power spectrum + survey mask measurements: from [2106.06324](https://arxiv.org/abs/2106.06324)

> eBOSS EFT likelihood: [2210.14931](https://arxiv.org/abs/2210.14931)
 
*** Disclaimer: due to updates in the data and the prior definition, it is possible that results obtained with up-to-date likelihoods differ slightly with the ones presented in the articles. 
