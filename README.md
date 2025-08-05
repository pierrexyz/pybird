# PyBird
**The Python code for Biased tracers in redshift space**  

- EFT predictions for correlators of biased tracers in redshift space  
- Likelihoods of galaxy-clustering data with EFT predictions  

[![](https://img.shields.io/badge/arXiv-2003.07956%20-red.svg)](https://arxiv.org/abs/2003.07956)
[![](https://img.shields.io/badge/arXiv-2507.20990%20-red.svg)](https://arxiv.org/abs/2507.20990)
[![](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/pierrexyz/pybird/blob/master/LICENSE)
[![](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://pierrexyz.github.io/pybird/)

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
- JAX acceleration with model-independent neural network emulators

#### Likelihoods with EFT predictions
Currently available: 
> [BOSS DR12 LRG 2pt full-shape + rec. bao](montepython/likelihoods/eftboss)  
> [BOSS-like simulations - PT|ABFG-D|patchy](montepython/likelihoods/mockboss)  
> [eBOSS DR16 QSO 2pt full-shape](montepython/likelihoods/efteboss)

Soon available: 
> [BOSS DR12 LRG 3pt full-shape]

## Documentation
Read the docs at [https://pierrexyz.github.io/pybird](https://pierrexyz.github.io/pybird).

## Installation

PyBird supports multiple installation modes for different use cases:

### Repository Installation

**For the up-to-date master version**, install directly from the GitHub repository:

```bash
git clone https://github.com/pierrexyz/pybird.git
cd pybird
```

Choose your installation mode:

**Core PyBird**
```bash
pip install -e .
```
- **Includes**: NumPy, SciPy, H5PY, emcee, iminuit, pyyaml, fftlog + documentation + development tools

**PyBird-JAX**
```bash
pip install -e ".[jax]"
```
- **Includes**: Everything in Core + JAX ecosystem (jax, jaxlib, flax, optax, numpyro, blackjax, nautilus-sampler)

**Backend Control**

Regardless of installation mode, you control which backend PyBird uses:

```python
from pybird.config import set_jax_enabled

# Use NumPy backend (works with both installation modes)
set_jax_enabled(False)

# Use JAX backend (works if JAX dependencies are available)
set_jax_enabled(True)
```

### PyPI Installation

PyBird is available on PyPI with both core and JAX installation modes:

**Core PyBird (NumPy backend)**
```bash
pip install --no-binary pybird-lss pybird-lss
```
- **Includes**: NumPy, SciPy, H5PY, emcee, iminuit, pyyaml, fftlog-lss + documentation + development tools

**PyBird-JAX (JAX acceleration)**
```bash
pip install --no-binary pybird-lss "pybird-lss[jax]"
```
- **Includes**: Everything in Core + JAX ecosystem (jax, jaxlib, flax, optax, numpyro, blackjax, nautilus-sampler)

**Note**: The `--no-binary` flag is required to ensure data files are included in the installation.

### Testing Your Installation

After installation, if you used the repo install you can test that everything works:

```bash
cd tests
python run_tests.py
```

Expected output: `🎉 All tests passed!`

For detailed installation instructions, troubleshooting, and testing options, see the [documentation](https://pybird.readthedocs.io).


## Getting started
Checkout our full list of tuturials [here](https://github.com/pierrexyz/pybird/tree/master/demo). 

The algebra on which PyBird is based is computed in the mathematica notebook [cbird.nb](demo/cbird). 


## Attribution
* Devs:
    * [Pierre Zhang](mailto:pierrexyz@protonmail.com) [main author]
    * [Guido D'Amico](mailto:damico.guido@gmail.com) [main author], 
    * [Alexander Reeves](alexcharlesreeves@gmail.com) [JAX/emulators]
* License: MIT
* Special thanks to: Marco Bonici, Thomas Colas, Yan Lai, Zhiyu Lu, Arnaud de Mattia, Théo Simon, Luis Ureña, Henry Zheng

When using PyBird in a publication, please acknowledge the code by citing the following paper:  
> G. D’Amico, L. Senatore, and P. Zhang, "Limits on wCDM from the EFTofLSS with the PyBird code", JCAP 01 (2021) 006, [2003.07956](https://arxiv.org/abs/2003.07956)

For PyBird-JAX: 
> A. Reeves, P. Zhang, and H. Zheng, "PyBird-JAX: Accelerated inference in large-scale structure with model-independent emulation of one-loop galaxy power spectra", [2507.20990](https://arxiv.org/abs/2507.20990)

The BibTeX entry are:
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

@article{Reeves:2025bxn,
    author = "Reeves, Alexander and Zhang, Pierre and Zheng, Henry",
    title = "{PyBird-JAX: Accelerated inference in large-scale structure with model-independent emulation of one-loop galaxy power spectra}",
    eprint = "2507.20990",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "7",
    year = "2025"
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

> PT challenge blind simulations: [2003.08277](https://arxiv.org/abs/2003.08277), with description and submitted results [here](https://www2.yukawa.kyoto-u.ac.jp/~takahiro.nishimichi/data/PTchallenge/)

> eBOSS DR16 data: [2007.08991](https://arxiv.org/abs/2007.08991), catalogs: [2007.09000](https://arxiv.org/abs/2007.09000), EZmocks (for covariance estimation): [1409.1124](https://arxiv.org/abs/1409.1124)

> eBOSS DR16 QSO power spectrum + survey mask measurements: from [2106.06324](https://arxiv.org/abs/2106.06324)

> eBOSS EFT likelihood: [2210.14931](https://arxiv.org/abs/2210.14931)
 
*** Disclaimer: due to updates in the data and the prior definition, it is possible that results obtained with up-to-date likelihoods differ slightly with the ones presented in the articles. 


