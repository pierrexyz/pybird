from setuptools import setup

# Core runtime dependencies only (without fftlog to avoid dependency conflicts)
core_deps = [
    'numpy>=1.19.0',
    'scipy>=1.5.0', 
    'h5py>=3.0.0',
    'pyyaml>=5.0.0',
    'emcee>=3.0.0',
    'iminuit>=2.0.0',
    'platformdirs>=3.0.0',
]

# fftlog dependency (install manually with --no-deps to avoid JAX conflicts)
fftlog_deps = [
    'fftlog @ git+https://github.com/pierrexyz/fftlog',
]

# Documentation dependencies
docs_deps = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=1.0.0',
    'nbsphinx>=0.8.0',
    'jupyter>=1.0.0',
    'ipython>=7.0.0',
    'matplotlib>=3.3.0',
]

# Development dependencies
dev_deps = [
    'pytest>=6.0.0',
    'pytest-cov>=2.0.0',
    'black>=21.0.0',
    'isort>=5.0.0'
]

# JAX ecosystem dependencies (added to core)
jax_deps = [
    'jax>=0.4.0',
    'jaxlib>=0.4.0',
    'interpax>=0.2.0',
    'optax>=0.1.0',
    'flax>=0.6.0',
    'numpyro>=0.12.0',
    'blackjax>=0.9.0',
    'nautilus-sampler>=0.5.0'
]

setup(
    name='pybird',
    version='0.3.0',
    description='EFT predictions for biased tracers in redshift space.',
    author="Pierre Zhang, Guido D'Amico and Alexander Reeves",
    license='MIT',
    packages=['pybird'],
    install_requires=core_deps,
    extras_require={
        'fftlog': fftlog_deps,
        'jax': jax_deps,
        'complete': fftlog_deps,
        'complete-jax': fftlog_deps + jax_deps,
        'docs': docs_deps,
        'dev': dev_deps,
        'all': fftlog_deps + jax_deps + docs_deps + dev_deps,
    },
    package_dir={'pybird': 'pybird'},
    zip_safe=False,
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
