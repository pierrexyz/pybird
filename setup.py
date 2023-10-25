from setuptools import setup

setup(
    name='pybird',
    version='0.2.0',
    description='EFT predictions for biased tracers in redshift space.',
#    url='https://github.com/pierrexyz/pybird',
    author="Pierre Zhang and Guido D'Amico",
    license='MIT',
    packages=['pybird'],
    install_requires=['numpy', 'scipy', 'h5py', 'pyyaml'],
    package_dir = {'pybird': 'pybird'},
    zip_safe=False,

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python",
    ],
)
