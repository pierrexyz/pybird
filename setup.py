from setuptools import setup

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Environment :: Console",
    "Programming Language :: Python",
]

setup(name='pybird',
      version='0.1',
      description='PyBird evaluates the power spectrum or its multipoles of matter or biased tracers in real or redshift space',
      url='https://github.com/pierrexyz/pybird',
      author="Pierre Zhang and Guido D'Amico",
      license='MIT',
      packages=['pybird'],
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      install_requires=[
        'numpy',
        'scipy'
      ],
      zip_safe=False)