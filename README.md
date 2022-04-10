# terrapyn

[![PyPI version](https://badge.fury.io/py/terrapyn.svg)](https://badge.fury.io/py/terrapyn)
[![Coverage](.github/coverage.svg])(.github/coverage.svg)
![versions](https://img.shields.io/pypi/pyversions/terrapyn.svg)
[![GitHub license](https://img.shields.io/pypi/l/terrapyn)](https://github.com/colinahill/terrapyn/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Toolkit to manipulate Earth observations and models. Designed to work with `Pandas` and `Xarray` data structures homogeneously, implementing `Dask` optimizations.

The name is pronounced the same as "terrapin", a type of [fresh water turtle](https://en.wikipedia.org/wiki/Terrapin)

- Documentation: https://colinahill.github.io/terrapyn.
- Free software: BSD-3-Clause

## Setup

An Anaconda Python distribution is required, as this significantly simplifies installation.

Either `Conda` or `Miniconda` are suitable: see [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Once the conda environment is installed, this repo can be installed:

Via pip:

```bash
pip install terrapyn
```

or from source:

```bash
git clone https://github.com/colinahill/terrapyn.git
cd terrapyn
pip install .

# OR
python setup.py install

# OR for development:
pip install -e .[dev]
```
