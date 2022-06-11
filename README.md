# terrapyn

[![PyPI version](https://badge.fury.io/py/terrapyn.svg)](https://badge.fury.io/py/terrapyn)
[Coverage](.github/coverage.svg)
![versions](https://img.shields.io/pypi/pyversions/terrapyn.svg)
[![GitHub license](https://img.shields.io/pypi/l/terrapyn)](https://github.com/colinahill/terrapyn/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Toolkit to manipulate Earth observations and models. Designed to work with `Pandas` and `Xarray` data structures homogeneously, implementing `Dask` optimizations where possible.

The name is pronounced the same as "terrapin", a type of [fresh water turtle](https://en.wikipedia.org/wiki/Terrapin)

- Documentation: https://colinahill.github.io/terrapyn.
- Free software: BSD-3-Clause

## Setup

Via pip:

```bash
pip install terrapyn
```

or from source:

```bash
git clone https://github.com/colinahill/terrapyn.git
cd terrapyn
pip install .

# OR for development:
pip install -e .[dev]
```

Then, the ipython kernel can be installed to use Jupyter notebooks
```
ipython kernel install --user --name=terrapyn
```
<!--

An Anaconda Python distribution is required. Either `Conda` or `Miniconda` are suitable: see [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Once (mini)conda is installed, a new environment can be created with all required dependencies:

```
conda env create -f environment.yml
```

Then the package can be installed

```
pip install .

# OR for development
pip install -e .
```


 -->
