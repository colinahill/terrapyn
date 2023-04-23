# terrapyn

![Code Coverage](https://img.shields.io/badge/Coverage-83%25-yellowgreen.svg)
[![PyPI version](https://badge.fury.io/py/terrapyn.svg)](https://badge.fury.io/py/terrapyn)
![versions](https://img.shields.io/pypi/pyversions/terrapyn.svg)
[![GitHub license](https://img.shields.io/pypi/l/terrapyn)](https://github.com/colinahill/terrapyn/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Toolkit to manipulate Earth Observation Data: Remote Sensing, Climate and Weather models. Designed to work with `Pandas`/`GeoPandas` and `Xarray` data structures, implementing `Dask` where possible.

The name is pronounced the same as "terrapin", a type of [fresh water turtle](https://en.wikipedia.org/wiki/Terrapin)

- Documentation: https://colinahill.github.io/terrapyn.
- Free software: BSD-3-Clause

## Setup/Installation

### Python environment setup
An environment with Python version `3.10` or later is required. If you don't have this, it can be created using [Pyenv](https://github.com/pyenv/pyenv) which should be installed first. After installing Pyenv, download and install Python `3.10` using

```bash
pyenv install 3.10
```

If you already have Python version `3.10` or later you can skip this step.

### Install

#### Via Pip
The package can be installed in an existing Python environment via pip:

```bash
pip install terrapyn
```

#### From source
Clone the repo and install the package:

```bash
git clone https://github.com/colinahill/terrapyn.git && cd terrapyn
```

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. In the case where Poetry doesn't automatically find the correct Python path, you can set it with

```bash
pyenv local 3.10
poetry env use 3.10
```

Then install the package
```bash
poetry install  # Creates a virtualenv and installs package into it
poetry shell  # Opens a sub-shell in the virtualenv
```



