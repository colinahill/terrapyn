# terrapyn

![coverage](https://raw.githubusercontent.com/colinahill/terrapyn/badges/.badges/main/coverage.svg)
[![PyPI version](https://badge.fury.io/py/terrapyn.svg)](https://badge.fury.io/py/terrapyn)
![versions](https://img.shields.io/pypi/pyversions/terrapyn.svg)
[![GitHub license](https://img.shields.io/pypi/l/terrapyn)](https://github.com/colinahill/terrapyn/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Toolkit to manipulate Earth observations and Climate / Weather models. Designed to work with `Pandas`/`GeoPandas` and `Xarray` data structures, implementing `Dask` where possible.

The name is pronounced the same as "terrapin", a type of [fresh water turtle](https://en.wikipedia.org/wiki/Terrapin)

- Documentation: https://colinahill.github.io/terrapyn.
- Free software: BSD-3-Clause

## Setup

The package can be installed in an existing Python environment via pip:

```bash
pip install terrapyn
```

OR from source:

```bash
git clone https://github.com/colinahill/terrapyn.git
cd terrapyn
pip install .

# OR for development:
pip install -e .[dev]
```
