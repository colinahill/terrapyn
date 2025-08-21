# terrapyn

![Code Coverage](https://img.shields.io/badge/Coverage-83%25-yellowgreen.svg)
[![PyPI version](https://badge.fury.io/py/terrapyn.svg)](https://badge.fury.io/py/terrapyn)
![versions](https://img.shields.io/pypi/pyversions/terrapyn.svg)
![GitHub license](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

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

The package can be installed in an existing Python environment via pip:

```bash
pip install terrapyn
```

Or the repo can be cloned from source:

```bash
git clone https://github.com/colinahill/terrapyn.git && cd terrapyn
```

## Development
This project uses [Astral uv](https://docs.astral.sh/uv/) to manage dependencies, which should be installed first. Then the environment is created with `uv venv`.

A `Makefile` contains most/all of the required tools for code quality, testing and publishing. Run `make help` to see available commands.