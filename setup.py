import os
import setuptools

version = os.getenv("VERSION", "0.0.1")

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "setuptools",
    "notebook",
    "numpy",
    "pandas",
    "xarray",
    "scipy",
    "dask",
    "typing",
    "netCDF4",
    "matplotlib",
    "seaborn",
    "bottleneck",
    # "geopandas",
    # "shapely",
]

dev_install_requires = [
    "mkdocs-material",
    "pytest",
    "pytest-cov",
    "flake8",
    "flake8-docstrings",
    "flake8-import-order",
    "flake8-colors",
    "black",
    "mypy",
    "pre-commit",
    "freezegun",
]

setuptools.setup(
    name="terrapyn",
    version=version,
    url="https://github.com/colinahill/terrapyn",
    description="Toolkit to manipulate Earth observations and models.",
    author="Colin Hill",
    author_email="colinalastairhill@gmail.com>",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # readme = "README.md",
    license="BSD-3-Clause",
    install_requires=install_requires,
    extras_require={"dev": dev_install_requires},
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    include=("LICENSE", "terrapyn/py.typed"),
)
