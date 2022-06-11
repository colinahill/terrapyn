import os

import setuptools

version = os.getenv("VERSION", "0.0.2")

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "setuptools",
    "matplotlib",
    "xarray",
    "pandas",
    "dask",
    "bottleneck",
    "geopandas",
    "netcdf4",
    "pygeos",
    "proj",
    "shapely",
    "scipy",
    "numpy",
    "ipykernel",
    "pyarrow",
    "notebook",
    # pyshp
    # shapely --no-binary shapely
    # "cfgrib",
    # "eccodes",
]

dev_install_requires = [
    "pytest",
    "pytest-cov",
    "flake8",
    "flake8-docstrings",
    "flake8-import-order",
    "flake8-colors",
    "black",
    "mypy",
    "isort",
    "pre-commit",
    "freezegun",
    "mkdocs-material",
    "mkdocs-gen-files",
    "mkdocstrings-python",
    "mkdocs-literate-nav",
    "mkdocs-section-index",
    "mkdocs-include-markdown-plugin",
    "coverage-badge",
]

setuptools.setup(
    name="terrapyn",
    version=version,
    url="https://github.com/colinahill/terrapyn",
    description="Toolkit to manipulate Earth observations and models.",
    author="Colin Hill",
    author_email="colinalastairhill@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    readme="README.md",
    license="BSD-3-Clause",
    install_requires=install_requires,
    packages=setuptools.find_packages(exclude=("site",)),
    extras_require={"dev": dev_install_requires},
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    include=("LICENSE", "terrapyn/py.typed"),
)
