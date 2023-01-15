import os

import setuptools

version = os.getenv("VERSION", "0.0.5")

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("dev-requirements.txt") as f:
    dev_install_requires = f.read().splitlines()

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
