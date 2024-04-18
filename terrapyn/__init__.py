from . import (
    bigquery,
    conversion,
    dask_utils,
    ee,
    indices,
    io,
    logger,
    params,
    space,
    stats,
    time,
    utils,
    validation,
)

__all__ = [
    "time",
    "utils",
    "stats",
    "validation",
    "indices",
    "dask_utils",
    "conversion",
    "space",
    "params",
    "io",
    "ee",
    "logger",
    "bigquery",
]
__version__ = "0.1.1"

from pathlib import Path

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PACKAGE_ROOT_DIR / "tests" / "data"
