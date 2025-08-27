import importlib.metadata

from . import (
	conversion,
	dask_utils,
	indices,
	io,
	logger,
	params,
	scoring,
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
	"logger",
	"scoring",
]

__version__ = importlib.metadata.version("terrapyn")

from pathlib import Path

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PACKAGE_ROOT_DIR / "tests" / "data"
