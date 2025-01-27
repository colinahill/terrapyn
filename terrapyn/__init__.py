import importlib.metadata

from . import (
	bq,
	conversion,
	dask_utils,
	# ee,
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
	# "ee",
	"logger",
	"bq",
	"scoring",
]

__version__ = importlib.metadata.version("terrapyn")

from pathlib import Path

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PACKAGE_ROOT_DIR / "tests" / "data"
