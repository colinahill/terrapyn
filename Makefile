# lint:
# 	poetry run ruff check terrapyn/ tests/ --extend-ignore=D1,D2,D4,TID

# .PHONY: test
# # Ignore earth engine directory 'ee'
# test:
# 	pytest --doctest-modules --cov=terrapyn --cov-branch --cov-report term-missing  -vv --color=yes --ignore=terrapyn/ee

.PHONY: help build format lint sync lock test upgrade all 

help:
	@echo "Available commands:"
	@echo "  lint     - Lint the code using ruff"
	@echo "  format   - Format the code using ruff"
	@echo "  test     - Run tests using pytest"
	@echo "  sync     - Sync and compile the project using uv"
	@echo "  lock     - Lock dependencies using uv"
	@echo "  build    - Build the project using uv"
	@echo "  upgrade  - Upgrade dependencies using uv"
	@echo "  all      - Run lock, sync, format, lint, and test"

build:
	uv build

format:
	uv run ruff format terrapyn/ tests/

lint:
	uv run ruff check --fix terrapyn/ tests/

sync:
	uv sync --compile

lock:
	uv lock

test:
	uv run pytest terrapyn/ tests/

upgrade:
	uv lock --upgrade

all: lock sync
	make format
	make lint
	make test
