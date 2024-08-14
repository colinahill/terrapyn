.PHONY: lint
lint:
	poetry run ruff check terrapyn/ tests/ --extend-ignore=D1,D2,D4,TID

.PHONY: test
# Ignore earth engine directory 'ee'
test:
	pytest --doctest-modules --cov=terrapyn --cov-branch --cov-report term-missing  -vv --color=yes --ignore=terrapyn/ee

