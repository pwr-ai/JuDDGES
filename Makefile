lint_dirs := juddges scripts dashboards tests
mypy_dirs := juddges scripts dashboards tests

fix:
	ruff check $(lint_dirs) --fix
	ruff format $(lint_dirs)

check:
	ruff check $(lint_dirs)
	ruff format $(lint_dirs) --check

test:
	coverage run -m pytest
	coverage report -mi


all: check test

nbdev: \
	all
	nbdev_prepare
	nbdev_update

install:
	pip install -r requirements.txt
