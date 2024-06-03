lint_dirs := juddges scripts dashboards tests
mypy_dirs := juddges scripts dashboards tests

fix:
	ruff check $(lint_dirs) setup.py --fix
	ruff format $(lint_dirs)

check:
	ruff check $(lint_dirs)
	ruff format $(lint_dirs) --check

test:
	coverage run -m pytest
	coverage report -mi

nbdev:
	nbdev_prepare
	nbdev_update

all: fix check test nbdev

install:
	pip install -r requirements.txt
