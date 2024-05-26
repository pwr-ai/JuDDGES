lint_dirs := juddges scripts dashboards tests
mypy_dirs := juddges scripts dashboards tests

fix:
	ruff check $(lint_dirs) --fix
	ruff format $(lint_dirs)

check:
	ruff check $(lint_dirs)
	ruff format $(lint_dirs) --check

check-types:
	mypy --install-types --non-interactive $(mypy_dirs)

test:
	coverage run -m pytest
	coverage report -mi

all: check test

install:
	pip install -r requirements.txt
	pip install flash-attn --no-build-isolation
