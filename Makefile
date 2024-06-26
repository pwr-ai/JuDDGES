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

install_cpu:
	pip install --find-links https://download.pytorch.org/whl/cpu -r requirements.txt

# unsloth requires python 3.10
# requires conda environment
install_unsloth:
	conda install --yes pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	conda install pyg -c pyg
	pip install flash-attn --no-build-isolation
	pip install -r requirements.txt
