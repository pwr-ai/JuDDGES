lint_dirs := juddges scripts dashboards tests
mypy_dirs := juddges scripts dashboards tests

fix:
	pre-commit run --all-files

check:
	pre-commit run --all-files

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

# unsloth requires python 3.10 and conda environment
install_unsloth:
	conda install \
		pytorch-cuda=12.1 \
		pytorch \
		cudatoolkit \
		'xformers<0.0.27' \
		-c pytorch \
		-c nvidia \
		-c xformers \
		--yes
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	pip install flash-attn --no-build-isolation
	pip install -r requirements_unsloth.txt
