lint_dirs := juddges scripts tests
mypy_dirs := juddges scripts tests

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

install: cuda := 128
install:
	pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu$(cuda)
	pip install -r requirements.txt
	pip install flash-attn==2.7.4.post1 --no-build-isolation

install_macos:
	# bitsandbytes are not supported on macOS (https://github.com/pwr-ai/JuDDGES/pull/41)
	grep -v "bitsandbytes" requirements.txt | pip install -r /dev/stdin

install_cpu:
	pip install --find-links https://download.pytorch.org/whl/cpu -r requirements.txt

install_unsloth: cuda := 128
install_unsloth:
	conda install \
		pytorch-cuda=$(cuda) \
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
