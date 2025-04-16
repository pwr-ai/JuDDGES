from contextlib import contextmanager

from datasets import disable_caching, enable_caching


@contextmanager
def disable_hf_dataset_cache():
    disable_caching()
    yield
    enable_caching()
