import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.absolute()
DATA_PATH = ROOT_PATH / "data"

EXPERIMENTS_PATH = DATA_PATH / "experiments"

# Dataset paths
DATASETS_PATH = DATA_PATH / "datasets"
FINE_TUNING_DATASETS_PATH = DATASETS_PATH / "fine_tuning"
