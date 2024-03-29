import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.absolute()
DATA_PATH = ROOT_PATH / "data"

EXPERIMENTS_PATH = DATA_PATH / "experiments"
