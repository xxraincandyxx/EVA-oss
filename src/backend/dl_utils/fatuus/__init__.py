# __init__.py

import os
from pathlib import Path

from ...paths import DATA_DIR as EVA_DATA_DIR

DATA_DIR = str(
  Path(os.getenv("EVA_TRAINING_DATA_DIR", EVA_DATA_DIR / "training")).expanduser()
)
CACHE_DIR = str(
  Path(
    os.getenv("EVA_TRAINING_CACHE_DIR", EVA_DATA_DIR / "cache" / "training")
  ).expanduser()
)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

IMAGENET_100_DIR = os.path.join(DATA_DIR, "imagenet100")
MINI_IMAGENET_DIR = os.path.join(DATA_DIR, "mini-imagenet")
NEU_CLS_DIR = os.path.join(DATA_DIR, "NEU-CLS")


# __init__.py
