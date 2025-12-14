from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetFile:
    path: Path
    label: int
    signal: np.typing.NDArray[np.float64]


@dataclass(frozen=True)
class DatasetSample:
    file: DatasetFile
    start: int
    length: int
