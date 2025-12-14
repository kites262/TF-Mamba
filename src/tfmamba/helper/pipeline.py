import os
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
from loguru import logger

from tfmamba.protocol.dataset import DatasetFile, DatasetSample
from tfmamba.protocol.experiment import ExperimentConfig


class CWRUPipeline:
    def __init__(self, root: Path, config: ExperimentConfig):
        self.root = root
        self.config = config

        self.files: list[DatasetFile] = []
        self.samples: dict[str, list[DatasetSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.on_load()

    def on_load(self):
        logger.info(f"Scanning dataset root: {self.root}")
        self._load_files()
        logger.info(f"Found {len(self.files)} dataset files.")

        for file in self.files:
            splits = self._create_splits(file)
            for split, split_samples in splits.items():
                self.samples[split].extend(split_samples)

        rng = random.Random(self.config.seed)
        for split, split_samples in self.samples.items():
            rng.shuffle(split_samples)
            logger.info(f"Total {split} samples: {len(split_samples)}")

    def get_samples(self, split: str) -> list[DatasetSample]:
        return self.samples[split]

    def _load_files(self):
        """
        Scan directory structure:
        root/
          ├── 0/
          ├── 1/
          └── ...
        """

        for label_dir in os.listdir(self.root):
            label_path = self.root / label_dir
            if not label_path.is_dir():
                logger.warning(f"Expected directory but found file: {label_path}")

            try:
                label = int(label_dir)
            except Exception:
                logger.warning(f"Invalid label directory name: {label_dir}")
                continue

            for file_name in os.listdir(label_path):
                if not file_name.endswith(".mat"):
                    continue

                file_path = label_path / file_name

                mat = sio.loadmat(file_path)
                signal = np.array([])

                for key, value in mat.items():
                    if key.endswith("_DE_time"):
                        signal = value.flatten()
                        break
                if signal is None or len(signal) == 0:
                    logger.warning(f"No valid signal found in file: {file_path}")
                    continue

                dataset_file = DatasetFile(
                    path=file_path,
                    label=label,
                    signal=signal,
                )
                self.files.append(dataset_file)

    def _create_splits(self, file: DatasetFile) -> dict[str, list[DatasetSample]]:
        signal = file.signal

        length = len(signal)
        win = self.config.window_size
        stride = self.config.stride

        samples: list[DatasetSample] = []
        for start in range(0, length - win + 1, stride):
            samples.append(
                DatasetSample(
                    file=file,
                    start=start,
                    length=win,
                ),
            )

        rng = random.Random(self.config.seed)
        rng.shuffle(samples)

        n = len(samples)
        n_train = int(n * self.config.train_samples)
        n_val = int(n * self.config.val_samples)

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :]

        return {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }
