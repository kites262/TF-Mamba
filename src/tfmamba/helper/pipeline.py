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
        self.splits: dict[str, list[DatasetSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.on_load()

    def on_load(self):
        logger.info(f"Scanning dataset root: {self.root}")
        self._load_files()
        logger.info(f"Found {len(self.files)} dataset files.")

        if self.config.split_level == "window":
            logger.info("Splitting dataset by window...")
            self._split_by_window()
        elif self.config.split_level == "file":
            logger.info("Splitting dataset by file...")
            self._split_by_file()
        else:
            raise ValueError(f"Unsupported split level: {self.config.split_level}")

    def get_samples(self, split: str) -> list[DatasetSample]:
        return self.splits[split]

    def _split_by_window(self):
        rng = random.Random(self.config.seed)

        for file in self.files:
            samples = self._create_splits(file)
            n = len(samples)

            n_train = int(n * self.config.train_samples)
            n_val = int(n * self.config.val_samples)

            train_samples = samples[:n_train]
            val_samples = samples[n_train : n_train + n_val]
            test_samples = samples[n_train + n_val :]

            self.splits["train"].extend(train_samples)
            self.splits["val"].extend(val_samples)
            self.splits["test"].extend(test_samples)

        for split, split_samples in self.splits.items():
            rng.shuffle(split_samples)
            logger.info(f"Total {split} samples: {len(split_samples)}")

    def _split_by_file(self):
        rng = random.Random(self.config.seed)

        label_files: dict[int, list[DatasetFile]] = {}
        for file in self.files:
            label_files.setdefault(file.label, []).append(file)

        split_files: dict[str, list[DatasetFile]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for _, files in label_files.items():
            rng.shuffle(files)

            n_train = 2
            n_val = 1

            split_files["train"].extend(files[:n_train])
            split_files["val"].extend(files[n_train : n_train + n_val])
            split_files["test"].extend(files[n_train + n_val :])

        for split, split_files_list in split_files.items():
            for file in split_files_list:
                samples = self._create_splits(file)
                self.splits[split].extend(samples)

        for split, split_samples in self.splits.items():
            rng.shuffle(split_samples)
            logger.info(f"Total {split} samples: {len(split_samples)}")

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

    def _create_splits(self, file: DatasetFile) -> list[DatasetSample]:
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
        return samples
