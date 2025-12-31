from dataclasses import dataclass


@dataclass
class ExperimentDataloaderConfig:
    num_workers: int
    pin_memory: bool


@dataclass
class ExperimentConfig:
    seed: int
    lr: float
    epochs: int

    train_samples: float
    val_samples: float
    test_samples: float
    split_level: str

    num_classes: int

    window_size: int
    stride: int
    batch_size: int

    dataloader: ExperimentDataloaderConfig


@dataclass
class ExperimentMetadata:
    name: str
    project: str
    description: str
