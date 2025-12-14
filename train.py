from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from tfmamba.helper.dataset import Dataset
from tfmamba.helper.pipeline import CWRUPipeline
from tfmamba.protocol.experiment import ExperimentConfig, ExperimentMetadata
from tfmamba.protocol.logger import ConsoleCommon
from tfmamba.utils.dataloader import build_dataloader
from tfmamba.utils.logger import setup_loguru_logger


def setup_console(cfg: DictConfig) -> ConsoleCommon:
    console = instantiate(cfg.console.instance)
    assert isinstance(console, ConsoleCommon)
    return console


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    metadata: ExperimentMetadata = instantiate(cfg.metadata.instance)
    config: ExperimentConfig = instantiate(cfg.config.instance)
    data_dir: Path = Path(cfg.data_dir)
    configDict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(configDict, dict)

    setup_loguru_logger(metadata)
    logger.info("Setting up experiment...")

    device = torch.device(cfg.device)

    console = setup_console(cfg)
    console.init(
        metadata=metadata,
        configDict=configDict,
    )

    pipline = CWRUPipeline(root=data_dir, config=config)
    train_ds = Dataset(pipline.get_samples(split="train"))
    val_ds = Dataset(pipline.get_samples(split="val"))
    test_ds = Dataset(pipline.get_samples(split="test"))

    train_loader = build_dataloader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )
    val_loader = build_dataloader(
        dataset=val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )
    test_loader = build_dataloader(
        dataset=test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )

    def sanity_check_dataloader(loader, name: str, max_batches: int = 2):
        logger.info(f"Sanity check for {name} loader")

        for i, (x, y) in enumerate(loader):
            logger.info(
                f"[{name}] batch {i} | "
                f"x.shape={tuple(x.shape)}, "
                f"x.dtype={x.dtype}, "
                f"y.shape={tuple(y.shape)}, "
                f"y.dtype={y.dtype}, "
                f"y.min={y.min().item()}, y.max={y.max().item()}"
            )

            # 数值检查
            if torch.isnan(x).any():
                raise ValueError(f"NaN found in {name} inputs")
            if torch.isinf(x).any():
                raise ValueError(f"Inf found in {name} inputs")

            if i + 1 >= max_batches:
                break

    sanity_check_dataloader(train_loader, "train")
    sanity_check_dataloader(val_loader, "val")
    sanity_check_dataloader(test_loader, "test")

    from tfmamba.model.dummy import DummyModel

    model = DummyModel(input_size=config.window_size, num_classes=config.num_classes).to(device=device)
    x, y = next(iter(train_loader))
    x = x.to(device=device)
    y = y.to(device=device)

    out = model(x)
    logger.info(f"Model forward pass successful. out.shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
