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
from tfmamba.protocol.metrics import Metrics
from tfmamba.utils.dataloader import build_dataloader
from tfmamba.utils.logger import setup_loguru_logger
from tfmamba.utils.metrics import compute_metrics


def setup_console(instance) -> ConsoleCommon:
    console = instantiate(instance)
    assert isinstance(console, ConsoleCommon)
    return console


def build_model(instance, num_classes: int, in_channels: int, hidden_dim: int) -> torch.nn.Module:
    model = instantiate(
        instance,
        num_classes=num_classes,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
    )
    return model


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    metadata: ExperimentMetadata = instantiate(cfg.metadata.instance)
    config: ExperimentConfig = instantiate(cfg.config.instance)
    data_dir: Path = Path(cfg.data_dir)
    device = torch.device(cfg.device)
    configDict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(configDict, dict)

    setup_loguru_logger(metadata)
    logger.info("Setting up experiment...")

    console = setup_console(cfg.console.instance)
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

    logger.info("Building model...")
    model = build_model(
        cfg.model.instance,
        num_classes=config.num_classes,
        in_channels=1,
        hidden_dim=64,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    logger.info("Starting training...")
    for epoch in range(config.epochs):
        # --------
        # Train
        # --------
        model.train()
        train_loss = 0.0

        all_train_logits = []
        all_train_targets = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            all_train_logits.append(logits)
            all_train_targets.append(y)

        avg_train_loss = train_loss / len(train_loader)

        train_logits = torch.cat(all_train_logits, dim=0)
        train_targets = torch.cat(all_train_targets, dim=0)

        train_metrics_dict = compute_metrics(
            logits=train_logits,
            targets=train_targets,
            num_classes=config.num_classes,
        )

        train_metrics = Metrics(
            step=epoch + 1,
            loss=avg_train_loss,
            oa=train_metrics_dict["oa"],
            recall=train_metrics_dict["recall"],
            precision=train_metrics_dict["precision"],
            f1_score=train_metrics_dict["f1_score"],
        )

        # --------
        # Val
        # --------
        model.eval()
        val_loss = 0.0

        all_val_logits = []
        all_val_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()

                all_val_logits.append(logits.cpu())
                all_val_targets.append(y.cpu())

        avg_val_loss = val_loss / len(val_loader)

        val_logits = torch.cat(all_val_logits, dim=0)
        val_targets = torch.cat(all_val_targets, dim=0)

        val_metrics_dict = compute_metrics(
            logits=val_logits,
            targets=val_targets,
            num_classes=config.num_classes,
        )

        val_metrics = Metrics(
            step=epoch + 1,
            loss=avg_val_loss,
            oa=val_metrics_dict["oa"],
            recall=val_metrics_dict["recall"],
            precision=val_metrics_dict["precision"],
            f1_score=val_metrics_dict["f1_score"],
        )

        console.log(metrics=train_metrics, stage="train")
        console.log(metrics=val_metrics, stage="val")

        logger.info(
            f"Epoch [{epoch + 1}/{config.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
            f"Train OA: {train_metrics.oa:.4f} | Val OA: {val_metrics.oa:.4f}"
        )


if __name__ == "__main__":
    main()
