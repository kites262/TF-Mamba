import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix

from tfmamba.helper.dataset import Dataset
from tfmamba.helper.pipeline import CWRUPipeline
from tfmamba.protocol.experiment import ExperimentConfig, ExperimentMetadata
from tfmamba.protocol.logger import ConsoleProtocol
from tfmamba.protocol.metrics import Metrics
from tfmamba.utils.dataloader import build_dataloader
from tfmamba.utils.logger import setup_loguru_logger
from tfmamba.utils.metrics import compute_metrics


def setup_console(instance) -> ConsoleProtocol:
    console = instantiate(instance)
    assert isinstance(console, ConsoleProtocol)
    return console


def build_model(instance, num_classes: int) -> torch.nn.Module:
    model = instantiate(
        instance,
        num_classes=num_classes,
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
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    logger.info("Starting training...")
    save_model_path = "best_model.pth"
    best_val_oa = 0.0
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
        console.log(metrics=train_metrics, stage="train")

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
        console.log(metrics=val_metrics, stage="val")

        logger.info(
            f"Epoch {(epoch + 1):02}/{config.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train OA: {train_metrics.oa:.4f} | Val OA: {val_metrics.oa:.4f}"
        )

        # Save best model
        if val_metrics.oa > max(best_val_oa, 0.750):
            best_val_oa = val_metrics.oa
            torch.save(model.state_dict(), save_model_path)

    # --------
    # Test
    # --------
    if not os.path.exists(save_model_path):
        logger.warning("No best model found, skipping testing.")
        return
    else:
        logger.info("Testing the best model...")

    model.load_state_dict(torch.load(save_model_path))

    model.eval()
    test_loss = 0.0

    all_test_logits = []
    all_test_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            test_loss += loss.item()

            all_test_logits.append(logits.cpu())
            all_test_targets.append(y.cpu())

    avg_test_loss = test_loss / len(test_loader)

    test_logits = torch.cat(all_test_logits, dim=0)
    test_targets = torch.cat(all_test_targets, dim=0)

    test_preds = torch.argmax(test_logits, dim=1)
    cm = confusion_matrix(test_targets.numpy(), test_preds.numpy())

    logger.info(f"Confusion Matrix:\n{cm}")

    test_metrics_dict = compute_metrics(
        logits=test_logits,
        targets=test_targets,
        num_classes=config.num_classes,
    )

    test_metrics = Metrics(
        step=0,  # No step for test metrics
        loss=avg_test_loss,
        oa=test_metrics_dict["oa"],
        recall=test_metrics_dict["recall"],
        precision=test_metrics_dict["precision"],
        f1_score=test_metrics_dict["f1_score"],
    )

    console.log(metrics=test_metrics, stage="test")
    logger.info(
        f"Test Loss: {avg_test_loss:.4f} | "
        f"OA: {test_metrics.oa:.4f} | "
        f"Recall: {test_metrics.recall:.4f} | "
        f"Precision: {test_metrics.precision:.4f} | "
        f"F1 Score: {test_metrics.f1_score:.4f}"
    )


if __name__ == "__main__":
    main()
