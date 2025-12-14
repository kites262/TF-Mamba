import swanlab
from loguru import logger as loguru_logger
from swanlab.data.sdk import MODES

from tfmamba.protocol.experiment import ExperimentMetadata
from tfmamba.protocol.metrics import Metrics


class SwanLabConsole:
    def __init__(self, mode: MODES = "offline"):
        self.mode: MODES = mode

    def init(
        self,
        metadata: ExperimentMetadata,
        configDict: dict,
    ):
        swanlab.init(
            mode=self.mode,
            project=metadata.project,
            name=metadata.name,
            description=metadata.description,
            config=configDict,
            logdir=".",
            settings=swanlab.Settings(
                backup=True,
            ),
        )

    def log(self, metrics: Metrics):
        swanlab.log(
            {
                "loss": metrics.loss,
                "OA": metrics.oa,
                "Recall": metrics.recall,
                "Precision": metrics.precision,
                "F1-Score": metrics.f1_score,
            },
            step=metrics.step,
        )


def setup_loguru_logger(metadata: ExperimentMetadata) -> None:
    LOG_FORMAT = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS}"
        "|"
        "<green>{level: <6}</green>"
        "|"
        "<cyan>{module: <20}</cyan>:<cyan>{line: <4}</cyan>"
        "> "
        "{message}"
    )
    loguru_logger.remove()
    loguru_logger.add(
        sink="train.log",
        format=LOG_FORMAT,
        enqueue=True,
        mode="w",
        level="DEBUG",
    )
    loguru_logger.add(
        sink=lambda msg: print(msg, end=""),
        format=LOG_FORMAT,
        colorize=True,
        level="INFO",
    )
    loguru_logger.info(f"Initialized experiment '{metadata.name}' in project '{metadata.project}'")
