from typing import Protocol, runtime_checkable

from tfmamba.protocol.experiment import ExperimentMetadata
from tfmamba.protocol.metrics import Metrics


@runtime_checkable
class ConsoleCommon(Protocol):
    def init(
        self,
        metadata: ExperimentMetadata,
        configDict: dict,
    ) -> None: ...

    def log(
        self,
        metrics: Metrics,
        stage: str,
    ) -> None: ...
