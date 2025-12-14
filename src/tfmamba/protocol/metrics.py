from dataclasses import dataclass


@dataclass
class Metrics:
    step: int
    loss: float

    oa: float
    recall: float
    precision: float
    f1_score: float
