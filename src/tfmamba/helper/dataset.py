import torch
from torch.utils.data import Dataset as TorchDataset

from tfmamba.protocol.dataset import DatasetSample


class Dataset(TorchDataset):
    def __init__(
        self,
        samples: list[DatasetSample],
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.samples = samples
        self.normalize = normalize
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        x = sample.file.signal[sample.start : sample.start + sample.length]

        if self.normalize:
            mean = x.mean()
            std = x.std()
            if std > 0:
                x = (x - mean) / std

        x = torch.tensor(x, dtype=self.dtype)
        x = x.unsqueeze(-1)  # (Seq, 1)

        y = sample.file.label
        y = torch.tensor(y, dtype=torch.long)

        return x, y
