"""Dataset preprocessing isolated from model training."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import packet_features_from_description
from .hdc import HypervectorEncoder
from .taxonomy import extract_authoritative_label


class HypervectorDataset(Dataset):
    def __init__(self, vectors: torch.Tensor, targets: torch.Tensor):
        self.vectors = vectors
        self.targets = targets

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vectors[index], self.targets[index]


def preprocess_dataset(rows: Iterable[Mapping[str, Any]], encoder: HypervectorEncoder,
                       packet_field: str = "Packet/Tags") -> tuple[torch.Tensor, torch.Tensor]:
    vectors = []
    targets = []
    for row_number, row in enumerate(rows):
        if packet_field not in row:
            raise ValueError(f"dataset row {row_number} is missing {packet_field!r}")
        features = packet_features_from_description(row[packet_field])
        vectors.append(encoder.encode_packet(features))
        targets.append(extract_authoritative_label(row).index)
    if not vectors:
        raise ValueError("dataset contains no rows")
    return (
        torch.tensor(np.stack(vectors), dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
