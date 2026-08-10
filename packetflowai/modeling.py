"""Local neural model definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .taxonomy import ATTACK_TYPES


class HVModel(nn.Module):
    def __init__(self, input_dim: int, num_categories: int,
                 hidden_dimensions: tuple[int, int] = (512, 256), dropout: float = 0.5):
        super().__init__()
        first, second = hidden_dimensions
        self.fc1 = nn.Linear(input_dim, first)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(first, second)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(second, num_categories)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        values = self.dropout1(F.relu(self.fc1(values)))
        values = self.dropout2(F.relu(self.fc2(values)))
        return self.fc3(values)


def build_model(config: ModelConfig) -> HVModel:
    return HVModel(
        input_dim=config.hv_dimension,
        num_categories=len(ATTACK_TYPES),
        hidden_dimensions=config.hidden_dimensions,
        dropout=config.dropout,
    )


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
