"""PyTorch models for ordered currency-pair forecasting."""

from __future__ import annotations

import torch
from torch import nn


class CurrencyLSTM(nn.Module):
    """Encode both ordered country feature branches and predict one exchange rate."""

    def __init__(
        self,
        first_input_size: int,
        second_input_size: int,
        branch_size: int = 32,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.first_projection = nn.Sequential(nn.Linear(first_input_size, branch_size), nn.LayerNorm(branch_size), nn.GELU())
        self.second_projection = nn.Sequential(nn.Linear(second_input_size, branch_size), nn.LayerNorm(branch_size), nn.GELU())
        self.sequence = nn.LSTM(
            input_size=branch_size * 2,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_branch = self.first_projection(first)
        second_branch = self.second_projection(second)
        sequence, _ = self.sequence(torch.cat((first_branch, second_branch), dim=-1))
        return self.output(sequence[:, -1]).squeeze(-1)
