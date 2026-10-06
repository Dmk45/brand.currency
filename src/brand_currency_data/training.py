"""Prepare ordered pair CSVs and train the currency LSTM."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
 
from .model import CurrencyLSTM

TARGET = "second_currency_per_first_currency"
_METADATA_NAMES = {"date", "country", "currency", "conversion_set", "feature", "source", "retrieved_at", "data_status"}
_METADATA_PREFIXES = ("source_", "requested_", "observation_", "is_", "data_status", "retrieved_")


@dataclass(frozen=True)
class PreparedPair:
    dates: pd.DatetimeIndex
    first: np.ndarray
    second: np.ndarray
    target: np.ndarray
    first_features: list[str]
    second_features: list[str]


def _is_value_column(column: str, branch: str) -> bool:
    if not column.startswith(f"{branch}_"):
        return False
    suffix = column[len(branch) + 1:]
    return suffix not in _METADATA_NAMES and not suffix.startswith(_METADATA_PREFIXES)


def _feature_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = f"first_date"
    if date_column not in frame:
        raise ValueError(f"{path.name} is missing first_date")
    frame["date"] = pd.to_datetime(frame[date_column], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    value_columns = [
        column for column in frame.columns
        if _is_value_column(column, "first") or _is_value_column(column, "second")
    ]
    if not value_columns:
        raise ValueError(f"{path.name} contains no numeric pair features")
    result = frame[["date", *value_columns]].copy()
    for column in value_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def prepare_pair(pair_dir: Path) -> PreparedPair:
    """Build a common target-date matrix using only observations available by each date."""
    target_frame = pd.read_csv(pair_dir / "target_exchange_rate.csv")
    target_frame["date"] = pd.to_datetime(target_frame["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    target_frame[TARGET] = pd.to_numeric(target_frame[TARGET], errors="coerce")
    target_frame = target_frame[["date", TARGET]].dropna().sort_values("date").drop_duplicates("date", keep="last")
    if target_frame.empty:
        raise ValueError("target_exchange_rate.csv contains no usable targets")

    first_frames: list[pd.DataFrame] = []
    second_frames: list[pd.DataFrame] = []
    for path in sorted(pair_dir.glob("*.csv")):
        if path.name in {"target_exchange_rate.csv", "validation_exchange_rate.csv"}:
            continue
        frame = _feature_frame(path)
        first_columns = [column for column in frame if column.startswith("first_")]
        second_columns = [column for column in frame if column.startswith("second_")]
        first_frames.append(frame[["date", *first_columns]])
        second_frames.append(frame[["date", *second_columns]])

    def align(frames: list[pd.DataFrame], branch: str) -> tuple[pd.DataFrame, list[str]]:
        result = target_frame[["date"]].copy()
        names: list[str] = []
        for frame in frames:
            columns = [column for column in frame.columns if column != "date"]
            result = pd.merge_asof(result, frame, on="date", direction="backward")
            names.extend(columns)
        if not names:
            raise ValueError(f"no {branch} features found in pair dataset")
        result[names] = result[names].ffill().bfill()
        if result[names].isna().any().any():
            missing = result[names].columns[result[names].isna().any()].tolist()
            raise ValueError(f"features have no observation usable at target dates: {missing}")
        return result, names

    first_frame, first_names = align(first_frames, "first")
    second_frame, second_names = align(second_frames, "second")
    return PreparedPair(
        dates=pd.DatetimeIndex(target_frame["date"]),
        first=first_frame[first_names].to_numpy(dtype=np.float32),
        second=second_frame[second_names].to_numpy(dtype=np.float32),
        target=target_frame[TARGET].to_numpy(dtype=np.float32),
        first_features=first_names,
        second_features=second_names,
    )


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, first: np.ndarray, second: np.ndarray, target: np.ndarray, window: int, starts: np.ndarray) -> None:
        self.first = first
        self.second = second
        self.target = target
        self.window = window
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        end = start + self.window
        return (
            torch.from_numpy(self.first[start:end]),
            torch.from_numpy(self.second[start:end]),
            torch.tensor(self.target[end], dtype=torch.float32),
        )


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / np.where(scale < 1e-8, 1.0, scale)


def _run_epoch(model: CurrencyLSTM, loader: DataLoader, loss_fn: nn.Module, optimizer: Any, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for first, second, target in loader:
        first, second, target = first.to(device), second.to(device), target.to(device)
        prediction = model(first, second)
        loss = loss_fn(prediction, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = target.shape[0]
        total += loss.item() * batch_size
        count += batch_size
    return total / max(count, 1)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prepared = prepare_pair(args.pair_dir)
    if len(prepared.target) <= args.window + 1:
        raise ValueError(f"need more than window + 1 target rows; found {len(prepared.target)}")
    split_row = int(len(prepared.target) * (1.0 - args.validation_fraction))
    starts = np.arange(len(prepared.target) - args.window, dtype=np.int64)
    train_starts = starts[starts + args.window < split_row]
    validation_starts = starts[starts + args.window >= split_row]
    if len(train_starts) == 0 or len(validation_starts) == 0:
        raise ValueError("window and validation_fraction leave no train or validation sequences")
    first_mean, first_scale = prepared.first[:split_row].mean(0), prepared.first[:split_row].std(0)
    second_mean, second_scale = prepared.second[:split_row].mean(0), prepared.second[:split_row].std(0)
    first = _standardize(prepared.first, first_mean, first_scale).astype(np.float32)
    second = _standardize(prepared.second, second_mean, second_scale).astype(np.float32)
    train_loader = DataLoader(SequenceDataset(first, second, prepared.target, args.window, train_starts), args.batch_size, shuffle=True)
    validation_loader = DataLoader(SequenceDataset(first, second, prepared.target, args.window, validation_starts), args.batch_size)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CurrencyLSTM(len(prepared.first_features), len(prepared.second_features), args.branch_size, args.hidden_size, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_loss = _run_epoch(model, validation_loader, loss_fn, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_validation:
            best_validation = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {"first_input_size": len(prepared.first_features), "second_input_size": len(prepared.second_features), "branch_size": args.branch_size, "hidden_size": args.hidden_size, "layers": args.layers, "dropout": args.dropout},
                "first_features": prepared.first_features,
                "second_features": prepared.second_features,
                "first_mean": first_mean.tolist(), "first_scale": first_scale.tolist(),
                "second_mean": second_mean.tolist(), "second_scale": second_scale.tolist(),
                "window": args.window, "target": TARGET, "history": history,
            }, args.output)
    report = {"checkpoint": str(args.output), "device": str(device), "train_sequences": len(train_starts), "validation_sequences": len(validation_starts), "best_validation_loss": best_validation, "history": history}
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Brand Currency ordered-pair LSTM.")
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/currency_lstm.pt"))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--branch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
