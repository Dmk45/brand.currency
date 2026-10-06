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
        result[names] = result[names].ffill()
        if result[names].isna().all().any():
            missing = result[names].columns[result[names].isna().all()].tolist()
            raise ValueError(f"features have no observation usable at target dates: {missing}")
        return result, names

    first_frame, first_names = align(first_frames, "first")
    second_frame, second_names = align(second_frames, "second")
    first_valid_dates = first_frame.dropna(subset=first_names)["date"]
    second_valid_dates = second_frame.dropna(subset=second_names)["date"]
    valid_dates = first_valid_dates[first_valid_dates.isin(second_valid_dates)]
    if valid_dates.empty:
        raise ValueError("no target dates have observations available for every pair feature")
    first_frame = first_frame[first_frame["date"].isin(valid_dates)].reset_index(drop=True)
    second_frame = second_frame[second_frame["date"].isin(valid_dates)].reset_index(drop=True)
    target_frame = target_frame[target_frame["date"].isin(valid_dates)].reset_index(drop=True)
    return PreparedPair(
        dates=pd.DatetimeIndex(first_frame["date"]),
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


def _evaluate(
    model: CurrencyLSTM,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for first, second, target in loader:
            first, second, target = first.to(device), second.to(device), target.to(device)
            prediction = model(first, second)
            loss = loss_fn(prediction, target)
            batch_size = target.shape[0]
            total += loss.item() * batch_size
            count += batch_size
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    return total / max(count, 1), np.concatenate(predictions), np.concatenate(targets)


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
    target_mean = float(prepared.target[:split_row].mean())
    target_scale = float(prepared.target[:split_row].std())
    if target_scale < 1e-8:
        target_scale = 1.0
    first = _standardize(prepared.first, first_mean, first_scale).astype(np.float32)
    second = _standardize(prepared.second, second_mean, second_scale).astype(np.float32)
    target = ((prepared.target - target_mean) / target_scale).astype(np.float32)
    train_loader = DataLoader(SequenceDataset(first, second, target, args.window, train_starts), args.batch_size, shuffle=True)
    validation_loader = DataLoader(SequenceDataset(first, second, target, args.window, validation_starts), args.batch_size)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CurrencyLSTM(len(prepared.first_features), len(prepared.second_features), args.branch_size, args.hidden_size, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    best_predictions: np.ndarray | None = None
    best_targets: np.ndarray | None = None
    best_metrics: dict[str, float] = {}
    previous_targets = prepared.target[validation_starts + args.window - 1]
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_loss, scaled_predictions, scaled_actual = _evaluate(model, validation_loader, loss_fn, device)
        predictions = scaled_predictions * target_scale + target_mean
        actual = scaled_actual * target_scale + target_mean
        mae = float(np.mean(np.abs(predictions - actual)))
        rmse = float(np.sqrt(np.mean(np.square(predictions - actual))))
        directional_accuracy = float(
            np.mean(np.sign(predictions - previous_targets) == np.sign(actual - previous_targets))
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_mae": mae,
            "validation_rmse": rmse,
            "validation_directional_accuracy": directional_accuracy,
        })
        is_best = validation_loss < best_validation
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_predictions = predictions.copy()
            best_targets = actual.copy()
            best_metrics = {
                "validation_mae": mae,
                "validation_rmse": rmse,
                "validation_directional_accuracy": directional_accuracy,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {"first_input_size": len(prepared.first_features), "second_input_size": len(prepared.second_features), "branch_size": args.branch_size, "hidden_size": args.hidden_size, "layers": args.layers, "dropout": args.dropout},
                "first_features": prepared.first_features,
                "second_features": prepared.second_features,
                "first_mean": first_mean.tolist(), "first_scale": first_scale.tolist(),
                "second_mean": second_mean.tolist(), "second_scale": second_scale.tolist(),
                "target_mean": target_mean, "target_scale": target_scale,
                "window": args.window, "target": TARGET, "history": history,
            }, args.output)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_huber_z={train_loss:.6f} "
            f"val_huber_z={validation_loss:.6f} "
            f"val_mae={mae:.6f} "
            f"val_rmse={rmse:.6f} "
            f"val_directional_accuracy={directional_accuracy:.1%} "
            f"{'best' if is_best else ''}",
            flush=True,
        )

    if best_predictions is None or best_targets is None:
        raise RuntimeError("training completed without producing a valid best checkpoint")
    target_rows = validation_starts + args.window
    prediction_path = args.output.with_name(f"{args.output.stem}_validation_predictions.csv")
    quote_metadata_path = args.pair_dir / "pair_metadata.json"
    quote_metadata = json.loads(quote_metadata_path.read_text(encoding="utf-8")) if quote_metadata_path.exists() else {}
    first_currency = quote_metadata.get("first_currency", "first_currency")
    second_currency = quote_metadata.get("second_currency", "second_currency")
    target_quote = f"{second_currency}_per_{first_currency}"
    baseline_mae = float(np.mean(np.abs(previous_targets - best_targets)))
    baseline_rmse = float(np.sqrt(np.mean(np.square(previous_targets - best_targets))))
    prediction_frame = pd.DataFrame({
        "date": prepared.dates[target_rows].strftime("%Y-%m-%d"),
        f"actual_{target_quote}": best_targets,
        f"predicted_{target_quote}": best_predictions,
        f"previous_{target_quote}": previous_targets,
        "absolute_error": np.abs(best_predictions - best_targets),
        "direction_correct": (
            np.sign(best_predictions - previous_targets) == np.sign(best_targets - previous_targets)
        ),
    })
    prediction_frame.to_csv(prediction_path, index=False)
    report = {
        "checkpoint": str(args.output),
        "validation_predictions": str(prediction_path),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "quote": target_quote,
        "window_observations": args.window,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "first_target_date": prepared.dates.min().strftime("%Y-%m-%d"),
        "last_target_date": prepared.dates.max().strftime("%Y-%m-%d"),
        "train_sequences": len(train_starts),
        "validation_sequences": len(validation_starts),
        "best_validation_loss": best_validation,
        **best_metrics,
        "persistence_baseline_mae": baseline_mae,
        "persistence_baseline_rmse": baseline_rmse,
        "mae_skill_vs_persistence": 1.0 - best_metrics["validation_mae"] / baseline_mae if baseline_mae else 0.0,
        "history": history,
    }
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
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
