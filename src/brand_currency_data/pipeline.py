"""Download and write model-ready, feature-specific currency datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

WORLD_BANK_INDICATORS = {
    "exports": "NE.EXP.GNFS.CD",
    "imports": "NE.IMP.GNFS.CD",
    "inflation": "FP.CPI.TOTL.ZG",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "interest_rate": "FR.INR.LEND",
    "currency_in_circulation": "FM.LBL.BMNY.CN",
    "debt_to_gdp": "GC.DOD.TOTL.GD.ZS",
}


@dataclass(frozen=True)
class DatasetConfig:
    country_code: str
    currency: str
    days: int
    output_dir: Path
    debt_years: int = 5
    as_of: date = date.today()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z]{3}", self.country_code):
            raise ValueError("country_code must be an uppercase ISO 3166-1 alpha-3 code")
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ValueError("currency must be an uppercase ISO 4217 code")
        if self.days < 1:
            raise ValueError("days must be at least 1")
        if self.debt_years < 1:
            raise ValueError("debt_years must be at least 1")

    @property
    def start_date(self) -> date:
        return self.as_of - timedelta(days=self.days - 1)


class SourceError(RuntimeError):
    """Raised when a public data source returns an unusable response."""


def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_world_bank(country_code: str, indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch annual World Bank observations for one indicator."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    payload = _get_json(url, {"format": "json", "per_page": 1000, "date": f"{start_year}:{end_year}"})
    if not isinstance(payload, list) or len(payload) < 2:
        raise SourceError(f"World Bank returned no observations for {indicator}")
    rows = [row for row in payload[1] if row.get("value") is not None]
    return pd.DataFrame(
        {"date": pd.to_datetime([f"{row['date']}-12-31" for row in rows]), "value": [row["value"] for row in rows]}
    ).sort_values("date")


def fetch_fx(currency: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch daily target-currency, CHF, and USD rates from Frankfurter/ECB."""
    if currency.upper() == "CHF":
        payload = _get_json(f"https://api.frankfurter.app/{start_date}..{end_date}", {"from": "CHF", "to": "USD"})
        rates = [
            {
                "date": pd.Timestamp(observation_date),
                "target_per_chf": 1.0,
                "target_per_usd": 1 / values["USD"],
                "chf_per_target": 1.0,
                "usd_per_target": values["USD"],
            }
            for observation_date, values in payload.get("rates", {}).items()
        ]
        if not rates:
            raise SourceError("Frankfurter returned no FX observations for CHF")
        return pd.DataFrame(
            rates
        ).sort_values("date")
    url = f"https://api.frankfurter.app/{start_date}..{end_date}"
    payload = _get_json(url, {"from": currency.upper(), "to": "CHF,USD"})
    rates = []
    for observation_date, values in payload.get("rates", {}).items():
        chf = values.get("CHF")
        usd = values.get("USD")
        rates.append(
            {
                "date": pd.Timestamp(observation_date),
                "target_per_chf": None if chf is None else 1 / chf,
                "target_per_usd": None if usd is None else 1 / usd,
                "chf_per_target": chf,
                "usd_per_target": usd,
            }
        )
    if not rates:
        raise SourceError(f"Frankfurter returned no FX observations for {currency}")
    return pd.DataFrame(rates).sort_values("date")


def _write_feature(frame: pd.DataFrame, path: Path, feature: str, source: str) -> None:
    frame = frame.copy()
    frame.insert(0, "feature", feature)
    frame["source"] = source
    frame.to_csv(path, index=False)


def collect_numeric_datasets(config: DatasetConfig) -> list[Path]:
    """Collect every numeric feature into its own CSV file."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start_year = config.start_date.year - config.debt_years
    end_year = config.as_of.year
    written: list[Path] = []

    for feature, indicator in WORLD_BANK_INDICATORS.items():
        observations = fetch_world_bank(config.country_code, indicator, start_year, end_year)
        if feature == "debt_to_gdp":
            observations["value"] = observations["value"].pct_change(config.debt_years) * 100
            observations = observations.rename(columns={"value": f"debt_growth_{config.debt_years}y_percent"}).dropna()
        else:
            observations = observations.rename(columns={"value": feature})
        output = config.output_dir / f"{feature}.csv"
        _write_feature(observations, output, feature, f"World Bank:{indicator}")
        written.append(output)

    fx = fetch_fx(config.currency, config.start_date, config.as_of)
    fx_output = config.output_dir / "currency_conversion.csv"
    fx.to_csv(fx_output, index=False)
    written.append(fx_output)
    return written


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:120] or "document"


def download_sentiment_documents(manifest_path: Path, output_dir: Path) -> list[Path]:
    """Download country PDF articles/reports from a user-maintained manifest."""
    manifest = pd.read_csv(manifest_path)
    required = {"country_code", "title", "url"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Sentiment manifest is missing columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    written: list[Path] = []
    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for row in manifest.to_dict(orient="records"):
            response = requests.get(row["url"], timeout=90, headers={"User-Agent": "brand-currency-data/0.1"})
            response.raise_for_status()
            content = response.content
            if not content.startswith(b"%PDF"):
                raise SourceError(f"Expected a PDF at {row['url']}")
            digest = hashlib.sha256(content).hexdigest()
            filename = f"{row['country_code']}_{digest[:16]}_{_safe_filename(row['title'])}.pdf"
            path = output_dir / filename
            path.write_bytes(content)
            record = {**row, "local_path": str(path), "sha256": digest, "content_type": response.headers.get("content-type")}
            metadata_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce Brand Currency feature datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    numeric = subparsers.add_parser("numeric", help="Download numeric country and FX datasets.")
    numeric.add_argument("--country", required=True, help="ISO 3166-1 alpha-3 code, such as USA")
    numeric.add_argument("--currency", required=True, help="ISO 4217 code, such as USD")
    numeric.add_argument("--days", required=True, type=int, help="Number of calendar days of FX history")
    numeric.add_argument("--out", type=Path, default=Path("data/processed"))
    numeric.add_argument("--debt-years", type=int, default=5)
    numeric.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    documents = subparsers.add_parser("documents", help="Download PDF sentiment documents from a CSV manifest.")
    documents.add_argument("--manifest", type=Path, required=True)
    documents.add_argument("--out", type=Path, default=Path("data/sentiment_documents"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "numeric":
        country_dir = args.out / args.country.upper()
        files = collect_numeric_datasets(DatasetConfig(args.country.upper(), args.currency.upper(), args.days, country_dir, args.debt_years, args.as_of))
    else:
        files = download_sentiment_documents(args.manifest, args.out)
    print(f"Wrote {len(files)} files")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()