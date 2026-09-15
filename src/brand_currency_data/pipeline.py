"""Download and write model-ready, feature-specific currency datasets."""

from __future__ import annotations

import argparse
import hashlib
from io import StringIO
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pycountry
import requests

WORLD_BANK_INDICATORS = {
    "exports": "NE.EXP.GNFS.CD",
    "imports": "NE.IMP.GNFS.CD",
    "inflation": "FP.CPI.TOTL.ZG",
    "interest_rate": "FR.INR.LEND",
    "currency_in_circulation": "FM.LBL.BMNY.CN",
    "debt_to_gdp": "GC.DOD.TOTL.GD.ZS",
}

FRED_USA_SERIES = {
    "inflation": "CPIAUCSL",
    "interest_rate": "DFF",
}

IMF_WEO_DATASET = "WEO:2024-10"

@dataclass(frozen=True)
class DatasetConfig:
    country_code: str
    currency: str
    days: int | None
    output_dir: Path
    debt_years: int = 5
    as_of: date = date.today()
    requested_start: date | None = None
    requested_end: date | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z]{3}", self.country_code):
            raise ValueError("country_code must be an uppercase ISO 3166-1 alpha-3 code")
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ValueError("currency must be an uppercase ISO 4217 code")
        if self.days is not None and self.days < 1:
            raise ValueError("days must be at least 1")
        if self.debt_years < 1:
            raise ValueError("debt_years must be at least 1")
        if (self.requested_start is None) != (self.requested_end is None):
            raise ValueError("requested_start and requested_end must be provided together")
        if self.days is None and self.requested_start is None:
            raise ValueError("provide days or an explicit start and end date")
        if self.days is not None and self.requested_start is not None:
            raise ValueError("use either days or an explicit start and end date, not both")
        if self.requested_start and self.requested_end and self.requested_start > self.requested_end:
            raise ValueError("requested_start must be on or before requested_end")

    @property
    def collection_start(self) -> date:
        if self.requested_start is not None:
            return self.requested_start
        return self.as_of - timedelta(days=self.days - 1)

    @property
    def collection_end(self) -> date:
        return self.requested_end or self.as_of


class SourceError(RuntimeError):
    """Raised when a public data source returns an unusable response."""


def _retrieved_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> date:
    """Parse an ISO date or timestamp and retain its calendar date."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


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


def fetch_fred(series_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch a public FRED CSV series without an API key."""
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id, "cosd": start_date.isoformat(), "coed": end_date.isoformat()},
        timeout=60,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), na_values=["."])
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    if date_column not in frame.columns or series_id not in frame.columns:
        raise SourceError(f"FRED returned an unexpected schema for {series_id}")
    frame = frame.rename(columns={date_column: "date", series_id: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values("date")
    if frame.empty:
        raise SourceError(f"FRED returned no observations for {series_id}")
    return frame[["date", "value"]]


def fetch_imf_monthly_inflation(country_code: str) -> pd.DataFrame:
    """Fetch monthly CPI percentage change from IMF IFS via DBnomics."""
    country = pycountry.countries.get(alpha_3=country_code)
    if country is None:
        raise SourceError(f"Unknown ISO-3 country code: {country_code}")
    series_code = f"M.{country.alpha_2}.PCPI_PC_CP_A_PT"
    url = f"https://api.db.nomics.world/v22/series/IMF/IFS/{series_code}"
    payload = _get_json(url, {"observations": 1})
    documents = payload.get("series", {}).get("docs", [])
    if not documents:
        raise SourceError(f"IMF IFS returned no monthly inflation series for {country_code}")
    document = documents[0]
    frame = pd.DataFrame({"date": pd.to_datetime(document.get("period", [])), "value": document.get("value", [])})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna().sort_values("date")
    if frame.empty:
        raise SourceError(f"IMF IFS returned no monthly inflation values for {country_code}")
    return frame


def fetch_imf_debt_to_gdp(country_code: str) -> pd.DataFrame:
    """Fetch annual IMF WEO general-government debt as a percent of GDP."""
    url = f"https://api.db.nomics.world/v22/series/IMF/{IMF_WEO_DATASET}/{country_code}.GGXWDG_NGDP.pcent_gdp"
    payload = _get_json(url, {"observations": 1})
    documents = payload.get("series", {}).get("docs", [])
    if not documents:
        raise SourceError(f"IMF WEO returned no debt data for {country_code}")
    document = documents[0]
    frame = pd.DataFrame({"date": pd.to_datetime(document.get("period", [])), "value": document.get("value", [])})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna().sort_values("date")


def complete_monthly_inflation(
    observations: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, bool]:
    """Fill missing requested months from a rolling mean of available history."""
    requested_months = pd.date_range(start_date, end_date, freq="MS")
    history = observations.sort_values("date").copy()
    rows: list[dict[str, Any]] = []
    projected = False
    for month in requested_months:
        exact = history[history["date"].dt.to_period("M") == month.to_period("M")]
        if not exact.empty:
            row = exact.iloc[-1]
            rows.append({"date": month, "value": row["value"], "is_projected": False})
            continue
        prior = history[history["date"] < month].tail(12)
        if prior.empty:
            continue
        rows.append(
            {
                "date": month,
                "value": prior["value"].mean(),
                "source_reference_date": prior["date"].max(),
                "is_projected": True,
            }
        )
        projected = True
    return pd.DataFrame(rows), projected


def fetch_fx(currency: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch daily target-currency, CHF, and USD rates from Frankfurter/ECB."""
    if currency.upper() in {"CHF", "USD"}:
        base_currency = currency.upper()
        other_currency = "USD" if base_currency == "CHF" else "CHF"
        payload = _get_json(
            f"https://api.frankfurter.app/{start_date}..{end_date}",
            {"from": base_currency, "to": other_currency},
        )
        rates = [
            {
                "date": pd.Timestamp(observation_date),
                "target_per_chf": 1.0 if base_currency == "CHF" else 1 / values["CHF"],
                "target_per_usd": 1 / values["USD"] if base_currency == "CHF" else 1.0,
                "chf_per_target": 1.0 if base_currency == "CHF" else values["CHF"],
                "usd_per_target": values["USD"] if base_currency == "CHF" else 1.0,
            }
            for observation_date, values in payload.get("rates", {}).items()
        ]
        if not rates:
            raise SourceError(f"Frankfurter returned no FX observations for {base_currency}")
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


def _write_feature(
    frame: pd.DataFrame,
    path: Path,
    feature: str,
    source: str,
    retrieved_at: str,
    requested_start: date,
    requested_end: date,
    source_frequency: str,
) -> None:
    frame = frame.copy()
    frame.insert(0, "feature", feature)
    frame["source"] = source
    frame["retrieved_at"] = retrieved_at
    frame["requested_start"] = requested_start.isoformat()
    frame["requested_end"] = requested_end.isoformat()
    frame["observation_frequency"] = "daily"
    frame["source_frequency"] = source_frequency
    frame.to_csv(path, index=False)


def _expand_to_daily(
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    value_columns: list[str],
) -> pd.DataFrame:
    """Align lower-frequency observations to every calendar date in the range."""
    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"])
    source = source[
        (source["date"] >= pd.Timestamp(start_date))
        & (source["date"] <= pd.Timestamp(end_date))
    ].sort_values("date")
    daily = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    if source.empty:
        for column in value_columns:
            daily[column] = pd.NA
        daily["source_observation_date"] = pd.NaT
        daily["is_forward_filled"] = False
        daily["is_projected"] = False
        daily["data_status"] = "unavailable_in_requested_range"
        return daily
    source = source.rename(columns={"date": "source_observation_date"})
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    source["source_observation_date"] = pd.to_datetime(source["source_observation_date"]).astype("datetime64[ns]")
    source_metadata = [column for column in ("is_projected", "source_reference_date") if column in source.columns]
    daily = pd.merge_asof(
        daily.sort_values("date"),
        source[["source_observation_date", *value_columns, *source_metadata]],
        left_on="date",
        right_on="source_observation_date",
        direction="backward",
    )
    daily["is_forward_filled"] = daily["source_observation_date"] != daily["date"]
    if "is_projected" not in daily.columns:
        daily["is_projected"] = False
    daily["is_projected"] = daily["is_projected"].fillna(False)
    daily["data_status"] = daily["is_projected"].map({True: "projected", False: "observed_or_aligned"})
    daily.loc[daily["source_observation_date"].isna(), "data_status"] = "unavailable_in_requested_range"
    return daily


def _project_daily(
    frame: pd.DataFrame,
    value_column: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Project the latest verified annual value across the requested daily range."""
    latest = frame.sort_values("date").dropna(subset=[value_column]).iloc[-1]
    daily = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    daily[value_column] = latest[value_column]
    daily["source_observation_date"] = pd.Timestamp(start_date)
    daily["source_reference_date"] = pd.Timestamp(
        latest.get("source_reference_date", latest["date"])
    )
    daily["is_forward_filled"] = True
    daily["is_projected"] = True
    daily["data_status"] = "projected"
    return daily


def collect_numeric_datasets(config: DatasetConfig) -> list[Path]:
    """Collect every numeric feature into its own CSV file."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start_year = config.collection_start.year - config.debt_years - 5
    end_year = config.collection_end.year
    retrieved_at = _retrieved_at()
    written: list[Path] = []
    unavailable_features: list[str] = []

    for feature, indicator in WORLD_BANK_INDICATORS.items():
        source_name = f"World Bank:{indicator}"
        source_frequency = "annual"
        projected = False
        if config.country_code == "USA" and feature in FRED_USA_SERIES:
            series_id = FRED_USA_SERIES[feature]
            fred_start = config.collection_start - timedelta(days=800) if feature == "inflation" else config.collection_start
            observations = fetch_fred(series_id, fred_start, config.collection_end)
            source_name = f"FRED:{series_id}"
            source_frequency = "monthly" if feature == "inflation" else "daily"
            if feature == "inflation":
                observations["value"] = observations["value"].pct_change(12) * 100
                observations = observations.dropna()
        elif feature == "inflation":
            try:
                raw_observations = fetch_imf_monthly_inflation(config.country_code)
                observations, projected = complete_monthly_inflation(
                    raw_observations,
                    config.collection_start,
                    config.collection_end,
                )
            except (requests.RequestException, SourceError):
                observations = pd.DataFrame(columns=["date", "value"])
                source_name = "IMF IFS monthly CPI unavailable"
                source_frequency = "unavailable"
            else:
                source_name = "IMF IFS monthly CPI with dynamic gap projection" if projected else "IMF IFS monthly CPI"
                source_frequency = "monthly-with-projection" if projected else "monthly"
        elif feature == "debt_to_gdp":
            observations = fetch_imf_debt_to_gdp(config.country_code)
            source_name = f"IMF WEO via DBnomics:{IMF_WEO_DATASET}"
            source_frequency = "annual"
        else:
            observations = fetch_world_bank(config.country_code, indicator, start_year, end_year)
        if feature == "debt_to_gdp":
            values_by_year = observations.assign(year=observations["date"].dt.year).set_index("year")["value"]
            observations["value"] = [
                ((value / values_by_year.get(year - config.debt_years)) - 1) * 100
                if values_by_year.get(year - config.debt_years) not in (None, 0)
                else None
                for year, value in zip(observations["date"].dt.year, observations["value"])
            ]
            observations = observations.rename(columns={"value": f"debt_growth_{config.debt_years}y_percent"}).dropna()
        else:
            observations = observations.rename(columns={"value": feature})
        value_columns = [f"debt_growth_{config.debt_years}y_percent"] if feature == "debt_to_gdp" else [feature]
        has_in_range_observation = (
            not observations.empty
            and observations["date"].between(
                pd.Timestamp(config.collection_start),
                pd.Timestamp(config.collection_end),
            ).any()
        )
        if feature != "inflation" and not has_in_range_observation and not observations.empty:
            observations = _project_daily(
                observations,
                value_columns[0],
                config.collection_start,
                config.collection_end,
            )
            source_name = f"{source_name} dynamic projection"
            source_frequency = f"{source_frequency}-projection"
        else:
            observations = _expand_to_daily(observations, config.collection_start, config.collection_end, value_columns)
        if (observations["data_status"] == "unavailable_in_requested_range").any():
            unavailable_features.append(feature)
        output = config.output_dir / f"{feature}.csv"
        _write_feature(
            observations,
            output,
            feature,
            source_name,
            retrieved_at,
            config.collection_start,
            config.collection_end,
            source_frequency,
        )
        written.append(output)

    fx = fetch_fx(config.currency, config.collection_start, config.collection_end)
    fx = _expand_to_daily(
        fx,
        config.collection_start,
        config.collection_end,
        ["target_per_chf", "target_per_usd", "chf_per_target", "usd_per_target"],
    )
    fx["retrieved_at"] = retrieved_at
    fx["requested_start"] = config.collection_start.isoformat()
    fx["requested_end"] = config.collection_end.isoformat()
    fx["source"] = "Frankfurter/ECB"
    fx["observation_frequency"] = "daily"
    fx["source_frequency"] = "business-day"
    fx_output = config.output_dir / "currency_conversion.csv"
    fx.to_csv(fx_output, index=False)
    written.append(fx_output)
    if unavailable_features:
        raise SourceError(
            "No in-range source data for: "
            + ", ".join(unavailable_features)
            + ". Add a provider for the requested frequency before training."
        )
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
    retrieved_at = _retrieved_at()
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
            record = {
                **row,
                "local_path": str(path),
                "sha256": digest,
                "content_type": response.headers.get("content-type"),
                "retrieved_at": retrieved_at,
            }
            metadata_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce Brand Currency feature datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    numeric = subparsers.add_parser("numeric", help="Download numeric country and FX datasets.")
    numeric.add_argument("--country", required=True, help="ISO 3166-1 alpha-3 code, such as USA")
    numeric.add_argument("--currency", required=True, help="ISO 4217 code, such as USD")
    range_group = numeric.add_mutually_exclusive_group(required=True)
    range_group.add_argument("--days", type=int, help="Number of calendar days ending on --as-of")
    range_group.add_argument("--start-date", type=_parse_timestamp, help="Inclusive ISO date/timestamp")
    numeric.add_argument("--end-date", type=_parse_timestamp, help="Inclusive ISO date/timestamp; required with --start-date")
    numeric.add_argument("--out", type=Path, default=Path("data/processed"))
    numeric.add_argument("--debt-years", type=int, default=5)
    numeric.add_argument("--as-of", type=_parse_timestamp, default=date.today())
    documents = subparsers.add_parser("documents", help="Download PDF sentiment documents from a CSV manifest.")
    documents.add_argument("--manifest", type=Path, required=True)
    documents.add_argument("--out", type=Path, default=Path("data/sentiment_documents"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "numeric":
        country_dir = args.out / args.country.upper()
        if (args.start_date is None) != (args.end_date is None):
            raise SystemExit("--start-date and --end-date must be provided together")
        files = collect_numeric_datasets(
            DatasetConfig(
                args.country.upper(),
                args.currency.upper(),
                args.days,
                country_dir,
                args.debt_years,
                args.as_of,
                args.start_date,
                args.end_date,
            )
        )
    else:
        files = download_sentiment_documents(args.manifest, args.out)
    print(f"Wrote {len(files)} files")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()