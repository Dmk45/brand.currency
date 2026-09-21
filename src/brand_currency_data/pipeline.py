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
    "electricity_price": "APU000072610",
}

YAHOO_GLOBAL_MARKET_TICKERS = {
    "oil_price": "CL=F",
    "gold_price": "GC=F",
}

DEFAULT_STOCK_TICKERS = {
    "USA": "^GSPC",
    "JPN": "^N225",
    "GBR": "^FTSE",
    "CHN": "000001.SS",
}

DEFAULT_BOND_YIELD_SERIES = {
    "USA": "DGS10",
    "JPN": "IRLTLT01JPM156N",
    "GBR": "IRLTLT01GBM156N",
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
    stock_ticker: str | None = None
    bond_yield_series: str | None = None

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


def fetch_yahoo_chart(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch daily adjusted close data from Yahoo Finance chart API."""
    start = int(datetime.combine(start_date, datetime.min.time(), timezone.utc).timestamp())
    end = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp())
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"period1": start, "period2": end, "interval": "1d", "events": "history"},
        headers={"User-Agent": "brand-currency-data/0.1"},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json().get("chart", {}).get("result", [None])[0]
    if not result:
        raise SourceError(f"Yahoo Finance returned no data for {ticker}")
    quotes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    dates = [pd.Timestamp.utcfromtimestamp(value).tz_localize(None) for value in result.get("timestamp", [])]
    frame = pd.DataFrame({"date": dates, "value": quotes}).dropna().sort_values("date")
    if frame.empty:
        raise SourceError(f"Yahoo Finance returned no usable data for {ticker}")
    return frame


def _write_feature(
    frame: pd.DataFrame,
    path: Path,
    feature: str,
    source: str,
    retrieved_at: str,
    requested_start: date,
    requested_end: date,
    source_frequency: str,
    observation_frequency: str,
) -> None:
    frame = frame.copy()
    frame.insert(0, "feature", feature)
    frame["source"] = source
    frame["retrieved_at"] = retrieved_at
    frame["requested_start"] = requested_start.isoformat()
    frame["requested_end"] = requested_end.isoformat()
    frame["observation_frequency"] = observation_frequency
    frame["source_frequency"] = source_frequency
    frame.to_csv(path, index=False)


def _align_to_native_frequency(
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    value_columns: list[str],
    frequency: str,
) -> pd.DataFrame:
    """Keep only source observations in range at their native frequency."""
    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["source_observation_date"] = source["date"]
    if frequency in {"annual", "annual-projection"}:
        source["period"] = source["date"].dt.to_period("Y")
        requested = pd.period_range(start_date, end_date, freq="Y")
    elif frequency in {"monthly", "monthly-with-projection", "monthly-projection"}:
        source["period"] = source["date"].dt.to_period("M")
        requested = pd.period_range(start_date, end_date, freq="M")
    else:
        result = source[source["date"].dt.date.between(start_date, end_date)].drop_duplicates("date", keep="last")
        result = result.drop(columns="period", errors="ignore")
        if "is_projected" not in result:
            result["is_projected"] = False
        result["is_projected"] = result["is_projected"].fillna(False)
        result["is_forward_filled"] = result["is_projected"]
        result["data_status"] = result["is_projected"].map({True: "projected", False: "observed_or_aligned"})
        return result[["date", "source_observation_date", *value_columns, "is_forward_filled", "is_projected", "data_status", *( ["source_reference_date"] if "source_reference_date" in result else [] )]]
    source = source[source["period"].isin(requested)].drop_duplicates("period", keep="last")
    result = source.drop(columns="period")
    result["date"] = source["period"].dt.start_time.clip(lower=pd.Timestamp(start_date)).to_numpy()
    if "is_projected" not in result:
        result["is_projected"] = False
    result["is_projected"] = result["is_projected"].fillna(False)
    result["is_forward_filled"] = result["is_projected"]
    result["data_status"] = result["is_projected"].map({True: "projected", False: "observed_or_aligned"})
    return result[["date", "source_observation_date", *value_columns, "is_forward_filled", "is_projected", "data_status", *( ["source_reference_date"] if "source_reference_date" in result else [] )]]


def _project_period(
    frame: pd.DataFrame,
    value_column: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Project the latest verified value at the source frequency."""
    latest = frame.sort_values("date").dropna(subset=[value_column]).iloc[-1]
    projected = pd.DataFrame({"date": [pd.Timestamp(start_date)]})
    projected[value_column] = latest[value_column]
    projected["source_observation_date"] = pd.Timestamp(start_date)
    projected["source_reference_date"] = pd.Timestamp(latest.get("source_reference_date", latest["date"]))
    projected["is_forward_filled"] = True
    projected["is_projected"] = True
    projected["data_status"] = "projected"
    return projected


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
        if source_frequency.startswith("annual"):
            requested_periods = pd.period_range(config.collection_start, config.collection_end, freq="Y")
            has_in_range_observation = not observations.empty and observations["date"].dt.to_period("Y").isin(requested_periods).any()
        elif source_frequency.startswith("monthly"):
            requested_periods = pd.period_range(config.collection_start, config.collection_end, freq="M")
            has_in_range_observation = not observations.empty and observations["date"].dt.to_period("M").isin(requested_periods).any()
        else:
            has_in_range_observation = (
                not observations.empty
                and observations["date"].between(
                    pd.Timestamp(config.collection_start),
                    pd.Timestamp(config.collection_end),
                ).any()
            )
        if feature != "inflation" and not has_in_range_observation and not observations.empty:
            observations = _project_period(
                observations,
                value_columns[0],
                config.collection_start,
                config.collection_end,
            )
            source_name = f"{source_name} dynamic projection"
            source_frequency = f"{source_frequency}-projection"
        else:
            observations = _align_to_native_frequency(
                observations,
                config.collection_start,
                config.collection_end,
                value_columns,
                source_frequency,
            )
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
            "daily" if source_frequency.startswith(("daily", "business-day")) else source_frequency.split("-")[0],
        )
        written.append(output)

    fx = fetch_fx(config.currency, config.collection_start, config.collection_end)
    fx["source_observation_date"] = fx["date"]
    fx["is_forward_filled"] = False
    fx["is_projected"] = False
    fx["data_status"] = "observed_or_aligned"
    fx["retrieved_at"] = retrieved_at
    fx["requested_start"] = config.collection_start.isoformat()
    fx["requested_end"] = config.collection_end.isoformat()
    fx["source"] = "Frankfurter/ECB"
    fx["observation_frequency"] = "business-day"
    fx["source_frequency"] = "business-day"
    fx_output = config.output_dir / "currency_conversion.csv"
    fx.to_csv(fx_output, index=False)
    written.append(fx_output)

    fluctuations = fx[["date", "target_per_chf", "target_per_usd"]].copy()
    reference_rate = "target_per_chf" if fluctuations["target_per_chf"].nunique() > 1 else "target_per_usd"
    fluctuations["currency_fluctuations"] = fluctuations[reference_rate].pct_change(fill_method=None) * 100
    fluctuations = fluctuations.dropna(subset=["currency_fluctuations"])
    fluctuations["source_observation_date"] = fluctuations["date"]
    fluctuations["is_forward_filled"] = False
    fluctuations["is_projected"] = False
    fluctuations["data_status"] = "observed_or_aligned"
    fluctuation_output = config.output_dir / "currency_fluctuations.csv"
    _write_feature(
        fluctuations[["date", "source_observation_date", "currency_fluctuations", "is_forward_filled", "is_projected", "data_status"]],
        fluctuation_output,
        "currency_fluctuations",
        "Derived from Frankfurter/ECB currency conversion rates",
        retrieved_at,
        config.collection_start,
        config.collection_end,
        "business-day",
        "business-day",
    )
    written.append(fluctuation_output)

    market_sources = [
        ("oil_price", "Yahoo Finance", YAHOO_GLOBAL_MARKET_TICKERS["oil_price"], "business-day"),
        ("gold_price", "Yahoo Finance", YAHOO_GLOBAL_MARKET_TICKERS["gold_price"], "business-day"),
    ]
    if config.country_code == "USA":
        market_sources.append(("electricity_price", "FRED", FRED_USA_SERIES["electricity_price"], "monthly"))
    if config.stock_ticker or config.country_code in DEFAULT_STOCK_TICKERS:
        market_sources.append(("stock_market_index", "Yahoo Finance", config.stock_ticker or DEFAULT_STOCK_TICKERS[config.country_code], "business-day"))
    if config.bond_yield_series or config.country_code in DEFAULT_BOND_YIELD_SERIES:
        market_sources.append(("bond_yield", "FRED", config.bond_yield_series or DEFAULT_BOND_YIELD_SERIES[config.country_code], "monthly"))
    for feature, provider, series_id, frequency in market_sources:
        if provider == "FRED_GLOBAL_MARKET_SERIES" or provider == "FRED":
            market = fetch_fred(series_id, config.collection_start, config.collection_end)
            source = f"FRED:{series_id}"
        else:
            market = fetch_yahoo_chart(series_id, config.collection_start, config.collection_end)
            source = f"Yahoo Finance:{series_id}"
        market = _align_to_native_frequency(market, config.collection_start, config.collection_end, ["value"], frequency)
        market = market.rename(columns={"value": feature})
        output = config.output_dir / f"{feature}.csv"
        _write_feature(market, output, feature, source, retrieved_at, config.collection_start, config.collection_end, frequency, frequency)
        written.append(output)
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


def build_country_pair_dataset(
    first_dir: Path,
    second_dir: Path,
    first_country: str,
    second_country: str,
    first_currency: str,
    second_currency: str,
    output_dir: Path,
) -> list[Path]:
    """Combine two country datasets in ordered model-input and quote order.

    The first dataset is conversion_set 0 (the denominator currency) and the
    second dataset is conversion_set 1 (the numerator currency). This makes
    the target ``second currency per first currency``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_files = [
        path.name
        for path in first_dir.glob("*.csv")
        if path.name != "currency_conversion.csv" and (second_dir / path.name).exists()
    ]
    written: list[Path] = []
    for filename in feature_files:
        first = pd.read_csv(first_dir / filename).add_prefix("first_")
        second = pd.read_csv(second_dir / filename).add_prefix("second_")
        first["first_date"] = pd.to_datetime(first["first_date"], utc=True).dt.tz_localize(None).dt.normalize()
        second["second_date"] = pd.to_datetime(second["second_date"], utc=True).dt.tz_localize(None).dt.normalize()
        joined = pd.merge(first, second, left_on="first_date", right_on="second_date", how="inner")
        joined.insert(0, "first_country", first_country)
        joined.insert(1, "second_country", second_country)
        joined.insert(2, "first_currency", first_currency)
        joined.insert(3, "second_currency", second_currency)
        joined.insert(4, "first_conversion_set", 0)
        joined.insert(5, "second_conversion_set", 1)
        output = output_dir / filename
        joined.to_csv(output, index=False)
        written.append(output)

    first_fx = pd.read_csv(first_dir / "currency_conversion.csv")
    second_fx = pd.read_csv(second_dir / "currency_conversion.csv")
    first_fx = first_fx[["date", "target_per_chf"]].rename(columns={"target_per_chf": "first_per_chf"})
    second_fx = second_fx[["date", "target_per_chf"]].rename(columns={"target_per_chf": "second_per_chf"})
    first_fx["date"] = pd.to_datetime(first_fx["date"], utc=True).dt.tz_localize(None).dt.normalize()
    second_fx["date"] = pd.to_datetime(second_fx["date"], utc=True).dt.tz_localize(None).dt.normalize()
    target = first_fx.merge(second_fx, on="date", how="inner")
    target["second_currency_per_first_currency"] = target["second_per_chf"] / target["first_per_chf"]
    target.insert(0, "first_country", first_country)
    target.insert(1, "second_country", second_country)
    target.insert(2, "first_currency", first_currency)
    target.insert(3, "second_currency", second_currency)
    target.insert(4, "denominator_conversion_set", 0)
    target.insert(5, "numerator_conversion_set", 1)
    target.to_csv(output_dir / "target_exchange_rate.csv", index=False)
    written.append(output_dir / "target_exchange_rate.csv")
    metadata = {
        "first_country": first_country,
        "second_country": second_country,
        "first_currency": first_currency,
        "second_currency": second_currency,
        "conversion_set": {
            "0": "first currency; denominator/source currency",
            "1": "second currency; numerator/target currency",
        },
        "target": "second_currency_per_first_currency",
        "target_formula": "second_per_chf / first_per_chf",
    }
    metadata_path = output_dir / "pair_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    written.append(metadata_path)
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
    numeric.add_argument("--stock-ticker", help="Yahoo Finance benchmark ticker for the target country")
    numeric.add_argument("--bond-yield-series", help="FRED series ID for the target-country bond yield")
    documents = subparsers.add_parser("documents", help="Download PDF sentiment documents from a CSV manifest.")
    documents.add_argument("--manifest", type=Path, required=True)
    documents.add_argument("--out", type=Path, default=Path("data/sentiment_documents"))
    pair = subparsers.add_parser("pair", help="Combine two country datasets in ordered model-input order.")
    pair.add_argument("--first-dir", type=Path, required=True)
    pair.add_argument("--second-dir", type=Path, required=True)
    pair.add_argument("--first-country", required=True)
    pair.add_argument("--second-country", required=True)
    pair.add_argument("--first-currency", required=True)
    pair.add_argument("--second-currency", required=True)
    pair.add_argument("--out", type=Path, required=True)
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
                args.stock_ticker,
                args.bond_yield_series,
            )
        )
    elif args.command == "documents":
        files = download_sentiment_documents(args.manifest, args.out)
    else:
        files = build_country_pair_dataset(
            args.first_dir,
            args.second_dir,
            args.first_country.upper(),
            args.second_country.upper(),
            args.first_currency.upper(),
            args.second_currency.upper(),
            args.out,
        )
    print(f"Wrote {len(files)} files")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()