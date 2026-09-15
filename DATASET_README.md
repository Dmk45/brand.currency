# Dataset Production Guide

This project currently produces datasets only. The PyTorch LSTM model will be added later. Each numeric input is written to a separate CSV so that feature validation, replacement, and future model assembly remain independent.

## Requirements

- Python 3.11 or newer. Python 3.14 is supported by the project code, but confirm that all pinned packages have wheels available in the selected environment.
- Internet access to the public World Bank API and Frankfurter/ECB API.
- A CSV manifest containing links to PDF articles and investment reports for sentiment data.

No single public provider supplies genuinely daily values for every requested economic parameter across every country. This implementation therefore uses the broadest suitable source per parameter and creates a daily model-alignment layer. For the USA, interest rate uses FRED's daily effective federal funds rate (`DFF`) and inflation uses FRED's monthly CPI (`CPIAUCSL`). China and Colombia use IMF IFS monthly CPI through DBnomics; when the requested month is not yet available, the latest 12 monthly observations are averaged dynamically and projected across the requested range. The native frequency and whether a value was carried forward are retained in every output row.

Install with the Windows Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Numeric Dataset Usage

Run the collector from the repository root. Use either `--days` with `--as-of`, or an explicit inclusive `--start-date` and `--end-date`. Both date options accept an ISO date or timestamp such as `2026-09-07T00:00:00Z`. Every output file contains one calendar-day row for each date in the requested range.

```powershell
py -m brand_currency_data.pipeline numeric `
  --country USA `
  --currency USD `
  --days 365 `
  --debt-years 5 `
  --out data/processed
```

For an explicit date range:

```powershell
py -m brand_currency_data numeric `
  --country USA `
  --currency USD `
  --start-date 2026-09-07T00:00:00Z `
  --end-date 2026-09-14T23:59:59Z `
  --out data/processed
```

The range is inclusive. FX providers may return only business days within it; macroeconomic providers return the observations available for the requested calendar years.

Because the package lives under `src`, use one of these approaches from the repository root:

```powershell
$env:PYTHONPATH = "src"
py -m brand_currency_data numeric --country USA --currency USD --days 365
```

or install the package in editable mode after adding a packaging configuration:

```powershell
py -m pip install -e .
py -m brand_currency_data numeric --country USA --currency USD --days 365
```

The output is written under `data/processed/<COUNTRY>/`:

```text
exports.csv
imports.csv
inflation.csv
interest_rate.csv
currency_in_circulation.csv
debt_to_gdp.csv
currency_conversion.csv
```

Every numeric row includes `retrieved_at`, `requested_start`, `requested_end`, `observation_frequency=daily`, `source_frequency`, `source_observation_date`, `is_forward_filled`, `is_projected`, and `data_status`. A source observation is accepted only when its observation date falls inside the requested range. For inflation, a dynamic projection uses the latest 12 monthly observations and records the actual source month in `source_reference_date`; the requested-period anchor is marked `data_status=projected` and `is_projected=True`. Other unavailable values remain empty with `data_status=unavailable_in_requested_range`. `currency_conversion.csv` contains both CHF and USD conversion fields. `target_per_chf` means one CHF expressed in target-currency units.

### Numeric source mapping

| Dataset | Public source | Indicator or interpretation |
| --- | --- | --- |
| Exports | World Bank | `NE.EXP.GNFS.CD`, current USD |
| Imports | World Bank | `NE.IMP.GNFS.CD`, current USD |
| Inflation | FRED USA; IMF IFS via DBnomics for other countries | Monthly CPI percentage change versus the same month one year earlier; dynamic projection from the latest 12 monthly observations for missing months. |
| Interest rate | FRED for USA; World Bank fallback | `DFF`, daily effective federal funds rate for USA; `FR.INR.LEND` annual lending rate otherwise |
| Currency in circulation | World Bank | `FM.LBL.BMNY.CN`, broad money in local currency as a practical proxy; it is not notes-and-coins-only circulation |
| Debt growth | IMF WEO via DBnomics | `GGXWDG_NGDP` debt-to-GDP, transformed into percent change over `--debt-years` |
| Currency conversion | Frankfurter/ECB | Business-day rates converted to target-per-CHF and target-per-USD, aligned to calendar days |

The collector records the source in each macro CSV. Check provider definitions and licensing before production use.

## Sentiment PDF Dataset

Create a copy of `data/sentiment_manifest.example.csv` and replace the example URL with direct PDF URLs. Keep one document per row:

```csv
country_code,title,url,published_date,source,document_type
GBR,Central bank investment outlook,https://publisher.example/outlook.pdf,2026-02-15,Publisher Name,investment_report
```

Download the documents:

```powershell
$env:PYTHONPATH = "src"
py -m brand_currency_data documents `
  --manifest data/sentiment_manifest.csv `
  --out data/sentiment_documents
```

The downloader accepts only files beginning with the PDF signature, stores a SHA-256 hash, adds a UTC `retrieved_at` timestamp, and writes `metadata.jsonl` alongside the raw documents. The later LLM scorer should read the PDFs and write a separate, versioned sentiment dataset; it should never overwrite the source documents.

## Data Handling Rules

- Use ISO 3166-1 alpha-3 country codes and ISO 4217 currency codes.
- Treat release dates as availability dates. Do not train on values before they were publicly available.
- Treat daily macro rows with `is_forward_filled=True` as aligned lower-frequency provider observations, not new daily measurements.
- Preserve raw downloads outside Git and retain source URLs, retrieval dates, and hashes.
- Validate units and quote direction before joining features for an LSTM sequence.
- Keep the forward-fill policy explicit and audit `source_observation_date` before training.
- Do not include PDFs containing personal or restricted information without checking the publisher's terms.

## Not Included Yet

- PyTorch or LSTM code
- Feature joining and sequence-window generation
- Sentiment scoring by an LLM
- Model training, evaluation, inference, and Docker orchestration