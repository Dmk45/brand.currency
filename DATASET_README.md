# Dataset Production Guide

This project currently produces datasets only. The PyTorch LSTM model will be added later. Each numeric input is written to a separate CSV so that feature validation, replacement, and future model assembly remain independent.

## Requirements

- Python 3.11 or newer. Python 3.14 is supported by the project code, but confirm that all pinned packages have wheels available in the selected environment.
- Internet access to the public World Bank API and Frankfurter/ECB API.
- A CSV manifest containing links to PDF articles and investment reports for sentiment data.

Install with the Windows Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Numeric Dataset Usage

Run the collector from the repository root. `--days` controls the requested FX history; annual macroeconomic observations are fetched for the same period plus the debt lookback window.

```powershell
py -m brand_currency_data.pipeline numeric `
  --country USA `
  --currency USD `
  --days 365 `
  --debt-years 5 `
  --out data/processed
```

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
gdp_growth.csv
interest_rate.csv
currency_in_circulation.csv
debt_to_gdp.csv
currency_conversion.csv
```

`currency_conversion.csv` contains both CHF and USD conversion fields. `target_per_chf` means one CHF expressed in target-currency units. The World Bank indicators are annual, while FX observations are daily and may omit weekends and market holidays.

### Numeric source mapping

| Dataset | Public source | Indicator or interpretation |
| --- | --- | --- |
| Exports | World Bank | `NE.EXP.GNFS.CD`, current USD |
| Imports | World Bank | `NE.IMP.GNFS.CD`, current USD |
| Inflation | World Bank | `FP.CPI.TOTL.ZG`, annual percent |
| GDP growth | World Bank | `NY.GDP.MKTP.KD.ZG`, annual real percent |
| Interest rate | World Bank | `FR.INR.LEND`, lending rate percent; replace with a central-bank series when a precise policy-rate definition is chosen |
| Currency in circulation | World Bank | `FM.LBL.BMNY.CN`, broad money in local currency as a practical proxy; it is not notes-and-coins-only circulation |
| Debt growth | World Bank | `GC.DOD.TOTL.GD.ZS`, transformed into percent change over `--debt-years` |
| Currency conversion | Frankfurter/ECB | Daily rates converted to target-per-CHF and target-per-USD |

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

The downloader accepts only files beginning with the PDF signature, stores a SHA-256 hash, and writes `metadata.jsonl` alongside the raw documents. The later LLM scorer should read the PDFs and write a separate, versioned sentiment dataset; it should never overwrite the source documents.

## Data Handling Rules

- Use ISO 3166-1 alpha-3 country codes and ISO 4217 currency codes.
- Treat release dates as availability dates. Do not train on values before they were publicly available.
- Preserve raw downloads outside Git and retain source URLs, retrieval dates, and hashes.
- Validate units and quote direction before joining features for an LSTM sequence.
- Do not silently forward-fill annual macro data into daily rows. Make the resampling policy an explicit model-preparation step.
- Do not include PDFs containing personal or restricted information without checking the publisher's terms.

## Not Included Yet

- PyTorch or LSTM code
- Feature joining and sequence-window generation
- Sentiment scoring by an LLM
- Model training, evaluation, inference, and Docker orchestration