# Dataset Production Guide

This project currently produces datasets only. The PyTorch LSTM model will be added later. Each numeric input is written to a separate CSV so that feature validation, replacement, and future model assembly remain independent.

## Requirements


No single public provider supplies genuinely daily values for every requested economic parameter across every country. The collector therefore preserves each source's native cadence: daily and business-day sources retain only their actual observations, while monthly and annual sources produce one row per source period intersecting the requested range. For the USA, interest rate uses FRED's daily effective federal funds rate (`DFF`), inflation uses FRED's monthly CPI (`CPIAUCSL`), and electricity price uses FRED's monthly average residential electricity price (`APU000072610`). The native frequency and whether a value was projected are retained in every output row.

Each numeric input is written to a separate CSV so that feature validation, replacement, and future model assembly remain independent.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Numeric Dataset Usage

Run the collector from the repository root. Use either `--days` with `--as-of`, or an explicit inclusive `--start-date` and `--end-date`. Both date options accept an ISO date or timestamp such as `2026-09-07T00:00:00Z`. Each output uses the provider's native acquisition frequency: daily sources produce daily rows, monthly sources produce monthly rows, quarterly sources produce quarterly rows, and annual sources produce annual rows.

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

The range is inclusive. FX providers return business-day rows. Lower-frequency providers return only the months, quarters, or years intersecting the requested range; they are not expanded into artificial daily rows.

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
currency_fluctuations.csv
oil_price.csv
gold_price.csv
stock_market_index.csv
electricity_price.csv
bond_yield.csv
```

Every numeric row includes `retrieved_at`, `requested_start`, `requested_end`, `observation_frequency`, `source_frequency`, `source_observation_date`, `is_forward_filled`, `is_projected`, and `data_status`. A source observation is kept at its native period. For inflation, a dynamic projection uses the latest 12 monthly observations and records the actual source month in `source_reference_date`; the requested-period anchor is marked `data_status=projected` and `is_projected=True`. `currency_conversion.csv` contains both CHF and USD conversion fields. `target_per_chf` means one CHF expressed in target-currency units.

### Numeric source mapping

| Dataset | Public source | Indicator or interpretation |
| --- | --- | --- |
| Exports | World Bank | `NE.EXP.GNFS.CD`, current USD |
| Imports | World Bank | `NE.IMP.GNFS.CD`, current USD |
| Inflation | FRED USA; IMF IFS via DBnomics for other countries | Monthly CPI percentage change versus the same month one year earlier; dynamic projection from the latest 12 monthly observations for missing months. |
| Interest rate | FRED for USA; World Bank fallback | `DFF`, daily effective federal funds rate for USA; `FR.INR.LEND` annual lending rate otherwise |
| Currency in circulation | World Bank | `FM.LBL.BMNY.CN`, broad money in local currency as a practical proxy; it is not notes-and-coins-only circulation |
| Debt growth | IMF WEO via DBnomics | `GGXWDG_NGDP` debt-to-GDP, transformed into percent change over `--debt-years` |
| Currency conversion | Frankfurter/ECB | Business-day rates converted to target-per-CHF and target-per-USD |
| Currency fluctuations | Derived from Frankfurter/ECB | Percent change in the average of the target-per-CHF and target-per-USD rates |
| Oil price | Yahoo Finance | `CL=F`, WTI crude oil futures price in USD per barrel |
| Gold price | Yahoo Finance | `GC=F`, gold futures price in USD per troy ounce |
| Stock market index | Yahoo Finance | Country benchmark ticker; defaults exist for USA, Japan, Great Britain, and China; use `--stock-ticker` for other countries |
| Electricity price | FRED USA | `APU000072610`, average residential electricity price in cents per kWh |
| Bond yield | FRED | Country long-term yield series where mapped; use `--bond-yield-series` for other countries |

The collector records the source in each macro CSV. Check provider definitions and licensing before production use.

Market-feature overrides:

```powershell
py -m brand_currency_data numeric `
  --country DEU `
  --currency EUR `
  --start-date 2025-01-01T00:00:00Z `
  --end-date 2025-01-31T23:59:59Z `
  --stock-ticker ^GDAXI `
  --bond-yield-series IRLTLT01DEM156N `
  --out data/processed
```

The stock ticker must represent the target country's largest or chosen benchmark exchange. The bond series must be documented with its maturity and unit. The program does not guess these for unsupported countries.

## Ordered Country-Pair Dataset

The future model consumes two country datasets with the same feature names. Order matters:

```text
first country features -> weighting linear layer -> sequence model
second country features -------------------------> sequence model
sequence output: second currency per one first currency
```

For example, with Great Britain first and Russia second, the target is `RUB per GBP`, meaning how many roubles equal one pound. Reversing the countries creates a different training example and target.

Build a pair after collecting both country datasets:

```powershell
$env:PYTHONPATH = "src"
py -m brand_currency_data pair `
  --first-dir data/processed/GBR `
  --second-dir data/processed/RUS `
  --first-country GBR `
  --second-country RUS `
  --first-currency GBP `
  --second-currency RUB `
  --out data/pairs/GBR_RUS
```

The pair output prefixes feature columns with `first_` and `second_`. Every paired feature row also contains `first_conversion_set=0` and `second_conversion_set=1`. Set 0 is the denominator/source currency and set 1 is the numerator/target currency. `target_exchange_rate.csv` contains `second_currency_per_first_currency`, calculated from both currencies' CHF-referenced rates, plus the same direction metadata. Dates are normalized to calendar days before joining, so timestamps from the same day align. `pair_metadata.json` records the quote direction for downstream sequence preparation.

For example, put RUB in `--first-dir` and USD in `--second-dir` to produce USD per RUB. Reversing the directories produces the reciprocal target and a separate ordered training example.

The LSTM input should receive both feature sets and their conversion-set labels as ordinary inputs or branch metadata; `conversion_set` should not be treated as hidden long-term memory. For a scalable implementation, use a learned two-value embedding or one-hot indicator alongside each branch, and include country/currency IDs separately when generalization across many currencies is required. Training batches must sample ordered country pairs without duplicate pairs inside a batch. The reverse order is required as a separate example: `(GBR, RUS)` and `(RUS, GBR)` have different inputs and reciprocal targets. Across epochs, pairs may recur, and the sampler should deliberately include both orientations rather than deduplicating them globally.

Use one directional exchange-rate target rather than the arithmetic mean of both conversion rates. The reverse rate is mathematically constrained to be the reciprocal, so averaging the two directions mixes incompatible units. If both orientations are trained, add a reciprocal-consistency loss or evaluate both directions after inversion; do not average rates unless they have first been converted to a common, explicitly defined quantity.

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
- Feature joining and sequence-window generation (the ordered pair export is available, but LSTM window assembly is not)
- Sentiment scoring by an LLM
- Model training, evaluation, inference, and Docker orchestration