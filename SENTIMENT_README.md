# Investor Sentiment Scoring for Currency Prediction

This document describes tools and quantitative methods for turning news articles and other public text into features for the ordered currency-pair model. The goal is not to produce a universal opinion score. The goal is to estimate whether newly available information should increase or decrease the value of the second currency relative to the first currency, and how surprising, credible, and persistent that information is.

## Recommended First Version

Build the first production-quality baseline as a **country-aware, event-level, supervised text model**:

1. Collect article metadata and full text with a publication timestamp, source, URL, country entities, and currency entities.
2. Score each article with a finance-language classifier for positive, neutral, or negative tone, while separately extracting economic events and their expected FX direction.
3. Convert article scores into hourly and daily country-pair features using recency decay, source weighting, novelty, and disagreement.
4. Train the sentiment features only on information available before each FX observation. Compare them against a numeric-only model using walk-forward validation.

The most useful target is usually **next-period excess log return**, not the article's abstract sentiment label:

$$
r_{t,h}^{(2/1)} = \log(S_{t+h}^{(2/1)}) - \log(S_t^{(2/1)})
$$

where $S^{(2/1)}$ is the number of units of the second currency per one unit of the first currency. For a directional feature, use the sign of the expected return after conditioning on the ordered pair.

## Tool Options

### Collection and provenance

| Need | Practical tools | Notes |
| --- | --- | --- |
| Licensed news feeds | GDELT 2.1, Event Registry, NewsAPI, Bloomberg/Refinitiv feeds, publisher RSS feeds | Check redistribution and historical-access terms. Prefer a feed with stable article IDs and publication timestamps. |
| Central-bank and government text | Central-bank websites, IMF, BIS, OECD, World Bank, national statistics offices | These sources are especially valuable for policy and macro-event language. Store release time and revision status. |
| Market and calendar context | FRED, ECB, central-bank calendars, economic-calendar provider, existing numeric CSVs | Join text to the numeric regime, surprise, rate, inflation, and yield features already collected here. |
| PDF and HTML extraction | `trafilatura`, `readability-lxml`, `BeautifulSoup`, `PyMuPDF` | Preserve raw files, extraction version, URL, hash, and failure status. Do not silently replace failed extraction with an empty document. |
| Deduplication | `simhash`, `datasketch` MinHash, normalized URL/title hashes | Wire stories and syndicated copies can otherwise dominate the daily score. |
| Language detection and translation | `fastText` language identification, `lingua`, approved translation API | Keep original text and language. Translation can change financial nuance and should be recorded as a feature. |

### NLP and modeling

| Tool | Best use | Recommendation |
| --- | --- | --- |
| `scikit-learn` TF-IDF + logistic regression | Auditable baseline | Start here. It is fast, strong on small labeled datasets, and exposes vocabulary and coefficients. |
| FinBERT or another finance-domain BERT classifier | Article-level financial tone | Use as a feature, not as the only source of direction. Validate on currency and central-bank text. |
| Hugging Face Transformers | Fine-tuning and batch inference | Pin model name and revision; cache model artifacts with the experiment metadata. |
| spaCy or a transformer NER model | Country, currency, institution, person, and organization entities | Entity linking is needed: “the Fed”, “Federal Reserve”, and “FOMC” should map to the same entity. |
| Sentence-transformers | Similarity, novelty, clustering, event deduplication | Compare each story with recent stories and with a rolling country-topic archive. |
| BERTopic or topic models | Topic discovery and monitoring | Use for exploration and drift detection; avoid treating discovered topics as stable features without versioning. |
| `statsmodels`, LightGBM, XGBoost, or PyTorch | Predictive layer | Use a simple linear/logistic model for the baseline, then test tree models or the existing LSTM with strict time splits. |
| MLflow or a versioned JSON/CSV registry | Reproducibility | Record corpus cutoff, model revision, label definitions, feature window, and training dates. |

## Quantification Methods

### 1. Lexicon score

A finance lexicon assigns each token a polarity and aggregates it:

$$
L(d) = \frac{\sum_{w \in d} c_w\,p_w}{\sum_{w \in d} c_w}
$$

where $p_w$ is token polarity and $c_w$ is an optional count or importance weight. This is useful as a transparent sanity check, but it misses negation, modality, context, and the difference between a strong economy and a strong currency.

Use it for diagnostics and fallback coverage, not as the primary currency signal.

### 2. TF-IDF with a supervised classifier

Represent an article with word and character n-grams, then fit logistic regression or a linear SVM to labels such as:

- currency-positive: likely appreciation of the named currency;
- currency-negative: likely depreciation;
- neutral or unclear;
- optionally: risk-on, risk-off, hawkish, dovish, inflationary, growth-positive, or intervention-related.

TF-IDF is the recommended first model because its errors are inspectable and it performs well when the labeled sample is modest. Labels must be defined relative to a currency and a forecast horizon, not generic positive or negative prose.

### 3. Finance-language transformer sentiment

FinBERT-style models produce contextual probabilities:

$$
P_{model}(y \mid d) = (p_{positive}, p_{neutral}, p_{negative})
$$

Use the expected tone $p_{positive}-p_{negative}$ and retain entropy as uncertainty. Fine-tune on a manually reviewed sample of currency-relevant articles when possible. Generic sentiment models should not be assumed to understand that “higher inflation” can be negative for a currency unless the central bank reaction is also considered.

### 4. Event and expectation extraction

For FX, event structure is generally more useful than tone alone. Extract:

```text
actor: FOMC
country/currency: USA / USD
event: policy-rate decision
direction: hawkish or dovish
surprise: above, inline, or below expectation
confidence: 0..1
effective_time: timestamp
```

Examples include rate decisions, intervention, sanctions, elections, fiscal announcements, inflation releases, employment releases, trade shocks, and geopolitical risk. A rule system can provide the first version; a supervised event classifier can replace it later.

For macro releases, calculate text or headline surprise against the market consensus when a consensus series exists:

$$
z_{surprise} = \frac{actual - consensus}{historical\ standard\ deviation}
$$

The sign must be mapped through the likely monetary-policy reaction. A higher inflation surprise is not automatically currency-positive or currency-negative.

### 5. Embedding and similarity features

Encode articles as embeddings and calculate:

- novelty versus the previous 24 hours or seven days;
- similarity to known event examples;
- cluster volume by country and topic;
- distance from the recent language regime.

Embeddings are strong for retrieval and event grouping. They are less auditable than TF-IDF and should initially be used alongside, rather than instead of, interpretable features.

## Turning Article Scores into FX Features

An article score is not yet a time-series feature. For country $c$, topic $k$, and time bucket $t$, use a recency-weighted aggregation:

$$
F_{c,k,t} = \frac{\sum_i w_i\,s_i\,e^{-\lambda (t - published_i)}}{\sum_i w_i\,e^{-\lambda (t - published_i)} + \epsilon}
$$

where:

- $s_i$ is the article's signed score or event direction;
- $w_i$ combines source reliability, article relevance, model confidence, and originality;
- $\lambda$ controls the half-life;
- only articles published by the feature cutoff are included.

For an ordered pair, form a relative feature rather than feeding two unrelated sentiment values:

$$
F_{pair,t}^{(2/1)} = F_{c_2,t} - F_{c_1,t}
$$

Recommended additional features:

| Feature | Why it matters |
| --- | --- |
| Article count and unique-source count | Measures attention and breadth. |
| Weighted mean sentiment | Measures direction. |
| Positive and negative mass separately | Prevents cancellation from hiding a news shock. |
| Sentiment dispersion and model entropy | Measures disagreement and uncertainty. |
| Novelty and duplicate-adjusted count | Measures new information rather than repeated headlines. |
| Topic/event proportions | Separates monetary policy, growth, inflation, politics, and risk. |
| Source-type mix | Distinguishes official releases, financial press, commentary, and social content. |
| Time since last major event | Captures fading shock effects. |
| Cross-country relative score | Aligns the feature with the ordered FX target. |

Use several half-lives, for example 6 hours, 24 hours, and 7 days. The half-lives must be selected inside training folds; choosing them on the full dataset leaks future information through model selection.

## Labels and Training Design

The preferred label is the future log return after the article or aggregate feature. Alternatives are a three-class direction label with a dead zone for small moves, or a volatility-scaled return:

$$
y_{t,h} = \frac{r_{t,h}^{(2/1)}}{\hat{\sigma}_t}
$$

Use publication or release availability time, not the date printed on a report. News published after the market close belongs to the next available trading session. For PDFs, record both the document date and the first verified download or release time; do not use retrieval time as a substitute when it is later than publication.

Validation should be walk-forward or expanding-window. Include a gap around the forecast horizon when overlapping windows could leak labels. Compare:

1. numeric-only features;
2. text-only features;
3. numeric plus text features;
4. shuffled-text and timestamp-shifted negative controls.

Report out-of-sample MAE/RMSE, directional accuracy, correlation with realized returns, calibration, and performance by currency, topic, volatility regime, and source type. A sentiment feature is useful only if it improves the numeric baseline after transaction-cost and publication-latency assumptions.

## Data Contract for a Sentiment Dataset

Keep raw documents immutable and write scored records separately. A minimum article-level schema is:

```text
article_id
url
title
text_hash
published_at_utc
retrieved_at_utc
source
source_type
language
country_codes
currency_codes
topics
event_type
model_name
model_revision
sentiment_positive
sentiment_neutral
sentiment_negative
sentiment_score
sentiment_entropy
fx_direction_by_currency
relevance_score
novelty_score
is_duplicate
extraction_status
```

The aggregated feature file should also include `feature_cutoff_utc`, `window_start_utc`, `window_end_utc`, `half_life_hours`, `country_code`, `currency_code`, `pair_direction`, and `source_document_count`. These fields make leakage and quote-direction mistakes auditable.

## Implementation Order

1. Add a manifest and immutable article/PDF store beside the existing sentiment manifest.
2. Implement extraction, hashing, language detection, entity normalization, and near-duplicate removal.
3. Create a small hand-labeled set focused on central-bank, macro, and FX articles. Label expected currency direction at a fixed horizon and allow “unclear”.
4. Train the TF-IDF logistic-regression baseline and compare it with a finance-language transformer.
5. Add recency-weighted, relative country-pair aggregation and join it by availability time to the existing ordered pair data.
6. Run walk-forward ablations against the numeric-only model.
7. Add event extraction, embeddings, and more sources only when an ablation shows incremental out-of-sample value.

## Common Failure Modes

- Generic positive/negative sentiment is confused with appreciation/depreciation.
- Article publication time is replaced with retrieval time, causing look-ahead bias.
- Repeated wire stories inflate confidence and volume.
- A single global sentiment score ignores which country and currency the article concerns.
- High-impact events are averaged away by a large volume of low-impact articles.
- Headlines are used without checking whether the full article reverses or qualifies the headline.
- Translation, revisions, and model upgrades are not versioned.
- Daily forward-filled numeric features are treated as new observations; follow the existing dataset's `source_observation_date` and `is_forward_filled` fields.
- The reverse currency pair is treated as the same training example without inverting the target and relative sentiment direction.

## Suggested Baseline Decision

For this repository, begin with **TF-IDF logistic regression plus a finance-language transformer score, event flags, source/relevance weights, duplicate-adjusted recency aggregation, and relative country-pair features**. This combination is cheap to audit, compatible with the current CSV-based pipeline, and better aligned with currency prediction than a single off-the-shelf sentiment number. Add embeddings and fine-tuning after the timestamped baseline proves that text contributes out-of-sample signal.