# Earnings Event Text & Return Pipeline

This repo builds a per-ticker earnings-event dataset with prices, returns, transcripts, news, fundamentals, and multiple text sentiment scorers (LM, FinBERT, OpenAI). Orchestrators fetch data, build event windows, compute returns/abnormal returns, extract and score text units, and emit analysis-ready panels plus audits.

## What’s produced
- `data/{T}/prices/` – daily OHLCV from Yahoo Finance.
- `data/{T}/calendar/` – earnings dates from FMP.
- `data/{T}/transcripts/{YYYY-MM-DD}/` – call transcript text + metadata.
- `data/{T}/financials/` – ratios + key metrics aligned to events.
- `data/{T}/news/{EARNINGS_DATE}/` – FMP stock news over ±10 business days (chunked paging with per-day fallback).
- `data/{T}/events/` – event windows, price paths, returns, abnormal returns, stable features, and text features:
  - `event_windows.csv` (trading-day window -10..+10; day0 handling BMO/AMC)
  - `event_price_path_m10_p10.csv`, `event_window_returns.csv`, `event_abnormal_windows.csv`
  - `text_units_transcripts.csv`, `text_units_news.csv`
  - LM scores: `text_lm_*`
  - FinBERT scores: `text_finbert_*`
  - OpenAI scores: `text_oa_*`
- Derived text panel and summaries: `data/_derived/text/`
- Audits: `data/_derived/audits/<timestamp>/`

## Core scripts
| Stage | Script | Notes |
| --- | --- | --- |
| Data prep | `scripts/10_ticker_prep_all.py` | One-stop fetch: prices, calendar, transcripts, financials, news (defaults ±10 bd). |
| Event pipeline | `scripts/20_run_event_pipeline_all.py` | Build windows, price paths, returns, market model, abnormal returns, stable features, panel. |
| Text extract/score | `scripts/40_run_text_pipeline_all.py` | Extract units (31), LM scores (32/33), optional FinBERT (34/35) and OpenAI (36/37), merge panels & correlations. |
| Audits | `scripts/00_audit_event_dataset.py` | Coverage/missingness reports. |

Utilities: `_eventlib.py` (window logic), `_text_utils.py`, `_finbert.py`, `_openai_sent.py`, `bert/` (FinBERT fine-tune utilities), `scripts/41_text_tone_vs_car_heatmap.py` (analysis viz).

## Quickstart
1) Env setup (example)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FMP_API_KEY=...
# for OpenAI scoring
export OPENAI_API_KEY=...
```

2) Run everything with defaults (DEFAULT_20 tickers, dates 2021-01-01..2025-12-31):
```bash
python scripts/10_ticker_prep_all.py
python scripts/20_run_event_pipeline_all.py
python scripts/40_run_text_pipeline_all.py --out-dir data/_derived/text --with-finbert --with-openai --overwrite
python scripts/00_audit_event_dataset.py --data-dir data
```

### Important defaults
- News windows: pre/post business days = 10/10 (in 05 and orchestrator 10).
- News fetch is chunked; pagination continues until empty page; auto per-day retry if coverage gaps are detected; warnings in `data/_derived/logs/news_warnings.log`.
- Event windows/trading offsets: [-10, +10]; windows defined on trading days (BMO day0 = same-day close; AMC day0 = next trading day).
- Text extraction drops leading roster headers and detects QA start with heuristics (line 3+ or “A - ” marker, fallbacks for “first question”, etc.).
- Transcript operator turns are dropped by default in scoring scripts; add `--keep-operator` to keep them.

## Per-script cheatsheet
### Data collection
- `05_fmp_news.py` – standalone news fetch; supports `--per-day` (default chunked), `--fallback` auto; warnings logged.
- `04_fmp_transcripts.py`, `03_fmp_earnings_calendar.py`, `06_fmp_financials.py` straightforward FMP pulls.

### Event construction (trading-day windows)
- `11_build_event_windows.py` – writes `event_windows.csv` with -10..+10 anchors.
- `12_extract_event_price_path.py` – price path long (-10..+10).
- `13_compute_event_returns.py`, `14_fit_event_market_model.py`, `15_compute_event_abnormal_returns.py`, `16_merge_event_features_stable.py`, `17_build_event_panel.py`, `18_export_event_long.py`, `19_event_feature_corrs.py`.

### Text extraction & scoring
- `31_extract_text_units.py` – builds `text_units_transcripts.csv` and `text_units_news.csv`.
- `32_score_text_LM_transcripts.py`, `33_score_text_LM_news.py` – LM dictionary scoring.
- `34_score_text_FinBERT_transcripts.py`, `35_score_text_FinBERT_news.py` – HF FinBERT sentiment (pos/neg/neu + tone).
- `36_score_text_OpenAI_transcripts.py`, `37_score_text_OpenAI_news.py` – OpenAI chat model sentiment with FinBERT-like probabilities; model via `OPENAI_MODEL` (default `gpt-4o-mini`).
- `40_run_text_pipeline_all.py` orchestrates extraction, scoring, merging, correlations; outputs:
  - `event_text_features.csv`
  - `summary_by_ticker.csv`, `summary_by_event_seq.csv`
  - `top_correlations.csv`, `top_correlations_by_target.csv`

### Audits
- `00_audit_event_dataset.py` – coverage/missingness per ticker/event, publisher counts, transcript section lengths.
- Logs under `data/_derived/audits/<timestamp>/`.

## Key environment variables
- `FMP_API_KEY` (required for FMP endpoints)
- `OPENAI_API_KEY` (required for OpenAI scoring)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `FINBERT_MODEL_NAME` (optional, default `ProsusAI/finbert`)
- `LM_DICT_PATH` / `LM_DICT_URL` (optional, LM dictionary source)

## Typical output paths (per ticker)
```
data/{T}/prices/yf_ohlcv_daily.csv
data/{T}/calendar/earnings_calendar.csv
data/{T}/transcripts/{CALL_DATE}/transcript.txt
data/{T}/news/{EARNINGS_DATE}/stock_news.csv
data/{T}/events/
  event_windows.csv
  event_price_path_m10_p10.csv
  event_window_returns.csv
  event_abnormal_windows.csv
  text_units_transcripts.csv
  text_units_news.csv
  text_lm_* (units/event_long/event_wide)
  text_finbert_* (units/event_long/event_wide)
  text_oa_* (units/event_long/event_wide)
data/_derived/text/event_text_features.csv
data/_derived/text/top_correlations.csv
data/_derived/audits/<run_id>/...
```

## Tips
- For full refresh after deleting `data/`, rerun 10 → 20 → 40 → 00.
- Use `--overwrite` in scoring scripts if rerunning after code changes.
- Check warning logs: `data/_derived/logs/news_warnings.log` and `transcript_warnings.log`.
- OpenAI scoring can be slow/costly; use `--with-openai` only when needed or subset tickers via `--tickers`.

## License
Internal research use; review data source terms (FMP, Yahoo Finance, OpenAI, HF models) before redistribution.
