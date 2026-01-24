# PEAD Sentiment Research Pipeline

This repository builds a research dataset for post–earnings announcement drift (PEAD) style analyses by assembling, **per ticker**, a consistent set of raw inputs:

- Daily prices from Yahoo Finance (via `yfinance`)
- Daily technical indicators computed from prices
- Earnings calendar events from Financial Modeling Prep (FMP)
- Earnings call transcripts from FMP
- Quarterly processed fundamentals from FMP (ratios + key metrics), **aligned to transcript events**
- Optional: stock news articles from FMP (can be fetched separately)

All data are saved under `data/{TICKER}/...` using a standardized folder layout so downstream modeling can be swapped in/out without changing collection logic.

---

## Repository layout

- `scripts/`  
  Data collection and feature building scripts.

- `data/`  
  Generated outputs and intermediate files, organized by ticker.

- `bert/`  
  Legacy FinBERT LoRA training/inference utilities (not required for the data-first pipeline).

---

## Data directory structure

After preparing one ticker, the folder structure looks like:

### `data/{TICKER}/prices/`
- `yf_ohlcv_daily_raw.csv`  
  Daily OHLCV (+ adjusted close) with a buffer window (indicator warmup).
- `yf_ohlcv_daily.csv`  
  Trimmed to the requested `[start, end]` window (canonical file used downstream).

### `data/{TICKER}/technicals/`
- `technicals_daily.csv`
- `technicals_daily.meta.json`

### `data/{TICKER}/calendar/`
- `earnings_calendar.csv`
- `earnings_calendar.raw.json`
- `earnings_calendar.meta.json`

### `data/{TICKER}/transcripts/{YYYY-MM-DD}/`
- `meta.json`
- `transcript.json`
- `transcript.txt`

> Note: transcript folders are keyed by the **earnings call date** returned by FMP.

### `data/{TICKER}/financials/`
- `ratios_quarter.csv`
- `key_metrics_quarter.csv`
- corresponding `.raw.json` and `.meta.json` files

These are **aligned to the transcript event dates** (the transcript dates define the “event cadence” used for alignment).

### `data/{TICKER}/news/`
Empty unless you run `05_fmp_news.py`.

### `data/{TICKER}/events/`
Empty unless you build an event-level panel downstream.

---

## Setup

### Create and activate environment

```bash
python -m venv .sentvenv
source .sentvenv/bin/activate
pip install -r requirements.txt
````

### Set your FMP API key

FMP endpoints require `FMP_API_KEY` in your environment.

Check if it is set:

```bash
echo $FMP_API_KEY
python -c "import os; print(os.getenv('FMP_API_KEY'))"
```

Set it for the current shell session:

```bash
export FMP_API_KEY="YOUR_KEY_HERE"
```

To make it permanent, add that line to your shell profile (e.g., `~/.zshrc`).

---

## Script overview (00 → 06)

### 00 Initialize ticker directories

Creates the folder structure under `data/{TICKER}/`.

```bash
python scripts/00_init_ticker_dirs.py --ticker NVDA
```

### 01 Download prices from Yahoo Finance

Downloads daily OHLCV (+ adjusted close). Saves two files:

* buffered raw file (indicator warmup)
* trimmed canonical file for the requested window

```bash
python scripts/01_yf_prices.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### 02 Compute technical indicators

Reads `prices/yf_ohlcv_daily.csv` and writes indicators:

```bash
python scripts/02_technicals.py --ticker NVDA
```

### 03 Download earnings calendar from FMP

Creates a clean earnings event table:

```bash
python scripts/03_fmp_earnings_calendar.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### 04 Download earnings call transcripts from FMP

Downloads transcripts for earnings calls in range and saves them under `data/{TICKER}/transcripts/{YYYY-MM-DD}/`.

```bash
python scripts/04_fmp_transcripts.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### 05 Download stock news from FMP (optional)

Fetches news items for a ticker on a single day:

```bash
python scripts/05_fmp_news.py --ticker NVDA --date 2024-11-20
```

### 06 Download processed quarterly fundamentals from FMP (aligned)

Fetches:

* quarterly ratios
* quarterly key metrics

Then aligns them to the transcript event dates and writes:

* `data/{TICKER}/financials/ratios_quarter.csv`
* `data/{TICKER}/financials/key_metrics_quarter.csv`

```bash
python scripts/06_fmp_financials.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

Useful options (if present in your current `06_fmp_financials.py -h`):

* `--limit N` : cap rows fetched from endpoints (default in runner is 400)
* `--refresh` : force re-fetch even if outputs exist
* `--save-as-is` : save raw endpoint payloads “as is” in addition to the aligned tables

Example:

```bash
python scripts/06_fmp_financials.py --ticker NVDA --start 2021-01-01 --end 2025-12-31 --limit 400 --save-as-is
```

**Important:** `06_fmp_financials.py` requires transcripts to exist first. If you run it before transcripts, it will error.

---

## End-to-end prep runner (10)

`scripts/10_ticker_prep_all.py` runs a full per-ticker pipeline:

* 00 init dirs
* 01 prices
* 02 technicals
* 03 earnings calendar
* 04 transcripts
* 06 aligned quarterly fundamentals

### Run for one ticker

```bash
python scripts/10_ticker_prep_all.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### Run for multiple tickers

```bash
python scripts/10_ticker_prep_all.py --tickers AAPL MSFT NVDA --start 2021-01-01 --end 2025-12-31
```

### Run from a ticker file

* CSV with a `ticker` column, OR
* text file with one ticker per line (`#` comments allowed)

```bash
python scripts/10_ticker_prep_all.py --tickers-file tickers.csv --start 2021-01-01 --end 2025-12-31
```

### Notes on runner behavior

* The runner **auto-detects** which flags `06_fmp_financials.py` supports and will only forward compatible flags.
* If financials are enabled, the runner forces transcripts/calendar steps because alignment depends on transcript event cadence.
* By default, the runner performs a **hard alignment check** (last transcript date matches last row date in the aligned financial CSVs). You can disable it with:

```bash
python scripts/10_ticker_prep_all.py --no-align-check
```

---

## Helper modules

* `scripts/_common.py`
  Shared utilities (IO helpers, HTTP helpers, timestamp helpers).

* `scripts/_indicators.py`
  Indicator functions used by `02_technicals.py`.

---

## Notes on dates and time zone

We treat U.S. equity market alignment in Eastern Time when building event-level datasets.
Earnings timing (AMC vs BMO) matters for defining pre/post windows in downstream panels.

---

## Common troubleshooting

### Missing / invalid FMP key

If you see “Missing FMP_API_KEY” or “Invalid API KEY”, confirm:

```bash
echo $FMP_API_KEY
```

### `events/` is empty

Normal until you run an event-panel builder downstream.

### `news/` is empty

Normal until you run `05_fmp_news.py`.

### `06_fmp_financials.py` says “No transcript events found…”

Run transcripts first:

```bash
python scripts/04_fmp_transcripts.py --ticker NFLX --start 2021-01-01 --end 2025-12-31
python scripts/06_fmp_financials.py --ticker NFLX --start 2021-01-01 --end 2025-12-31
```