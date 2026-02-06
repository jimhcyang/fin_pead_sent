# PEAD Sentiment Research Pipeline

This repository builds a research dataset for post–earnings announcement drift (PEAD) style analyses by assembling, **per ticker**, a consistent set of raw inputs and a standardized **event-level panel**.

## Raw inputs (per ticker)
- Daily prices from Yahoo Finance (via `yfinance`)
- Daily technical indicators computed from prices
- Earnings calendar events from Financial Modeling Prep (FMP)
- Earnings call transcripts from FMP
- Quarterly processed fundamentals from FMP (ratios + key metrics), **aligned to transcript events**
- Optional: stock news articles from FMP (can be fetched separately)

## Event-level outputs (PEAD-ready)
From the raw inputs, the pipeline builds:
- Trading-day **earnings anchors** (pre/react/+5/+10/+20)
- Close-only (adjusted close) **event price path**
- Minimal **event returns** (price levels, $ deltas, % returns)
- A merged **numeric event panel** (returns + surprises + selected fundamentals)
- A final **event panel** with **market (SPX) returns** and **market-adjusted abnormal returns**

All data are saved under `data/{TICKER}/...` using a standardized folder layout so downstream modeling can be swapped in/out without changing collection logic.

---

## Repository layout

- `scripts/`  
  Data collection, panel building, and visualization scripts.

- `data/`  
  Generated outputs and intermediate files, organized by ticker.

- `bert/`  
  Legacy FinBERT LoRA training/inference utilities (not required for the data-first pipeline).

---

## Data directory structure

After preparing one ticker, the folder structure looks like:

### `data/{TICKER}/prices/`
- `yf_ohlcv_daily.csv`  
  Daily OHLCV (+ adjusted close) from Yahoo Finance. Starts at `--price-start`
  (default 2019-01-01) and keeps the full window—no separate raw/trimmed files.

> Note: the event-level PEAD pipeline (scripts 21–26) uses **adjusted close only** and ignores intraday OHLC and volume.

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

> Transcript folders are keyed by the **earnings call date** returned by FMP.

### `data/{TICKER}/financials/`
- `ratios_quarter.csv`
- `key_metrics_quarter.csv`
- corresponding `.raw.json` and `.meta.json` files

These are **aligned to the transcript event dates** (the transcript dates define the “event cadence” used for alignment).

### `data/{TICKER}/news/`
Empty unless you run `05_fmp_news.py`.

### `data/{TICKER}/events/`  (created by scripts 21–26 / 30)
- `earnings_anchors.csv` (+ `.meta.json`)  
  Event anchors per earnings event:
  - `pre_date`: “close before the announcement”
  - `react_date`: “close after the announcement is incorporated”
  - `d5/d10/d20`: trading-day offsets from `react_date`

  Timing rules:
  - **BMO**: `pre_date = previous trading day`, `react_date = same day`
  - **AMC**: `pre_date = same day`, `react_date = next trading day`

- `event_price_path.csv` (+ `.meta.json`)  
  Close-only adjusted-close levels:
  - `pre_adj_close`, `react_adj_close`, `adj_close_p5/p10/p20`

- `event_returns.csv` (+ `.meta.json`)  
  Minimal PEAD/event-study outputs:
  - price levels (adj close)
  - $ deltas vs pre and vs react
  - simple returns in percent: `100*(P2/P1 - 1)`
  - “drift” returns are measured from `react_date` (e.g., `react→d20`)

- `event_panel_numeric.csv` (+ `.meta.json`)  
  Merged panel:
  - event returns
  - earnings surprises (EPS and revenue surprise %)
  - selected quarterly fundamentals:
    - `km_*` (key metrics)
    - `rt_*` (ratios)

- `event_panel.csv` (+ `.meta.json`)  
  Final panel adds:
  - market (SPX) adjusted-close levels at the same anchor dates
  - market returns (percent) over the same windows
  - abnormal returns (percent): `abn = stock_return - market_return`

---

## Market data cache (SPX)

The event-panel builder uses SPX for abnormal return construction:

- `data/_tmp_market/spx/prices/yf_ohlcv_daily.csv`

This file is **close-only** (date + adjusted close).

**Important behavior**
- If you run the full orchestrator `30_build_events_all.py`, it will **re-download SPX as needed**.
- If you run `26_compute_abnormal_returns.py` directly and SPX is missing, run `25_yf_download_spx.py` (or rerun `30`).

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

## Script overview (00 → 06): raw inputs

### 00 Initialize ticker directories

```bash
python scripts/00_init_ticker_dirs.py --ticker NVDA
```

### 01 Download prices from Yahoo Finance

```bash
python scripts/01_yf_prices.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### 02 Compute technical indicators

```bash
python scripts/02_technicals.py --ticker NVDA
```

### 03 Download earnings calendar from FMP

```bash
python scripts/03_fmp_earnings_calendar.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### 04 Download earnings call transcripts from FMP

```bash
python scripts/04_fmp_transcripts.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

### 05 Download stock news from FMP (optional)

```bash
python scripts/05_fmp_news.py --ticker NVDA --date 2024-11-20
```

### 06 Download processed quarterly fundamentals from FMP (aligned)

```bash
python scripts/06_fmp_financials.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

> Important: `06_fmp_financials.py` requires transcripts to exist first.

---

## End-to-end raw prep runner (10)

Runs:

* 00 init dirs
* 01 prices
* 02 technicals
* 03 earnings calendar
* 04 transcripts
* 06 aligned quarterly fundamentals

```bash
python scripts/10_ticker_prep_all.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

---

## Event-level PEAD panel builder (21 → 26, orchestrated by 30)

These scripts build PEAD-ready, close-only (adjusted close) event panels.

### 21 Build event anchors

```bash
python scripts/21_build_event_anchors.py --ticker NVDA
```

### 22 Extract close-only event price path (adj_close only)

```bash
python scripts/22_extract_event_price_path.py --ticker NVDA
```

### 23 Compute minimal event returns (levels, $ deltas, % returns)

```bash
python scripts/23_compute_event_returns.py --ticker NVDA
```

### 24 Merge event returns with surprises + selected fundamentals

```bash
python scripts/24_merge_event_fundamentals.py --ticker NVDA
```

### 25 Download SPX close-only data (for abnormal returns)

```bash
python scripts/25_yf_download_spx.py --symbol ^GSPC --start 2021-01-01 --end 2025-12-31 --out-rel _tmp_market/spx/prices/yf_ohlcv_daily.csv
```

### 26 Compute market-adjusted abnormal returns

```bash
python scripts/26_compute_abnormal_returns.py --ticker NVDA
```

### 30 Orchestrator: build everything above for many tickers

```bash
python scripts/30_build_events_all.py --tickers AAPL MSFT NVDA
```

---

## Visualization scripts (31 → 38) and runner (40)

Figures are saved to:

* `data/{TICKER}/viz/` (ticker-level overview figures + matrices)
* `data/{TICKER}/viz/events/` (per-event cards)

### 31 — Event overview visuals (per ticker)

Narrative-first figures (time ordered):

* Price series with earnings markers
* Average cumulative abnormal return curve (from PRE)
* Average abnormal drift bars (from REACT)
* Histogram of 20-day abnormal drift

```bash
python scripts/31_viz_event_overview.py --ticker AAPL
```

### 32 — Surprise vs drift (scatter)

Scatterplots:

* EPS surprise vs 20d abnormal drift
* Revenue surprise vs 20d abnormal drift

```bash
python scripts/32_viz_surprise_vs_drift.py --ticker AAPL
```

### 33 — Surprise buckets (heatmaps)

Heatmaps of mean abnormal returns by surprise buckets (when sample size permits):

* Drift returns (REACT→+h)
* Total cumulative abnormal returns (PRE→+h)

```bash
python scripts/33_viz_surprise_heatmaps.py --ticker AAPL
```

### 34 — Feature correlation + ranked CSV

* Correlation scan vs target abnormal drift
* Correlation heatmap (top correlated predictors)
* Writes a ranked CSV used by later plots

```bash
python scripts/34_viz_feature_correlation.py --ticker AAPL
```

### 35 — Event “spaghetti” paths (each event is its own line)

Event-ordered paths (no reordering):

* Abnormal cumulative return from PRE across horizons
* Abnormal drift from REACT across horizons

```bash
python scripts/35_viz_event_paths.py --ticker AAPL
```

### 36 — Per-event “cards” (one PNG per earnings event)

Each earnings event gets its own figure containing:

* Adj-close price path at anchors (PRE/REACT/+5/+10/+20)
* Return heatmap (stock vs market vs abnormal) across windows
* Text summary of anchors and key drift metrics

```bash
python scripts/36_viz_event_cards.py --ticker AAPL
```

Outputs:

* `data/AAPL/viz/events/YYYY-MM-DD_event_card.png`

### 37 — Event-by-event matrix heatmaps (each event is a row)

“Scan views” in **date order**:

* Abnormal totals from PRE (events × horizons)
* Abnormal drift from REACT (events × horizons)
* Surprises (events × surprise metrics)

```bash
python scripts/37_viz_event_matrix.py --ticker AAPL
```

### 38 — Top predictor timelines (z-scored, date ordered)

Uses the ranked CSV from script 34 (if present) to plot the top predictors through time.

```bash
python scripts/38_viz_top_feature_timelines.py --ticker AAPL
```

---

## 40 — Generate all figures for many tickers (recommended)

Runs 31–38 (skips any missing viz scripts instead of crashing).

```bash
python scripts/40_make_figures_all.py --tickers AAPL MSFT NVDA
```

If you don’t want many per-event PNGs:

```bash
python scripts/40_make_figures_all.py --tickers AAPL MSFT NVDA --skip-event-cards
```

---

## Notes on dates and time zone

We treat U.S. equity market alignment in Eastern Time when building event-level datasets.
Earnings timing (AMC vs BMO) matters for defining pre/post windows in the event anchors.

---

## Common troubleshooting

### Missing / invalid FMP key

```bash
echo $FMP_API_KEY
```

### `events/` is empty

Normal until you run `30_build_events_all.py` (or scripts 21–26 individually).

### `news/` is empty

Normal until you run `05_fmp_news.py`.

### `06_fmp_financials.py` says “No transcript events found…”

Run transcripts first:

```bash
python scripts/04_fmp_transcripts.py --ticker NFLX --start 2021-01-01 --end 2025-12-31
python scripts/06_fmp_financials.py --ticker NFLX --start 2021-01-01 --end 2025-12-31
```

### Abnormal returns fail because SPX is missing

If you deleted the SPX cache and are running script 26 directly:

```bash
python scripts/25_yf_download_spx.py --symbol ^GSPC --start 2021-01-01 --end 2025-12-31 --out-rel _tmp_market/spx/prices/yf_ohlcv_daily.csv
python scripts/26_compute_abnormal_returns.py --ticker NVDA
```

If you run `30_build_events_all.py`, SPX will be re-fetched automatically.
