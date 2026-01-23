# Sentiment Analysis on PEAD

This repo builds a per-ticker dataset that combines:
- Yahoo Finance daily prices (adjusted OHLCV)
- Technical indicators computed from prices
- Financial Modeling Prep (FMP) earnings calendar (EPS/Revenue est/actual + timing)
- FMP earnings call transcripts (json + plain text)

Outputs are written under `data/{TICKER}/...` in a consistent folder structure.

## Setup

```bash
python -m venv .sentvenv
source .sentvenv/bin/activate
pip install -r requirements.txt
````

Set your FMP key (required for calendar/transcripts/news):

```bash
export FMP_API_KEY="YOUR_KEY"
```

## Directory Layout (per ticker)

* `data/{TICKER}/prices/yf_ohlcv_daily.csv`
* `data/{TICKER}/technicals/technicals_daily.csv`
* `data/{TICKER}/calendar/earnings_calendar.csv` (+ raw/meta json)
* `data/{TICKER}/transcripts/YYYY-MM-DD/{meta.json, transcript.json, transcript.txt}`
* `data/{TICKER}/news/` (optional, if you run the news script)
* `data/{TICKER}/events/` (pilot merged panel, if you run the event panel builder)

## Single-ticker prep

```bash
python scripts/00_init_ticker_dirs.py --ticker NVDA
python scripts/01_yf_prices.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
python scripts/02_technicals.py --ticker NVDA
python scripts/03_fmp_earnings_calendar.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
python scripts/04_fmp_transcripts.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

Or use the orchestrator:

```bash
python scripts/06_ticker_prep_all.py --ticker NVDA --start 2021-01-01 --end 2025-12-31
```

## Batch prep (custom ticker list)

```bash
python scripts/06_ticker_prep_batch.py --tickers AAPL MSFT NVDA AMD AVGO
```

Or pass a CSV/text file of tickers:

```bash
python scripts/06_ticker_prep_batch.py --tickers-file tickers.csv
```

## Optional: news by date (one day)

```bash
python scripts/05_fmp_news.py --ticker NVDA --date 2024-11-20
```

## Optional: build pilot earnings-event panel

This merges calendar + prices + sentiment summaries into a single per-event dataset:

```bash
python scripts/10_build_event_panel_pilot.py --ticker NVDA
```

It writes:

* `data/NVDA/events/event_panel_pilot.csv`

## BERT folder

`bert/` contains FinBERT LoRA training/eval/infer scripts and stored adapters. This is separate from the core data download pipeline.