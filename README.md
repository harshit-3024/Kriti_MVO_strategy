# Quantitative Trading Strategy Backtest

A long-only quantitative equity backtester for Indian NSE stocks, built as a competition entry for BeyondIRR Group / LevUp. The strategy combines relative strength filtering, residual momentum scoring, and mean-variance portfolio optimisation, benchmarked against the Nifty 500 index.

---

## How It Works

At each rebalance, the strategy runs through four steps:

1. **Relative Strength Filter** — Ranks all stocks against the Nifty 500 and keeps the top 200.
2. **Residual Momentum Score** — From those 200, selects the top 20 using a log-linear regression slope × R² score.
3. **Mean-Variance Optimisation** — Allocates weights across the 20 stocks to maximise the Sharpe ratio (weights constrained to 2%–10% per stock).
4. **Rebalancing & Risk Rules** — Rebalances every 21 trading days (or earlier on a volume spike), and cuts any position that drops 30% from entry.

Benchmark data (Nifty 500 / `^CRSLDX`) is downloaded automatically from Yahoo Finance at runtime.

---

## Requirements

**Python 3.8+** and an active internet connection (for benchmark download).

Install dependencies:

```bash
pip install pandas numpy scipy matplotlib tqdm yfinance
```

---

## Input Data

A `.csv` or `.parquet` file with one row per stock per trading day.

| Column | Accepted names |
|---|---|
| Date | `date`, `tradedate`, `trading_date` |
| Symbol | `symbol`, `ticker`, `fid` |
| OHLCV | `open`, `high`, `low`, `close`, `volume` |
| Sector | `sector` |

> **Note:** The script requires ~252 trading days of warmup data *before* the active trading start date. If your active window begins on `2010-01-01`, your data file should include data from at least `2009-01-01`.

---

## Usage

**Basic:**
```bash
python strategy.py --data NSE_Data_2010_2020.csv
```

**With all options:**
```bash
python strategy.py \
  --data data.csv \
  --output my_results \
  --start 2020-01-01 \
  --end 2025-01-01 \
  --capital 1000000
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to data file (`.csv` or `.parquet`) | **Required** |
| `--output` | Folder to save results | `results` |
| `--start` | Active trading start date (`YYYY-MM-DD`) | `2020-01-01` |
| `--end` | Active trading end date (`YYYY-MM-DD`) | `2025-01-01` |
| `--capital` | Starting capital in rupees | `5,000,000` |

---

## Outputs

All files are saved to the `--output` directory.

**Charts**
- `backtest_performance.png` — NAV curve, drawdown, Sharpe, and turnover
- `sector_allocation_rebalance.png` — Sector mix at every rebalance
- `strategy_vs_nifty500.png` — Side-by-side benchmark comparison

**CSV Files**
- `portfolio_values.csv` — Daily portfolio value and returns
- `rebalance_log.csv` — Every rebalance event and its trigger
- `trade_log.csv` — Every individual buy/sell trade
- `portfolio_snapshots.csv` — Holdings at each rebalance
- `sector_allocations.csv` — Daily sector weights
- `nifty500_benchmark.csv` — Strategy vs. benchmark side by side
- `rolling_performance_metrics.csv` — 1Y / 3Y / 5Y rolling outperformance

**Report**
- `performance_report.txt` — Full stats summary (CAGR, Sharpe, max drawdown, etc.)

---

## Configuration

Key parameters can be adjusted in the `CONFIG` dict at the top of `strategy.py`:

| Parameter | Default | Description |
|---|---|---|
| `top_k_rs` | `200` | Stocks kept after Relative Strength filter |
| `top_n_final` | `20` | Final portfolio size |
| `min_weight` | `0.02` | Minimum weight per stock (2%) |
| `max_weight` | `0.10` | Maximum weight per stock (10%) |
| `static_rebalance_days` | `21` | Rebalance every N days |
| `volume_spike_threshold` | `1.35` | Multiplier to trigger early rebalance |
| `transaction_cost` | `0.00268` | Per-side cost per trade |
| `stop_loss_pct` | `0.30` | Cut position if it falls 30% from entry |
| `use_trailing_stop` | `False` | Set `True` to use a trailing stop instead |
| `scoring_lookback_days` | `252` | Lookback window for momentum scoring |
| `optimization_lookback_days` | `252` | Lookback window for covariance estimation |
