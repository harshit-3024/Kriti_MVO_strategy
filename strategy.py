
"""
WHAT THIS SCRIPT DOES
----------------------
This script runs a full backtest of a quantitative long-only equity strategy
on Indian NSE stocks from 2010 to 2020 (or any date range you choose).

At each rebalance, the strategy:
  1. Filters the top 200 stocks by Relative Strength vs the Nifty 500
  2. Picks the top 20 of those using a residual momentum score
     (log-linear regression slope × R²)
  3. Allocates weights using Mean-Variance Optimisation (Max Sharpe)
  4. Rebalances every 21 days, or earlier if a volume spike is detected
  5. Cuts any position that falls 30% from entry (stop-loss)

Results are benchmarked against the Nifty 500 index (downloaded automatically
from Yahoo Finance).

REQUIREMENTS
----------------------
  pip install pandas numpy scipy matplotlib tqdm yfinance

  Python 3.8+
  Active internet connection (to download the Nifty 500 benchmark)

INPUT DATA FORMAT
----------------------
A CSV or Parquet file with one row per stock per trading day.

Required columns (names are flexible — the script auto-detects them):
  - date / tradedate / trading_date
  - symbol / ticker / fid
  - open, high, low, close
  - volume 
  - sector 


HOW TO RUN
----------------------
Basic usage:
  python strategy.py --data NSE_Data_2010_2020.csv

With custom options:
  python strategy.py --data data.csv --output my_results
  python strategy.py --data data.csv --start 2010-01-01 --end 2020-01-01
  python strategy.py --data data.csv --capital 1000000

Arguments:
  --data      Path to your data file (.csv or .parquet)   [REQUIRED]
  --output    Folder to save results in                   [default: results]
  --start     Active trading start date  (YYYY-MM-DD)     [default: CONFIG['start_date']]
  --end       Active trading end date    (YYYY-MM-DD)     [default: CONFIG['end_date']]
  --capital   Starting capital in rupees                  [default: 5000000]


OUTPUTS (saved to --output folder)
----------------------
  Charts:
    backtest_performance.png       — NAV curve, drawdown, Sharpe, turnover
    sector_allocation_rebalance.png — Sector mix at every rebalance
    strategy_vs_nifty500.png       — Side-by-side benchmark comparison

  CSV Files:
    portfolio_values.csv           — Daily portfolio value and returns
    rebalance_log.csv              — Every rebalance event and its trigger
    trade_log.csv                  — Every individual buy/sell trade
    sector_allocations.csv         — Daily sector weights
    nifty500_benchmark.csv         — Strategy vs benchmark side by side
    rolling_performance_metrics.csv — 1Y / 3Y / 5Y rolling outperformance

  Report:
    performance_report.txt         — Full stats summary (CAGR, Sharpe, etc.)


KEY CONFIGURATION (edit CONFIG dict at top of script)
----------------------
  top_k_rs              = 200     # Stocks kept after Relative Strength filter
  top_n_final           = 20      # Final portfolio size
  min_weight            = 0.02    # Minimum weight per stock (2%)
  max_weight            = 0.10    # Maximum weight per stock (10%)
  static_rebalance_days = 21      # Rebalance every N days
  volume_spike_threshold= 1.35    # Trigger early rebalance if volume spikes
  transaction_cost      = 0.00268 # Per-side cost per trade
  stop_loss_pct         = 0.30    # Cut position if it falls 30% from entry
  use_trailing_stop     = False   # Set True for trailing stop-loss


NOTES
----------------------
  - The script needs ~252 trading days of warmup data before active trading
    begins. If your file starts on 2010-01-01, make sure it includes data
    from at least one year earlier for best results.
  - Internet is required only for downloading the Nifty 500 benchmark.
    Your stock data is read entirely from the local file.
  - All performance metrics are calculated on active trading days only,
    excluding the warmup period.

================================================================================
"""
import sys
import gc
import socket
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from tqdm import tqdm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


CONFIG = {
    'date_column': 'date',
    'symbol_column': 'symbol',
    
    'start_date': '2010-01-01',
    'end_date': '2020-01-01',
    'initial_capital': 5000000,
    'scoring_lookback_days': 252,
    'optimization_lookback_days': 252,
    
    'top_k_rs': 200,
    'top_n_final': 20,
    'min_weight': 0.02,
    'max_weight': 0.10,
    
    'static_rebalance_days': 21,
    'volume_spike_threshold': 1.35,
    'min_days_between_rebal': 7,
    
    'transaction_cost': 0.00268,
    'stop_loss_pct': 0.30,
    'use_trailing_stop': False,
    'output_dir': 'results',
}

def check_internet_connection(timeout=5):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except (socket.error, OSError):
        return False

def require_internet_for_nifty500():
    print("      Checking internet connectivity for Nifty 500 download...")
    if not check_internet_connection():
        print()
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║                   ⚠  NO INTERNET CONNECTION DETECTED  ⚠                  ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  The backtest requires an active internet connection to download the     ║")
        print("║  Nifty 500 benchmark data (^CRSLDX) from Yahoo Finance.                  ║")
        print("║                                                                          ║")
        print("║  Please check your network and try again:                                ║")
        print("║    1. Ensure your machine is connected to the internet.                  ║")
        print("║    3. Re-run the script once connectivity is restored.                   ║")
        print("╚══════════════════════════════════════════════════════════════════════════╝")
        print()
        sys.exit(1)
    print("      ✓ Internet connection confirmed.")

def clear_memory(*objects, verbose=True):
    if verbose:
        print("\n[Memory] Clearing large objects from RAM...")
    deleted = 0
    for obj in objects:
        try:
            del obj
            deleted += 1
        except Exception:
            pass
    collected = gc.collect()
    if verbose:
        print(f"[Memory] ✓ Deleted {deleted} object(s), GC collected {collected} cycles.")


def prompt_clear_memory_after_run(exec_px_df, close_df, low_df, volume_df,
                                   bt, nifty_df, sector_df, trade_df):
    print()
    print("─" * 60)
    print("  MEMORY CLEANUP")
    print("─" * 60)
    try:
        answer = input("  Free large dataframes from memory now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  (Non-interactive mode – skipping memory cleanup prompt)")
        return
    if answer in ("", "y", "yes"):
        clear_memory(exec_px_df, close_df, low_df, volume_df,
                     bt, nifty_df, sector_df, trade_df, verbose=True)
        print("  ✓ Memory freed.  Results have already been saved to disk.")
    else:
        print("  Memory NOT cleared – variables remain accessible in this session.")
    print("─" * 60)


def detect_column_names(df):
    date_candidates = ['date', 'tradedate', 'trading_date', 'timestamp', 'datetime']
    symbol_candidates = ['symbol', 'fid', 'stock_id', 'ticker', 'code']
    
    date_col = None
    symbol_col = None
    
    for col in df.columns:
        if col.lower() in date_candidates:
            date_col = col
            break
    
    for col in df.columns:
        if col.lower() in symbol_candidates:
            symbol_col = col
            break
    
    if not date_col or not symbol_col:
        raise ValueError(f"Could not auto-detect columns. Found: {list(df.columns)}")
    
    return date_col, symbol_col


def load_master_data(filepath, start_date, end_date, lookback_days):
    print(f"\n{'='*80}")
    print(f"LOADING DATA FROM: {filepath}")
    print(f"{'='*80}")
    
    filepath = Path(filepath)
    
    print("\n[1/6] Reading master file...")
    if filepath.suffix.lower() == '.parquet':
        df = pd.read_parquet(filepath)
        print(f"      ✓ Loaded {len(df):,} rows from Parquet")
    elif filepath.suffix.lower() in ['.csv', '.txt']:
        df = pd.read_csv(filepath)
        print(f"      ✓ Loaded {len(df):,} rows from CSV")
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}. Use .csv or .parquet")
    
    print("\n[2/6] Detecting column names...")
    date_col, symbol_col = detect_column_names(df)
    print(f"      ✓ Date column: '{date_col}'")
    print(f"      ✓ Symbol column: '{symbol_col}'")
    
    df = df.rename(columns={date_col: 'date', symbol_col: 'symbol'})
    
    print("\n[3/6] Processing dates...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['symbol', 'date'])
    print(f"      ✓ Full data range in file: {df['date'].min()} to {df['date'].max()}")
    
    print("\n[4/6] Calculating execution prices...")
    required_price_cols = ['open', 'high', 'low', 'close']
    
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in required_price_cols:
            col_mapping[col] = col_lower
    df = df.rename(columns=col_mapping)
    
    df['exec_price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
    
    volume_cols = ['volume', 'traded_volume', 'vol', 'quantity']
    volume_col = None
    for col in df.columns:
        if col.lower() in volume_cols:
            volume_col = col
            break
    
    if volume_col:
        df = df.rename(columns={volume_col: 'volume'})
    else:
        df['volume'] = 0
        print(f"      Warning: No volume column found, using 0")
    
    sector_cols = ['sector', 'gics_sector', 'industry', 'gics']
    sector_col = None
    for col in df.columns:
        if col.lower() in sector_cols:
            sector_col = col
            break
    
    if sector_col:
        df = df.rename(columns={sector_col: 'sector'})
    else:
        df['sector'] = 0.0
        print(f"      Warning: No sector column found, using 0")
    
    if 'in_nse500' not in df.columns:
        df['in_nse500'] = 1
    
    df = df[['date', 'symbol', 'exec_price', 'close', 'volume', 'low', 'sector', 'in_nse500']].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
    
    print(f"      ✓ Processed {df['symbol'].nunique():,} unique stocks")
    
    print("\n[5/6] Filtering date range with lookback warmup...")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    all_trading_dates_in_file = sorted(df['date'].unique())
    dates_before_start = [d for d in all_trading_dates_in_file if d < start_dt]
    available_lookback = len(dates_before_start)

    if available_lookback == 0:
        print(f"      ⚠ WARNING: No data available before {start_date} for lookback warmup.")
        print(f"               Active trading will begin from the date when {lookback_days} warmup days are available.")
        warmup_start_dt = start_dt
    elif available_lookback < lookback_days:
        print(f"      ⚠ WARNING: Only {available_lookback} trading days available before {start_date}.")
        print(f"               Required lookback: {lookback_days} days.")
        print(f"               Trading will start once {lookback_days} warmup days have accumulated for sufficient lookback.")
        warmup_start_dt = pd.Timestamp(dates_before_start[0])
    else:
        warmup_days_to_use = dates_before_start[-lookback_days:]
        warmup_start_dt = pd.Timestamp(warmup_days_to_use[0])
        print(f"      ✓ Found {available_lookback} pre-start trading days, using last {lookback_days} for warmup.")

    df = df[(df['date'] >= warmup_start_dt) & (df['date'] <= end_dt)]
    print(f"      ✓ {len(df):,} rows, {df['symbol'].nunique():,} stocks")

    print("\n[6/6] Creating pivot tables...")
    exec_px_df = df.pivot(index='date', columns='symbol', values='exec_price')
    close_df = df.pivot(index='date', columns='symbol', values='close')
    low_df = df.pivot(index='date', columns='symbol', values='low')
    volume_df = df.pivot(index='date', columns='symbol', values='volume')

    sector_map = df.drop_duplicates(subset=['symbol'])[['symbol', 'sector']].set_index('symbol')['sector'].to_dict()

    all_dates = exec_px_df.index.sort_values()

    dates_before_active = sorted([d for d in all_dates if d < start_dt])
    actual_warmup_days = len(dates_before_active)

    remaining_lookback = max(0, lookback_days - actual_warmup_days)
    dates_from_start = sorted([d for d in all_dates if d >= start_dt])
    if remaining_lookback < len(dates_from_start):
        real_active_start = dates_from_start[remaining_lookback]
    elif dates_from_start:
        real_active_start = dates_from_start[-1]
    else:
        real_active_start = start_dt

    if actual_warmup_days >= lookback_days:
        active_start_label = str(start_dt.date())
    else:
        active_start_label = str(pd.Timestamp(real_active_start).date())

    print(f"      ✓ Created pivot tables")
    print(f"      ✓ Data window (incl. warmup): {warmup_start_dt.date()} to {end_date}")
    print(f"      ✓ Active trading start:       {active_start_label}")
    print(f"      ✓ Active trading end:          {end_date}")
    print(f"      ✓ Total days in data window:  {len(all_dates):,} (warmup: {lookback_days}, active: {len(all_dates) - actual_warmup_days - remaining_lookback})")
    print(f"      ✓ Stocks: {len(exec_px_df.columns):,}")
    
    print(f"\n{'='*80}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*80}\n")
    
    return exec_px_df, close_df, low_df, volume_df, all_dates, sector_map, start_dt, actual_warmup_days, pd.Timestamp(real_active_start)

def download_nifty500_benchmark(start_date, end_date, warmup_start_dt=None):
    print("      Downloading Nifty 500 benchmark data...")
    require_internet_for_nifty500()

    fetch_start = warmup_start_dt if warmup_start_dt is not None else pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    nifty_ticker = "^CRSLDX"
    
    try:
        nifty_data = yf.download(nifty_ticker, start=fetch_start, end=end_dt, progress=False)
        if nifty_data.empty:
            raise ValueError("No data downloaded")
    except:
        print("      Warning: Could not download Nifty 500 (^CRSLDX), trying alternative...")
        nifty_ticker = "NIFTY500.NS"
        try:
            nifty_data = yf.download(nifty_ticker, start=fetch_start, end=end_dt, progress=False)
            if nifty_data.empty:
                raise ValueError("No data downloaded")
        except:
            print("      Error: Could not download Nifty 500.")
            print("      Please check your internet connection and try again.")
            sys.exit(1)
    
    if isinstance(nifty_data['Close'], pd.DataFrame):
        close_prices = nifty_data['Close'].iloc[:, 0]
    else:
        close_prices = nifty_data['Close']
    
    bench_df = pd.DataFrame({'nifty500': close_prices})
    
    return bench_df.ffill().bfill()


def calculate_rs_momentum(stocks_df, bench_series):
    rs_df = stocks_df.div(bench_series, axis=0)
    rs_momentum = (rs_df.iloc[-1] / rs_df.iloc[0]) - 1
    return rs_momentum

def calculate_residual_alpha(stock_prices):
    y = np.log(stock_prices.values)
    x = np.arange(len(y))
    
    slope, intercept = np.polyfit(x, y, 1)
    
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    if ss_tot == 0:
        r_squared = 0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    alpha = slope * r_squared
    
    return alpha

def get_max_sharpe_weights(returns, min_weight, max_weight, n_stocks, risk_free_rate=0.0):
    n = returns.shape[1]
    if n < 2:
        return np.ones(n) / n
    
    cov = returns.cov() * 252
    mean_returns = returns.mean() * 252
    
    def neg_sharpe(w):
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((min_weight, max_weight) for _ in range(n))
    initial_guess = np.ones(n) / n
    
    try:
        res = minimize(neg_sharpe, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints, tol=1e-6)
        if res.success:
            return res.x
    except Exception as e:
        pass
    
    return initial_guess

def select_portfolio_stocks(close_prices, bench_series, scoring_lookback, top_k_rs, top_n_final):
    rs_momentum = calculate_rs_momentum(close_prices, bench_series)
    rs_momentum = rs_momentum.dropna()
    
    if rs_momentum.empty:
        return []
    
    top_rs_stocks = rs_momentum.nlargest(top_k_rs).index.tolist()
    
    alpha_scores = {}
    for stock in top_rs_stocks:
        stock_prices = close_prices[stock].dropna()
        if len(stock_prices) >= scoring_lookback:
            alpha = calculate_residual_alpha(stock_prices)
            alpha_scores[stock] = alpha
    
    if not alpha_scores:
        return []
    
    alpha_series = pd.Series(alpha_scores)
    selected_stocks = alpha_series.nlargest(top_n_final).index.tolist()
    
    return selected_stocks


def apply_integer_share_rounding(weights, exec_prices, portfolio_value):
    exec_prices = pd.to_numeric(exec_prices, errors="coerce").replace([np.inf, -np.inf], np.nan)
    weights = weights.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    active = weights[weights > 0].index
    if len(active) == 0:
        return weights, pd.Series(0.0, index=weights.index), 0.0
    
    s = float(weights.loc[active].sum())
    if s <= 0:
        return weights, pd.Series(0.0, index=weights.index), 0.0
    
    weights.loc[active] = weights.loc[active] / s
    
    shares = pd.Series(0.0, index=weights.index)
    for sid in active:
        px = exec_prices.get(sid, np.nan)
        if pd.isna(px) or px <= 0:
            continue
        alloc = portfolio_value * float(weights.loc[sid])
        sh = np.floor(alloc / float(px))
        shares.loc[sid] = sh
    
    value = shares * exec_prices
    value = value.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    total_invested = float(value.sum())
    cash_remaining = portfolio_value - total_invested
    
    if total_invested <= 0:
        return pd.Series(0.0, index=weights.index), shares, portfolio_value
    
    w_int = (value / portfolio_value).fillna(0.0)
    w_final = pd.Series(0.0, index=weights.index)
    w_final.loc[w_int.index] = w_int
    
    return w_final, shares, cash_remaining


def check_rebalance_triggers(current_date, last_rebal_date, avg_volume_recent, 
                            avg_volume_baseline, params):
    days_since_rebal = (current_date - last_rebal_date).days if last_rebal_date else 999
    
    if days_since_rebal >= params['static_rebalance_days']:
        return True, "static_period"
    
    if days_since_rebal < params['min_days_between_rebal']:
        return False, None
    
    if avg_volume_baseline > 0:
        volume_ratio = avg_volume_recent / avg_volume_baseline
        if volume_ratio >= params['volume_spike_threshold']:
            return True, "volume_spike"
    
    return False, None
def run_backtest(exec_px_df, close_df, low_df, volume_df, bench_df,
                 all_dates, sector_map, params, active_start_dt, pre_start_warmup_days):

    portfolio_value = params['initial_capital']
    cash = 0.0

    daily_values = []
    daily_returns = []
    daily_cash = []
    dates = []

    all_stocks = close_df.columns
    w = pd.Series(0.0, index=all_stocks)
    shares = pd.Series(0.0, index=all_stocks)

    n_rebalances = 0
    total_trades = 0
    total_turnover = 0.0
    total_tcost_paid = 0.0
    total_sl_triggers = 0

    rebal_log = []
    portfolio_snapshots = []
    sector_allocations = []
    trade_log = []

    prev_close_df = close_df.shift(1)

    last_rebal_date = None
    volume_window = 20
    scoring_lookback = params['scoring_lookback_days']
    optimization_lookback = params['optimization_lookback_days']
    effective_lookback = max(scoring_lookback, optimization_lookback)
    stop_loss_pct = params['stop_loss_pct']

    bench_series = bench_df['nifty500'].reindex(all_dates).ffill()
    pending_sl = []
    pending_rebalance = None

    entry_prices = pd.Series(np.nan, index=all_stocks)
    highest_prices = pd.Series(np.nan, index=all_stocks)

    active_start_idx = None
    for i, d in enumerate(all_dates):
        if d >= active_start_dt:
            active_start_idx = i
            break
    if active_start_idx is None:
        active_start_idx = 0

    warmup_end_idx = active_start_idx

    total_active_days = sum(
        1 for i in range(len(all_dates))
        if (pre_start_warmup_days + (i - warmup_end_idx)) >= effective_lookback
    )

    active_day_counter = 0
    pbar = tqdm(enumerate(all_dates), desc="Running backtest", total=total_active_days)

    for idx, current_date in pbar:

        days_since_active_start = idx - warmup_end_idx
        accumulated_lookback = pre_start_warmup_days + days_since_active_start
        is_warmup = accumulated_lookback < effective_lookback

        val_start = portfolio_value

        if is_warmup:
            dates.append(current_date)
            daily_returns.append(0.0)
            daily_values.append(portfolio_value)
            daily_cash.append(cash)
            sector_allocations.append({'date': current_date})
            continue

        exec_today = exec_px_df.loc[current_date]
        low_today = low_df.loc[current_date]

        if pending_sl:
            total_sl_recovered = 0.0
            for stock in pending_sl:
                exit_px = exec_today.get(stock, np.nan)
                if pd.isna(exit_px):
                    continue
                trade_log.append({
                    'date': current_date,
                    'stock': stock,
                    'action': 'SELL_SL',
                    'shares': shares.loc[stock],
                    'price': exit_px,
                    'old_weight': w.loc[stock],
                    'new_weight': 0.0,
                    'trigger': 'stop_loss_T+1'
                })
                gross_value = shares.loc[stock] * exit_px
                tcost = params['transaction_cost'] * gross_value
                recovered_value = gross_value - tcost
                total_tcost_paid += tcost
                total_sl_triggers += 1
                total_sl_recovered += recovered_value
                w.loc[stock] = 0.0
                shares.loc[stock] = 0.0
                entry_prices.loc[stock] = np.nan
                highest_prices.loc[stock] = np.nan
            
            if total_sl_recovered > 0 and w.sum() > 0:
                remaining_stocks = all_stocks[w > 0]
                if len(remaining_stocks) > 0:
                    current_total_value = (shares * exec_today).sum() + cash
                    reinvest_weights = w[remaining_stocks] / w[remaining_stocks].sum()
                    for stock in remaining_stocks:
                        px = exec_today.get(stock, np.nan)
                        if pd.isna(px) or px <= 0:
                            continue
                        add_shares = np.floor((total_sl_recovered * reinvest_weights.loc[stock]) / px)
                        if add_shares > 0:
                            shares.loc[stock] += add_shares
                            trade_log.append({
                                'date': current_date,
                                'stock': stock,
                                'action': 'BUY',
                                'shares': add_shares,
                                'price': px,
                                'old_weight': w.loc[stock],
                                'new_weight': w.loc[stock],
                                'trigger': 'sl_reinvest'
                            })
                    reinvested_value = (shares * exec_today).sum() + cash - current_total_value
                    reinvest_cost = params['transaction_cost'] * reinvested_value
                    total_tcost_paid += reinvest_cost
                    cash = cash + total_sl_recovered - reinvested_value - reinvest_cost
                else:
                    cash += total_sl_recovered
            else:
                cash += total_sl_recovered
            
            pending_sl = []

        if pending_rebalance is not None:
            exec_curr = exec_today
            w_target = pending_rebalance['w_target']
            trigger_reason = pending_rebalance['trigger_reason']
            val_before_trade = portfolio_value
            w_new, shares_new, cash = apply_integer_share_rounding(
                w_target, exec_curr, portfolio_value
            )
            turnover_fraction = (w_new - w).abs().sum()
            tcost = params['transaction_cost'] * portfolio_value * turnover_fraction
            trade_count = int(((w_new - w).abs() > 1e-12).sum())
            portfolio_value -= tcost
            total_tcost_paid += tcost
            for stock in all_stocks:
                old_pos = shares.loc[stock]
                new_pos = shares_new.loc[stock]
                if abs(new_pos - old_pos) > 0:
                    trade_log.append({
                        'date': current_date,
                        'stock': stock,
                        'action': 'BUY' if new_pos > old_pos else 'SELL',
                        'shares': abs(new_pos - old_pos),
                        'price': exec_curr.get(stock, np.nan),
                        'old_weight': w.loc[stock],
                        'new_weight': w_new.loc[stock],
                        'trigger': trigger_reason
                    })
                    if new_pos > old_pos:
                        px = exec_curr.get(stock, np.nan)
                        if not pd.isna(px) and px > 0:
                            entry_prices.loc[stock] = px
                            highest_prices.loc[stock] = px
                    if new_pos == 0:
                        entry_prices.loc[stock] = np.nan
                        highest_prices.loc[stock] = np.nan
            n_rebalances += 1
            total_trades += trade_count
            total_turnover += float(turnover_fraction)
            w = w_new
            shares = shares_new
            last_rebal_date = current_date
            rebal_log.append({
                "date": current_date,
                "trigger": trigger_reason,
                "turnover": float(turnover_fraction),
                "trades": trade_count,
                "n_stocks": int((w_new > 0.001).sum()),
                "return": (portfolio_value / val_before_trade) - 1.0,
            })
            pending_rebalance = None

        if params['use_trailing_stop']:
            for stock in all_stocks:
                if w.loc[stock] > 0:
                    close_px = close_df.loc[current_date].get(stock, np.nan)
                    if not pd.isna(close_px) and close_px > 0:
                        hp = highest_prices.loc[stock]
                        if pd.isna(hp) or close_px > hp:
                            highest_prices.loc[stock] = close_px

        sl_triggered_today = []
        for stock in all_stocks:
            if w.loc[stock] <= 0:
                continue
            ep = entry_prices.loc[stock]
            if pd.isna(ep) or ep <= 0:
                continue
            
            if params['use_trailing_stop']:
                hp = highest_prices.loc[stock]
                if pd.isna(hp) or hp <= 0:
                    sl_level = ep * (1.0 - stop_loss_pct)
                else:
                    sl_level = hp * (1.0 - stop_loss_pct)
            else:
                sl_level = ep * (1.0 - stop_loss_pct)
                
            low_px = low_today.get(stock, np.nan)
            if not pd.isna(low_px) and low_px <= sl_level:
                sl_triggered_today.append(stock)

        if sl_triggered_today and idx + 1 < len(all_dates):
            pending_sl = list(set(pending_sl).union(sl_triggered_today))

        close_ret = close_df.loc[current_date] / prev_close_df.loc[current_date] - 1.0
        close_ret = close_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        r_port = float((w * close_ret).sum())
        if pd.isna(r_port):
            r_port = 0.0

        portfolio_value = (portfolio_value - cash) * (1.0 + r_port) + cash

        if idx > volume_window:
            recent_dates = all_dates[idx-volume_window:idx]
            avg_volume_recent = volume_df.loc[recent_dates].mean().mean()
            if idx > 2 * volume_window:
                baseline_dates = all_dates[idx-2*volume_window:idx-volume_window]
                avg_volume_baseline = volume_df.loc[baseline_dates].mean().mean()
            else:
                avg_volume_baseline = avg_volume_recent
        else:
            avg_volume_recent = 0
            avg_volume_baseline = 1

        should_rebal, trigger_reason = check_rebalance_triggers(
            current_date, last_rebal_date,
            avg_volume_recent, avg_volume_baseline,
            params
        )

        if should_rebal and idx + 1 < len(all_dates):
            lookback_start_idx = max(0, idx - scoring_lookback + 1)
            scoring_dates = all_dates[lookback_start_idx:idx+1]
            opt_lookback_start_idx = max(0, idx - optimization_lookback + 1)
            optimization_dates = all_dates[opt_lookback_start_idx:idx+1]

            hist_close_scoring = close_df.loc[scoring_dates]
            hist_close_optimization = close_df.loc[optimization_dates]
            hist_bench = bench_series.loc[scoring_dates]

            selected_stocks = select_portfolio_stocks(
                hist_close_scoring, hist_bench, scoring_lookback,
                params['top_k_rs'], params['top_n_final']
            )
            if selected_stocks:
                past_rets = hist_close_optimization[selected_stocks].pct_change().dropna()
                if len(past_rets) > 20:
                    opt_w = get_max_sharpe_weights(
                        past_rets,
                        params['min_weight'],
                        params['max_weight'],
                        len(selected_stocks)
                    )
                else:
                    opt_w = np.ones(len(selected_stocks)) / len(selected_stocks)
                w_target = pd.Series(0.0, index=all_stocks)
                w_target.loc[selected_stocks] = opt_w
                pending_rebalance = {
                    'w_target': w_target,
                    'trigger_reason': trigger_reason
                }

        day_ret = (portfolio_value / val_start) - 1.0 if val_start != 0 else 0.0

        dates.append(current_date)
        daily_returns.append(day_ret)
        daily_values.append(portfolio_value)
        daily_cash.append(cash)

        sector_weights = {}
        for sid in all_stocks:
            if w.loc[sid] > 0:
                sector = sector_map.get(sid, 0.0)
                sector_weights[sector] = sector_weights.get(sector, 0) + w.loc[sid]

        sector_allocations.append({
            'date': current_date,
            **{f'sector_{int(s)}': v for s, v in sector_weights.items()}
        })

        active_day_counter += 1
        pbar.n = active_day_counter
        pbar.set_postfix_str(
            f"PV: ₹{portfolio_value/1e6:.2f}M | Ret: {day_ret*100:+.2f}% | SL: {total_sl_triggers}",
            refresh=True
        )

    pbar.close()

    bt = pd.DataFrame({
        "date": dates,
        "daily_ret": daily_returns,
        "portfolio_value": daily_values,
        "cash": daily_cash,
    })

    bt["cum_ret"] = bt["portfolio_value"] / bt["portfolio_value"].iloc[0] - 1.0

    remaining_lookback = max(0, effective_lookback - pre_start_warmup_days)
    dates_from_active_start = sorted(bt.loc[bt['date'] >= active_start_dt, 'date'].unique())
    if remaining_lookback < len(dates_from_active_start):
        real_active_start = pd.Timestamp(dates_from_active_start[remaining_lookback])
    elif dates_from_active_start:
        real_active_start = pd.Timestamp(dates_from_active_start[-1])
    else:
        real_active_start = active_start_dt

    actual_warmup_count = bt[bt['date'] < real_active_start].shape[0]

    stats = calculate_performance_stats(
        bt,
        n_rebalances,
        total_trades,
        total_turnover,
        total_tcost_paid,
        total_sl_triggers,
        params['initial_capital'],
        real_active_start=real_active_start
    )

    sector_df = pd.DataFrame(sector_allocations)
    trade_df = pd.DataFrame(trade_log)

    return bt, stats, rebal_log, portfolio_snapshots, sector_df, trade_df, actual_warmup_count


def calculate_performance_stats(bt, n_rebalances, total_trades, total_turnover,
                                total_tcost_paid, total_sl_triggers, initial_capital,
                                real_active_start=None, effective_lookback=0):
    if real_active_start is not None:
        bt = bt[bt['date'] >= real_active_start].copy().reset_index(drop=True)
    else:
        bt = bt.iloc[effective_lookback:].copy().reset_index(drop=True)
    bt["cum_ret"] = bt["portfolio_value"] / bt["portfolio_value"].iloc[0] - 1.0

    daily_ret = bt["portfolio_value"].pct_change().dropna()
    mean_daily = daily_ret.mean()
    vol_daily = daily_ret.std()
    trading_days = bt.shape[0]
    ann_factor = np.sqrt(252)

    total_return = bt["portfolio_value"].iloc[-1] / bt["portfolio_value"].iloc[0] - 1.0
    cagr = (bt["portfolio_value"].iloc[-1] / bt["portfolio_value"].iloc[0]) ** (252 / trading_days) - 1
    ann_vol = vol_daily * ann_factor
    sharpe = mean_daily / vol_daily * ann_factor if vol_daily > 0 else np.nan

    downside = daily_ret.copy()
    downside[downside > 0] = 0
    downside_std = downside.std()
    sortino = mean_daily / downside_std * ann_factor if downside_std > 0 else np.nan

    win_rate = (daily_ret > 0).sum() / len(daily_ret) if len(daily_ret) > 0 else np.nan
    
    cum_max = bt["portfolio_value"].cummax()
    drawdown = bt["portfolio_value"] / cum_max - 1.0
    max_dd = drawdown.min()
    
    avg_turnover = total_turnover / n_rebalances if n_rebalances > 0 else np.nan
    
    tcost_pct_of_equity = (total_tcost_paid / initial_capital) * 100

    active_start = bt["date"].iloc[0]
    active_end = bt["date"].iloc[-1]
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'final_value': bt["portfolio_value"].iloc[-1],
        'trading_days': trading_days,
        'n_rebalances': n_rebalances,
        'total_trades': total_trades,
        'avg_turnover': avg_turnover,
        'total_tcost_paid': total_tcost_paid,
        'tcost_pct_of_equity': tcost_pct_of_equity,
        'total_sl_triggers': total_sl_triggers,
        'initial_capital': initial_capital,
        'active_start': active_start,
        'active_end': active_end,
        'win_rate': win_rate,
    }


def get_nifty500_data(start_date, end_date, initial_capital):
    print("      Downloading Nifty 500 data from Yahoo Finance...")
    require_internet_for_nifty500()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    nifty_ticker = "^CRSLDX"
    
    try:
        nifty_data = yf.download(nifty_ticker, start=start_dt, end=end_dt, progress=False)
        if nifty_data.empty:
            raise ValueError("No data downloaded")
    except:
        print("      Warning: Could not download Nifty 500 (^CRSLDX), trying alternative ticker...")
        nifty_ticker = "NIFTY500.NS"
        try:
            nifty_data = yf.download(nifty_ticker, start=start_dt, end=end_dt, progress=False)
        except:
            print("      Error: Could not download Nifty 500 data. Using synthetic benchmark.")
            dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
            nifty_data = pd.DataFrame({
                'Close': 10000 * (1 + 0.12 * np.arange(len(dates)) / 252 + np.random.randn(len(dates)) * 0.01)
            }, index=dates)
    
    if isinstance(nifty_data['Close'], pd.DataFrame):
        close_prices = nifty_data['Close'].iloc[:, 0].values
    else:
        close_prices = nifty_data['Close'].values
    
    nifty_df = pd.DataFrame({
        'date': nifty_data.index,
        'nifty500_price': close_prices
    })
    
    nifty_df = nifty_df.sort_values('date').reset_index(drop=True)
    nifty500_initial_price = nifty_df['nifty500_price'].iloc[0]
    nifty_df['value'] = initial_capital * (nifty_df['nifty500_price'] / nifty500_initial_price)
    
    return nifty_df[['date', 'value']]


def calculate_rolling_metrics(bt, nifty_df):
    portfolio_df = bt[['date', 'portfolio_value']].copy()
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df = portfolio_df.set_index('date')
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
    
    nifty_df_copy = nifty_df.copy()
    nifty_df_copy['date'] = pd.to_datetime(nifty_df_copy['date'])
    nifty_df_copy = nifty_df_copy.set_index('date')
    nifty_df_copy['returns'] = nifty_df_copy['value'].pct_change()
    
    results = []
    
    for window_days, window_name in [(252, '1 Year'), (756, '3 Year'), (1260, '5 Year')]:
        if len(portfolio_df) >= window_days:
            rolling_data = []
            
            for i in range(window_days, len(portfolio_df)):
                port_window = portfolio_df.iloc[i-window_days:i]
                
                strat_ret = (port_window['portfolio_value'].iloc[-1] / port_window['portfolio_value'].iloc[0]) - 1
                
                start_date = port_window.index[0]
                end_date = port_window.index[-1]
                bench_window = nifty_df_copy.loc[start_date:end_date]
                
                if len(bench_window) > 0:
                    bench_ret = (bench_window['value'].iloc[-1] / bench_window['value'].iloc[0]) - 1
                    outperformance = strat_ret - bench_ret
                    rolling_data.append(outperformance)
            
            if rolling_data:
                avg_outperf = np.mean(rolling_data) * 100
                worst_underperf = np.min(rolling_data) * 100
                
                results.append({
                    'window': window_name,
                    'avg_outperformance': avg_outperf,
                    'worst_underperformance': worst_underperf
                })
    
    return pd.DataFrame(results)


def calculate_benchmark_metrics(portfolio_df, nifty_df):
    portfolio_df = portfolio_df.copy()
    nifty_df = nifty_df.copy()
    
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
    nifty_df['returns'] = nifty_df['value'].pct_change()
    
    merged = pd.merge(
        portfolio_df[['date', 'portfolio_value', 'returns']],
        nifty_df[['date', 'value', 'returns']],
        left_on='date',
        right_on='date',
        how='inner',
        suffixes=('_strategy', '_benchmark')
    )
    
    benchmark_returns = merged['returns_benchmark'].dropna()

    strategy_returns_full = portfolio_df['portfolio_value'].pct_change().dropna()
    strategy_returns = merged['returns_strategy'].dropna()

    strategy_total = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
    benchmark_total = (nifty_df['value'].iloc[-1] / nifty_df['value'].iloc[0]) - 1

    strategy_trading_days = len(portfolio_df)
    benchmark_trading_days = len(nifty_df)
    strategy_cagr = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) ** (252 / strategy_trading_days) - 1
    benchmark_cagr = (nifty_df['value'].iloc[-1] / nifty_df['value'].iloc[0]) ** (252 / benchmark_trading_days) - 1

    ann_factor = np.sqrt(252)

    strategy_vol = strategy_returns_full.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)

    s_mean = strategy_returns_full.mean()
    s_std  = strategy_returns_full.std()
    b_mean = benchmark_returns.mean()
    b_std  = benchmark_returns.std()
    strategy_sharpe  = s_mean / s_std  * ann_factor if s_std  > 0 else np.nan
    benchmark_sharpe = b_mean / b_std  * ann_factor if b_std  > 0 else np.nan

    strategy_win_rate = (strategy_returns_full > 0).sum() / len(strategy_returns_full) if len(strategy_returns_full) > 0 else np.nan
        
    strategy_cumulative = (1 + strategy_returns).cumprod()
    strategy_running_max = strategy_cumulative.expanding().max()
    strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
    strategy_max_dd = strategy_drawdown.min()
    
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_running_max = benchmark_cumulative.expanding().max()
    benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
    benchmark_max_dd = benchmark_drawdown.min()
    
    excess_returns = strategy_returns - benchmark_returns
    information_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252) + 1e-6)
    
    up_periods = benchmark_returns > 0
    down_periods = benchmark_returns < 0
    
    up_capture = (strategy_returns[up_periods].sum() / (benchmark_returns[up_periods].sum() + 1e-6))
    down_capture = (strategy_returns[down_periods].sum() / (benchmark_returns[down_periods].sum() + 1e-6))
    
    print("\n" + "="*90)
    print("STRATEGY vs NIFTY 500 BENCHMARK")
    print("="*90)
    print("\nRETURNS:")
    print(f"  Strategy:       {strategy_total*100:>10.2f}%")
    print(f"  Nifty 500:      {benchmark_total*100:>10.2f}%")
    print(f"  Alpha:          {(strategy_total - benchmark_total)*100:>10.2f}%")
    print("\nANNUALIZED:")
    print(f"  Strategy CAGR:  {strategy_cagr*100:>10.2f}%")
    print(f"  Nifty 500 CAGR: {benchmark_cagr*100:>10.2f}%")
    print(f"  Strategy Vol:   {strategy_vol*100:>10.2f}%")
    print(f"  Nifty 500 Vol:  {benchmark_vol*100:>10.2f}%")
    print("\nRISK-ADJUSTED:")
    print(f"  Strategy Sharpe:{strategy_sharpe:>10.3f}")
    print(f"  Nifty 500 Sharpe:{benchmark_sharpe:>10.3f}")
    print(f"  Info Ratio:     {information_ratio:>10.3f}")
    print("\nDRAWDOWN:")
    print(f"  Strategy:       {strategy_max_dd*100:>10.2f}%")
    print(f"  Nifty 500:      {benchmark_max_dd*100:>10.2f}%")
    print("\nWIN RATE:")
    print(f"  Strategy:       {strategy_win_rate*100:>10.2f}%")
    print("\nCAPTURE:")
    print(f"  Up Capture:     {up_capture*100:>10.2f}%")
    print(f"  Down Capture:   {down_capture*100:>10.2f}%")
    print("\nFINAL VALUES:")
    print(f"  Strategy:       ₹{portfolio_df['portfolio_value'].iloc[-1]:>12,.0f}")
    print(f"  Nifty 500:      ₹{nifty_df['value'].iloc[-1]:>12,.0f}")
    print(f"  Difference:     ₹{portfolio_df['portfolio_value'].iloc[-1] - nifty_df['value'].iloc[-1]:>12,.0f}")
    print("="*90 + "\n")
    
    return {
        'strategy_total': strategy_total, 'benchmark_total': benchmark_total,
        'alpha': strategy_total - benchmark_total,
        'strategy_cagr': strategy_cagr, 'benchmark_cagr': benchmark_cagr,
        'strategy_vol': strategy_vol, 'benchmark_vol': benchmark_vol,
        'strategy_sharpe': strategy_sharpe, 'benchmark_sharpe': benchmark_sharpe,
        'information_ratio': information_ratio,
        'strategy_max_dd': strategy_max_dd, 'benchmark_max_dd': benchmark_max_dd,
        'up_capture': up_capture, 'down_capture': down_capture,
        'strategy_win_rate': strategy_win_rate,
    }


def print_performance_stats(stats):
    print("\n" + "="*80)
    print("CORE METRICS - PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Active Trading Start:           {stats['active_start'].date()}")
    print(f"Active Trading End:             {stats['active_end'].date()}")
    print(f"CAGR:                          {stats['cagr']*100:6.2f}%")
    print(f"Annualized Std Dev:            {stats['ann_vol']*100:6.2f}%")
    print(f"Maximum Drawdown:              {stats['max_dd']*100:6.2f}%")
    print(f"Sharpe Ratio:                  {stats['sharpe']:6.2f}")
    print(f"\nTotal Return:                  {stats['total_return']*100:6.2f}%")
    print(f"Sortino Ratio:                 {stats['sortino']:6.2f}")
    print(f"\nTrading Days:                  {stats['trading_days']}")
    print(f"Rebalance Events:              {stats['n_rebalances']}")
    print(f"Total Trades Taken:            {stats['total_trades']}")
    print(f"Stop Loss Triggers:            {stats['total_sl_triggers']}")
    print(f"Win Rate (daily):              {stats['win_rate']*100:6.2f}%")
    print(f"Avg Turnover/Rebalance:        {stats['avg_turnover']*100:6.2f}%")
    print(f"Total Transaction Cost:        ₹{stats['total_tcost_paid']:,.2f}  ({stats['tcost_pct_of_equity']:.4f}% of initial equity)")
    print(f"\nInitial Capital:               ₹{stats['initial_capital']:,.0f}")
    print(f"Final Portfolio Value:         ₹{stats['final_value']:,.0f}")
    print("="*80)


def save_performance_report(stats, metrics, rolling_metrics_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / 'performance_report.txt'

    lines = []
    lines.append("=" * 80)
    lines.append("KRITI 2026 - QUANTITATIVE TRADING STRATEGY BACKTEST REPORT")
    lines.append("BeyondIRR Group / LevUp")
    lines.append("=" * 80)
    lines.append("")
    lines.append("CORE METRICS - PERFORMANCE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Active Trading Start:           {stats['active_start'].date()}")
    lines.append(f"Active Trading End:             {stats['active_end'].date()}")
    lines.append(f"CAGR:                          {stats['cagr']*100:6.2f}%")
    lines.append(f"Annualized Std Dev:            {stats['ann_vol']*100:6.2f}%")
    lines.append(f"Maximum Drawdown:              {stats['max_dd']*100:6.2f}%")
    lines.append(f"Sharpe Ratio:                  {stats['sharpe']:6.2f}")
    lines.append(f"Total Return:                  {stats['total_return']*100:6.2f}%")
    lines.append(f"Sortino Ratio:                 {stats['sortino']:6.2f}")
    lines.append(f"Trading Days:                  {stats['trading_days']}")
    lines.append(f"Rebalance Events:              {stats['n_rebalances']}")
    lines.append(f"Total Trades Taken:            {stats['total_trades']}")
    lines.append(f"Stop Loss Triggers:            {stats['total_sl_triggers']}")
    lines.append(f"Win Rate (daily):              {stats['win_rate']*100:6.2f}%")
    lines.append(f"Avg Turnover/Rebalance:        {stats['avg_turnover']*100:6.2f}%")
    lines.append(f"Total Transaction Cost:        ₹{stats['total_tcost_paid']:,.2f}  ({stats['tcost_pct_of_equity']:.4f}% of initial equity)")
    lines.append(f"Initial Capital:               ₹{stats['initial_capital']:,.0f}")
    lines.append(f"Final Portfolio Value:         ₹{stats['final_value']:,.0f}")
    lines.append("")
    lines.append("=" * 90)
    lines.append("STRATEGY vs NIFTY 500 BENCHMARK")
    lines.append("=" * 90)
    lines.append("")
    lines.append("RETURNS:")
    lines.append(f"  Strategy:       {metrics['strategy_total']*100:>10.2f}%")
    lines.append(f"  Nifty 500:      {metrics['benchmark_total']*100:>10.2f}%")
    lines.append(f"  Alpha:          {metrics['alpha']*100:>10.2f}%")
    lines.append("")
    lines.append("ANNUALIZED:")
    lines.append(f"  Strategy CAGR:  {metrics['strategy_cagr']*100:>10.2f}%")
    lines.append(f"  Nifty 500 CAGR: {metrics['benchmark_cagr']*100:>10.2f}%")
    lines.append(f"  Strategy Vol:   {metrics['strategy_vol']*100:>10.2f}%")
    lines.append(f"  Nifty 500 Vol:  {metrics['benchmark_vol']*100:>10.2f}%")
    lines.append("")
    lines.append("RISK-ADJUSTED:")
    lines.append(f"  Strategy Sharpe:{metrics['strategy_sharpe']:>10.3f}")
    lines.append(f"  Nifty 500 Sharpe:{metrics['benchmark_sharpe']:>10.3f}")
    lines.append(f"  Info Ratio:     {metrics['information_ratio']:>10.3f}")
    lines.append("")
    lines.append("DRAWDOWN:")
    lines.append(f"  Strategy:       {metrics['strategy_max_dd']*100:>10.2f}%")
    lines.append(f"  Nifty 500:      {metrics['benchmark_max_dd']*100:>10.2f}%")
    lines.append("")
    lines.append("CAPTURE RATIOS:")
    lines.append(f"  Up Capture:     {metrics['up_capture']*100:>10.2f}%")
    lines.append(f"  Down Capture:   {metrics['down_capture']*100:>10.2f}%")
    lines.append("")
    if not rolling_metrics_df.empty:
        lines.append("=" * 90)
        lines.append("ROLLING PERFORMANCE METRICS (vs Nifty 500)")
        lines.append("=" * 90)
        for _, row in rolling_metrics_df.iterrows():
            lines.append(
                f"{row['window']:8s}: Avg Outperformance = {row['avg_outperformance']:>7.2f}% | "
                f"Worst Underperformance = {row['worst_underperformance']:>7.2f}%"
            )
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"      ✓ Saved: {report_path}")


def save_backtest_plots(bt, rebal_log, sector_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    
    axes[0].plot(bt['date'], bt['portfolio_value'] / 1e6, linewidth=2.5, color='#27ae60')
    axes[0].set_title('Portfolio NAV / Equity Curve', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_ylabel('Value (₹ Millions)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=bt['portfolio_value'].iloc[0] / 1e6, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[0].legend(loc='upper left', fontsize=11)
    
    cum_max = bt['portfolio_value'].cummax()
    drawdown = (bt['portfolio_value'] / cum_max - 1.0) * 100
    axes[1].fill_between(bt['date'], drawdown, 0, alpha=0.4, color='red')
    axes[1].plot(bt['date'], drawdown, linewidth=2, color='darkred')
    
    rebal_df = pd.DataFrame(rebal_log)
    if not rebal_df.empty:
        trigger_colors = {
            'static_period': ('green', 'Static Period'),
            'volume_spike': ('blue', 'Volume Spike')
        }
        
        plotted_triggers = set()
        for _, row in rebal_df.iterrows():
            dd_at_rebal = drawdown[bt['date'] == row['date']].values
            if len(dd_at_rebal) > 0:
                trigger = row['trigger']
                color, label = trigger_colors.get(trigger, ('purple', trigger))
                
                if trigger not in plotted_triggers:
                    axes[1].scatter(row['date'], dd_at_rebal[0], color=color, s=50, 
                                  zorder=5, alpha=0.7, label=label)
                    plotted_triggers.add(trigger)
                else:
                    axes[1].scatter(row['date'], dd_at_rebal[0], color=color, s=50, 
                                  zorder=5, alpha=0.7)
        
        if plotted_triggers:
            axes[1].legend(loc='lower left', fontsize=10, framealpha=0.95)
    
    axes[1].set_title('Drawdown Curve', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    rolling_ret = bt['daily_ret'].rolling(30).mean() * 252
    rolling_vol = bt['daily_ret'].rolling(30).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol
    axes[2].plot(bt['date'], rolling_sharpe, linewidth=2, color='#2980b9')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling 30-Day Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Sharpe', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    if not rebal_df.empty and 'turnover' in rebal_df.columns:
        rebal_df['turnover_pct'] = rebal_df['turnover'] * 100
        axes[3].bar(range(len(rebal_df)), rebal_df['turnover_pct'], color='steelblue', alpha=0.7)
        axes[3].axhline(y=rebal_df['turnover_pct'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Average: {rebal_df["turnover_pct"].mean():.1f}%')
        axes[3].set_title('Portfolio Turnover per Rebalance', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Turnover (%)', fontsize=12)
        axes[3].set_xlabel('Rebalance Event #', fontsize=12)
        axes[3].legend(loc='upper right', fontsize=10)
        axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'backtest_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: {output_path}")

    if not sector_df.empty:
        rebal_dates = rebal_df['date'].values if not rebal_df.empty else sector_df['date'].unique()
        
        sector_at_rebal = sector_df[sector_df['date'].isin(rebal_dates)].copy()
        
        if not sector_at_rebal.empty:
            sector_cols = [col for col in sector_at_rebal.columns if col.startswith('sector_')]
            sector_at_rebal[sector_cols] = sector_at_rebal[sector_cols].fillna(0) * 100
            
            fig, ax = plt.subplots(figsize=(16, 8))
            
            bottom = np.zeros(len(sector_at_rebal))
            for col in sector_cols:
                ax.bar(range(len(sector_at_rebal)), sector_at_rebal[col], 
                       bottom=bottom, label=col, alpha=0.8)
                bottom += sector_at_rebal[col].values
            
            ax.set_title('Sector Allocation at Each Rebalance', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Allocation (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Rebalance Event #', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            sector_path = output_dir / 'sector_allocation_rebalance.png'
            plt.savefig(sector_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: {sector_path}")


def plot_benchmark_comparison(bt, nifty_df, rolling_metrics_df, output_dir):
    output_dir = Path(output_dir)
    
    portfolio_df = bt.copy()
    portfolio_df.columns = ['Date' if c == 'date' else c for c in portfolio_df.columns]
    portfolio_df.columns = ['Portfolio_Value' if c == 'portfolio_value' else c for c in portfolio_df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    norm_strategy = portfolio_df['Portfolio_Value'] / portfolio_df['Portfolio_Value'].iloc[0] * 100
    norm_nifty = nifty_df['value'] / nifty_df['value'].iloc[0] * 100
    axes[0, 0].plot(portfolio_df['Date'], norm_strategy, label='Strategy', linewidth=2.5, color='#27ae60')
    axes[0, 0].plot(nifty_df['date'], norm_nifty, label='Nifty 500', linewidth=2.5, color='#e74c3c', alpha=0.8)
    axes[0, 0].set_title('Performance: Strategy vs Nifty 500', fontsize=16, fontweight='bold', pad=20)
    axes[0, 0].set_ylabel('Normalized (Base=100)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='upper left', fontsize=12, framealpha=0.95)
    axes[0, 0].grid(True, alpha=0.3)
    
    portfolio_df['returns'] = portfolio_df['Portfolio_Value'].pct_change()
    nifty_df['returns'] = nifty_df['value'].pct_change()
    strategy_returns = portfolio_df['returns'].dropna()
    benchmark_returns = nifty_df['returns'].dropna()
    
    strategy_cumulative = (1 + strategy_returns).cumprod()
    strategy_running_max = strategy_cumulative.expanding().max()
    strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_running_max = benchmark_cumulative.expanding().max()
    benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
    
    axes[0, 1].fill_between(range(len(strategy_drawdown)), strategy_drawdown*100, 0, alpha=0.4, color='#27ae60', label='Strategy')
    axes[0, 1].plot(strategy_drawdown*100, color='#27ae60', linewidth=2)
    axes[0, 1].fill_between(range(len(benchmark_drawdown)), benchmark_drawdown*100, 0, alpha=0.4, color='#e74c3c', label='Nifty 500')
    axes[0, 1].plot(benchmark_drawdown*100, color='#e74c3c', linewidth=2)
    axes[0, 1].set_title('Drawdown Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].set_ylabel('%', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='lower left', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    excess_returns = strategy_returns - benchmark_returns
    cumulative_excess = excess_returns.cumsum() * 100
    axes[1, 0].fill_between(range(len(cumulative_excess)), cumulative_excess, 0, where=(cumulative_excess >= 0), alpha=0.4, color='green', label='Outperformance')
    axes[1, 0].fill_between(range(len(cumulative_excess)), cumulative_excess, 0, where=(cumulative_excess < 0), alpha=0.4, color='red', label='Underperformance')
    axes[1, 0].plot(cumulative_excess, linewidth=2.5, color='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title('Cumulative Excess Returns', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].set_ylabel('Excess Return (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Trading Days', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='upper left', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    rolling_excess = excess_returns.rolling(252).sum() * 100
    axes[1, 1].plot(rolling_excess.values, linewidth=2, color='#2980b9')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[1, 1].set_title('1-Year Rolling Outperformance', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Excess Return (%)', fontsize=12)
    axes[1, 1].set_xlabel('Trading Days', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'strategy_vs_nifty500.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"      ✓ Saved: {output_path}")
    plt.close()


def save_results(bt, rebal_log, portfolio_snapshots, sector_df, trade_df, output_dir, active_start, active_end):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    active_start = pd.Timestamp(active_start)
    active_end = pd.Timestamp(active_end)

    bt_output = bt[['date', 'portfolio_value', 'cum_ret', 'daily_ret', 'cash']].copy()
    bt_output = bt_output[(bt_output['date'] >= active_start) & (bt_output['date'] <= active_end)]
    bt_output.columns = ['Date', 'Portfolio_Value', 'Cumulative_Return', 'Daily_Return', 'Cash']
    bt_output.to_csv(output_dir / 'portfolio_values.csv', index=False)
    print(f"      ✓ Saved: {output_dir / 'portfolio_values.csv'}")

    if rebal_log:
        rebal_df = pd.DataFrame(rebal_log)
        rebal_df['date'] = pd.to_datetime(rebal_df['date'])
        rebal_df = rebal_df[(rebal_df['date'] >= active_start) & (rebal_df['date'] <= active_end)]
        rebal_df = rebal_df.drop(columns=['return'], errors='ignore')
        rebal_df.to_csv(output_dir / 'rebalance_log.csv', index=False)
        print(f"      ✓ Saved: {output_dir / 'rebalance_log.csv'}")

    if portfolio_snapshots:
        snapshot_df = pd.DataFrame(portfolio_snapshots)
        snapshot_df['date'] = pd.to_datetime(snapshot_df['date'])
        snapshot_df = snapshot_df[(snapshot_df['date'] >= active_start) & (snapshot_df['date'] <= active_end)]
        snapshot_df.to_csv(output_dir / 'portfolio_snapshots.csv', index=False)
        print(f"      ✓ Saved: {output_dir / 'portfolio_snapshots.csv'}")

    if not sector_df.empty:
        sector_out = sector_df.copy()
        sector_out['date'] = pd.to_datetime(sector_out['date'])
        sector_out = sector_out[(sector_out['date'] >= active_start) & (sector_out['date'] <= active_end)]
        sector_out.to_csv(output_dir / 'sector_allocations.csv', index=False)
        print(f"      ✓ Saved: {output_dir / 'sector_allocations.csv'}")

    if not trade_df.empty:
        trade_out = trade_df.copy()
        trade_out['date'] = pd.to_datetime(trade_out['date'])
        trade_out = trade_out[(trade_out['date'] >= active_start) & (trade_out['date'] <= active_end)]
        if 'trigger' not in trade_out.columns:
            trade_out['trigger'] = ''
        trade_out['trigger'] = trade_out['trigger'].fillna('')
        trade_out.to_csv(output_dir / 'trade_log.csv', index=False)
        print(f"      ✓ Saved: {output_dir / 'trade_log.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description='Kriti 2026 Quantitative Trading Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy.py --data NSE_Data_2010_2020.csv
  python strategy.py --data NSE_Data_2010_2020.parquet
  python strategy.py --data data.csv --output my_results
  python strategy.py --data data.csv --start 2010-01-01 --end 2020-01-01
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to master data file (.csv or .parquet)')
    parser.add_argument('--output', default=CONFIG['output_dir'], help='Output directory for results')
    parser.add_argument('--start', default=CONFIG['start_date'], help='Active trading start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=CONFIG['end_date'], help='Active trading end date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=CONFIG['initial_capital'], help='Initial capital')
    
    args = parser.parse_args()
    
    params = CONFIG.copy()
    params['start_date'] = args.start
    params['end_date'] = args.end
    params['initial_capital'] = args.capital
    params['output_dir'] = args.output
    
    print("\n" + "="*80)
    print("KRITI 2026 - QUANTITATIVE TRADING STRATEGY BACKTEST")
    print("BeyondIRR Group / LevUp")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data file:              {args.data}")
    print(f"  Active trading range:   {args.start} to {args.end}")
    print(f"  Initial capital:        ₹{args.capital:,.0f}")
    print(f"  Output dir:             {args.output}")
    
    effective_lookback_days = max(params['scoring_lookback_days'], params['optimization_lookback_days'])

    exec_px_df, close_df, low_df, volume_df, all_dates, sector_map, active_start_dt, pre_start_warmup_days, data_active_start = load_master_data(
        args.data, params['start_date'], params['end_date'], effective_lookback_days
    )
    
    print(f"\n{'='*80}")
    print("DOWNLOADING BENCHMARK")
    print(f"{'='*80}\n")
    warmup_start_dt = all_dates[0] if len(all_dates) > 0 else pd.to_datetime(params['start_date'])
    bench_df = download_nifty500_benchmark(params['start_date'], params['end_date'], warmup_start_dt)
    print(f"      ✓ Loaded {len(bench_df)} benchmark days\n")
    
    print(f"\n{'='*80}")
    print("RUNNING BACKTEST")
    print(f"{'='*80}\n")
    print(f"Strategy Parameters:")
    print(f"  Active Trading Start:   {data_active_start.date()}")
    print(f"  Active Trading End:     {params['end_date']}")
    print(f"  Scoring Lookback:       {params['scoring_lookback_days']} days")
    print(f"  Optimization Lookback:  {params['optimization_lookback_days']} days")
    print(f"  Pre-start warmup days:  {pre_start_warmup_days}")
    print(f"  Top RS stocks:          {params['top_k_rs']}")
    print(f"  Final portfolio:        {params['top_n_final']} stocks")
    print(f"  Rebalance:              Every {params['static_rebalance_days']} days")
    print(f"  Stop Loss Type:         {'Trailing' if params['use_trailing_stop'] else 'Fixed'} ({params['stop_loss_pct']*100:.1f}%)")
    print(f"  Transaction cost:       {params['transaction_cost']*100:.3f}% per side\n")

    bt, stats, rebal_log, portfolio_snapshots, sector_df, trade_df, effective_lookback = run_backtest(
        exec_px_df, close_df, low_df, volume_df, bench_df, all_dates, sector_map, params,
        active_start_dt, pre_start_warmup_days
    )

    real_active_start = stats['active_start']

    bt_trimmed = bt[bt['date'] >= real_active_start].copy().reset_index(drop=True)
    bt_trimmed["cum_ret"] = bt_trimmed["portfolio_value"] / bt_trimmed["portfolio_value"].iloc[0] - 1.0
    
    print_performance_stats(stats)
    print(f"  (Metrics computed on {stats['trading_days']} active trading days, excluding {effective_lookback}-day warmup period)")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}\n")
    
    print("Fetching Nifty 500 data...")
    nifty_df = get_nifty500_data(params['start_date'], params['end_date'], params['initial_capital'])
    
    active_start_date = real_active_start
    nifty_df = nifty_df[nifty_df['date'] >= active_start_date].copy().reset_index(drop=True)
    nifty_initial = nifty_df['value'].iloc[0]
    nifty_df['value'] = params['initial_capital'] * (nifty_df['value'] / nifty_initial)
    
    print("Calculating rolling metrics...")
    rolling_metrics_df = calculate_rolling_metrics(bt_trimmed, nifty_df)
    
    print("\nROLLING PERFORMANCE METRICS (vs Nifty 500):")
    print("="*90)
    if not rolling_metrics_df.empty:
        for _, row in rolling_metrics_df.iterrows():
            print(f"{row['window']:8s}: Avg Outperformance = {row['avg_outperformance']:>7.2f}% | "
                  f"Worst Underperformance = {row['worst_underperformance']:>7.2f}%")
    else:
        print("Insufficient data for rolling calculations")
    print("="*90)
    
    print("\nCalculating metrics...")
    metrics = calculate_benchmark_metrics(bt_trimmed, nifty_df)
    
    print(f"\n{'='*80}")
    print("GENERATING OUTPUTS")
    print(f"{'='*80}\n")
    
    print("Creating charts...")
    save_backtest_plots(bt_trimmed, rebal_log, sector_df, params['output_dir'])
    plot_benchmark_comparison(bt_trimmed, nifty_df, rolling_metrics_df, params['output_dir'])
    
    print("\nSaving CSV files...")
    save_results(bt_trimmed, rebal_log, portfolio_snapshots, sector_df, trade_df,
                 params['output_dir'], real_active_start, stats['active_end'])

    strat_out = bt_trimmed[['date', 'portfolio_value', 'daily_ret', 'cum_ret']].copy()
    strat_out.columns = ['Date', 'Strategy_Value', 'Strategy_Daily_Return', 'Strategy_Cum_Return']
    nifty_out = nifty_df.copy()
    nifty_out['Nifty_Daily_Return'] = nifty_out['value'].pct_change()
    nifty_out['Nifty_Cum_Return'] = nifty_out['value'] / nifty_out['value'].iloc[0] - 1.0
    nifty_out = nifty_out.rename(columns={'date': 'Date', 'value': 'Nifty_Value'})
    combined = pd.merge(strat_out, nifty_out, on='Date', how='outer').sort_values('Date')
    combined = combined[(combined['Date'] >= pd.Timestamp(real_active_start)) &
                        (combined['Date'] <= pd.Timestamp(stats['active_end']))]
    combined.to_csv(Path(params['output_dir']) / 'nifty500_benchmark.csv', index=False)
    print(f"      ✓ Saved: {params['output_dir']}/nifty500_benchmark.csv")
    
    if not rolling_metrics_df.empty:
        rolling_metrics_df.to_csv(Path(params['output_dir']) / 'rolling_performance_metrics.csv', index=False)
        print(f"      ✓ Saved: {params['output_dir']}/rolling_performance_metrics.csv")

    print("\nSaving performance report...")
    save_performance_report(stats, metrics, rolling_metrics_df, params['output_dir'])
    
    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {params['output_dir']}/")
    print("\nGenerated files:")
    print("   Charts:")
    print("     - backtest_performance.png")
    print("     - sector_allocation_rebalance.png")
    print("     - strategy_vs_nifty500.png")
    print("   CSV Files:")
    print("     - portfolio_values.csv")
    print("     - rebalance_log.csv")
    print("     - portfolio_snapshots.csv")
    print("     - sector_allocations.csv")
    print("     - trade_log.csv")
    print("     - nifty500_benchmark.csv")
    print("     - rolling_performance_metrics.csv")
    print("   Report:")
    print("     - performance_report.txt")
    print(f"\n{'='*80}\n")

    prompt_clear_memory_after_run(
        exec_px_df, close_df, low_df, volume_df,
        bt, nifty_df, sector_df, trade_df
    )


if __name__ == "__main__":
    main()