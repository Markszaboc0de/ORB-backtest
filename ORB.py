import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:  # yfinance optional; handled at runtime
    yf = None

# --- 1. Data Loading and Setup (Conceptual) ---
# ASSUME: 'df' is a pandas DataFrame with 1-minute data
# Index must be a datetime object (localized to EST or UTC/EST-aware)
# Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
# For demonstration, we'll assume the index is the trading date and time.
#
# Example Data Loading (You would replace this with actual data loading)
# data = {
#     'DateTime': pd.to_datetime(['2024-01-02 09:30:00', ..., '2024-01-02 16:00:00', ...]),
#     'Open': [...], 'High': [...], 'Low': [...], 'Close': [...], 'Volume': [...]
# }
# df = pd.DataFrame(data).set_index('DateTime')
# df = df.tz_localize('EST') # Ensure timezone is correct

# --- 2. Configuration and Constants ---
# Strategy parameters
COMMISSION_ROUND_TURN = 2.00
RISK_REWARD_RATIO = 3.0
TIME_STOP_HOUR = 10
TIME_STOP_MINUTE = 30
TIME_EOD_EXIT_HOUR = 15
TIME_EOD_EXIT_MINUTE = 50

# Range filter parameters
RANGE_FILTER_MIN = 0.0005  # unused in engulfing version
RANGE_FILTER_MAX = 0.0100  # unused in engulfing version

# --- 3. Core Strategy Backtest Function ---

def backtest_orb_strategy(df: pd.DataFrame, initial_capital: float = 10000.0, return_trades: bool = False):
    # Prepare the DataFrame for backtesting
    df['Date'] = df.index.date
    trades = [] # List to store trade outcomes
    
    for date, daily_data in df.groupby('Date'):
        # --- 1. Opening Range (first 5 minutes 9:30-9:34) ---
        orb_start = pd.to_datetime(f"{date} 09:30:00").tz_localize(df.index.tz)
        orb_after = pd.to_datetime(f"{date} 09:35:00").tz_localize(df.index.tz)
        orb_data = daily_data.loc[orb_start:orb_after - pd.Timedelta(minutes=1)]
        if orb_data.empty:
            continue
        or_high = orb_data['High'].max()
        or_low = orb_data['Low'].min()

        # --- 2. Engulfing Breakout Entry (scan post 9:35 close) ---
        scan_start = pd.to_datetime(f"{date} 09:36:00").tz_localize(df.index.tz)
        day_after_orb = daily_data.loc[scan_start:]

        def body_low(open_price, close_price):
            return min(open_price, close_price)

        def body_high(open_price, close_price):
            return max(open_price, close_price)

        executed = False

        # Iterate with window of 3 candles: breakout (i), engulf (i+1), entry on open of (i+2)
        bars = day_after_orb.copy()
        idx_list = list(bars.index)
        for i in range(0, len(idx_list) - 2):
            t0 = idx_list[i]
            t1 = idx_list[i + 1]
            t2 = idx_list[i + 2]

            b0 = bars.loc[t0]
            b1 = bars.loc[t1]

            # Breakout candle close beyond range
            long_breakout = b0['Close'] > or_high
            short_breakout = b0['Close'] < or_low

            if not (long_breakout or short_breakout):
                continue

            # Engulfing (same-direction) on next candle
            b0_low = body_low(b0['Open'], b0['Close'])
            b0_high = body_high(b0['Open'], b0['Close'])

            if long_breakout:
                # Bullish engulfing candle: bullish and body covers previous body
                bullish = b1['Close'] > b1['Open']
                b1_low = body_low(b1['Open'], b1['Close'])
                b1_high = body_high(b1['Open'], b1['Close'])
                engulf = bullish and (b1_low <= b0_low) and (b1_high >= b0_high)
                if not engulf:
                    continue
                position_type = 'Long'
                entry_time = t2
                entry_price = bars.loc[t2]['Open']
                stop_loss = float(b1['Low'])
                risk = max(0.0, entry_price - stop_loss)
                if risk <= 0:
                    continue
                take_profit = entry_price + RISK_REWARD_RATIO * risk
            else:
                # Short: bearish engulfing candle
                bearish = b1['Close'] < b1['Open']
                b1_low = body_low(b1['Open'], b1['Close'])
                b1_high = body_high(b1['Open'], b1['Close'])
                engulf = bearish and (b1_low <= b0_low) and (b1_high >= b0_high)
                if not engulf:
                    continue
                position_type = 'Short'
                entry_time = t2
                entry_price = bars.loc[t2]['Open']
                stop_loss = float(b1['High'])
                risk = max(0.0, stop_loss - entry_price)
                if risk <= 0:
                    continue
                take_profit = entry_price - RISK_REWARD_RATIO * risk

            # From entry_time to EOD, check TP/SL; force EOD close if neither hit
            exits = daily_data.loc[entry_time:]
            eod_exit_time = pd.to_datetime(f"{date} {TIME_EOD_EXIT_HOUR}:{TIME_EOD_EXIT_MINUTE}:00").tz_localize(df.index.tz)
            exit_time = None
            exit_price = None

            for e_index, e_bar in exits.iterrows():
                if e_index >= eod_exit_time:
                    exit_time = e_index
                    exit_price = e_bar['Close']
                    break
                if position_type == 'Long':
                    if e_bar['Low'] <= stop_loss:
                        exit_time = e_index
                        exit_price = stop_loss
                        break
                    if e_bar['High'] >= take_profit:
                        exit_time = e_index
                        exit_price = take_profit
                        break
                else:  # Short
                    if e_bar['High'] >= stop_loss:
                        exit_time = e_index
                        exit_price = stop_loss
                        break
                    if e_bar['Low'] <= take_profit:
                        exit_time = e_index
                        exit_price = take_profit
                        break

            if exit_price is not None:
                point_pnl = (exit_price - entry_price) if position_type == 'Long' else (entry_price - exit_price)
                dollar_pnl = point_pnl
                net_pnl = dollar_pnl - COMMISSION_ROUND_TURN

                trades.append({
                    'Date': date,
                    'EntryTime': entry_time,
                    'ExitTime': exit_time,
                    'Position': position_type,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'OR_High': or_high,
                    'OR_Low': or_low,
                    'StopLoss': stop_loss,
                    'TakeProfit': take_profit,
                    'PointPNL': point_pnl,
                    'DollarPNL': dollar_pnl,
                    'NetPNL': net_pnl,
                    'Outcome': 'Win' if net_pnl > 0 else ('Loss' if net_pnl < -COMMISSION_ROUND_TURN else 'BreakEven'),
                    'Risk': risk
                })
                executed = True
            # One trade per day
            if executed:
                break

    # --- 6. Performance Metrics to Output ---
    if not trades:
        empty_result = {"Note": "No trades executed based on filters and time constraints."}
        if return_trades:
            return empty_result, pd.DataFrame(), pd.DataFrame()
        return empty_result
        
    trades_df = pd.DataFrame(trades)
    
    # Basic Metrics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['NetPNL'] > 0]
    losing_trades = trades_df[trades_df['NetPNL'] < 0]

    gross_profit = winning_trades['NetPNL'].sum()
    gross_loss = losing_trades['NetPNL'].sum() * -1 # Make positive for ratio
    total_net_profit = trades_df['NetPNL'].sum()
    
    # Calculate Metrics
    profit_factor = gross_profit / gross_loss if gross_loss else (float('inf') if gross_profit > 0 else 0)
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades else 0

    avg_winner_points = winning_trades['PointPNL'].mean()
    avg_loser_points = losing_trades['PointPNL'].mean()
    avg_winner_dollars = winning_trades['DollarPNL'].mean()
    avg_loser_dollars = losing_trades['DollarPNL'].mean()
    
    # Drawdown Calculation (Cumulative P&L)
    trades_df['CumulativePNL'] = trades_df['NetPNL'].cumsum()
    trades_df['Peak'] = trades_df['CumulativePNL'].cummax()
    trades_df['Drawdown'] = trades_df['Peak'] - trades_df['CumulativePNL']
    
    max_drawdown_dollars = trades_df['Drawdown'].max()
    # Max Drawdown Percentage (from the highest peak)
    max_drawdown_percent = (max_drawdown_dollars / trades_df['Peak'].max()) * 100 if trades_df['Peak'].max() > 0 else 0

    results = {
        "Total Net Profit / Loss ($)": round(total_net_profit, 2),
        "Gross Profit ($)": round(gross_profit, 2),
        "Gross Loss ($)": round(gross_loss, 2),
        "Profit Factor (GP/GL)": round(profit_factor, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Maximum Drawdown ($)": round(max_drawdown_dollars, 2),
        "Maximum Drawdown (%)": round(max_drawdown_percent, 2),
        "Average Winner (Points)": round(avg_winner_points, 4),
        "Average Loser (Points)": round(avg_loser_points, 4),
        "Average Winner ($)": round(avg_winner_dollars, 2),
        "Average Loser ($)": round(avg_loser_dollars, 2),
        "Number of Trades Executed": total_trades,
        "Commission per Round-Turn ($)": COMMISSION_ROUND_TURN
    }
    
    if not return_trades:
        return results
    
    # Build equity curve (trade-by-trade, sequential, 1 unit position sizing)
    trades_df = trades_df.sort_values(by=['ExitTime']).copy()
    trades_df['Equity'] = initial_capital + trades_df['NetPNL'].cumsum()
    # Also compute a daily equity series using last known equity of each trade's exit date
    daily_equity = trades_df.groupby('Date')['Equity'].last().to_frame(name='StrategyEquity')
    return results, trades_df, daily_equity

############################################
# Data Loading Utilities for SPY/S&P 500   #
############################################

def _ensure_tz_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        # Assume timestamps are in US/Eastern if naive
        return df.tz_localize('America/New_York')
    # Convert to US/Eastern
    return df.tz_convert('America/New_York')


def load_spy_minute_polygon(start: datetime, end: datetime, api_key: str) -> pd.DataFrame:
    """
    Load SPY 1-minute bars from Polygon.io within [start, end]. Requires paid plan for 1y depth.
    """
    import requests

    url = "https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{start}/{end}"
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }
    resp = requests.get(url.format(start=start_str, end=end_str), params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = data.get('results', []) or []
    if not results:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    records = []
    for r in results:
        # Polygon returns epoch ms in 't'
        ts = pd.to_datetime(r['t'], unit='ms', utc=True).tz_convert('America/New_York')
        records.append({
            'DateTime': ts,
            'Open': r['o'],
            'High': r['h'],
            'Low': r['l'],
            'Close': r['c'],
            'Volume': r.get('v', np.nan)
        })
    df = pd.DataFrame.from_records(records).set_index('DateTime').sort_index()
    return _ensure_tz_ny(df)


def load_spy_intraday_yf(period: str = '60d', interval: str = '5m') -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    ticker = yf.Ticker('SPY')
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return df
    # yfinance index is tz-aware (UTC); convert to NY
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
    df = df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
    })[['Open','High','Low','Close','Volume']]
    return df


def load_spy_daily_yf(start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame(columns=['Close'])
    df = yf.download('SPY', start=start.date(), end=end.date(), auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize('America/New_York')
    return df[['Close']].rename(columns={'Close': 'SPY_Close'})


def fetch_minute_data_one_year(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Try Polygon (if POLYGON_API_KEY available) for 1-minute over ~1y. Fallback to yfinance 5m/60d.
    """
    api_key = os.environ.get('POLYGON_API_KEY', '').strip()
    if api_key:
        try:
            df_min = load_spy_minute_polygon(start, end, api_key)
            if not df_min.empty:
                return df_min
        except Exception:
            pass
    # Fallback: yfinance intraday limitations (up to ~60 days at 5m)
    df_5m = load_spy_intraday_yf(period='60d', interval='5m')
    return df_5m


def build_equity_vs_index_plot(strategy_daily_equity: pd.DataFrame, spy_daily: pd.DataFrame, initial_capital: float = 10000.0):
    if strategy_daily_equity.empty or spy_daily.empty:
        print("Insufficient data to plot equity vs index.")
        return
    # Work on copies
    sdf = strategy_daily_equity.copy()
    idf = spy_daily.copy()
    
    # Strategy: enforce naive midnight DatetimeIndex
    if not isinstance(sdf.index, pd.DatetimeIndex):
        sdf.index = pd.to_datetime(sdf.index, errors='coerce')
    sdf.index = pd.to_datetime(sdf.index)
    if sdf.index.tz is not None:
        sdf.index = sdf.index.tz_convert(None)
    sdf.index = sdf.index.normalize()

    # Index (SPY): compute index equity and normalize index
    if not isinstance(idf.index, pd.DatetimeIndex):
        idf.index = pd.to_datetime(idf.index, errors='coerce')
    if idf.index.tz is not None:
        try:
            idf.index = idf.index.tz_convert('America/New_York')
        except Exception:
            pass
        idf.index = idf.index.tz_convert(None)
    idf = idf.sort_index()
    # Pick a single Close series robustly (handles possible MultiIndex columns and numpy.str_ labels)
    spy_close = None
    # If spy_daily was provided as a Series
    if isinstance(idf, pd.Series):
        spy_close = idf
        idf = spy_daily.to_frame(name='SPY_Close')
    else:
        for cand in ['SPY_Close', 'Close', 'Adj Close']:
            if cand in idf.columns:
                spy_close = idf[cand]
                break
        if spy_close is None and isinstance(idf.columns, pd.MultiIndex):
            for cand in ['SPY_Close', 'Close', 'Adj Close']:
                matches = [col for col in idf.columns if isinstance(col, tuple) and (col[-1] == cand)]
                if matches:
                    spy_close = idf[matches[0]]
                    break
        if spy_close is None:
            raise ValueError("spy_daily does not contain a recognizable close column.")

    # Normalize index to naive midnight dates
    idf.index = pd.to_datetime(idf.index).normalize()
    # Ensure Series
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    else:
        spy_close = pd.Series(spy_close, index=idf.index) if not isinstance(spy_close, pd.Series) else spy_close
    # Buy & Hold: invest initial capital at start, hold constant shares
    first_price = float(spy_close.iloc[0])
    shares = initial_capital / first_price
    index_equity = (shares * spy_close).rename('IndexEquity')

    # Align via concat with outer join (both Series) for strategy plot
    merged = pd.concat([
        index_equity,
        sdf['StrategyEquity']
    ], axis=1, join='outer')
    merged['StrategyEquity'] = merged['StrategyEquity'].ffill()
    strategy_series = merged['StrategyEquity'].dropna()

    # Build a daily buy & hold series (use last value per date)
    ie = index_equity.copy()
    if isinstance(ie.index, pd.DatetimeIndex):
        if ie.index.tz is not None:
            ie.index = ie.index.tz_convert(None)
        ie.index = ie.index.normalize()
    index_equity_daily = ie.groupby(ie.index).last()
    # Plot as separate subplots for clarity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Limit both series to the last 60 days
    latest_date = max(strategy_series.index.max(), index_equity_daily.index.max())
    cutoff_date = latest_date - pd.Timedelta(days=60)
    strategy_series_60 = strategy_series[strategy_series.index >= cutoff_date]
    index_equity_60 = index_equity_daily[index_equity_daily.index >= cutoff_date]

    # Strategy
    ax1.plot(strategy_series_60.index, strategy_series_60, color='tab:blue', linewidth=2)
    ax1.set_title('Strategy Equity (ORB)')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    # SPY Buy & Hold
    ax2.plot(index_equity_60.index, index_equity_60, color='tab:green', linewidth=2)
    ax2.set_title('SPY Buy & Hold (invested at start)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity ($)')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define a 1-year window ending today (NY)
    today_ny = datetime.now(timezone.utc).astimezone()
    end_dt = today_ny
    start_dt = end_dt - timedelta(days=365)

    print("Fetching SPY minute data (tries Polygon 1m, falls back to yfinance 5m/60d)...")
    minute_df = fetch_minute_data_one_year(start_dt, end_dt)
    if minute_df.empty:
        print("Failed to load intraday SPY data. Provide POLYGON_API_KEY for full 1y minute bars.")
    else:
        # Backtest the ORB strategy; return trades and daily equity
        results, trades_df, daily_equity = backtest_orb_strategy(minute_df, initial_capital=10000.0, return_trades=True)
        print("Backtest Results:")
        print(results)

        # Fetch SPY daily closes for benchmark across the same period
        spy_daily = load_spy_daily_yf(start_dt, end_dt)
        if spy_daily.empty:
            print("Failed to load SPY daily data for benchmark plot.")
        else:
            build_equity_vs_index_plot(daily_equity, spy_daily, initial_capital=10000.0)
