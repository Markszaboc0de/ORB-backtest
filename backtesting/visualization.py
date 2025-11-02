import pandas as pd
import matplotlib.pyplot as plt

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
