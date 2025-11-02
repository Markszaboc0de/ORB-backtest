import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

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
