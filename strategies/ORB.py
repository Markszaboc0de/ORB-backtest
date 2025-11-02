import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# TODO: incorporate these as parameters of the strategy
COMMISSION_ROUND_TURN = 2.00
RISK_REWARD_RATIO = 3.0
TIME_STOP_HOUR = 10
TIME_STOP_MINUTE = 30
TIME_EOD_EXIT_HOUR = 15
TIME_EOD_EXIT_MINUTE = 50
DAILY_MAX_LOSSES = 2  # Stop trading for the day after this many losses
REQUIRE_PREV_DAY_TREND_ALIGNMENT = True  # Only trade with prior day's trend
LEVERAGE = 10.0  # Apply 10x leverage to each trade's notional
RANGE_FILTER_MIN = 0.0005  # unused in engulfing version
RANGE_FILTER_MAX = 0.0100  # unused in engulfing version

def orb_strategy(df: pd.DataFrame):

    for date, daily_data in df.groupby('Date'):
        # --- 1. Opening Range (first 5 minutes 9:30-9:34) ---
        orb_start = pd.to_datetime(f"{date} 09:30:00").tz_localize(df.index.tz)
        orb_after = pd.to_datetime(f"{date} 09:35:00").tz_localize(df.index.tz)
        orb_data = daily_data.loc[orb_start:orb_after - pd.Timedelta(minutes=1)]
        if orb_data.empty:
            continue
        or_high = orb_data['High'].max()
        or_low = orb_data['Low'].min()

        # --- Prev-day trend filter ---
        prior_trend = None  # 'Up' or 'Down' or None if unavailable/flat
        if REQUIRE_PREV_DAY_TREND_ALIGNMENT:
            try:
                current_idx = unique_dates.index(date)
                if current_idx > 0:
                    prev_date = unique_dates[current_idx - 1]
                    prev_day_data = df[df['Date'] == prev_date]
                    if not prev_day_data.empty:
                        prev_open = float(prev_day_data['Open'].iloc[0])
                        prev_close = float(prev_day_data['Close'].iloc[-1])
                        if prev_close > prev_open:
                            prior_trend = 'Up'
                        elif prev_close < prev_open:
                            prior_trend = 'Down'
                        else:
                            prior_trend = None
            except ValueError:
                prior_trend = None

        # --- 2. Engulfing Breakout Entry (scan post 9:35 close) ---
        scan_start = pd.to_datetime(f"{date} 09:36:00").tz_localize(df.index.tz)
        day_after_orb = daily_data.loc[scan_start:]

        def body_low(open_price, close_price):
            return min(open_price, close_price)

        def body_high(open_price, close_price):
            return max(open_price, close_price)

        executed = False
        losses_today = 0

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

            # Prev-day trend direction alignment (if required)
            if REQUIRE_PREV_DAY_TREND_ALIGNMENT and prior_trend is not None:
                if long_breakout and prior_trend != 'Up':
                    continue
                if short_breakout and prior_trend != 'Down':
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
                # Position sizing with leverage: trade 10% of equity times leverage
                notional = LEVERAGE * 0.10 * current_equity
                position_qty = notional / entry_price if entry_price > 0 else 0.0
                point_pnl = (exit_price - entry_price) if position_type == 'Long' else (entry_price - exit_price)
                dollar_pnl = point_pnl * position_qty
                net_pnl = dollar_pnl - COMMISSION_ROUND_TURN

                trades.append({
                    'Date': date,
                    'EntryTime': entry_time,
                    'ExitTime': exit_time,
                    'Position': position_type,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PositionQty': position_qty,
                    'NotionalAtEntry': notional,
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
                # Track daily loss count for potential daily stop (future multi-trade support)
                if net_pnl < 0:
                    losses_today += 1
                executed = True
                # Update current equity after trade closes
                current_equity += net_pnl
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
    
    # Build equity curve (trade-by-trade, sequential, 1 unit position sizing)
    trades_df = trades_df.sort_values(by=['ExitTime']).copy()
    return results, trades_df
