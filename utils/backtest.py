import numpy as np
import pandas as pd


def backtest_topk(pred_df, initial_cash=1_000_000.0, topk=1,
                  commission=0.0003, stamp=0.001, risk_free=0.03):
    if pred_df.empty:
        return {
            'final_capital': initial_cash,
            'cumulative_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe': 0.0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'trade_days': 0
        }, pd.DataFrame(columns=['date', 'capital'])

    df = pred_df.copy()
    df = df.sort_values('trade_date')
    df = df.dropna(subset=['pred_return', 'true_return'])
    df = df[np.isfinite(df['pred_return']) & np.isfinite(df['true_return'])]
    df = df[df['true_return'] > -0.999]
    df = df[df['suspendFlag'] == 0]

    cash = float(initial_cash)
    equity_curve = []
    trade_returns = []
    trade_profits = []
    trade_days = 0

    grouped = df.groupby('trade_date', sort=True)
    for trade_date, group in grouped:
        if group.empty:
            equity_curve.append((trade_date, cash))
            continue
        picks = group.sort_values('pred_return', ascending=False).head(topk)
        if picks.empty:
            equity_curve.append((trade_date, cash))
            continue
        gross_returns = picks['true_return'].values
        gross_returns = gross_returns[np.isfinite(gross_returns) & (gross_returns > -0.999)]
        if gross_returns.size == 0:
            equity_curve.append((trade_date, cash))
            continue
        net_factors = (1 - commission) * (1 + gross_returns) * (1 - commission - stamp)
        net_factors = net_factors[net_factors > 0]
        if net_factors.size == 0:
            equity_curve.append((trade_date, cash))
            continue
        net_factor = float(np.mean(net_factors))
        net_return = net_factor - 1.0
        profit = cash * net_return
        cash *= net_factor
        trade_returns.append(net_return)
        trade_profits.append(profit)
        trade_days += 1
        equity_curve.append((trade_date, cash))

    curve_df = pd.DataFrame(equity_curve, columns=['date', 'capital'])
    if len(curve_df) < 2:
        total_years = 0.0
    else:
        total_days = (curve_df['date'].iloc[-1] - curve_df['date'].iloc[0]).days
        total_years = total_days / 365.25 if total_days > 0 else 0.0

    final_capital = cash
    cumulative_return = (final_capital - initial_cash) / initial_cash if initial_cash > 0 else 0.0
    if total_years > 0 and initial_cash > 0 and final_capital > 0:
        annualized_return = (final_capital / initial_cash) ** (1 / total_years) - 1
    elif final_capital <= 0:
        annualized_return = -1.0
    else:
        annualized_return = 0.0

    capital_values = curve_df['capital'].values
    peak = np.maximum.accumulate(capital_values)
    peak = np.where(peak > 0, peak, np.nan)
    drawdowns = (peak - capital_values) / peak
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    if len(capital_values) > 1:
        denom = capital_values[:-1]
        num = np.diff(capital_values)
        daily_returns = np.zeros_like(num, dtype=float)
        valid = denom > 0
        daily_returns[valid] = num[valid] / denom[valid]
    else:
        daily_returns = np.array([0.0])
    annual_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) else 0.0
    sharpe = (annualized_return - risk_free) / annual_vol if annual_vol > 0 else 0.0

    wins = [r for r in trade_returns if r > 0]
    win_rate = (len(wins) / len(trade_returns)) if trade_returns else 0.0
    total_profit = sum(p for p in trade_profits if p > 0)
    total_loss = sum(-p for p in trade_profits if p < 0)
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf') if total_profit > 0 else 0.0

    total_trades = trade_days * topk * 2

    metrics = {
        'final_capital': float(final_capital),
        'cumulative_return_pct': float(cumulative_return * 100),
        'annualized_return_pct': float(annualized_return * 100),
        'max_drawdown_pct': float(max_drawdown * 100),
        'sharpe': float(sharpe),
        'win_rate_pct': float(win_rate * 100),
        'profit_factor': float(profit_factor),
        'total_trades': int(total_trades),
        'trade_days': int(trade_days)
    }
    return metrics, curve_df
