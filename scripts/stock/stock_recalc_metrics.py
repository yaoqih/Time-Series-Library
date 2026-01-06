import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.wandb_utils import init_wandb, log_wandb, finish_wandb


REQUIRED_COLUMNS = {
    'code', 'trade_date', 'pred_return', 'true_return', 'suspendFlag'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Recalculate stock backtest metrics from predictions.csv')
    parser.add_argument('--results_root', type=str, default='stock_results', help='root directory for results')
    parser.add_argument('--models', type=str, default='', help='comma-separated model list; empty = all')
    parser.add_argument('--splits', type=str, default='test,val', help='comma-separated splits')

    parser.add_argument('--initial_cash', type=float, default=1_000_000.0, help='initial capital')
    parser.add_argument('--commission', type=float, default=0.0003, help='commission rate')
    parser.add_argument('--stamp', type=float, default=0.001, help='stamp tax rate')
    parser.add_argument('--risk_free', type=float, default=0.03, help='risk free rate')

    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default='stock_recalc', help='wandb group')
    parser.add_argument('--wandb_mode', type=str, default=None, help='wandb mode')

    return parser.parse_args()


def _prepare_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(pred_df.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    df = pred_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    if 'pred_date' in df.columns:
        df['pred_date'] = pd.to_datetime(df['pred_date'])
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'])

    df = df.dropna(subset=['pred_return', 'true_return'])
    df = df[np.isfinite(df['pred_return']) & np.isfinite(df['true_return'])]
    df = df[df['true_return'] > -0.999]
    df = df[df['suspendFlag'] == 0]
    return df.sort_values('trade_date')


def _compute_sharpe_series(dates: pd.Series, capital: np.ndarray, initial_cash: float, risk_free: float) -> np.ndarray:
    sharpe_values = np.zeros(len(capital), dtype=float)
    if len(capital) <= 1:
        return sharpe_values

    daily_returns = np.zeros(len(capital) - 1, dtype=float)
    for i in range(1, len(capital)):
        denom = capital[i - 1]
        if denom > 0:
            daily_returns[i - 1] = (capital[i] - denom) / denom
        else:
            daily_returns[i - 1] = 0.0

        elapsed_days = (dates.iloc[i] - dates.iloc[0]).days
        total_years = elapsed_days / 365.25 if elapsed_days > 0 else 0.0
        if total_years > 0 and capital[i] > 0 and initial_cash > 0:
            annualized_return = (capital[i] / initial_cash) ** (1 / total_years) - 1
        elif capital[i] <= 0:
            annualized_return = -1.0
        else:
            annualized_return = 0.0

        vol = np.std(daily_returns[:i]) * np.sqrt(252) if i > 0 else 0.0
        sharpe_values[i] = (annualized_return - risk_free) / vol if vol > 0 else 0.0

    return sharpe_values


def backtest_top1(pred_df: pd.DataFrame,
                  initial_cash: float,
                  commission: float,
                  stamp: float,
                  risk_free: float) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pred_df.empty:
        metrics = {
            'final_capital': float(initial_cash),
            'cumulative_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe': 0.0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'trade_days': 0
        }
        empty_curve = pd.DataFrame(columns=['date', 'capital'])
        empty_picks = pd.DataFrame(columns=[
            'trade_date', 'code', 'pred_return', 'true_return',
            'net_return', 'net_factor', 'capital'
        ])
        empty_daily = pd.DataFrame(columns=[
            'trade_date', 'code', 'pred_return', 'true_return',
            'net_return', 'net_factor', 'capital', 'return_pct',
            'drawdown_pct', 'max_drawdown_pct', 'profit_factor', 'sharpe'
        ])
        return metrics, empty_curve, empty_picks, empty_daily

    df = _prepare_predictions(pred_df)
    if df.empty:
        return backtest_top1(pd.DataFrame(), initial_cash, commission, stamp, risk_free)

    cash = float(initial_cash)
    equity_curve = []
    trade_returns = []
    trade_profits = []
    trade_days = 0
    picks_records = []
    daily_records = []
    total_profit = 0.0
    total_loss = 0.0

    grouped = df.groupby('trade_date', sort=True)
    for trade_date, group in grouped:
        if group.empty:
            equity_curve.append((trade_date, cash))
            continue
        pick = group.sort_values('pred_return', ascending=False).iloc[0]
        gross_return = float(pick['true_return'])
        net_factor = (1 - commission) * (1 + gross_return) * (1 - commission - stamp)
        if not np.isfinite(net_factor) or net_factor <= 0:
            equity_curve.append((trade_date, cash))
            continue
        net_return = net_factor - 1.0
        profit = cash * net_return
        cash *= net_factor
        trade_returns.append(net_return)
        trade_profits.append(profit)
        trade_days += 1
        equity_curve.append((trade_date, cash))
        if profit > 0:
            total_profit += profit
        elif profit < 0:
            total_loss += -profit
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0
        picks_records.append({
            'trade_date': trade_date,
            'code': pick['code'],
            'pred_return': float(pick['pred_return']),
            'true_return': float(pick['true_return']),
            'net_return': float(net_return),
            'net_factor': float(net_factor),
            'capital': float(cash)
        })
        daily_records.append({
            'trade_date': trade_date,
            'code': pick['code'],
            'pred_return': float(pick['pred_return']),
            'true_return': float(pick['true_return']),
            'net_return': float(net_return),
            'net_factor': float(net_factor),
            'capital': float(cash),
            'profit_factor': float(profit_factor)
        })

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

    metrics = {
        'final_capital': float(final_capital),
        'cumulative_return_pct': float(cumulative_return * 100),
        'annualized_return_pct': float(annualized_return * 100),
        'max_drawdown_pct': float(max_drawdown * 100),
        'sharpe': float(sharpe),
        'win_rate_pct': float(win_rate * 100),
        'profit_factor': float(profit_factor),
        'total_trades': int(trade_days * 2),
        'trade_days': int(trade_days)
    }

    picks_df = pd.DataFrame(picks_records)
    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty:
        capital = daily_df['capital'].values
        daily_df['return_pct'] = (capital / initial_cash - 1.0) * 100.0
        peak = np.maximum.accumulate(capital)
        peak = np.where(peak > 0, peak, np.nan)
        drawdown = (peak - capital) / peak
        daily_df['drawdown_pct'] = drawdown * 100.0
        daily_df['max_drawdown_pct'] = np.maximum.accumulate(drawdown) * 100.0
        sharpe_series = _compute_sharpe_series(daily_df['trade_date'], capital, initial_cash, risk_free)
        daily_df['sharpe'] = sharpe_series
        daily_df['final_capital'] = daily_df['capital']
    return metrics, curve_df, picks_df, daily_df


def _write_outputs(split_dir: str,
                   metrics: Dict,
                   curve_df: pd.DataFrame,
                   picks_df: pd.DataFrame,
                   daily_df: pd.DataFrame,
                   risk_free: float,
                   initial_cash: float) -> pd.DataFrame:
    os.makedirs(split_dir, exist_ok=True)
    metrics_fp = os.path.join(split_dir, 'metrics_recalc.json')
    curve_fp = os.path.join(split_dir, 'equity_curve_recalc.csv')
    picks_fp = os.path.join(split_dir, 'daily_picks.csv')
    sharpe_fp = os.path.join(split_dir, 'sharpe_curve.csv')
    daily_fp = os.path.join(split_dir, 'daily_metrics.csv')

    with open(metrics_fp, 'w') as f:
        json.dump(metrics, f, indent=2)
    curve_df.to_csv(curve_fp, index=False)
    picks_df.to_csv(picks_fp, index=False)
    daily_df.to_csv(daily_fp, index=False)

    if curve_df.empty:
        sharpe_df = pd.DataFrame(columns=['date', 'capital', 'drawdown', 'sharpe'])
        sharpe_df.to_csv(sharpe_fp, index=False)
        return sharpe_df

    capital = curve_df['capital'].values
    peak = np.maximum.accumulate(capital)
    peak = np.where(peak > 0, peak, np.nan)
    drawdown = (peak - capital) / peak
    sharpe_series = _compute_sharpe_series(curve_df['date'], capital, initial_cash, risk_free)

    sharpe_df = pd.DataFrame({
        'date': curve_df['date'],
        'capital': capital,
        'drawdown': drawdown,
        'sharpe': sharpe_series
    })
    sharpe_df.to_csv(sharpe_fp, index=False)
    return sharpe_df


def _iter_models(results_root: str, models_arg: str):
    if models_arg:
        return [m.strip() for m in models_arg.split(',') if m.strip()]
    if not os.path.isdir(results_root):
        return []
    return [m for m in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, m))]


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    models = _iter_models(args.results_root, args.models)

    for model_name in models:
        run_name = f"stock_recalc_{model_name}"
        init_wandb(args, name=run_name, group=args.wandb_group, tags=[model_name, 'recalc'])
        try:
            wandb_mod = None
            if args.use_wandb:
                try:
                    import wandb as wandb_mod
                    for split in splits:
                        wandb_mod.define_metric(f'{split}/step')
                        wandb_mod.define_metric(f'{split}/*', step_metric=f'{split}/step')
                except ImportError:
                    wandb_mod = None
            for split in splits:
                pred_fp = os.path.join(args.results_root, model_name, split, 'predictions.csv')
                if not os.path.exists(pred_fp):
                    continue
                pred_df = pd.read_csv(pred_fp)
                metrics, curve_df, picks_df, daily_df = backtest_top1(
                    pred_df,
                    initial_cash=args.initial_cash,
                    commission=args.commission,
                    stamp=args.stamp,
                    risk_free=args.risk_free
                )

                split_dir = os.path.join(args.results_root, model_name, split)
                _write_outputs(
                    split_dir,
                    metrics,
                    curve_df,
                    picks_df,
                    daily_df,
                    risk_free=args.risk_free,
                    initial_cash=args.initial_cash
                )

                if wandb_mod is not None and not daily_df.empty:
                    daily_df = daily_df.replace([np.inf, -np.inf], np.nan)
                    for step_idx, row in daily_df.iterrows():
                        log_wandb({
                            f'{split}/step': int(step_idx),
                            f'{split}/trade_date': str(row['trade_date']),
                            f'{split}/code': str(row['code']),
                            f'{split}/final_capital': float(row['final_capital']),
                            f'{split}/return_pct': float(row['return_pct']),
                            f'{split}/profit_factor': float(row['profit_factor']),
                            f'{split}/drawdown_pct': float(row['drawdown_pct']),
                            f'{split}/max_drawdown_pct': float(row['max_drawdown_pct']),
                            f'{split}/sharpe': float(row['sharpe'])
                        })
        finally:
            finish_wandb()


if __name__ == '__main__':
    main()
