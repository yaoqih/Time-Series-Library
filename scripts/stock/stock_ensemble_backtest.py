import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.wandb_utils import init_wandb, log_wandb, finish_wandb


REQUIRED_COLUMNS = {
    'code', 'trade_date', 'pred_return', 'true_return', 'suspendFlag'
}

SEARCH_METRIC_KEYS = [
    'cumulative_return_pct',
    'annualized_return_pct',
    'max_drawdown_pct',
    'sharpe',
    'win_rate_pct',
    'profit_factor'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble rolling backtest by rank-sum of predictions')
    parser.add_argument('--results_root', type=str, default='stock_results_rolling',
                        help='root directory for individual model results')
    parser.add_argument('--output_root', type=str, default='stock_results_rolling',
                        help='root directory for ensemble outputs')
    parser.add_argument('--models', type=str, default='',
                        help='comma-separated model list; empty = all in results_root')
    parser.add_argument('--ensemble_name', type=str, default='EnsembleRankSum',
                        help='output folder name for the ensemble')
    parser.add_argument('--splits', type=str, default='test,val', help='comma-separated splits')

    parser.add_argument('--initial_cash', type=float, default=1_000_000.0, help='initial capital')
    parser.add_argument('--commission', type=float, default=0.0003, help='commission rate')
    parser.add_argument('--stamp', type=float, default=0.001, help='stamp tax rate')
    parser.add_argument('--risk_free', type=float, default=0.03, help='risk free rate')

    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='tslib-roll', help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default='stock_rolling_ensemble', help='wandb group')
    parser.add_argument('--wandb_mode', type=str, default=None, help='wandb mode')

    parser.add_argument('--search', action='store_true', default=False,
                        help='search best ensemble combinations without writing per-window outputs')
    parser.add_argument('--combo_sizes', type=str, default='2,3,4,5',
                        help='comma-separated combination sizes for search')
    parser.add_argument('--search_split', type=str, default='test',
                        help='split used for search metrics')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of worker processes for search')

    args = parser.parse_args()
    if args.no_wandb:
        args.use_wandb = False
    return args


def _iter_models(results_root: str, models_arg: str) -> List[str]:
    if models_arg:
        return [m.strip() for m in models_arg.split(',') if m.strip()]
    if not os.path.isdir(results_root):
        return []
    return sorted([m for m in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, m))])


def _window_sort_key(name: str):
    match = re.search(r"win(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def _list_windows(results_root: str, models: List[str]) -> List[str]:
    if not models:
        return []
    windows = None
    for model in models:
        model_dir = os.path.join(results_root, model)
        if not os.path.isdir(model_dir):
            continue
        items = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if windows is None:
            windows = set(items)
        else:
            windows &= set(items)
    if not windows:
        return []
    return sorted(windows, key=_window_sort_key)


def _parse_combo_sizes(sizes_arg: str) -> List[int]:
    sizes = []
    for raw in sizes_arg.split(','):
        raw = raw.strip()
        if not raw:
            continue
        sizes.append(int(raw))
    return sorted({s for s in sizes if s > 0})


def _collect_windows_by_model(results_root: str, models: List[str]) -> Dict[str, set]:
    windows_by_model = {}
    for model in models:
        model_dir = os.path.join(results_root, model)
        if not os.path.isdir(model_dir):
            windows_by_model[model] = set()
            continue
        items = [
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
        ]
        windows_by_model[model] = set(items)
    return windows_by_model


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


def _merge_predictions(pred_map: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    models = list(pred_map.keys())
    base_model = models[0]
    base_df = pred_map[base_model]
    join_keys = ['code', 'trade_date']
    if all('pred_date' in df.columns for df in pred_map.values()):
        join_keys.append('pred_date')

    merged = base_df[join_keys + ['pred_return', 'true_return']].copy()
    merged = merged.rename(columns={'pred_return': f'pred_return_{base_model}',
                                    'true_return': f'true_return_{base_model}'})

    for model in models[1:]:
        df = pred_map[model]
        block = df[join_keys + ['pred_return', 'true_return']].copy()
        block = block.rename(columns={'pred_return': f'pred_return_{model}',
                                      'true_return': f'true_return_{model}'})
        merged = merged.merge(block, on=join_keys, how='inner')

    true_cols = [c for c in merged.columns if c.startswith('true_return_')]
    if not true_cols:
        raise ValueError("no true_return columns found after merge")
    base_true = merged[true_cols[0]].values
    for col in true_cols[1:]:
        diff = np.nanmax(np.abs(merged[col].values - base_true))
        if diff > 1e-6:
            print(f"[warn] true_return mismatch across models; max diff {diff:.6g}")
            break
    merged = merged.rename(columns={true_cols[0]: 'true_return'})
    drop_cols = [c for c in true_cols if c != true_cols[0]]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    pred_cols = [f'pred_return_{m}' for m in models]
    return merged, pred_cols


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


def backtest_rank_sum(
    merged: pd.DataFrame,
    pred_cols: List[str],
    initial_cash: float,
    commission: float,
    stamp: float,
    risk_free: float
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if merged.empty:
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
            'trade_date', 'net_return', 'net_factor', 'capital',
            'return_pct', 'drawdown_pct', 'max_drawdown_pct',
            'profit_factor', 'sharpe'
        ])
        return metrics, empty_curve, empty_picks, empty_daily

    df = merged.copy()
    df['mean_pred_return'] = df[pred_cols].mean(axis=1)
    for col in pred_cols:
        rank_col = f'rank_{col}'
        df[rank_col] = df.groupby('trade_date')[col].rank(ascending=False, method='min')
    rank_cols = [c for c in df.columns if c.startswith('rank_pred_return_')]
    df['rank_sum'] = df[rank_cols].sum(axis=1)

    df = df.sort_values(
        ['trade_date', 'rank_sum', 'mean_pred_return', 'code'],
        ascending=[True, True, False, True]
    )
    picks = df.groupby('trade_date', sort=False).head(1).copy()
    picks['pred_return'] = picks['mean_pred_return']

    all_trade_dates = df['trade_date'].drop_duplicates().sort_values()
    if all_trade_dates.empty:
        return backtest_rank_sum(pd.DataFrame(), pred_cols, initial_cash, commission, stamp, risk_free)

    net_factor = (1 - commission) * (1 + picks['true_return'].values) * (1 - commission - stamp)
    net_factor = np.where(np.isfinite(net_factor) & (net_factor > 0), net_factor, np.nan)
    picks['net_factor'] = net_factor
    picks['net_return'] = picks['net_factor'] - 1.0

    net_factor_by_date = (
        picks.groupby('trade_date', sort=True)['net_factor']
             .mean()
             .dropna()
    )

    net_factor_all = pd.Series(1.0, index=all_trade_dates)
    net_factor_all.loc[net_factor_by_date.index] = net_factor_by_date.values
    capital_curve = float(initial_cash) * net_factor_all.cumprod()
    curve_df = pd.DataFrame({'date': net_factor_all.index, 'capital': capital_curve.values})

    trade_dates = net_factor_by_date.index
    trade_net_factor = net_factor_by_date.values
    trade_days = int(len(trade_net_factor))
    if trade_days == 0:
        picks_df = pd.DataFrame(columns=[
            'trade_date', 'code', 'pred_return', 'true_return',
            'net_return', 'net_factor', 'capital'
        ])
        daily_df = pd.DataFrame(columns=[
            'trade_date', 'net_return', 'net_factor', 'capital',
            'return_pct', 'drawdown_pct', 'max_drawdown_pct',
            'profit_factor', 'sharpe'
        ])
        cash = float(initial_cash)
    else:
        trade_capital = float(initial_cash) * np.cumprod(trade_net_factor)
        prev_cash = np.concatenate([[float(initial_cash)], trade_capital[:-1]])
        trade_net_return = trade_net_factor - 1.0
        trade_profit = prev_cash * trade_net_return

        total_profit_cum = np.cumsum(np.maximum(trade_profit, 0.0))
        total_loss_cum = np.cumsum(np.maximum(-trade_profit, 0.0))
        profit_factor_series = np.zeros_like(total_profit_cum, dtype=float)
        np.divide(
            total_profit_cum,
            total_loss_cum,
            out=profit_factor_series,
            where=total_loss_cum > 0
        )
        profit_factor_series = np.where(
            total_loss_cum > 0,
            profit_factor_series,
            np.where(total_profit_cum > 0, np.inf, 0.0)
        )

        picks = picks[picks['trade_date'].isin(trade_dates)].copy()
        picks['capital'] = picks['trade_date'].map(
            pd.Series(trade_capital, index=trade_dates)
        ).values
        picks_df = picks[[
            'trade_date', 'code', 'pred_return', 'true_return',
            'net_return', 'net_factor', 'capital'
        ]].sort_values(['trade_date', 'pred_return'], ascending=[True, False])

        daily_df = pd.DataFrame({
            'trade_date': trade_dates,
            'net_return': trade_net_return,
            'net_factor': trade_net_factor,
            'capital': trade_capital,
            'profit_factor': profit_factor_series
        })
        cash = float(trade_capital[-1])

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

    if trade_days > 0:
        trade_net_return = daily_df['net_return'].values
        wins = trade_net_return[trade_net_return > 0]
        win_rate = (len(wins) / len(trade_net_return)) if trade_net_return.size else 0.0
        trade_profit = (daily_df['capital'].values / daily_df['net_factor'].values) * trade_net_return
        total_profit_sum = float(np.sum(trade_profit[trade_profit > 0]))
        total_loss_sum = float(np.sum(-trade_profit[trade_profit < 0]))
        profit_factor_final = (total_profit_sum / total_loss_sum) if total_loss_sum > 0 else float('inf') if total_profit_sum > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor_final = 0.0

    total_trades = trade_days * 2
    metrics = {
        'final_capital': float(final_capital),
        'cumulative_return_pct': float(cumulative_return * 100),
        'annualized_return_pct': float(annualized_return * 100),
        'max_drawdown_pct': float(max_drawdown * 100),
        'sharpe': float(sharpe),
        'win_rate_pct': float(win_rate * 100),
        'profit_factor': float(profit_factor_final),
        'total_trades': int(total_trades),
        'trade_days': int(trade_days)
    }

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
    return metrics, curve_df, picks_df, daily_df


def _load_predictions(results_root: str, model: str, window_tag: str, split: str) -> pd.DataFrame:
    pred_fp = os.path.join(results_root, model, window_tag, split, 'predictions.csv')
    if not os.path.exists(pred_fp):
        return pd.DataFrame()
    return pd.read_csv(pred_fp)


def _evaluate_combo_search_task(task):
    (
        combo,
        size,
        windows_by_model,
        results_root,
        search_split,
        initial_cash,
        commission,
        stamp,
        risk_free
    ) = task
    try:
        combo_windows = None
        for model in combo:
            model_windows = windows_by_model.get(model, set())
            combo_windows = model_windows if combo_windows is None else combo_windows & model_windows
        if not combo_windows:
            return None

        window_tags = sorted(combo_windows, key=_window_sort_key)
        window_metrics = {key: [] for key in SEARCH_METRIC_KEYS}
        window_metric_map = {}
        for window_tag in window_tags:
            pred_map = {}
            for model in combo:
                df_raw = _load_predictions(results_root, model, window_tag, search_split)
                if df_raw.empty:
                    return None
                pred_map[model] = _prepare_predictions(df_raw)

            merged, pred_cols = _merge_predictions(pred_map)
            metrics, _, _, _ = backtest_rank_sum(
                merged,
                pred_cols,
                initial_cash=initial_cash,
                commission=commission,
                stamp=stamp,
                risk_free=risk_free
            )
            per_window = {}
            for key in SEARCH_METRIC_KEYS:
                value = float(metrics.get(key, 0.0))
                window_metrics[key].append(value)
                per_window[key] = value
            window_metric_map[window_tag] = per_window

        if not window_metrics['cumulative_return_pct']:
            return None

        mean_value = float(np.mean(window_metrics['cumulative_return_pct']))
        row = {
            'size': size,
            'models': ','.join(combo),
            'num_windows': int(len(window_metrics['cumulative_return_pct'])),
            'mean_test_cumulative_return_pct': mean_value,
            'std_test_cumulative_return_pct': float(np.std(window_metrics['cumulative_return_pct'])),
            'min_test_cumulative_return_pct': float(np.min(window_metrics['cumulative_return_pct'])),
            'max_test_cumulative_return_pct': float(np.max(window_metrics['cumulative_return_pct']))
        }
        for key in SEARCH_METRIC_KEYS:
            row[f'mean_test_{key}'] = float(np.mean(window_metrics[key]))
        detail = {
            'size': size,
            'models': list(combo),
            'window_metrics': window_metric_map
        }
        return row, detail
    except Exception as exc:
        return ("__error__", combo, str(exc))


def _search_best_combinations(args, models: List[str]) -> None:
    args.use_wandb = False
    sizes = _parse_combo_sizes(args.combo_sizes)
    sizes = [s for s in sizes if s >= 1 and s <= len(models)]
    if not sizes:
        raise ValueError("no valid combination sizes for search")

    windows_by_model = _collect_windows_by_model(args.results_root, models)
    search_root = os.path.join(args.output_root, args.ensemble_name, 'search')
    os.makedirs(search_root, exist_ok=True)

    combos = [(combo, size) for size in sizes for combo in itertools.combinations(models, size)]
    total_combos = len(combos)
    print(f"[search] evaluating {total_combos} combinations across sizes {sizes}")

    summary_rows = []
    detail_rows = []
    best_by_size: Dict[int, Dict] = {}
    combo_index = 0

    if args.num_workers <= 1:
        pred_cache: Dict[Tuple[str, str, str], Optional[pd.DataFrame]] = {}
        for size in sizes:
            for combo in itertools.combinations(models, size):
                combo_index += 1
                if combo_index % 50 == 0:
                    print(f"[search] progress {combo_index}/{total_combos}")

                combo_windows = None
                for model in combo:
                    model_windows = windows_by_model.get(model, set())
                    combo_windows = model_windows if combo_windows is None else combo_windows & model_windows
                if not combo_windows:
                    continue

                window_tags = sorted(combo_windows, key=_window_sort_key)
                window_metrics = {key: [] for key in SEARCH_METRIC_KEYS}
                window_metric_map = {}
                missing_window = False
                for window_tag in window_tags:
                    pred_map = {}
                    for model in combo:
                        cache_key = (model, window_tag, args.search_split)
                        if cache_key not in pred_cache:
                            df_raw = _load_predictions(args.results_root, model, window_tag, args.search_split)
                            if df_raw.empty:
                                pred_cache[cache_key] = None
                            else:
                                pred_cache[cache_key] = _prepare_predictions(df_raw)
                        df = pred_cache[cache_key]
                        if df is None or df.empty:
                            missing_window = True
                            break
                        pred_map[model] = df
                    if missing_window:
                        break

                    merged, pred_cols = _merge_predictions(pred_map)
                    metrics, _, _, _ = backtest_rank_sum(
                        merged,
                        pred_cols,
                        initial_cash=args.initial_cash,
                        commission=args.commission,
                        stamp=args.stamp,
                        risk_free=args.risk_free
                    )
                    per_window = {}
                    for key in SEARCH_METRIC_KEYS:
                        value = float(metrics.get(key, 0.0))
                        window_metrics[key].append(value)
                        per_window[key] = value
                    window_metric_map[window_tag] = per_window

                if missing_window or not window_metrics['cumulative_return_pct']:
                    continue

                mean_value = float(np.mean(window_metrics['cumulative_return_pct']))
                row = {
                    'size': size,
                    'models': ','.join(combo),
                    'num_windows': int(len(window_metrics['cumulative_return_pct'])),
                    'mean_test_cumulative_return_pct': mean_value,
                    'std_test_cumulative_return_pct': float(np.std(window_metrics['cumulative_return_pct'])),
                    'min_test_cumulative_return_pct': float(np.min(window_metrics['cumulative_return_pct'])),
                    'max_test_cumulative_return_pct': float(np.max(window_metrics['cumulative_return_pct']))
                }
                for key in SEARCH_METRIC_KEYS:
                    row[f'mean_test_{key}'] = float(np.mean(window_metrics[key]))
                summary_rows.append(row)
                detail_rows.append({
                    'size': size,
                    'models': list(combo),
                    'window_metrics': window_metric_map
                })
                _write_search_result(search_root, row, detail_rows[-1], combo_index)
                if size not in best_by_size or mean_value > best_by_size[size]['mean_test_cumulative_return_pct']:
                    best_by_size[size] = row.copy()
    else:
        tasks = [
            (
                combo,
                size,
                windows_by_model,
                args.results_root,
                args.search_split,
                args.initial_cash,
                args.commission,
                args.stamp,
                args.risk_free
            )
            for combo, size in combos
        ]
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_evaluate_combo_search_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 50 == 0:
                    print(f"[search] progress {completed}/{total_combos}")
                result = future.result()
                if result is None:
                    continue
                if isinstance(result, tuple) and len(result) == 3 and result[0] == "__error__":
                    print(f"[search] error combo={result[1]}: {result[2]}")
                    continue
                row, detail = result
                combo_index += 1
                summary_rows.append(row)
                detail_rows.append(detail)
                _write_search_result(search_root, row, detail, combo_index)
                size = int(row['size'])
                mean_value = float(row['mean_test_cumulative_return_pct'])
                if size not in best_by_size or mean_value > best_by_size[size]['mean_test_cumulative_return_pct']:
                    best_by_size[size] = row.copy()

    if not summary_rows:
        raise ValueError("no valid combinations produced search results")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ['mean_test_cumulative_return_pct', 'num_windows'],
        ascending=[False, False]
    )
    summary_df.to_csv(os.path.join(search_root, 'combo_summary.csv'), index=False)
    summary_df.to_json(os.path.join(search_root, 'combo_summary.json'), orient='records', indent=2)

    best_df = pd.DataFrame(best_by_size.values()).sort_values('size')
    best_df.to_csv(os.path.join(search_root, 'best_by_size.csv'), index=False)
    best_df.to_json(os.path.join(search_root, 'best_by_size.json'), orient='records', indent=2)

    with open(os.path.join(search_root, 'combo_details.json'), 'w') as f:
        json.dump(detail_rows, f, indent=2)

    overall_best = summary_df.iloc[0].to_dict()
    print("[search] best overall:", overall_best)
    print("[search] best by size:")
    for _, row in best_df.iterrows():
        print(f"  size={int(row['size'])} mean={row['mean_test_cumulative_return_pct']:.4f} models={row['models']}")


def _combo_folder_name(models: List[str], size: int) -> str:
    joined = "_".join(models)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", joined).strip("_")
    if not slug:
        slug = "combo"
    slug = slug[:80]
    digest = hashlib.md5(",".join(models).encode("utf-8")).hexdigest()[:8]
    return f"size{size}_{slug}_{digest}"


def _write_search_result(search_root: str, row: Dict, detail: Dict, index: int) -> None:
    size = int(row.get('size', 0))
    models = detail.get('models', [])
    if not models:
        return
    folder_name = _combo_folder_name(models, size)
    combo_root = os.path.join(search_root, "combos", f"{index:05d}_{folder_name}")
    os.makedirs(combo_root, exist_ok=True)

    summary_df = pd.DataFrame([row])
    summary_df.to_csv(os.path.join(combo_root, "summary.csv"), index=False)
    with open(os.path.join(combo_root, "summary.json"), "w") as f:
        json.dump(row, f, indent=2)

    window_metrics = detail.get("window_metrics") or {}
    if window_metrics:
        window_rows = []
        for window_tag, metrics in window_metrics.items():
            row_out = {"window_tag": window_tag}
            row_out.update(metrics)
            window_rows.append(row_out)
        pd.DataFrame(window_rows).sort_values("window_tag").to_csv(
            os.path.join(combo_root, "window_metrics.csv"), index=False
        )
        with open(os.path.join(combo_root, "window_metrics.json"), "w") as f:
            json.dump(window_metrics, f, indent=2)


def _write_outputs(split_dir: str, metrics: Dict, curve_df: pd.DataFrame, picks_df: pd.DataFrame, daily_df: pd.DataFrame):
    os.makedirs(split_dir, exist_ok=True)
    curve_df.to_csv(os.path.join(split_dir, 'equity_curve.csv'), index=False)
    picks_df.to_csv(os.path.join(split_dir, 'daily_picks.csv'), index=False)
    daily_df.to_csv(os.path.join(split_dir, 'daily_metrics.csv'), index=False)
    with open(os.path.join(split_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    models = _iter_models(args.results_root, args.models)
    if not models:
        raise ValueError("no models found for ensemble")

    if args.search:
        _search_best_combinations(args, models)
        return

    windows = _list_windows(args.results_root, models)
    if not windows:
        raise ValueError("no common windows across models")

    summary_rows = []
    ensemble_root = os.path.join(args.output_root, args.ensemble_name)

    for window_tag in windows:
        run_name = f"ensemble_{args.ensemble_name}_{window_tag}"
        if args.use_wandb:
            init_wandb(args, name=run_name, group=args.wandb_group, tags=models + ['ensemble'])
        try:
            row_summary = {
                'window_tag': window_tag
            }
            for split in splits:
                pred_map = {}
                missing = []
                for model in models:
                    df_raw = _load_predictions(args.results_root, model, window_tag, split)
                    if df_raw.empty:
                        missing.append(model)
                        continue
                    pred_map[model] = _prepare_predictions(df_raw)
                if missing:
                    print(f"[skip] {window_tag} {split}: missing predictions for {missing}")
                    continue
                merged, pred_cols = _merge_predictions(pred_map)
                metrics, curve_df, picks_df, daily_df = backtest_rank_sum(
                    merged,
                    pred_cols,
                    initial_cash=args.initial_cash,
                    commission=args.commission,
                    stamp=args.stamp,
                    risk_free=args.risk_free
                )

                split_dir = os.path.join(ensemble_root, window_tag, split)
                _write_outputs(split_dir, metrics, curve_df, picks_df, daily_df)
                for key, value in metrics.items():
                    row_summary[f'{split}_{key}'] = value
                if args.use_wandb:
                    log_wandb({f'{split}/{k}': v for k, v in metrics.items()})
            summary_rows.append(row_summary)
        finally:
            if args.use_wandb:
                finish_wandb()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        os.makedirs(ensemble_root, exist_ok=True)
        summary_csv = os.path.join(ensemble_root, 'window_summary.csv')
        summary_json = os.path.join(ensemble_root, 'window_summary.json')
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_json(summary_json, orient='records', indent=2)

        if args.use_wandb:
            try:
                init_wandb(
                    args,
                    name=f"ensemble_summary_{args.ensemble_name}",
                    group=f"{args.wandb_group}_summary",
                    tags=models + ['ensemble', 'summary']
                )
                import wandb
                wandb.log({'window_summary': wandb.Table(dataframe=summary_df)})
            finally:
                finish_wandb()


if __name__ == '__main__':
    main()
