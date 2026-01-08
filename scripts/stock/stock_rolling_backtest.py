import argparse
import copy
import json
import os
import random
import sys
import math

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import models as models_pkg

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Stock, Dataset_StockPacked, STOCK_RETURN_LIMIT
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.wandb_utils import init_wandb, log_wandb, finish_wandb

ZERO_SHOT_MODELS = {
    'Chronos', 'Chronos2', 'Moirai', 'Sundial', 'TiRex', 'TimeMoE', 'TimesFM'
}
RANK_TARGETS = {'lag_return_rank', 'lag_return_cs_rank'}

_STOCK_TRADE_CAL_CACHE = None


def _load_stock_trade_calendar(args) -> pd.DatetimeIndex:
    global _STOCK_TRADE_CAL_CACHE
    if _STOCK_TRADE_CAL_CACHE is not None:
        return _STOCK_TRADE_CAL_CACHE

    data_fp = os.path.abspath(os.path.join(args.root_path, args.data_path))
    if not os.path.exists(data_fp):
        raise FileNotFoundError(f"stock data not found: {data_fp}")
    cache_dir = getattr(args, 'stock_cache_dir', './cache')
    os.makedirs(cache_dir, exist_ok=True)
    mtime = os.path.getmtime(data_fp)
    cache_key = hashlib.md5(f"{data_fp}|{mtime}".encode("utf-8")).hexdigest()
    cache_fp = os.path.join(cache_dir, f"stock_trade_calendar_{cache_key}.npy")

    if os.path.exists(cache_fp):
        try:
            dates = np.load(cache_fp, allow_pickle=False)
            if dates is not None and len(dates) > 0:
                cal = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
                _STOCK_TRADE_CAL_CACHE = cal
                return cal
        except Exception:
            pass

    # Build from parquet time column (requires pyarrow/fastparquet).
    df = pd.read_parquet(data_fp, columns=["time"])
    ts = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        raise ValueError("cannot build trade calendar from stock_data.parquet: no valid 'time'")
    cal = pd.DatetimeIndex(ts.unique()).sort_values().unique()

    try:
        np.save(cache_fp, cal.values.astype('datetime64[ns]'), allow_pickle=False)
    except Exception:
        pass
    _STOCK_TRADE_CAL_CACHE = cal
    return cal


def _shift_by_trading_days(cal: pd.DatetimeIndex, anchor: pd.Timestamp, offset: int, *, ceil: bool) -> pd.Timestamp:
    if cal is None or len(cal) == 0:
        raise ValueError("empty trade calendar")
    anchor = pd.Timestamp(anchor)
    if pd.isna(anchor):
        raise ValueError("invalid anchor date")
    if ceil:
        base = int(cal.searchsorted(anchor, side="left"))
        if base >= len(cal):
            base = len(cal) - 1
    else:
        base = int(cal.searchsorted(anchor, side="right")) - 1
        if base < 0:
            base = 0
    idx = base + int(offset)
    if idx < 0:
        idx = 0
    if idx >= len(cal):
        idx = len(cal) - 1
    return pd.Timestamp(cal[idx])


def build_setting(args, tag):
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_"
        f"fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{tag}"
    )


def normalize_model_name(name):
    return name.replace('-', '').replace(' ', '')


def _model_available(model_name):
    module = getattr(models_pkg, model_name, None)
    return module is not None and hasattr(module, "Model")


def _adjust_segrnn_seg_len(args_run):
    if args_run.model != 'SegRNN':
        return
    seg_len = math.gcd(args_run.seq_len, args_run.pred_len)
    if seg_len <= 0:
        seg_len = args_run.pred_len
    if args_run.seq_len % seg_len != 0 or args_run.pred_len % seg_len != 0:
        raise ValueError(
            f"SegRNN requires seg_len to divide seq_len and pred_len; "
            f"got seq_len={args_run.seq_len}, pred_len={args_run.pred_len}, seg_len={seg_len}"
        )
    if args_run.seg_len != seg_len:
        print(f"[config] SegRNN requires seg_len dividing seq_len/pred_len; setting seg_len={seg_len}")
        args_run.seg_len = seg_len


def _years_spec(start, end):
    if start is None or end is None:
        return ''
    years = list(range(start.year, end.year + 1))
    if not years:
        return ''
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def _window_tag(train_start, train_end, test_start, test_end, val_start, val_end, idx):
    return (
        f"win{idx:03d}_"
        f"tr{train_start:%Y%m%d}-{train_end:%Y%m%d}_"
        f"te{test_start:%Y%m%d}-{test_end:%Y%m%d}_"
        f"va{val_start:%Y%m%d}-{val_end:%Y%m%d}"
    )


def _build_windows(start_date, end_date, train_years, test_months, val_months, step_months, *, window_order: str):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start_date/end_date must be valid dates")
    if start >= end:
        raise ValueError("start_date must be before end_date")
    if train_years <= 0 or test_months <= 0 or val_months <= 0 or step_months <= 0:
        raise ValueError("train_years/test_months/val_months/step_months must be > 0")

    train_offset = pd.DateOffset(years=train_years)
    test_offset = pd.DateOffset(months=test_months)
    val_offset = pd.DateOffset(months=val_months)
    step_offset = pd.DateOffset(months=step_months)
    window_order = str(window_order or '').strip().lower()
    if window_order not in {'train_test_val', 'train_val_test'}:
        raise ValueError("window_order must be one of: train_val_test, train_test_val")

    windows = []
    idx = 0
    train_start = start
    one_day = pd.Timedelta(days=1)
    while True:
        train_end = train_start + train_offset - one_day
        if window_order == 'train_val_test':
            val_start = train_end + one_day
            val_end = val_start + val_offset - one_day
            test_start = val_end + one_day
            test_end = test_start + test_offset - one_day
        else:
            test_start = train_end + one_day
            test_end = test_start + test_offset - one_day
            val_start = test_end + one_day
            val_end = val_start + val_offset - one_day

        if test_end > end or val_end > end:
            break
        windows.append((idx, train_start, train_end, test_start, test_end, val_start, val_end))
        idx += 1
        train_start = train_start + step_offset

    return windows


def _parse_device_list(devices):
    if devices is None:
        return []
    devices = devices.replace(' ', '')
    if not devices:
        return []
    return [int(d) for d in devices.split(',') if d != '']


def _normalize_multi_gpu_args(args):
    devices = _parse_device_list(args.devices)
    if not devices:
        return
    # Use visible indices for DataParallel; CUDA_VISIBLE_DEVICES will map them.
    args.devices = ",".join(str(d) for d in devices)
    args.device_ids = list(range(len(devices)))
    args.gpu = 0


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_compile_model(model, use_gpu, gpu_type):
    if not use_gpu or gpu_type != 'cuda':
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    try:
        return compile_fn(model)
    except Exception as exc:
        print(f"[compile] skip: {exc}")
        return model


def _compute_sharpe_series(dates, capital, initial_cash, risk_free):
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


def backtest_topk_detailed(
    pred_df,
    initial_cash,
    topk,
    commission,
    stamp,
    risk_free,
    *,
    nan_true_return_policy: str = "drop",
):
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
            'trade_date', 'net_return', 'net_factor', 'capital',
            'return_pct', 'drawdown_pct', 'max_drawdown_pct',
            'profit_factor', 'sharpe'
        ])
        return metrics, empty_curve, empty_picks, empty_daily

    df = pred_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    if 'pred_date' in df.columns:
        df['pred_date'] = pd.to_datetime(df['pred_date'])
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'])
    if 'exit_date' in df.columns:
        df['exit_date'] = pd.to_datetime(df['exit_date'])

    nan_policy = str(nan_true_return_policy or "drop").strip().lower()
    if nan_policy not in {"drop", "zero"}:
        raise ValueError(f"nan_true_return_policy must be one of: drop, zero (got {nan_true_return_policy!r})")

    df['pred_return'] = pd.to_numeric(df.get('pred_return'), errors='coerce')
    df['true_return'] = pd.to_numeric(df.get('true_return'), errors='coerce')
    df['pred_return'] = df['pred_return'].where(np.isfinite(df['pred_return']), np.nan)
    df['true_return'] = df['true_return'].where(np.isfinite(df['true_return']), np.nan)

    before_rows = int(len(df))
    before_nan_true = int(df['true_return'].isna().sum()) if 'true_return' in df.columns else before_rows

    df = df.dropna(subset=['pred_return'])
    if nan_policy == "drop":
        df = df.dropna(subset=['true_return'])
    else:
        df['true_return'] = df['true_return'].fillna(0.0)

    after_rows = int(len(df))
    if nan_policy == "drop" and before_nan_true > 0 and after_rows > 0:
        dropped_pct = before_nan_true / max(1, before_rows) * 100.0
        if dropped_pct >= 1.0:
            print(
                f"[warn] backtest dropped {before_nan_true}/{before_rows} rows ({dropped_pct:.2f}%) due to NaN true_return; "
                f"this can be optimistic (implicit lookahead). Consider --backtest_nan_true_return=zero."
            )
    df = df[df['true_return'] > -0.999]
    df = df[df['suspendFlag'] == 0]
    df = df.sort_values('trade_date')

    if df.empty:
        return backtest_topk_detailed(pd.DataFrame(), initial_cash, topk, commission, stamp, risk_free)

    all_trade_dates = df['trade_date'].drop_duplicates().sort_values()
    if all_trade_dates.empty:
        return backtest_topk_detailed(pd.DataFrame(), initial_cash, topk, commission, stamp, risk_free)

    if 'exit_date' in df.columns:
        # Warn if exit_date doesn't line up with next trade_date (common sign of multi-day horizon with daily compounding).
        try:
            exit_map = (
                df[['trade_date', 'exit_date']]
                  .dropna()
                  .drop_duplicates()
                  .groupby('trade_date', sort=True)['exit_date']
                  .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            )
            trade_dates_sorted = list(all_trade_dates)
            mismatches = 0
            checked = 0
            for i in range(len(trade_dates_sorted) - 1):
                td = pd.Timestamp(trade_dates_sorted[i])
                expected_exit = pd.Timestamp(trade_dates_sorted[i + 1])
                got_exit = exit_map.get(td)
                if got_exit is None:
                    continue
                checked += 1
                if pd.Timestamp(got_exit) != expected_exit:
                    mismatches += 1
            if checked > 0 and mismatches / checked >= 0.2:
                print(
                    f"[warn] exit_date != next trade_date for {mismatches}/{checked} trade days; "
                    "current backtest compounds per trade_date (assumes ~1-day holding). "
                    "If trade_horizon>2, results may be overstated."
                )
        except Exception:
            pass

    picks = (
        df.sort_values(['trade_date', 'pred_return'], ascending=[True, False])
          .groupby('trade_date', sort=False, as_index=False)
          .head(topk)
    )
    pick_factor = (1 - commission) * (1 + picks['true_return'].values) * (1 - commission - stamp)
    pick_factor = np.where(np.isfinite(pick_factor) & (pick_factor > 0), pick_factor, np.nan)
    picks = picks.copy()
    picks['net_factor'] = pick_factor
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
        profit_factor_series = np.where(
            total_loss_cum > 0,
            total_profit_cum / total_loss_cum,
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

    total_trades = trade_days * topk * 2

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


def predict_dataset(exp, data_set, data_loader, args):
    exp.model.eval()
    records = []
    offset = 0
    packed = bool(getattr(data_set, 'packed', False))
    need_true_open_return = getattr(args, 'target', '') in RANK_TARGETS
    trade_horizon = int(getattr(args, 'trade_horizon', 2) or 2)
    true_return_limit = getattr(args, 'stock_true_return_limit', None)
    if true_return_limit is None:
        true_return_limit = STOCK_RETURN_LIMIT
    if true_return_limit is not None:
        true_return_limit = float(true_return_limit)

    open_pos = None
    open_scale = None
    open_mean = None
    if need_true_open_return and not packed:
        try:
            open_pos = data_set.feature_cols.index('open')
        except (AttributeError, ValueError):
            open_pos = None
        if open_pos is not None and getattr(data_set, 'scale', False) and getattr(data_set, '_scaler_fitted', False):
            open_scale = float(data_set.scaler.scale_[open_pos])
            open_mean = float(data_set.scaler.mean_[open_pos])
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if data_set.scale and args.inverse:
                outputs = data_set.inverse_transform(outputs)
                batch_y = data_set.inverse_transform(batch_y)

            if packed:
                target_slice = getattr(data_set, 'target_slice', None)
                if target_slice and isinstance(target_slice, (list, tuple)) and len(target_slice) == 2:
                    target_start, target_end = int(target_slice[0]), int(target_slice[1])
                else:
                    n_codes = int(getattr(data_set, 'n_codes', 0) or len(getattr(data_set, 'universe_codes', []) or []))
                    n_groups = int(getattr(data_set, 'n_groups', 0) or 1)
                    target_start, target_end = (n_groups - 1) * n_codes, n_groups * n_codes

                outputs_t = outputs[:, :, target_start:target_end]
                batch_y_t = batch_y[:, :, target_start:target_end]

                step_idx = trade_horizon - 1
                pred_step = outputs_t[:, step_idx, :]  # [B, n_codes]
                true_step = batch_y_t[:, step_idx, :]  # [B, n_codes]
                codes = getattr(data_set, 'universe_codes', None)
                if not codes:
                    raise ValueError("packed dataset missing universe_codes")
                open_mat = getattr(data_set, 'open', None)
                suspend_mat = getattr(data_set, 'suspend', None)
                dates = getattr(data_set, 'dates', None)
                if dates is None:
                    raise ValueError("packed dataset missing dates")

                for i in range(pred_step.shape[0]):
                    end_idx = int(data_set.sample_index[offset + i])
                    trade_idx = end_idx + 1
                    exit_idx = end_idx + step_idx + 1
                    meta = data_set.sample_meta[offset + i]
                    exit_date = pd.Timestamp(dates[exit_idx])

                    if need_true_open_return:
                        if open_mat is None:
                            raise ValueError("packed dataset missing open matrix required for true return")
                        open_trade = open_mat[trade_idx]
                        open_pred = open_mat[exit_idx]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            true_ret = open_pred / open_trade - 1.0
                        true_ret = np.where((open_trade > 0) & (open_pred > 0), true_ret, np.nan)
                        if true_return_limit is not None:
                            true_ret = np.where(np.abs(true_ret) <= true_return_limit, true_ret, np.nan)
                    else:
                        true_ret = true_step[i]

                    suspend_flags = suspend_mat[trade_idx] if suspend_mat is not None else np.zeros(len(codes), dtype=int)
                    preds = pred_step[i]
                    for j, code in enumerate(codes):
                        records.append({
                            'code': code,
                            'end_date': meta['end_date'],
                            'trade_date': meta['trade_date'],
                            'pred_date': meta['pred_date'],
                            'exit_date': exit_date,
                            'suspendFlag': int(suspend_flags[j]),
                            'pred_return': float(preds[j]),
                            'true_return': float(true_ret[j]) if np.isfinite(true_ret[j]) else float('nan'),
                        })
                offset += pred_step.shape[0]
            else:
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                step_idx = trade_horizon - 1
                pred_step = outputs[:, step_idx, -1]
                true_step = batch_y[:, step_idx, -1]

                for i in range(len(pred_step)):
                    meta = data_set.sample_meta[offset + i]
                    true_return = float(true_step[i])
                    exit_date = meta.get('pred_date')
                    if need_true_open_return:
                        code, end_idx = data_set.sample_index[offset + i]
                        trade_idx = int(end_idx) + 1
                        exit_idx = int(end_idx) + step_idx + 1
                        payload = data_set.data_by_code.get(code)
                        if payload is None or open_pos is None:
                            true_return = float('nan')
                        else:
                            data = payload['data']
                            open_trade = float(data[trade_idx, open_pos])
                            open_pred = float(data[exit_idx, open_pos])
                            if open_scale is not None and open_mean is not None:
                                open_trade = open_trade * open_scale + open_mean
                                open_pred = open_pred * open_scale + open_mean
                            if open_trade > 0 and open_pred > 0:
                                true_return = open_pred / open_trade - 1.0
                                if true_return_limit is not None and abs(true_return) > true_return_limit:
                                    true_return = float('nan')
                            else:
                                true_return = float('nan')
                            try:
                                exit_date = pd.Timestamp(payload['dates'][exit_idx])
                            except Exception:
                                exit_date = meta.get('pred_date')
                    records.append({
                        'code': meta['code'],
                        'end_date': meta['end_date'],
                        'trade_date': meta['trade_date'],
                        'pred_date': meta['pred_date'],
                        'exit_date': exit_date,
                        'suspendFlag': meta['suspendFlag'],
                        'pred_return': float(pred_step[i]),
                        'true_return': float(true_return)
                    })
                offset += len(pred_step)
    return pd.DataFrame.from_records(records)


def parse_args():
    parser = argparse.ArgumentParser(description='Rolling stock backtest runner')

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return True
        text = str(v).strip().lower()
        if text in {'1', 'true', 't', 'yes', 'y', 'on'}:
            return True
        if text in {'0', 'false', 'f', 'no', 'n', 'off'}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")

    parser.add_argument('--root_path', type=str, default='.', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='stock_data.parquet', help='data file')
    parser.add_argument('--model', type=str, default='WPMixer', help='model name')
    parser.add_argument('--model_id', type=str, default='stock', help='model id')
    parser.add_argument('--des', type=str, default='stock_rolling', help='exp description')

    parser.add_argument('--start_date', type=str, default='2010-01-01', help='rolling backtest start date')
    parser.add_argument('--end_date', type=str, default='2025-12-31', help='rolling backtest end date')
    parser.add_argument('--train_years', type=int, default=5, help='training window length in years')
    parser.add_argument('--test_months', type=int, default=3, help='test window length in months')
    parser.add_argument('--val_months', type=int, default=3, help='validation window length in months')
    parser.add_argument('--step_months', type=int, default=6, help='rolling step in months')
    parser.add_argument('--window_order', type=str, default='train_val_test',
                        choices=['train_val_test', 'train_test_val'],
                        help='rolling window order (train_val_test avoids lookahead for early stopping; train_test_val is legacy)')
    parser.add_argument('--max_windows', type=int, default=0, help='limit number of windows (0 = no limit)')

    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')
    parser.add_argument('--trade_horizon', type=int, default=2,
                        help='which forecast horizon step to trade on (2 = next-next open return); must be <= pred_len')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='lag_return_cs_rank', help='target feature')
    parser.add_argument('--freq', type=str, default='b', help='freq for time features encoding')

    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--loss', type=str, default='HYBRID_WIC', help='loss function (e.g., MSE, CCC)')
    parser.add_argument('--ic_weight_beta', type=float, default=5.0,
                        help='beta for Weighted IC loss softmax weighting')
    parser.add_argument('--hybrid_ic_weight', type=float, default=0.7,
                        help='IC weight for Hybrid loss (ic_weight * IC + (1-ic_weight) * CCC)')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--persistent_workers', action='store_true', default=True,
                        help='keep data loader workers alive between epochs')
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false',
                        help='disable persistent data loader workers')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use amp')

    parser.add_argument('--d_model', type=int, default=256, help='model dim')
    parser.add_argument('--n_heads', type=int, default=8, help='num heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='ffn dim')
    parser.add_argument('--moving_avg', type=int, default=25, help='moving avg window')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='distil')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='use norm')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear individual mode')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='down sampling layers')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='number of hidden layers in projector')

    parser.add_argument('--use_gpu', type=_str2bool, nargs='?', const=True, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids')
    parser.add_argument('--parallel_windows', action='store_true', default=False,
                        help='run rolling windows in parallel across GPUs listed in --devices')

    parser.add_argument('--augmentation_ratio', type=int, default=0, help='augmentation ratio')
    parser.add_argument('--lradj', type=str, default='type1', help='lr adjust')
    parser.add_argument('--inverse', action='store_true', default=True, help='inverse output data')

    parser.add_argument('--initial_cash', type=float, default=1_000_000.0, help='initial capital')
    parser.add_argument('--topk', type=int, default=1, help='topk')
    parser.add_argument('--commission', type=float, default=0.0003, help='commission rate')
    parser.add_argument('--stamp', type=float, default=0.001, help='stamp tax rate')
    parser.add_argument('--risk_free', type=float, default=0.03, help='risk free rate')
    parser.add_argument('--stock_pack', action='store_true', default=True,
                        help="pack all stocks into one huge tensor; features='S' packs target only, otherwise packs base features + target")
    parser.add_argument('--stock_universe_size', type=int, default=0,
                        help='number of stocks to include when --stock_pack (0 = all codes with full coverage)')
    parser.add_argument('--stock_pack_start', type=str, default=None,
                        help='packed calendar start (YYYY-MM-DD); default derived per rolling window')
    parser.add_argument('--stock_pack_end', type=str, default=None,
                        help='packed calendar end (YYYY-MM-DD); default derived per rolling window')
    parser.add_argument('--stock_pack_select_end', type=str, default='train_end',
                        help="packed universe selection coverage end: 'train_end'(default, no-lookahead), "
                             "'pack_end'(legacy survivorship bias), 'none'(allow partial), or an explicit date (YYYY-MM-DD)")
    parser.add_argument('--stock_pack_extra_td', type=int, default=2,
                        help='extra trading days after max(test_end, val_end) for packed calendar end (in addition to pred_len-1)')
    parser.add_argument('--stock_pack_fill_value', type=float, default=0.0,
                        help='fill value for missing/invalid targets when --stock_pack')
    parser.add_argument('--stock_strict_pred_end', type=_str2bool, nargs='?', const=True, default=True,
                        help='require pred_date to also fall within the split range (prevents label leakage across train/val/test)')
    parser.add_argument('--stock_cache_dir', type=str, default='./cache', help='cache dir for stock preprocessing')
    parser.add_argument('--disable_stock_cache', action='store_true', default=False, help='disable stock cache')
    parser.add_argument('--stock_preprocess_workers', type=int, default=0,
                        help='parallel workers for stock preprocessing (0 = single process)')
    parser.add_argument('--stock_true_return_limit', type=float, default=None,
                        help='abs limit for true_return computed from open prices; default uses STOCK_RETURN_LIMIT')
    parser.add_argument('--backtest_nan_true_return', type=str, default='drop',
                        choices=['drop', 'zero'],
                        help="backtest policy for NaN true_return: 'drop' (current, can be optimistic), "
                             "'zero' (keep rows and assume 0 return)")

    parser.add_argument('--results_root', type=str, default='stock_results_rolling', help='results directory')
    parser.add_argument('--resume', type=_str2bool, nargs='?', const=True, default=True,
                        help='skip training/eval if outputs already exist (set False to re-run)')

    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='tslib-roll', help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default='stock_rolling', help='wandb group')
    parser.add_argument('--wandb_mode', type=str, default=None, help='wandb mode')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    args = parser.parse_args()
    if args.no_wandb:
        args.use_wandb = False
    return args


def results_complete(result_dir, split):
    split_dir = os.path.join(result_dir, split)
    required = ['predictions.csv', 'equity_curve.csv', 'metrics.json', 'daily_picks.csv', 'daily_metrics.csv']
    return all(os.path.exists(os.path.join(split_dir, fname)) for fname in required)

def _load_metrics(result_dir, split):
    metrics_fp = os.path.join(result_dir, split, 'metrics.json')
    if not os.path.exists(metrics_fp):
        return None
    with open(metrics_fp, 'r') as f:
        return json.load(f)


def _run_window(base_args, model_name, window, use_window_seed=False):
    idx, train_start, train_end, test_start, test_end, val_start, val_end = window
    if use_window_seed:
        _seed_everything(getattr(base_args, 'seed', 2021) + idx)

    args_run = copy.deepcopy(base_args)
    args_run.model = model_name
    args_run.task_name = 'zero_shot_forecast' if model_name in ZERO_SHOT_MODELS else 'long_term_forecast'
    args_run.train_start = train_start.strftime('%Y-%m-%d')
    args_run.train_end = train_end.strftime('%Y-%m-%d')
    args_run.test_start = test_start.strftime('%Y-%m-%d')
    args_run.test_end = test_end.strftime('%Y-%m-%d')
    args_run.val_start = val_start.strftime('%Y-%m-%d')
    args_run.val_end = val_end.strftime('%Y-%m-%d')
    args_run.train_years = _years_spec(train_start, train_end)
    args_run.test_years = _years_spec(test_start, test_end)
    args_run.val_years = _years_spec(val_start, val_end)

    if model_name == 'ETSformer' and args_run.e_layers != args_run.d_layers:
        print(f"[config] ETSformer requires e_layers == d_layers; setting d_layers={args_run.e_layers}")
        args_run.d_layers = args_run.e_layers
    _adjust_segrnn_seg_len(args_run)

    if getattr(args_run, 'stock_pack', False):
        trade_cal = _load_stock_trade_calendar(args_run)
        if not getattr(args_run, 'stock_pack_start', None):
            pack_start = _shift_by_trading_days(trade_cal, train_start, -int(args_run.seq_len), ceil=True)
            args_run.stock_pack_start = pack_start.strftime('%Y-%m-%d')
        if not getattr(args_run, 'stock_pack_end', None):
            extra_td = int(getattr(args_run, 'stock_pack_extra_td', 3) or 0)
            lookahead = int(args_run.pred_len) - 1 + max(0, extra_td)
            eval_end = max(test_end, val_end)
            pack_end = _shift_by_trading_days(trade_cal, eval_end, lookahead, ceil=False)
            args_run.stock_pack_end = pack_end.strftime('%Y-%m-%d')
        else:
            extra_td = int(getattr(args_run, 'stock_pack_extra_td', 3) or 0)

        probe_dataset = Dataset_StockPacked(
            args_run,
            root_path=args_run.root_path,
            flag='train',
            size=[args_run.seq_len, args_run.label_len, args_run.pred_len],
            features=args_run.features,
            data_path=args_run.data_path,
            target=args_run.target,
            scale=True,
            timeenc=0 if args_run.embed != 'timeF' else 1,
            freq=args_run.freq
        )
        args_run.enc_in = probe_dataset.c_in
        args_run.dec_in = probe_dataset.c_in
        args_run.c_out = probe_dataset.c_in

    window_tag = _window_tag(train_start, train_end, test_start, test_end, val_start, val_end, idx)
    strict_pred_end = bool(getattr(args_run, 'stock_strict_pred_end', True))
    window_tag = f"{window_tag}_SL{int(strict_pred_end)}"
    trade_horizon = int(getattr(args_run, 'trade_horizon', 2) or 2)
    true_return_limit = getattr(args_run, 'stock_true_return_limit', None)
    if true_return_limit is None:
        true_return_limit = STOCK_RETURN_LIMIT
    trl_val = "none" if true_return_limit is None else f"{float(true_return_limit):g}".replace('-', 'm').replace('.', 'p')
    window_tag = f"{window_tag}_TH{trade_horizon}_TRL{trl_val}"
    if getattr(args_run, 'stock_pack', False):
        sel = str(getattr(args_run, 'stock_pack_select_end', 'train_end') or 'train_end').strip().lower()
        sel = sel.replace('-', '')
        window_tag = f"{window_tag}_SE{sel}"
        try:
            pk_start = pd.Timestamp(getattr(args_run, 'stock_pack_start', None))
            pk_end = pd.Timestamp(getattr(args_run, 'stock_pack_end', None))
        except Exception:
            pk_start = None
            pk_end = None
        fill = float(getattr(args_run, 'stock_pack_fill_value', 0.0))
        fill_str = f"{fill:g}".replace('-', 'm').replace('.', 'p')
        n_codes = int(getattr(probe_dataset, 'n_codes', 0) or 0)
        n_groups = int(getattr(probe_dataset, 'n_groups', 0) or 0)
        date_str = ""
        if pk_start is not None and pk_end is not None and not (pd.isna(pk_start) or pd.isna(pk_end)):
            date_str = f"_pk{pk_start:%Y%m%d}-{pk_end:%Y%m%d}"
        window_tag = f"{window_tag}{date_str}_N{n_codes}_G{n_groups}_E{int(extra_td)}_F{fill_str}"
    setting = build_setting(args_run, window_tag)
    checkpoint_path = os.path.join(args_run.checkpoints, setting, 'checkpoint.pth')
    result_dir = os.path.join(args_run.results_root, model_name, window_tag)

    split_done = {split: results_complete(result_dir, split) for split in ['test', 'val']}
    needs_eval = any(not done for done in split_done.values())
    checkpoint_exists = os.path.exists(checkpoint_path)
    needs_train = model_name not in ZERO_SHOT_MODELS and (not args_run.resume or not checkpoint_exists)

    if args_run.resume and not needs_train and not needs_eval:
        print(f"[resume] skip window {window_tag}: checkpoint and results already exist.")
        test_metrics = _load_metrics(result_dir, 'test')
        val_metrics = _load_metrics(result_dir, 'val')
        summary_row = {
            'window_tag': window_tag,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'val_start': val_start.strftime('%Y-%m-%d'),
            'val_end': val_end.strftime('%Y-%m-%d'),
            'test_metrics': test_metrics,
            'val_metrics': val_metrics
        }
        return summary_row, None

    run_name = f"rolling_{model_name}_{window_tag}"
    if args_run.use_wandb:
        init_wandb(args_run, name=run_name, group=args_run.wandb_group, tags=[model_name, 'rolling'])
    try:
        exp = Exp_Long_Term_Forecast(args_run)
        exp.model = _maybe_compile_model(exp.model, args_run.use_gpu, args_run.gpu_type)
        if model_name not in ZERO_SHOT_MODELS:
            if needs_train:
                exp.train(setting)
            else:
                exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))

        os.makedirs(result_dir, exist_ok=True)
        wandb_mod = None
        if args_run.use_wandb:
            try:
                import wandb as wandb_mod
                for split in ['test', 'val']:
                    wandb_mod.define_metric(f'{split}/step')
                    wandb_mod.define_metric(f'{split}/*', step_metric=f'{split}/step')
            except ImportError:
                wandb_mod = None

        for split in ['test', 'val']:
            if args_run.resume and split_done[split]:
                print(f"[resume] skip {window_tag} {split}: results already exist.")
                continue
            data_set, _ = data_provider(args_run, split)
            data_loader = DataLoader(
                data_set,
                batch_size=args_run.batch_size,
                shuffle=False,
                num_workers=args_run.num_workers,
                pin_memory=args_run.use_gpu and args_run.gpu_type == 'cuda',
                drop_last=False
            )
            pred_df = predict_dataset(exp, data_set, data_loader, args_run)
            metrics, curve_df, picks_df, daily_df = backtest_topk_detailed(
                pred_df,
                initial_cash=args_run.initial_cash,
                topk=args_run.topk,
                commission=args_run.commission,
                stamp=args_run.stamp,
                risk_free=args_run.risk_free,
                nan_true_return_policy=getattr(args_run, 'backtest_nan_true_return', 'drop')
            )

            split_dir = os.path.join(result_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            pred_df.to_csv(os.path.join(split_dir, 'predictions.csv'), index=False)
            curve_df.to_csv(os.path.join(split_dir, 'equity_curve.csv'), index=False)
            picks_df.to_csv(os.path.join(split_dir, 'daily_picks.csv'), index=False)
            daily_df.to_csv(os.path.join(split_dir, 'daily_metrics.csv'), index=False)
            with open(os.path.join(split_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            if args_run.use_wandb:
                log_wandb({f'{split}/{k}': v for k, v in metrics.items()})

            if wandb_mod is not None and not daily_df.empty:
                daily_df = daily_df.replace([np.inf, -np.inf], np.nan)
                for step_idx, row in daily_df.iterrows():
                    log_wandb({
                        f'{split}/step': int(step_idx),
                        f'{split}/trade_date': str(row['trade_date']),
                        f'{split}/final_capital': float(row['capital']),
                        f'{split}/return_pct': float(row['return_pct']),
                        f'{split}/profit_factor': float(row['profit_factor']),
                        f'{split}/drawdown_pct': float(row['drawdown_pct']),
                        f'{split}/max_drawdown_pct': float(row['max_drawdown_pct']),
                        f'{split}/sharpe': float(row['sharpe'])
                    })
        test_metrics = _load_metrics(result_dir, 'test')
        val_metrics = _load_metrics(result_dir, 'val')
        summary_row = {
            'window_tag': window_tag,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'val_start': val_start.strftime('%Y-%m-%d'),
            'val_end': val_end.strftime('%Y-%m-%d'),
            'test_metrics': test_metrics,
            'val_metrics': val_metrics
        }
        return summary_row, None
    except Exception as exc:
        print(f"[error] window {window_tag}: {exc}")
        return None, (window_tag, repr(exc))
    finally:
        if args_run.use_wandb:
            finish_wandb()


def _split_windows(windows, num_shards):
    if num_shards <= 1:
        return [windows]
    return [windows[i::num_shards] for i in range(num_shards)]


def _worker_process(gpu_id, windows, base_args_dict, model_name, queue):
    summary_rows = []
    failures = []
    try:
        # Bind this worker to a single GPU so cuda:0 maps to the assigned device.
        if base_args_dict.get('use_gpu', True) and base_args_dict.get('gpu_type') == 'cuda':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        args = argparse.Namespace(**base_args_dict)
        args.disable_cuda_env = True
        # Keep use_multi_gpu=True so Exp_Basic won't overwrite CUDA_VISIBLE_DEVICES to "0".
        args.use_multi_gpu = True
        args.devices = str(gpu_id)
        args.device_ids = [0]
        args.gpu = 0
        for window in windows:
            row, failure = _run_window(args, model_name, window, use_window_seed=True)
            if row is not None:
                summary_rows.append(row)
            if failure is not None:
                failures.append(failure)
    except Exception as exc:
        failures.append((f"worker_gpu{gpu_id}", repr(exc)))
    finally:
        queue.put((summary_rows, failures))


def main():
    args = parse_args()
    _seed_everything(args.seed)

    if args.use_gpu and args.gpu_type == 'cuda' and not torch.cuda.is_available():
        args.use_gpu = False
    if args.use_gpu and args.gpu_type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.use_gpu and args.gpu_type == 'mps':
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            args.use_gpu = False
    if args.use_multi_gpu:
        _normalize_multi_gpu_args(args)
    if args.pred_len < 2:
        raise ValueError("pred_len must be >= 2 to forecast next-next open return")
    args.trade_horizon = int(getattr(args, 'trade_horizon', 2) or 2)
    if args.trade_horizon < 2:
        raise ValueError("trade_horizon must be >= 2 (2 = next-next open return)")
    if args.trade_horizon > args.pred_len:
        raise ValueError(f"trade_horizon ({args.trade_horizon}) must be <= pred_len ({args.pred_len})")
    if args.trade_horizon != 2:
        print(
            "[warn] trade_horizon!=2 implies multi-day holding; current backtest compounds per trade_date "
            "(assumes ~1-day holding) and may overstate results."
        )
    if args.window_order == 'train_test_val':
        print("[warn] window_order=train_test_val uses future val after test for early stopping; prefer train_val_test.")

    args.task_name = 'long_term_forecast'
    args.data = 'stock'
    args.checkpoints = './checkpoints/'
    args.seasonal_patterns = 'Monthly'
    args.mask_rate = 0.25
    args.anomaly_ratio = 0.25
    args.channel_independence = 1
    args.decomp_method = 'moving_avg'
    args.down_sampling_window = 1
    args.down_sampling_method = None
    args.seg_len = args.seq_len
    args.alpha = 0.1
    args.top_p = 0.5
    args.pos = 1

    overall_start = pd.Timestamp(args.start_date)
    overall_end = pd.Timestamp(args.end_date)
    if not getattr(args, 'stock_pack', False):
        probe_args = copy.deepcopy(args)
        probe_args.train_years = _years_spec(overall_start, overall_end)
        probe_args.test_years = probe_args.train_years
        probe_args.val_years = probe_args.train_years
        probe_args.train_start = None
        probe_args.train_end = None
        probe_args.test_start = None
        probe_args.test_end = None
        probe_args.val_start = None
        probe_args.val_end = None
        probe_dataset = Dataset_Stock(
            probe_args,
            root_path=args.root_path,
            flag='train',
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            scale=True,
            timeenc=0 if probe_args.embed != 'timeF' else 1,
            freq=args.freq
        )
        args.enc_in = probe_dataset.c_in
        args.dec_in = probe_dataset.c_in
        args.c_out = probe_dataset.c_in

    model_name = normalize_model_name(args.model)
    if not _model_available(model_name):
        raise ValueError(f"model {model_name} missing or dependencies unavailable")

    windows = _build_windows(
        args.start_date,
        args.end_date,
        args.train_years,
        args.test_months,
        args.val_months,
        args.step_months,
        window_order=args.window_order
    )
    if args.max_windows and args.max_windows > 0:
        windows = windows[:args.max_windows]

    if not windows:
        raise ValueError("no rolling windows generated; check date range and window sizes")

    failures = []
    summary_rows = []
    device_list = _parse_device_list(args.devices)
    parallel_ok = args.parallel_windows and args.use_gpu and len(device_list) > 1
    if args.parallel_windows and not parallel_ok:
        print("[parallel] disabled: need --use_gpu and multiple --devices.")

    if parallel_ok:
        base_args_dict = vars(args).copy()
        window_shards = _split_windows(windows, len(device_list))
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = []
        for gpu_id, shard in zip(device_list, window_shards):
            proc = ctx.Process(
                target=_worker_process,
                args=(gpu_id, shard, base_args_dict, model_name, queue)
            )
            prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                proc.start()
            finally:
                if prev_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible
            procs.append((gpu_id, proc))

        for _ in procs:
            rows, errs = queue.get()
            summary_rows.extend(rows)
            failures.extend(errs)

        for gpu_id, proc in procs:
            proc.join()
            if proc.exitcode != 0:
                failures.append((f"worker_gpu{gpu_id}", f"exitcode {proc.exitcode}"))
    else:
        for window in windows:
            row, failure = _run_window(args, model_name, window)
            if row is not None:
                summary_rows.append(row)
            if failure is not None:
                failures.append(failure)

    if summary_rows:
        summary_out = []
        for row in summary_rows:
            flat = {
                'window_tag': row['window_tag'],
                'train_start': row['train_start'],
                'train_end': row['train_end'],
                'test_start': row['test_start'],
                'test_end': row['test_end'],
                'val_start': row['val_start'],
                'val_end': row['val_end']
            }
            test_metrics = row.get('test_metrics') or {}
            val_metrics = row.get('val_metrics') or {}
            for key, value in test_metrics.items():
                flat[f'test_{key}'] = value
            for key, value in val_metrics.items():
                flat[f'val_{key}'] = value
            summary_out.append(flat)

        summary_df = pd.DataFrame(summary_out)
        summary_dir = os.path.join(args.results_root, model_name)
        os.makedirs(summary_dir, exist_ok=True)
        summary_csv = os.path.join(summary_dir, 'window_summary.csv')
        summary_json = os.path.join(summary_dir, 'window_summary.json')
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_json(summary_json, orient='records', indent=2)
        if args.use_wandb:
            try:
                import wandb
                init_wandb(
                    args,
                    name=f"rolling_summary_{model_name}",
                    group=f"{args.wandb_group}_summary",
                    tags=[model_name, 'rolling', 'summary']
                )
                wandb.log({'window_summary': wandb.Table(dataframe=summary_df)})
            finally:
                finish_wandb()

    if failures:
        print("[summary] failed windows:")
        for window_tag, err in failures:
            print(f" - {window_tag}: {err}")


if __name__ == '__main__':
    main()
