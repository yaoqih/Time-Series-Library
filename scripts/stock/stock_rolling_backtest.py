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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import models as models_pkg

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Stock
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.wandb_utils import init_wandb, log_wandb, finish_wandb

ZERO_SHOT_MODELS = {
    'Chronos', 'Chronos2', 'Moirai', 'Sundial', 'TiRex', 'TimeMoE', 'TimesFM'
}


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


def _build_windows(start_date, end_date, train_years, test_months, val_months, step_months):
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

    windows = []
    idx = 0
    train_start = start
    one_day = pd.Timedelta(days=1)
    while True:
        train_end = train_start + train_offset - one_day
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


def backtest_topk_detailed(pred_df, initial_cash, topk, commission, stamp, risk_free):
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

    df = df.dropna(subset=['pred_return', 'true_return'])
    df = df[np.isfinite(df['pred_return']) & np.isfinite(df['true_return'])]
    df = df[df['true_return'] > -0.999]
    df = df[df['suspendFlag'] == 0]
    df = df.sort_values('trade_date')

    if df.empty:
        return backtest_topk_detailed(pd.DataFrame(), initial_cash, topk, commission, stamp, risk_free)

    all_trade_dates = df['trade_date'].drop_duplicates().sort_values()
    if all_trade_dates.empty:
        return backtest_topk_detailed(pd.DataFrame(), initial_cash, topk, commission, stamp, risk_free)

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

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]
            pred_step = outputs[:, -1, -1]
            true_step = batch_y[:, -1, -1]

            for i in range(len(pred_step)):
                meta = data_set.sample_meta[offset + i]
                records.append({
                    'code': meta['code'],
                    'end_date': meta['end_date'],
                    'trade_date': meta['trade_date'],
                    'pred_date': meta['pred_date'],
                    'suspendFlag': meta['suspendFlag'],
                    'pred_return': float(pred_step[i]),
                    'true_return': float(true_step[i])
                })
            offset += len(pred_step)
    return pd.DataFrame.from_records(records)


def parse_args():
    parser = argparse.ArgumentParser(description='Rolling stock backtest runner')

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
    parser.add_argument('--max_windows', type=int, default=0, help='limit number of windows (0 = no limit)')

    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='lag_return', help='target feature')
    parser.add_argument('--freq', type=str, default='b', help='freq for time features encoding')

    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
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

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
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
    parser.add_argument('--stock_cache_dir', type=str, default='./cache', help='cache dir for stock preprocessing')
    parser.add_argument('--disable_stock_cache', action='store_true', default=False, help='disable stock cache')
    parser.add_argument('--stock_preprocess_workers', type=int, default=0,
                        help='parallel workers for stock preprocessing (0 = single process)')

    parser.add_argument('--results_root', type=str, default='stock_results_rolling', help='results directory')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='skip training/eval if outputs already exist')

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

    window_tag = _window_tag(train_start, train_end, test_start, test_end, val_start, val_end, idx)
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
                risk_free=args_run.risk_free
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
        timeenc=0,
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
        args.step_months
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
