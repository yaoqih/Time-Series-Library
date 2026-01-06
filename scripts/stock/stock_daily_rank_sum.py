import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import models as models_pkg
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Stock
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


ZERO_SHOT_MODELS = {
    'Chronos', 'Chronos2', 'Moirai', 'Sundial', 'TiRex', 'TimeMoE', 'TimesFM'
}
DEFAULT_MODELS = 'FreTS,PatchTST,WPMixer,iTransformer'


def build_setting(args):
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_"
        f"fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}"
    )


def normalize_model_name(name: str) -> str:
    return name.replace('-', '').replace(' ', '')


def _model_available(model_name: str) -> bool:
    module = getattr(models_pkg, model_name, None)
    return module is not None and hasattr(module, "Model")


def _adjust_segrnn_seg_len(args_run):
    if args_run.model != 'SegRNN':
        return
    # SegRNN requires seg_len to divide both seq_len and pred_len.
    import math
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


def _years_spec(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start is None or end is None:
        return ''
    years = list(range(start.year, end.year + 1))
    if not years:
        return ''
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def _parse_possible_date_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype='datetime64[ns]')
    if np.issubdtype(series.dtype, np.integer):
        sample = series.dropna().astype(str).iloc[:3]
        if sample.str.len().eq(8).all():
            return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')
    return pd.to_datetime(series, errors='coerce')


def _latest_data_date(path: str) -> pd.Timestamp:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = pf.schema.names
    except Exception:
        cols = []

    if 'time' in cols:
        df = pd.read_parquet(path, columns=['time'])
        ts = pd.to_datetime(df['time'], unit='ms', errors='coerce')
    elif 'date' in cols:
        df = pd.read_parquet(path, columns=['date'])
        ts = _parse_possible_date_series(df['date'])
    else:
        df = pd.read_parquet(path)
        if 'time' in df.columns:
            ts = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        elif 'date' in df.columns:
            ts = _parse_possible_date_series(df['date'])
        else:
            raise ValueError("stock_data.parquet must contain 'time' or 'date' column")

    latest = ts.max()
    if pd.isna(latest):
        raise ValueError("cannot infer latest date from stock_data.parquet")
    return pd.Timestamp(latest).normalize()


def _compute_window(
    data_end: pd.Timestamp,
    months: int,
    test_months: int,
    val_months: int,
    train_start: Optional[str],
    train_end: Optional[str],
) -> Dict[str, pd.Timestamp]:
    end_date = pd.Timestamp(train_end).normalize() if train_end else pd.Timestamp(data_end).normalize()
    if train_start:
        start_date = pd.Timestamp(train_start).normalize()
    else:
        start_date = end_date - pd.DateOffset(months=months) + pd.Timedelta(days=1)

    if test_months < 0 or val_months < 0:
        raise ValueError("test_months/val_months must be >= 0")
    if months <= 0:
        raise ValueError("months must be > 0")

    train_end_date = end_date - pd.DateOffset(months=test_months + val_months)
    if train_end_date < start_date:
        raise ValueError(
            "train window too small; reduce test/val months or expand training range"
        )

    if test_months == 0:
        test_start = train_end_date + pd.Timedelta(days=1)
        test_end = train_end_date
    else:
        test_start = train_end_date + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

    if val_months == 0:
        val_start = test_end + pd.Timedelta(days=1)
        val_end = test_end
    else:
        val_start = test_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)

    if val_end > end_date:
        val_end = end_date
    if val_start > val_end:
        val_start = val_end

    return {
        'train_start': start_date,
        'train_end': train_end_date,
        'test_start': test_start,
        'test_end': test_end,
        'val_start': val_start,
        'val_end': val_end,
        'data_end': end_date
    }


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


def _ensure_args(args, window: Dict[str, pd.Timestamp]):
    args.data = 'stock'
    args.task_name = 'long_term_forecast'
    args.train_start = window['train_start'].strftime('%Y-%m-%d')
    args.train_end = window['train_end'].strftime('%Y-%m-%d')
    args.test_start = window['test_start'].strftime('%Y-%m-%d') if window['test_start'] <= window['test_end'] else None
    args.test_end = window['test_end'].strftime('%Y-%m-%d') if window['test_start'] <= window['test_end'] else None
    args.val_start = window['val_start'].strftime('%Y-%m-%d') if window['val_start'] <= window['val_end'] else None
    args.val_end = window['val_end'].strftime('%Y-%m-%d') if window['val_start'] <= window['val_end'] else None
    args.train_years = _years_spec(window['train_start'], window['train_end'])
    if window['test_start'] <= window['test_end']:
        args.test_years = _years_spec(window['test_start'], window['test_end'])
    else:
        args.test_years = ''
    if window['val_start'] <= window['val_end']:
        args.val_years = _years_spec(window['val_start'], window['val_end'])
    else:
        args.val_years = ''


@dataclass
class ModelArtifacts:
    model_name: str
    pred_df: pd.DataFrame
    pred_date: pd.Timestamp
    output_path: str


def _load_checkpoint(exp: Exp_Long_Term_Forecast, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location=exp.device)
    exp.model.load_state_dict(state)


def train_models(args, models: List[str], window: Dict[str, pd.Timestamp], resume: bool) -> Dict[str, str]:
    _ensure_args(args, window)

    probe_args = copy.deepcopy(args)
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

    checkpoint_map = {}
    for model_name in models:
        args_run = copy.deepcopy(args)
        args_run.model = model_name
        if not _model_available(model_name):
            print(f"[skip] {model_name}: model module missing or dependencies unavailable.")
            continue
        _adjust_segrnn_seg_len(args_run)

        setting = build_setting(args_run)
        checkpoint_path = os.path.join(args_run.checkpoints, setting, 'checkpoint.pth')
        checkpoint_map[model_name] = checkpoint_path

        if resume and os.path.exists(checkpoint_path):
            print(f"[resume] {model_name}: checkpoint exists, skipping training.")
            continue

        exp = Exp_Long_Term_Forecast(args_run)
        exp.train(setting)

    return checkpoint_map


def infer_latest(args, models: List[str], window: Dict[str, pd.Timestamp], checkpoint_map: Dict[str, str],
                 output_dir: str, pred_date_arg: Optional[str]) -> Tuple[str, List[ModelArtifacts]]:
    _ensure_args(args, window)

    data_path = os.path.join(args.root_path, args.data_path)
    data_end = _latest_data_date(data_path)
    if args.infer_start:
        infer_start = pd.Timestamp(args.infer_start).normalize()
    else:
        infer_start = data_end - pd.DateOffset(months=args.infer_months) + pd.Timedelta(days=1)

    args.test_start = infer_start.strftime('%Y-%m-%d')
    args.test_end = data_end.strftime('%Y-%m-%d')
    args.test_years = _years_spec(infer_start, data_end)

    probe_args = copy.deepcopy(args)
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

    pred_frames: Dict[str, pd.DataFrame] = {}
    max_dates: Dict[str, pd.Timestamp] = {}

    for model_name in models:
        args_run = copy.deepcopy(args)
        args_run.model = model_name
        if not _model_available(model_name):
            print(f"[skip] {model_name}: model module missing or dependencies unavailable.")
            continue
        _adjust_segrnn_seg_len(args_run)
        setting = build_setting(args_run)
        checkpoint_path = checkpoint_map.get(model_name) or os.path.join(args_run.checkpoints, setting, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found for {model_name}: {checkpoint_path}")

        exp = Exp_Long_Term_Forecast(args_run)
        _load_checkpoint(exp, checkpoint_path)

        data_set, _ = data_provider(args_run, 'test')
        data_loader = DataLoader(
            data_set,
            batch_size=args_run.batch_size,
            shuffle=False,
            num_workers=args_run.num_workers,
            drop_last=False,
            persistent_workers=bool(getattr(args_run, 'persistent_workers', False)) and args_run.num_workers > 0
        )
        pred_df = predict_dataset(exp, data_set, data_loader, args_run)
        if pred_df.empty:
            print(f"[warn] {model_name}: predictions empty.")
            continue
        pred_df['pred_date'] = pd.to_datetime(pred_df['pred_date'])
        pred_frames[model_name] = pred_df
        max_dates[model_name] = pred_df['pred_date'].max()

    if not pred_frames:
        raise RuntimeError("no predictions generated; check data and checkpoints")

    if pred_date_arg:
        target_date = pd.Timestamp(pred_date_arg).normalize()
    else:
        target_date = min(max_dates.values())
    date_str = target_date.strftime('%Y-%m-%d')

    os.makedirs(output_dir, exist_ok=True)
    artifacts: List[ModelArtifacts] = []

    for model_name, pred_df in pred_frames.items():
        day_df = pred_df[pred_df['pred_date'] == target_date].copy()
        if day_df.empty:
            print(f"[warn] {model_name}: no rows for pred_date {date_str}")
            continue
        day_df = day_df[['code', 'pred_date', 'pred_return']].sort_values('pred_return', ascending=False)
        out_path = os.path.join(output_dir, f"{model_name}_{date_str}.csv")
        day_df.to_csv(out_path, index=False)
        artifacts.append(ModelArtifacts(model_name, day_df, target_date, out_path))

    if not artifacts:
        raise RuntimeError("no per-model predictions for target date")

    return date_str, artifacts


def build_rank_sum(artifacts: List[ModelArtifacts], output_dir: str, date_str: str) -> str:
    merged = None
    for art in artifacts:
        df = art.pred_df.copy()
        df = df.rename(columns={'pred_return': f"pred_return_{art.model_name}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=['code', 'pred_date'], how='inner')

    if merged is None or merged.empty:
        raise RuntimeError("no common codes across models for rank-sum")

    rank_cols = []
    for art in artifacts:
        col = f"pred_return_{art.model_name}"
        rank_col = f"rank_{art.model_name}"
        merged[rank_col] = merged.groupby('pred_date')[col].rank(ascending=False, method='min')
        rank_cols.append(rank_col)

    merged['rank_sum'] = merged[rank_cols].sum(axis=1)
    merged['rank_sum_rank'] = merged.groupby('pred_date')['rank_sum'].rank(ascending=True, method='min')
    merged = merged.sort_values(['rank_sum', 'code']).reset_index(drop=True)

    output_path = os.path.join(output_dir, f"{date_str}.csv")
    merged.to_csv(output_path, index=False)
    return output_path


def _save_config(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train once on last N months and run daily rank-sum inference')
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'train_infer'], default='infer')
    parser.add_argument('--root_path', type=str, default='.', help='root path of data file')
    parser.add_argument('--data_path', type=str, default='stock_data.parquet', help='data file name')
    parser.add_argument('--models', type=str, default=DEFAULT_MODELS, help='comma-separated model list')
    parser.add_argument('--months', type=int, default=6, help='months for training window')
    parser.add_argument('--train_start', type=str, default=None, help='override train start date YYYY-MM-DD')
    parser.add_argument('--train_end', type=str, default=None, help='override train end date YYYY-MM-DD')
    parser.add_argument('--test_months', type=int, default=1, help='months for test split inside window')
    parser.add_argument('--val_months', type=int, default=1, help='months for val split inside window')
    parser.add_argument('--pred_date', type=str, default=None, help='force prediction date YYYY-MM-DD')
    parser.add_argument('--infer_months', type=int, default=6, help='months for inference window')
    parser.add_argument('--infer_start', type=str, default=None, help='override inference start date YYYY-MM-DD')

    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--label_len', type=int, default=1)
    parser.add_argument('--pred_len', type=int, default=2)
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='lag_return')
    parser.add_argument('--freq', type=str, default='b')

    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false')
    parser.add_argument('--use_amp', action='store_true', default=False)

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0')

    parser.add_argument('--inverse', action='store_true', default=True)

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/stock_last6m', help='checkpoint root')
    parser.add_argument('--output_dir', type=str, default='stock_results_daily', help='output directory')
    parser.add_argument('--config', type=str, default=None, help='path to saved train config')
    parser.add_argument('--resume', action='store_true', default=False, help='skip training if checkpoint exists')
    parser.add_argument('--tag', type=str, default='last6m', help='tag for config/checkpoints')

    parser.add_argument('--model_id', type=str, default='stock', help='model id')
    parser.add_argument('--des', type=str, default='daily', help='exp description')

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")

    args.checkpoints = os.path.join(args.checkpoints, args.tag)
    if args.config is None:
        args.config = os.path.join(args.checkpoints, 'train_config.json')

    models = [normalize_model_name(m) for m in args.models.split(',') if m.strip()]

    if args.mode in ('train', 'train_infer'):
        data_end = _latest_data_date(data_path)
        window = _compute_window(
            data_end=data_end,
            months=args.months,
            test_months=args.test_months,
            val_months=args.val_months,
            train_start=args.train_start,
            train_end=args.train_end,
        )

        window_payload = {
            'train_start': window['train_start'].strftime('%Y-%m-%d'),
            'train_end': window['train_end'].strftime('%Y-%m-%d'),
            'test_start': window['test_start'].strftime('%Y-%m-%d') if window['test_start'] <= window['test_end'] else None,
            'test_end': window['test_end'].strftime('%Y-%m-%d') if window['test_start'] <= window['test_end'] else None,
            'val_start': window['val_start'].strftime('%Y-%m-%d') if window['val_start'] <= window['val_end'] else None,
            'val_end': window['val_end'].strftime('%Y-%m-%d') if window['val_start'] <= window['val_end'] else None,
            'data_end': window['data_end'].strftime('%Y-%m-%d'),
            'months': args.months,
            'test_months': args.test_months,
            'val_months': args.val_months,
            'models': models,
            'seq_len': args.seq_len,
            'label_len': args.label_len,
            'pred_len': args.pred_len
        }
        _save_config(args.config, window_payload)
        print(f"[config] saved {args.config}")

        checkpoint_map = train_models(args, models, window, args.resume)
    else:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"config not found: {args.config}")
        cfg = _load_config(args.config)
        window = {
            'train_start': pd.Timestamp(cfg['train_start']),
            'train_end': pd.Timestamp(cfg['train_end']),
            'test_start': pd.Timestamp(cfg['test_start']) if cfg.get('test_start') else pd.Timestamp(cfg['train_end']),
            'test_end': pd.Timestamp(cfg['test_end']) if cfg.get('test_end') else pd.Timestamp(cfg['train_end']),
            'val_start': pd.Timestamp(cfg['val_start']) if cfg.get('val_start') else pd.Timestamp(cfg['train_end']),
            'val_end': pd.Timestamp(cfg['val_end']) if cfg.get('val_end') else pd.Timestamp(cfg['train_end']),
            'data_end': pd.Timestamp(cfg.get('data_end', cfg['train_end']))
        }
        checkpoint_map = {}

    if args.mode in ('infer', 'train_infer'):
        date_str, artifacts = infer_latest(
            args=args,
            models=models,
            window=window,
            checkpoint_map=checkpoint_map,
            output_dir=args.output_dir,
            pred_date_arg=args.pred_date
        )
        output_path = build_rank_sum(artifacts, args.output_dir, date_str)
        print(f"[output] rank-sum saved: {output_path}")


if __name__ == '__main__':
    main()
