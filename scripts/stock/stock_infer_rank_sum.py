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


def _latest_time_ts(path: str) -> pd.Timestamp:
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
        raise ValueError("cannot infer latest timestamp from stock_data.parquet")
    return pd.Timestamp(latest)


def _years_spec(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start is None or end is None:
        return ''
    years = list(range(start.year, end.year + 1))
    if not years:
        return ''
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


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


def _time_features_from_dates(dates: pd.DatetimeIndex) -> np.ndarray:
    stamp_df = pd.DataFrame({'date': pd.to_datetime(dates)})
    stamp_df['month'] = stamp_df.date.dt.month
    stamp_df['day'] = stamp_df.date.dt.day
    stamp_df['weekday'] = stamp_df.date.dt.weekday
    return stamp_df.drop(['date'], axis=1).values


@dataclass
class ModelArtifacts:
    model_name: str
    pred_df: pd.DataFrame
    pred_date: pd.Timestamp
    output_path: str


def _load_checkpoint(exp: Exp_Long_Term_Forecast, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location=exp.device)
    exp.model.load_state_dict(state)


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _apply_infer_window(args, infer_start: pd.Timestamp, infer_end: pd.Timestamp):
    args.data = 'stock'
    args.task_name = 'long_term_forecast'
    args.test_start = infer_start.strftime('%Y-%m-%d')
    args.test_end = infer_end.strftime('%Y-%m-%d')
    args.test_years = _years_spec(infer_start, infer_end)
    if not hasattr(args, 'seasonal_patterns'):
        args.seasonal_patterns = None


def infer_latest(args, models: List[str], output_dir: str, pred_date_arg: Optional[str]) -> Tuple[str, List[ModelArtifacts]]:
    data_path = os.path.join(args.root_path, args.data_path)
    data_end = _latest_data_date(data_path)
    if args.infer_start:
        infer_start = pd.Timestamp(args.infer_start).normalize()
    else:
        infer_start = data_end - pd.DateOffset(months=args.infer_months) + pd.Timedelta(days=1)

    _apply_infer_window(args, infer_start, data_end)

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
    if not hasattr(args, 'channel_independence'):
        args.channel_independence = 1

    pred_frames: Dict[str, pd.DataFrame] = {}
    max_dates: Dict[str, pd.Timestamp] = {}

    for model_name in models:
        args_run = copy.deepcopy(args)
        args_run.model = model_name
        if not _model_available(model_name):
            print(f"[skip] {model_name}: model module missing or dependencies unavailable.")
            continue
        if model_name in ZERO_SHOT_MODELS:
            print(f"[skip] {model_name}: zero-shot model")
            continue
        _adjust_segrnn_seg_len(args_run)
        setting = build_setting(args_run)
        checkpoint_path = os.path.join(args_run.checkpoints, setting, 'checkpoint.pth')
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


def infer_latest_predict(args, models: List[str], output_dir: str) -> Tuple[str, List[ModelArtifacts]]:
    data_path = os.path.join(args.root_path, args.data_path)
    last_ts = _latest_time_ts(data_path)
    offset = pd.Timedelta(hours=args.time_offset_hours)
    last_date = (last_ts + offset).normalize()

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

    data_by_code = probe_dataset.data_by_code
    if not data_by_code:
        raise RuntimeError("no stock data loaded for prediction")

    future_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=args.pred_len)
    pred_date = pd.Timestamp(last_date).normalize()
    date_str = pred_date.strftime('%Y-%m-%d')

    os.makedirs(output_dir, exist_ok=True)
    artifacts: List[ModelArtifacts] = []

    for model_name in models:
        args_run = copy.deepcopy(args)
        args_run.model = model_name
        if not _model_available(model_name):
            print(f"[skip] {model_name}: model module missing or dependencies unavailable.")
            continue
        if model_name in ZERO_SHOT_MODELS:
            print(f"[skip] {model_name}: zero-shot model")
            continue
        _adjust_segrnn_seg_len(args_run)
        setting = build_setting(args_run)
        checkpoint_path = os.path.join(args_run.checkpoints, setting, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found for {model_name}: {checkpoint_path}")

        exp = Exp_Long_Term_Forecast(args_run)
        _load_checkpoint(exp, checkpoint_path)

        records = []
        batch_x = []
        batch_x_mark = []
        batch_dec_inp = []
        batch_dec_mark = []
        metas = []

        for code, payload in data_by_code.items():
            dates = pd.to_datetime(payload['dates'])
            if dates.size == 0:
                continue
            dates_norm = (dates + offset).normalize()
            idx = np.where(dates_norm == last_date)[0]
            if idx.size == 0:
                continue
            end_idx = int(idx[-1])
            s_end = end_idx + 1
            s_begin = s_end - args.seq_len
            if s_begin < 0:
                continue
            data = payload['data']
            stamp = payload['stamp']
            if s_end > len(data):
                continue

            seq_x = data[s_begin:s_end]
            seq_x_mark = stamp[s_begin:s_end]
            label = data[s_end - args.label_len:s_end]
            zeros = np.zeros((args.pred_len, data.shape[1]), dtype=label.dtype)
            dec_inp = np.concatenate([label, zeros], axis=0)
            label_mark = stamp[s_end - args.label_len:s_end]
            future_mark = _time_features_from_dates(future_dates)
            dec_mark = np.concatenate([label_mark, future_mark], axis=0)

            batch_x.append(seq_x)
            batch_x_mark.append(seq_x_mark)
            batch_dec_inp.append(dec_inp)
            batch_dec_mark.append(dec_mark)
            metas.append(code)

            if len(batch_x) >= args.batch_size:
                records.extend(_run_predict_batch(
                    exp, probe_dataset, args_run,
                    batch_x, batch_x_mark, batch_dec_inp, batch_dec_mark,
                    metas, pred_date
                ))
                batch_x, batch_x_mark, batch_dec_inp, batch_dec_mark, metas = [], [], [], [], []

        if batch_x:
            records.extend(_run_predict_batch(
                exp, probe_dataset, args_run,
                batch_x, batch_x_mark, batch_dec_inp, batch_dec_mark,
                metas, pred_date
            ))

        if not records:
            print(f"[warn] {model_name}: no prediction rows")
            continue

        pred_df = pd.DataFrame.from_records(records)
        out_path = os.path.join(output_dir, f"{model_name}_{date_str}.csv")
        pred_df.to_csv(out_path, index=False)
        artifacts.append(ModelArtifacts(model_name, pred_df, pred_date, out_path))

    if not artifacts:
        raise RuntimeError("no predictions generated for any model")

    return date_str, artifacts


def _run_predict_batch(exp, data_set, args, batch_x, batch_x_mark, batch_dec_inp, batch_dec_mark, metas, pred_date):
    batch_x = torch.tensor(np.asarray(batch_x), dtype=torch.float32).to(exp.device)
    batch_x_mark = torch.tensor(np.asarray(batch_x_mark), dtype=torch.float32).to(exp.device)
    dec_inp = torch.tensor(np.asarray(batch_dec_inp), dtype=torch.float32).to(exp.device)
    dec_mark = torch.tensor(np.asarray(batch_dec_mark), dtype=torch.float32).to(exp.device)

    exp.model.eval()
    with torch.no_grad():
        outputs = exp.model(batch_x, batch_x_mark, dec_inp, dec_mark)
        outputs = outputs[:, -args.pred_len:, :]
        outputs = outputs.detach().cpu().numpy()

        if data_set.scale and args.inverse:
            outputs = data_set.inverse_transform(outputs)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]
        pred_step = outputs[:, -1, -1]

    records = []
    for i, code in enumerate(metas):
        records.append({
            'code': code,
            'pred_date': pred_date,
            'pred_return': float(pred_step[i])
        })
    return records


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


def parse_args():
    parser = argparse.ArgumentParser(description='Daily inference and rank-sum from trained checkpoints')
    parser.add_argument('--config', type=str, required=True, help='path to train_config.json')
    parser.add_argument('--root_path', type=str, default=None, help='override data root path')
    parser.add_argument('--data_path', type=str, default=None, help='override data file name')
    parser.add_argument('--models', type=str, default=None, help='override model list')
    parser.add_argument('--infer_months', type=int, default=6, help='months for inference window')
    parser.add_argument('--infer_start', type=str, default=None, help='override inference start date YYYY-MM-DD')
    parser.add_argument('--pred_date', type=str, default=None, help='force prediction date YYYY-MM-DD')
    parser.add_argument('--infer_mode', type=str, default='backtest', choices=['backtest', 'predict'],
                        help='backtest uses true_return; predict uses latest history only')
    parser.add_argument('--time_offset_hours', type=int, default=8,
                        help='timezone offset hours to map time->trade date (e.g. 8 for China)')
    parser.add_argument('--output_dir', type=str, default='stock_results_daily', help='output directory')
    parser.add_argument('--batch_size', type=int, default=None, help='override batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='override num_workers')
    return parser.parse_args()


def main():
    args_cli = parse_args()
    if not os.path.exists(args_cli.config):
        raise FileNotFoundError(f"config not found: {args_cli.config}")

    cfg = _load_config(args_cli.config)
    train_args = cfg.get('train_args', {})

    args = argparse.Namespace(**train_args)
    args.checkpoints = train_args.get('checkpoints', args.checkpoints)

    if args_cli.root_path is not None:
        args.root_path = args_cli.root_path
    if args_cli.data_path is not None:
        args.data_path = args_cli.data_path
    if args_cli.batch_size is not None:
        args.batch_size = args_cli.batch_size
    if args_cli.num_workers is not None:
        args.num_workers = args_cli.num_workers

    args.infer_months = args_cli.infer_months
    args.infer_start = args_cli.infer_start
    args.pred_date = args_cli.pred_date
    args.infer_mode = args_cli.infer_mode
    args.time_offset_hours = args_cli.time_offset_hours
    args.output_dir = args_cli.output_dir

    if args_cli.models:
        models = [normalize_model_name(m) for m in args_cli.models.split(',') if m.strip()]
    else:
        models = cfg.get('models', [])

    if not models:
        raise ValueError("model list is empty")

    if 'train_start' in cfg and 'train_end' in cfg:
        args.train_start = cfg['train_start']
        args.train_end = cfg['train_end']
        args.val_start = cfg['train_start']
        args.val_end = cfg['train_end']
        args.train_years = _years_spec(pd.Timestamp(cfg['train_start']), pd.Timestamp(cfg['train_end']))
        args.val_years = args.train_years

    if args.infer_mode == 'predict':
        if args.pred_date:
            print(f"[warn] predict mode ignores --pred_date={args.pred_date}")
        date_str, artifacts = infer_latest_predict(
            args=args,
            models=models,
            output_dir=args.output_dir
        )
    else:
        date_str, artifacts = infer_latest(
            args=args,
            models=models,
            output_dir=args.output_dir,
            pred_date_arg=args.pred_date
        )
    output_path = build_rank_sum(artifacts, args.output_dir, date_str)
    print(f"[output] rank-sum saved: {output_path}")


if __name__ == '__main__':
    main()
