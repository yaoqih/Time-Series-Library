import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import models as models_pkg
from data_provider.data_loader import Dataset_Stock
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


ZERO_SHOT_MODELS = {
    'Chronos', 'Chronos2', 'Moirai', 'Sundial', 'TiRex', 'TimeMoE', 'TimesFM'
}
DEFAULT_MODELS = 'FreTS,PatchTST,WPMixer,iTransformer'


def _parse_device_list(devices):
    if devices is None:
        return []
    devices = str(devices).replace(' ', '')
    if not devices:
        return []
    return [int(d) for d in devices.split(',') if d != '']


def _normalize_multi_gpu_args(args):
    devices = _parse_device_list(args.devices)
    if not devices:
        return
    args.devices = ",".join(str(d) for d in devices)
    args.device_ids = list(range(len(devices)))
    args.gpu = 0


def _seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _years_spec(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start is None or end is None:
        return ''
    years = list(range(start.year, end.year + 1))
    if not years:
        return ''
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def _set_full_window(args, train_start: pd.Timestamp, train_end: pd.Timestamp):
    args.data = 'stock'
    args.task_name = 'long_term_forecast'
    args.train_start = train_start.strftime('%Y-%m-%d')
    args.train_end = train_end.strftime('%Y-%m-%d')
    args.test_start = args.train_start
    args.test_end = args.train_end
    args.val_start = args.train_start
    args.val_end = args.train_end
    years = _years_spec(train_start, train_end)
    args.train_years = years
    args.test_years = years
    args.val_years = years


def _save_config(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description='Train models on last N months without test/val split')
    parser.add_argument('--root_path', type=str, default='.', help='root path of data file')
    parser.add_argument('--data_path', type=str, default='stock_data.parquet', help='data file name')
    parser.add_argument('--models', type=str, default=DEFAULT_MODELS, help='comma-separated model list')
    parser.add_argument('--months', type=int, default=6, help='months for training window')
    parser.add_argument('--train_start', type=str, default=None, help='override train start date YYYY-MM-DD')
    parser.add_argument('--train_end', type=str, default=None, help='override train end date YYYY-MM-DD')

    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--label_len', type=int, default=1)
    parser.add_argument('--pred_len', type=int, default=2)
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='lag_return')
    parser.add_argument('--freq', type=str, default='b')
    parser.add_argument('--seasonal_patterns', type=str, default=None)

    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--augmentation_ratio', type=int, default=0)

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
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='series decomposition method for some models')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    parser.add_argument('--seg_len', type=int, default=96)
    parser.add_argument('--mask_rate', type=float, default=0.25)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--pos', type=int, default=1)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0')

    parser.add_argument('--inverse', action='store_true', default=True)
    parser.add_argument('--initial_cash', type=float, default=1_000_000.0)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--commission', type=float, default=0.0003)
    parser.add_argument('--stamp', type=float, default=0.001)
    parser.add_argument('--risk_free', type=float, default=0.03)
    parser.add_argument('--stock_cache_dir', type=str, default='./cache')
    parser.add_argument('--disable_stock_cache', action='store_true', default=False)
    parser.add_argument('--stock_preprocess_workers', type=int, default=0)

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/stock_last6m', help='checkpoint root')
    parser.add_argument('--tag', type=str, default='last6m', help='checkpoint tag')
    parser.add_argument('--config', type=str, default=None, help='path to save train config')
    parser.add_argument('--resume', action='store_true', default=False, help='skip training if checkpoint exists')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='tslib-roll')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default='stock_rolling')
    parser.add_argument('--wandb_mode', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2021)

    parser.add_argument('--model_id', type=str, default='stock', help='model id')
    parser.add_argument('--des', type=str, default='daily', help='exp description')

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")

    if args.no_wandb:
        args.use_wandb = False
    _seed_everything(args.seed)
    if args.use_gpu and args.gpu_type == 'cuda' and not torch.cuda.is_available():
        args.use_gpu = False
    if args.use_gpu and args.gpu_type == 'mps':
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            args.use_gpu = False
    if args.use_multi_gpu:
        _normalize_multi_gpu_args(args)
    if args.pred_len < 2:
        raise ValueError("pred_len must be >= 2 to forecast next-next open return")

    args.checkpoints = os.path.join(args.checkpoints, args.tag)
    if args.config is None:
        args.config = os.path.join(args.checkpoints, 'train_config.json')

    if args.seasonal_patterns is None:
        args.seasonal_patterns = 'Monthly'
    args.channel_independence = getattr(args, 'channel_independence', 1)
    args.decomp_method = getattr(args, 'decomp_method', 'moving_avg')
    args.down_sampling_window = getattr(args, 'down_sampling_window', 1)
    args.down_sampling_method = getattr(args, 'down_sampling_method', None)
    args.seg_len = args.seq_len
    args.mask_rate = getattr(args, 'mask_rate', 0.25)
    args.anomaly_ratio = getattr(args, 'anomaly_ratio', 0.25)
    args.alpha = getattr(args, 'alpha', 0.1)
    args.top_p = getattr(args, 'top_p', 0.5)
    args.pos = getattr(args, 'pos', 1)

    data_end = _latest_data_date(data_path)
    train_end = pd.Timestamp(args.train_end).normalize() if args.train_end else data_end
    train_start = pd.Timestamp(args.train_start).normalize() if args.train_start else (
        train_end - pd.DateOffset(months=args.months) + pd.Timedelta(days=1)
    )
    if train_start > train_end:
        raise ValueError("train_start must be <= train_end")

    _set_full_window(args, train_start, train_end)

    models = [normalize_model_name(m) for m in args.models.split(',') if m.strip()]

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
        if args.resume and os.path.exists(checkpoint_path):
            print(f"[resume] {model_name}: checkpoint exists, skipping training.")
            continue

        exp = Exp_Long_Term_Forecast(args_run)
        exp.train(setting)

    cfg = {
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_end': train_end.strftime('%Y-%m-%d'),
        'months': args.months,
        'models': models,
        'train_args': vars(args)
    }
    _save_config(args.config, cfg)
    print(f"[config] saved {args.config}")


if __name__ == '__main__':
    main()
