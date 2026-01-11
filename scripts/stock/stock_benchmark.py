import argparse
import copy
import json
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import math
import models as models_pkg

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Stock
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.backtest import backtest_topk
from utils.wandb_utils import init_wandb, log_wandb, finish_wandb

ZERO_SHOT_MODELS = {'Chronos2', 'TiRex', 'TimeMoE'}
DEFAULT_MODELS = [
    'TimeXer', 'iTransformer', 'PatchTST', 'TSMixer', 'DLinear',
    'TimeMixer', 'FEDformer', 'Chronos2', 'TiRex', 'TimeMoE'
]
ZERO_SHOT_MODELS = {
    'Chronos', 'Chronos2', 'Moirai', 'Sundial', 'TiRex', 'TimeMoE', 'TimesFM'
}
DEFAULT_MODELS = [
    'Autoformer', 'Transformer', 'Nonstationary_Transformer',
    'DLinear', 'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer',
    'PatchTST', 'Pyraformer', 'MICN', 'Crossformer', 'FiLM', 'iTransformer',
    'TiDE', 'FreTS', 'TimeMixer', 'TSMixer', 'SegRNN',
    'SCINet', 'PAttn', 'TimeXer', 'WPMixer',
    'MultiPatchFormer', 'TimeFilter', 'Sundial', 'TimeMoE',
    'Chronos', 'Moirai', 'TiRex', 'TimesFM', 'Chronos2','KANAD',  'TemporalFusionTransformer',
]
High_DEFAULT_MODELS = [
    'TimesNet','Koopa','MSGNet', "MambaSimple"
]
select=['TimeMixer','iTransformer','WPMixer','FreTS','PathchTST']

def build_setting(args, ii=0):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii
    )


def normalize_model_name(name):
    return name.replace('-', '').replace(' ', '')


def _model_available(model_name):
    module = getattr(models_pkg, model_name, None)
    return module is not None and hasattr(module, "Model")


def _adjust_segrnn_seg_len(args_run):
    if args_run.model != 'SegRNN':
        return
    # SegRNN requires seg_len to divide both seq_len and pred_len.
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


def predict_dataset(exp, data_set, data_loader, args):
    exp.model.eval()
    records = []
    offset = 0
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
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
    parser = argparse.ArgumentParser(description='Stock benchmark runner')

    parser.add_argument('--root_path', type=str, default='.', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='stock_data.parquet', help='data file')
    parser.add_argument('--models', type=str, default=','.join(DEFAULT_MODELS), help='comma-separated model list')
    parser.add_argument('--model_id', type=str, default='stock', help='model id')
    parser.add_argument('--des', type=str, default='stock', help='exp description')

    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='lag_return', help='target feature')
    parser.add_argument('--freq', type=str, default='b', help='freq for time features encoding')

    parser.add_argument('--train_years', type=str, default='2014-2023', help='train years')
    parser.add_argument('--test_years', type=str, default='2024', help='test years')
    parser.add_argument('--val_years', type=str, default='2025', help='val years')

    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
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

    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default='stock', help='wandb group')
    parser.add_argument('--wandb_mode', type=str, default=None, help='wandb mode')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='skip training/eval if outputs already exist')

    return parser.parse_args()


def results_complete(result_dir, split):
    split_dir = os.path.join(result_dir, split)
    required = ['predictions.csv', 'equity_curve.csv', 'metrics.json']
    return all(os.path.exists(os.path.join(split_dir, fname)) for fname in required)


def main():
    args = parse_args()
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    if args.use_gpu and args.gpu_type == 'cuda' and not torch.cuda.is_available():
        args.use_gpu = False
    if args.use_gpu and args.gpu_type == 'mps':
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            args.use_gpu = False
    if args.pred_len < 2:
        raise ValueError("pred_len must be >= 2 to forecast next-next open return")

    args.task_name = 'long_term_forecast'
    args.data = 'stock'
    args.checkpoints = './checkpoints/'
    args.seasonal_patterns = 'Monthly'
    args.mask_rate = 0.25
    args.anomaly_ratio = 0.25
    # keep CLI-provided defaults when present
    args.channel_independence = 1
    args.decomp_method = 'moving_avg'
    args.down_sampling_window = 1
    args.down_sampling_method = None
    args.seg_len = args.seq_len
    args.alpha = 0.1
    args.top_p = 0.5
    args.pos = 1

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

    models = [normalize_model_name(m) for m in args.models.split(',') if m.strip()]
    failures = []
    for model_name in models:
        args_run = copy.deepcopy(args)
        args_run.model = model_name
        if not _model_available(model_name):
            print(f"[skip] {model_name}: model module missing or dependencies unavailable.")
            continue
        if model_name == 'ETSformer' and args_run.e_layers != args_run.d_layers:
            print(f"[config] ETSformer requires e_layers == d_layers; "
                  f"setting d_layers={args_run.e_layers}")
            args_run.d_layers = args_run.e_layers
        _adjust_segrnn_seg_len(args_run)
        args_run.task_name = 'zero_shot_forecast' if model_name in ZERO_SHOT_MODELS else 'long_term_forecast'

        setting = build_setting(args_run)
        checkpoint_path = os.path.join(args_run.checkpoints, setting, 'checkpoint.pth')
        result_dir = os.path.join('stock_results', model_name)
        split_done = {split: results_complete(result_dir, split) for split in ['test', 'val']}
        needs_eval = any(not done for done in split_done.values())
        checkpoint_exists = os.path.exists(checkpoint_path)
        needs_train = model_name not in ZERO_SHOT_MODELS and (not args_run.resume or not checkpoint_exists)

        if args_run.resume and not needs_train and not needs_eval:
            print(f"[resume] skip {model_name}: checkpoint and results already exist.")
            continue

        run_name = f"stock_{model_name}"
        init_wandb(args_run, name=run_name, group=args_run.wandb_group, tags=[model_name])
        try:
            exp = Exp_Long_Term_Forecast(args_run)
            if model_name not in ZERO_SHOT_MODELS:
                if needs_train:
                    exp.train(setting)
                else:
                    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))

            os.makedirs(result_dir, exist_ok=True)

            for split in ['test', 'val']:
                if args_run.resume and split_done[split]:
                    print(f"[resume] skip {model_name} {split}: results already exist.")
                    continue
                data_set, _ = data_provider(args_run, split)
                data_loader = DataLoader(
                    data_set,
                    batch_size=args_run.batch_size,
                    shuffle=False,
                    num_workers=args_run.num_workers,
                    drop_last=False
                )
                pred_df = predict_dataset(exp, data_set, data_loader, args_run)
                metrics, curve_df = backtest_topk(
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
                with open(os.path.join(split_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)

                log_wandb({f'{split}/{k}': v for k, v in metrics.items()})
        except Exception as exc:
            failures.append((model_name, repr(exc)))
            print(f"[error] {model_name}: {exc}")
        finally:
            finish_wandb()

    if failures:
        print("[summary] failed models:")
        for model_name, err in failures:
            print(f" - {model_name}: {err}")


if __name__ == '__main__':
    main()
