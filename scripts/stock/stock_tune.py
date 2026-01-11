import argparse
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import math

import optuna
from optuna.samplers import TPESampler

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


def suggest_transformer(trial, d_model_opts, n_head_opts, e_layer_opts, d_ff_opts):
    d_model = trial.suggest_categorical('d_model', d_model_opts)
    valid_heads = [h for h in n_head_opts if d_model % h == 0]
    n_heads = trial.suggest_categorical('n_heads', valid_heads)
    e_layers = trial.suggest_categorical('e_layers', e_layer_opts)
    d_ff = trial.suggest_categorical('d_ff', d_ff_opts)
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    return {
        'd_model': d_model,
        'n_heads': n_heads,
        'e_layers': e_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'learning_rate': lr
    }


def suggest_params(model_name, trial):
    if model_name == 'TimeXer':
        params = suggest_transformer(trial, [128, 256, 512], [4, 8], [2, 3], [256, 512, 1024])
        params['patch_len'] = trial.suggest_categorical('patch_len', [8, 16, 32])
        params['use_norm'] = trial.suggest_categorical('use_norm', [0, 1])
        return params
    if model_name == 'iTransformer':
        return suggest_transformer(trial, [128, 256, 512], [4, 8], [2, 3], [256, 512, 1024])
    if model_name == 'PatchTST':
        params = suggest_transformer(trial, [128, 256, 512], [4, 8], [2, 3], [256, 512, 1024])
        patch_len = trial.suggest_categorical('patch_len', [8, 16, 32])
        stride_opts = [s for s in [4, 8, 16] if s <= patch_len]
        params['patch_len'] = patch_len
        params['stride'] = trial.suggest_categorical('stride', stride_opts)
        return params
    if model_name == 'TSMixer':
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        return {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'e_layers': trial.suggest_categorical('e_layers', [2, 4, 6]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
            'learning_rate': lr
        }
    if model_name == 'DLinear':
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        return {
            'moving_avg': trial.suggest_categorical('moving_avg', [7, 25, 49]),
            'individual': trial.suggest_categorical('individual', [False, True]),
            'learning_rate': lr
        }
    if model_name == 'TimeMixer':
        params = suggest_transformer(trial, [128, 256, 512], [4, 8], [2, 3], [256, 512, 1024])
        params['moving_avg'] = trial.suggest_categorical('moving_avg', [7, 25])
        params['down_sampling_layers'] = trial.suggest_categorical('down_sampling_layers', [0, 1, 2])
        return params
    if model_name == 'FEDformer':
        params = suggest_transformer(trial, [128, 256], [4, 8], [2, 3], [256, 512])
        params['d_layers'] = trial.suggest_categorical('d_layers', [1, 2])
        params['moving_avg'] = trial.suggest_categorical('moving_avg', [7, 25])
        return params
    if model_name in ZERO_SHOT_MODELS:
        return {
            'seq_len': trial.suggest_categorical('seq_len', [32, 64, 96, 128]),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32])
        }
    return suggest_transformer(trial, [128, 256], [4, 8], [2, 3], [256, 512])


def parse_args():
    parser = argparse.ArgumentParser(description='Stock hyperparameter tuning')

    parser.add_argument('--root_path', type=str, default='.', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='stock_data.parquet', help='data file')
    parser.add_argument('--models', type=str, default=','.join(DEFAULT_MODELS), help='comma-separated model list')
    parser.add_argument('--model_id', type=str, default='stock', help='model id')
    parser.add_argument('--des', type=str, default='stock_tune', help='exp description')

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
    parser.add_argument('--tune_epochs', type=int, default=5, help='tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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

    parser.add_argument('--max_trials', type=int, default=20, help='max trials per model')
    parser.add_argument('--timeout', type=int, default=None, help='timeout seconds per model')
    parser.add_argument('--objective', type=str, default='annualized_return_pct',
                        choices=['annualized_return_pct', 'cumulative_return_pct', 'sharpe', 'final_capital'],
                        help='tuning objective')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--include_zero_shot', action='store_true', default=False, help='tune zero-shot models too')
    parser.add_argument('--run_benchmark_after_tune', action='store_true', default=False,
                        help='run benchmark after tuning with best params')
    parser.add_argument('--benchmark_tag', type=str, default='tuned', help='tag for benchmark output folder')
    parser.add_argument('--benchmark_wandb_group', type=str, default='stock_benchmark', help='wandb group for benchmark')

    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_group', type=str, default='stock_tune', help='wandb group')
    parser.add_argument('--wandb_mode', type=str, default=None, help='wandb mode')

    return parser.parse_args()


def sync_dependent_args(args):
    if getattr(args, 'model', None) == 'SegRNN':
        seg_len = math.gcd(args.seq_len, args.pred_len)
        if seg_len <= 0:
            seg_len = args.pred_len
        if args.seq_len % seg_len != 0 or args.pred_len % seg_len != 0:
            raise ValueError(
                f"SegRNN requires seg_len to divide seq_len and pred_len; "
                f"got seq_len={args.seq_len}, pred_len={args.pred_len}, seg_len={seg_len}"
            )
        if args.seg_len != seg_len:
            print(f"[config] SegRNN requires seg_len dividing seq_len/pred_len; setting seg_len={seg_len}")
        args.seg_len = seg_len
    else:
        args.seg_len = args.seq_len


def build_probe_dataset(args):
    return Dataset_Stock(
        args,
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


def run_benchmark_with_params(model_name, base_args, best_params, tag):
    args_run = copy.deepcopy(base_args)
    args_run.model = model_name
    args_run.task_name = 'zero_shot_forecast' if model_name in ZERO_SHOT_MODELS else 'long_term_forecast'
    for key, value in best_params.items():
        setattr(args_run, key, value)
    sync_dependent_args(args_run)

    if args_run.label_len >= args_run.seq_len:
        raise ValueError("label_len must be < seq_len")
    if getattr(args_run, 'patch_len', 0) > 0 and args_run.seq_len < args_run.patch_len:
        raise ValueError("seq_len must be >= patch_len")

    probe_dataset = build_probe_dataset(args_run)
    args_run.enc_in = probe_dataset.c_in
    args_run.dec_in = probe_dataset.c_in
    args_run.c_out = probe_dataset.c_in

    run_name = f"stock_benchmark_{model_name}_{tag}"
    init_wandb(args_run, name=run_name, group=args_run.benchmark_wandb_group, tags=[model_name, 'benchmark', tag])

    exp = Exp_Long_Term_Forecast(args_run)
    setting = build_setting(args_run)
    if model_name not in ZERO_SHOT_MODELS:
        exp.train(setting)

    result_dir = os.path.join('stock_results', model_name, tag)
    os.makedirs(result_dir, exist_ok=True)

    for split in ['test', 'val']:
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

    finish_wandb()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    args.individual = False
    args.channel_independence = 1
    args.decomp_method = 'moving_avg'
    args.down_sampling_layers = 0
    args.down_sampling_window = 1
    args.down_sampling_method = None
    args.seg_len = args.seq_len
    args.alpha = 0.1
    args.top_p = 0.5
    args.pos = 1

    models = [normalize_model_name(m) for m in args.models.split(',') if m.strip()]
    if not args.include_zero_shot:
        models = [m for m in models if m not in ZERO_SHOT_MODELS]

    for model_name in models:
        args_run = copy.deepcopy(args)
        args_run.model = model_name
        args_run.task_name = 'zero_shot_forecast' if model_name in ZERO_SHOT_MODELS else 'long_term_forecast'
        args_run.train_epochs = args_run.tune_epochs
        args_run.test_years = args_run.val_years

        run_name = f"stock_tune_{model_name}"
        init_wandb(args_run, name=run_name, group=args_run.wandb_group, tags=[model_name, 'tune'])

        def objective(trial):
            trial_args = copy.deepcopy(args_run)
            params = suggest_params(model_name, trial)
            for key, value in params.items():
                setattr(trial_args, key, value)
            sync_dependent_args(trial_args)

            if trial_args.label_len >= trial_args.seq_len:
                raise optuna.TrialPruned()
            if getattr(trial_args, 'patch_len', 0) > 0 and trial_args.seq_len < trial_args.patch_len:
                raise optuna.TrialPruned()

            probe_dataset = build_probe_dataset(trial_args)
            trial_args.enc_in = probe_dataset.c_in
            trial_args.dec_in = probe_dataset.c_in
            trial_args.c_out = probe_dataset.c_in

            setting = build_setting(trial_args)
            exp = Exp_Long_Term_Forecast(trial_args)
            if model_name not in ZERO_SHOT_MODELS:
                exp.train(setting)

            data_set, _ = data_provider(trial_args, 'val')
            data_loader = DataLoader(
                data_set,
                batch_size=trial_args.batch_size,
                shuffle=False,
                num_workers=trial_args.num_workers,
                drop_last=False
            )
            pred_df = predict_dataset(exp, data_set, data_loader, trial_args)
            metrics, _ = backtest_topk(
                pred_df,
                initial_cash=trial_args.initial_cash,
                topk=trial_args.topk,
                commission=trial_args.commission,
                stamp=trial_args.stamp,
                risk_free=trial_args.risk_free
            )
            score = float(metrics[trial_args.objective])
            trial.set_user_attr('metrics', metrics)
            log_wandb({f'tune/{trial_args.objective}': score, 'tune/trial': trial.number})
            if trial_args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif trial_args.gpu_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return score

        sampler = TPESampler(seed=args_run.seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=args_run.max_trials, timeout=args_run.timeout)

        result_dir = os.path.join('stock_tuning', model_name)
        os.makedirs(result_dir, exist_ok=True)
        best = {
            'model': model_name,
            'objective': args_run.objective,
            'best_value': study.best_value,
            'best_params': study.best_trial.params,
            'best_metrics': study.best_trial.user_attrs.get('metrics', {})
        }
        with open(os.path.join(result_dir, 'best_params.json'), 'w') as f:
            json.dump(best, f, indent=2)
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(result_dir, 'study.csv'), index=False)

        finish_wandb()

        if args.run_benchmark_after_tune:
            run_benchmark_with_params(model_name, args, best['best_params'], args.benchmark_tag)


if __name__ == '__main__':
    main()
