import os
import hashlib
import pickle
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
try:
    from sktime.datasets import load_from_tsfile_to_dataframe
except ImportError:  # optional dependency (UEA datasets)
    load_from_tsfile_to_dataframe = None
import warnings
from utils.augmentation import run_augmentation_single
try:
    from datasets import load_dataset
except ImportError:  # optional dependency (HF datasets)
    load_dataset = None
try:
    from huggingface_hub import hf_hub_download
except ImportError:  # optional dependency (HF datasets)
    hf_hub_download = None
warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"
STOCK_CACHE_VERSION = "chaining_v2_clip30"
STOCK_BASE_CACHE_VERSION = "base_v2_clip30"
STOCK_PACKED_CACHE_VERSION = "packed_v2_union_multifeat"
STOCK_RETURN_LIMIT = 0.30
_STOCK_BASE_MEM_CACHE = {}


def parse_years_spec(spec):
    if spec is None:
        return set()
    if isinstance(spec, (list, tuple, set)):
        return {int(x) for x in spec}
    years = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return years


def _parse_date(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, errors='coerce')
    return None if pd.isna(ts) else ts


def parse_date_range(start, end):
    start_ts = _parse_date(start)
    end_ts = _parse_date(end)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"date range start {start_ts.date()} after end {end_ts.date()}")
    return start_ts, end_ts


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _stock_process_group(payload):
    (
        code,
        group,
        cols,
        scale,
        scaler,
        scaler_fitted,
        timeenc,
        freq,
        split_years_list,
        split_start,
        split_end,
        seq_len,
        pred_len,
        strict_pred_end,
    ) = payload

    group = group.dropna(subset=cols + ['datetime'])
    if group.empty:
        return None
    data = group[cols].values
    if scale and scaler_fitted:
        data = scaler.transform(data)
    dates = group['datetime'].values
    if timeenc == 0:
        stamp_df = pd.DataFrame({'date': pd.to_datetime(group['datetime'])})
        stamp_df['month'] = stamp_df.date.dt.month
        stamp_df['day'] = stamp_df.date.dt.day
        stamp_df['weekday'] = stamp_df.date.dt.weekday
        data_stamp = stamp_df.drop(['date'], axis=1).values
    else:
        data_stamp = time_features(pd.to_datetime(group['datetime'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)

    data_len = len(data)
    sample_index = []
    sample_meta = []
    if data_len >= seq_len + pred_len:
        end_idx = np.arange(seq_len - 1, data_len - pred_len)
        trade_idx = end_idx + 1
        trade_years = pd.DatetimeIndex(dates).year
        if split_years_list:
            mask = np.isin(trade_years[trade_idx], split_years_list)
            end_idx = end_idx[mask]
            trade_idx = trade_idx[mask]
        if split_start is not None or split_end is not None:
            trade_dates = pd.to_datetime(dates[trade_idx])
            date_mask = np.ones(len(trade_idx), dtype=bool)
            if split_start is not None:
                date_mask &= trade_dates >= split_start
            if split_end is not None:
                date_mask &= trade_dates <= split_end
            end_idx = end_idx[date_mask]
            trade_idx = trade_idx[date_mask]
        if end_idx.size > 0:
            pred_idx = end_idx + pred_len
            if strict_pred_end:
                pred_mask = np.ones(len(pred_idx), dtype=bool)
                if split_years_list:
                    pred_mask &= np.isin(trade_years[pred_idx], split_years_list)
                if split_start is not None or split_end is not None:
                    pred_dates = pd.to_datetime(dates[pred_idx])
                    if split_start is not None:
                        pred_mask &= pred_dates >= split_start
                    if split_end is not None:
                        pred_mask &= pred_dates <= split_end
                end_idx = end_idx[pred_mask]
                trade_idx = trade_idx[pred_mask]
                pred_idx = pred_idx[pred_mask]
            sample_index = [(code, int(idx)) for idx in end_idx]
            end_dates = dates[end_idx]
            trade_dates = dates[trade_idx]
            pred_dates = dates[pred_idx]
            suspend_flags = group['suspendFlag'].values[trade_idx]
            sample_meta = [
                {
                    'code': code,
                    'end_date': pd.Timestamp(end_dates[i]),
                    'trade_date': pd.Timestamp(trade_dates[i]),
                    'pred_date': pd.Timestamp(pred_dates[i]),
                    'suspendFlag': int(suspend_flags[i])
                }
                for i in range(len(end_idx))
            ]

    return code, data, data_stamp, dates, group['suspendFlag'].values, sample_index, sample_meta


def _stock_process_group_cached(payload):
    (
        code,
        data_all,
        dates_all,
        suspend_all,
        col_idx,
        scale,
        scaler,
        scaler_fitted,
        timeenc,
        freq,
        split_years_list,
        split_start,
        split_end,
        seq_len,
        pred_len,
        strict_pred_end,
    ) = payload

    if data_all is None or len(data_all) == 0:
        return None

    data = data_all[:, col_idx]
    valid = np.isfinite(data).all(axis=1)
    if isinstance(dates_all, np.ndarray):
        valid &= ~pd.isna(dates_all)
    data = data[valid]
    dates = dates_all[valid]
    suspend = suspend_all[valid] if suspend_all is not None else np.zeros(len(data), dtype=int)
    if data.size == 0:
        return None

    if scale and scaler_fitted:
        data = scaler.transform(data)

    if timeenc == 0:
        stamp_df = pd.DataFrame({'date': pd.to_datetime(dates)})
        stamp_df['month'] = stamp_df.date.dt.month
        stamp_df['day'] = stamp_df.date.dt.day
        stamp_df['weekday'] = stamp_df.date.dt.weekday
        data_stamp = stamp_df.drop(['date'], axis=1).values
    else:
        data_stamp = time_features(pd.to_datetime(dates), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)

    data_len = len(data)
    sample_index = []
    sample_meta = []
    if data_len >= seq_len + pred_len:
        end_idx = np.arange(seq_len - 1, data_len - pred_len)
        trade_idx = end_idx + 1
        trade_years = pd.DatetimeIndex(dates).year
        if split_years_list:
            mask = np.isin(trade_years[trade_idx], split_years_list)
            end_idx = end_idx[mask]
            trade_idx = trade_idx[mask]
        if split_start is not None or split_end is not None:
            trade_dates = pd.to_datetime(dates[trade_idx])
            date_mask = np.ones(len(trade_idx), dtype=bool)
            if split_start is not None:
                date_mask &= trade_dates >= split_start
            if split_end is not None:
                date_mask &= trade_dates <= split_end
            end_idx = end_idx[date_mask]
            trade_idx = trade_idx[date_mask]
        if end_idx.size > 0:
            pred_idx = end_idx + pred_len
            if strict_pred_end:
                pred_mask = np.ones(len(pred_idx), dtype=bool)
                if split_years_list:
                    pred_mask &= np.isin(trade_years[pred_idx], split_years_list)
                if split_start is not None or split_end is not None:
                    pred_dates = pd.to_datetime(dates[pred_idx])
                    if split_start is not None:
                        pred_mask &= pred_dates >= split_start
                    if split_end is not None:
                        pred_mask &= pred_dates <= split_end
                end_idx = end_idx[pred_mask]
                trade_idx = trade_idx[pred_mask]
                pred_idx = pred_idx[pred_mask]
            sample_index = [(code, int(idx)) for idx in end_idx]
            end_dates = dates[end_idx]
            trade_dates = dates[trade_idx]
            pred_dates = dates[pred_idx]
            suspend_flags = suspend[trade_idx] if suspend is not None else np.zeros(len(trade_idx), dtype=int)
            sample_meta = [
                {
                    'code': code,
                    'end_date': pd.Timestamp(end_dates[i]),
                    'trade_date': pd.Timestamp(trade_dates[i]),
                    'pred_date': pd.Timestamp(pred_dates[i]),
                    'suspendFlag': int(suspend_flags[i])
                }
                for i in range(len(end_idx))
            ]

    return code, data, data_stamp, dates, suspend, sample_index, sample_meta

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path  = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label  = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path  = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path  = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)
            
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path   = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data   = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data  = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class Dataset_Stock(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='stock_data.parquet',
                 target='lag_return', scale=True, timeenc=0, freq='b', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 64
            self.label_len = 1
            self.pred_len = 2
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.train_years = parse_years_spec(getattr(args, 'train_years', '2014-2023'))
        self.test_years = parse_years_spec(getattr(args, 'test_years', '2024'))
        self.val_years = parse_years_spec(getattr(args, 'val_years', '2025'))
        self.train_date_range = parse_date_range(
            getattr(args, 'train_start', None),
            getattr(args, 'train_end', None)
        )
        self.test_date_range = parse_date_range(
            getattr(args, 'test_start', None),
            getattr(args, 'test_end', None)
        )
        self.val_date_range = parse_date_range(
            getattr(args, 'val_start', None),
            getattr(args, 'val_end', None)
        )
        self.strict_pred_end = _parse_bool(getattr(args, 'stock_strict_pred_end', None), default=True)
        self.sample_index = []
        self.sample_meta = []
        self.data_by_code = {}
        self.feature_cols = []
        self.c_in = 0
        self.__read_data__()

    def _build_datetime(self, df, time_col):
        time_series = df[time_col]
        return pd.to_datetime(time_series, unit='ms', errors='coerce')

    def _select_split_years(self):
        if self.flag == 'train':
            return self.train_years
        if self.flag == 'test':
            return self.test_years
        return self.val_years

    def _select_split_date_range(self):
        if self.flag == 'train':
            return self.train_date_range
        if self.flag == 'test':
            return self.test_date_range
        return self.val_date_range

    def _cache_key(self, data_path):
        key_data = {
            'data_path': os.path.abspath(data_path),
            'mtime': os.path.getmtime(data_path),
            'cache_version': STOCK_CACHE_VERSION,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'flag': self.flag,
            'features': self.features,
            'target': self.target,
            'scale': self.scale,
            'timeenc': self.timeenc,
            'freq': self.freq,
            'train_years': sorted(self.train_years),
            'test_years': sorted(self.test_years),
            'val_years': sorted(self.val_years),
            'train_date_range': tuple(str(v) for v in self.train_date_range),
            'test_date_range': tuple(str(v) for v in self.test_date_range),
            'val_date_range': tuple(str(v) for v in self.val_date_range),
            'strict_pred_end': bool(self.strict_pred_end),
        }
        raw = str(sorted(key_data.items())).encode('utf-8')
        return hashlib.md5(raw).hexdigest()

    def _base_cache_key(self, data_path):
        key_data = {
            'data_path': os.path.abspath(data_path),
            'mtime': os.path.getmtime(data_path),
            'target': self.target,
            'cache_version': STOCK_BASE_CACHE_VERSION
        }
        raw = str(sorted(key_data.items())).encode('utf-8')
        return hashlib.md5(raw).hexdigest()

    def _cache_path(self, data_path):
        cache_dir = getattr(self.args, 'stock_cache_dir', './cache')
        return os.path.join(cache_dir, f"stock_{self._cache_key(data_path)}.pkl")

    def _base_cache_path(self, data_path):
        cache_dir = getattr(self.args, 'stock_cache_dir', './cache')
        return os.path.join(cache_dir, f"stock_base_{self._base_cache_key(data_path)}.pkl")

    def _load_cache(self, cache_fp):
        try:
            with open(cache_fp, 'rb') as f:
                payload = pickle.load(f)
        except Exception:
            return False
        self.feature_cols = payload['feature_cols']
        self.c_in = payload['c_in']
        self.data_by_code = payload['data_by_code']
        self.sample_index = payload['sample_index']
        self.sample_meta = payload['sample_meta']
        self.scaler = payload['scaler']
        self._scaler_fitted = payload.get('scaler_fitted', True)
        return True

    def _save_cache(self, cache_fp):
        cache_dir = os.path.dirname(cache_fp)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        payload = {
            'feature_cols': self.feature_cols,
            'c_in': self.c_in,
            'data_by_code': self.data_by_code,
            'sample_index': self.sample_index,
            'sample_meta': self.sample_meta,
            'scaler': self.scaler,
            'scaler_fitted': self._scaler_fitted
        }
        with open(cache_fp, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_base_cache(self, cache_fp):
        try:
            with open(cache_fp, 'rb') as f:
                payload = pickle.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get('version') != STOCK_BASE_CACHE_VERSION:
            return None
        return payload

    def _save_base_cache(self, cache_fp, payload):
        cache_dir = os.path.dirname(cache_fp)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_fp, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_base_cache(self, data_path):
        try:
            read_columns = None
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(data_path)
                available = set(pf.schema.names)
                base_cols = {
                    'open', 'high', 'low', 'close', 'volume', 'amount',
                    'settle', 'openInterest', 'preClose', 'suspendFlag'
                }
                wanted = {'code', 'time', self.target} | base_cols
                if self.target in {'lag_return', 'lag_return_rank', 'lag_return_cs_rank'}:
                    wanted |= {'open', 'close', 'preClose'}
                read_columns = [c for c in wanted if c in available]
            except Exception:
                read_columns = None
            df_raw = pd.read_parquet(data_path, columns=read_columns)
        except Exception as exc:
            raise ImportError("pandas.read_parquet requires pyarrow or fastparquet") from exc

        if 'code' not in df_raw.columns:
            raise ValueError("stock data must include 'code' column")
        if 'time' not in df_raw.columns:
            raise ValueError("stock data must include 'time' column with ms timestamp")

        df_raw = df_raw.copy()
        df_raw['datetime'] = self._build_datetime(df_raw, 'time')
        df_raw = df_raw.dropna(subset=['datetime', 'code'])
        df_raw = df_raw.sort_values(['code', 'datetime'])

        if 'open' not in df_raw.columns:
            raise ValueError("stock data must include 'open' column")
        if self.target in {'lag_return', 'lag_return_rank', 'lag_return_cs_rank'}:
            if 'close' not in df_raw.columns:
                raise ValueError("stock data must include 'close' column for lag_return chaining")
            if 'preClose' not in df_raw.columns:
                raise ValueError("stock data must include 'preClose' column for lag_return chaining")
            prev_open = df_raw.groupby('code')['open'].shift(1)
            prev_close = df_raw.groupby('code')['close'].shift(1)
            pre_close = df_raw['preClose']
            open_px = df_raw['open']
            with np.errstate(divide='ignore', invalid='ignore'):
                intraday = prev_close / prev_open
                overnight = open_px / pre_close
                lag_return = intraday * overnight - 1
            valid = (prev_open > 0) & (prev_close > 0) & (pre_close > 0) & (open_px > 0)
            if 'suspendFlag' in df_raw.columns:
                valid &= df_raw['suspendFlag'] == 0
            if STOCK_RETURN_LIMIT is not None:
                valid &= np.abs(lag_return) <= STOCK_RETURN_LIMIT
            df_raw['lag_return'] = lag_return.where(valid)
            if self.target != 'lag_return':
                df_raw[self.target] = df_raw.groupby('datetime')['lag_return'].rank(pct=True)

        if self.target not in df_raw.columns:
            raise ValueError(f"target '{self.target}' not found in stock data")
        if 'suspendFlag' not in df_raw.columns:
            df_raw['suspendFlag'] = 0

        base_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'settle', 'openInterest', 'preClose', 'suspendFlag'
        ]
        base_feature_cols = [col for col in base_cols if col in df_raw.columns]
        columns = base_feature_cols + ([self.target] if self.target not in base_feature_cols else [])
        if not columns:
            raise ValueError("no usable stock columns found for base cache")

        data_by_code = {}
        for code, group in df_raw.groupby('code', sort=True):
            data_by_code[code] = {
                'data': group[columns].values.astype(np.float32, copy=False),
                'dates': group['datetime'].values,
                'suspend': group['suspendFlag'].values.astype(int, copy=False)
            }

        return {
            'version': STOCK_BASE_CACHE_VERSION,
            'columns': columns,
            'data_by_code': data_by_code
        }

    def __read_data__(self):
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        local_fp = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(local_fp):
            raise FileNotFoundError(f"stock data not found: {local_fp}")
        use_cache = getattr(self.args, 'use_stock_cache', True)
        if getattr(self.args, 'disable_stock_cache', False):
            use_cache = False
        workers = int(getattr(self.args, 'stock_preprocess_workers', 0) or 0)
        write_cache = use_cache and workers <= 1
        cache_fp = self._cache_path(local_fp)
        if use_cache and os.path.exists(cache_fp):
            if self._load_cache(cache_fp):
                return
        base_payload = None
        base_key = None
        base_fp = None
        if use_cache:
            base_key = self._base_cache_key(local_fp)
            base_payload = _STOCK_BASE_MEM_CACHE.get(base_key)
            if base_payload is None:
                base_fp = self._base_cache_path(local_fp)
                if os.path.exists(base_fp):
                    base_payload = self._load_base_cache(base_fp)
            if base_payload is not None:
                _STOCK_BASE_MEM_CACHE[base_key] = base_payload

        if base_payload is None:
            base_payload = self._build_base_cache(local_fp)
            if use_cache:
                if base_key is None:
                    base_key = self._base_cache_key(local_fp)
                _STOCK_BASE_MEM_CACHE[base_key] = base_payload
                if base_fp is None:
                    base_fp = self._base_cache_path(local_fp)
                self._save_base_cache(base_fp, base_payload)

        base_columns = base_payload['columns']
        base_data_by_code = base_payload['data_by_code']
        base_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'settle', 'openInterest', 'preClose', 'suspendFlag'
        ]
        feature_cols = [col for col in base_cols if col in base_columns and col != self.target]
        if self.features == 'S':
            cols = [self.target]
        else:
            if not feature_cols:
                raise ValueError("no feature columns found in stock data")
            cols = feature_cols + [self.target]
        self.feature_cols = cols
        self.c_in = len(self.feature_cols)
        self.target_idx = len(self.feature_cols) - 1
        col_idx = [base_columns.index(c) for c in cols]

        if use_cache and workers > 1:
            print("[cache] base cache enabled; using single-process preprocessing.")
            workers = 0
            write_cache = use_cache

        if self.scale:
            for payload in base_data_by_code.values():
                data_all = payload['data'][:, col_idx]
                dates_all = payload['dates']
                valid = np.isfinite(data_all).all(axis=1)
                if isinstance(dates_all, np.ndarray):
                    valid &= ~pd.isna(dates_all)
                if not valid.any():
                    continue
                data = data_all[valid]
                dates = dates_all[valid]
                if self.train_years:
                    train_mask = pd.DatetimeIndex(dates).year.isin(self.train_years)
                else:
                    train_mask = np.ones(len(dates), dtype=bool)
                train_start, train_end = self.train_date_range
                if train_start is not None:
                    train_mask &= dates >= train_start
                if train_end is not None:
                    train_mask &= dates <= train_end
                if train_mask.any():
                    self.scaler.partial_fit(data[train_mask])
                    self._scaler_fitted = True

        split_years = self._select_split_years()
        split_start, split_end = self._select_split_date_range()
        split_years_list = sorted(split_years)

        if workers > 1:
            ctx = mp.get_context("spawn")
            payloads = [
                (
                    code,
                    payload['data'],
                    payload['dates'],
                    payload.get('suspend'),
                    col_idx,
                    self.scale,
                    self.scaler,
                    self._scaler_fitted,
                    self.timeenc,
                    self.freq,
                    split_years_list,
                    split_start,
                    split_end,
                    self.seq_len,
                    self.pred_len,
                    bool(self.strict_pred_end),
                )
                for code, payload in base_data_by_code.items()
            ]
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(_stock_process_group_cached, payloads)
            for result in results:
                if result is None:
                    continue
                code, data, stamp, dates, suspend, sample_index, sample_meta = result
                self.data_by_code[code] = {
                    'data': data,
                    'stamp': stamp,
                    'dates': dates,
                    'suspend': suspend
                }
                self.sample_index.extend(sample_index)
                self.sample_meta.extend(sample_meta)
        else:
            for code, payload in base_data_by_code.items():
                result = _stock_process_group_cached((
                    code,
                    payload['data'],
                    payload['dates'],
                    payload.get('suspend'),
                    col_idx,
                    self.scale,
                    self.scaler,
                    self._scaler_fitted,
                    self.timeenc,
                    self.freq,
                    split_years_list,
                    split_start,
                    split_end,
                    self.seq_len,
                    self.pred_len,
                    bool(self.strict_pred_end),
                ))
                if result is None:
                    continue
                code, data, stamp, dates, suspend, sample_index, sample_meta = result
                self.data_by_code[code] = {
                    'data': data,
                    'stamp': stamp,
                    'dates': dates,
                    'suspend': suspend
                }
                self.sample_index.extend(sample_index)
                self.sample_meta.extend(sample_meta)

        if write_cache:
            self._save_cache(cache_fp)

    def __getitem__(self, index):
        code, end_idx = self.sample_index[index]
        payload = self.data_by_code[code]
        data = payload['data']
        stamp = payload['stamp']
        s_end = end_idx + 1
        s_begin = s_end - self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data[s_begin:s_end]
        seq_y = data[r_begin:r_end]
        seq_x_mark = stamp[s_begin:s_end]
        seq_y_mark = stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_index)

    def inverse_transform(self, data):
        if not self.scale or not getattr(self, '_scaler_fitted', False):
            return data
        n_features = getattr(self.scaler, 'n_features_in_', data.shape[-1])
        if data.shape[-1] == n_features:
            shape = data.shape
            data = data.reshape(-1, shape[-1])
            data = self.scaler.inverse_transform(data)
            return data.reshape(shape)
        if data.shape[-1] == 1 and hasattr(self, 'target_idx'):
            mean = self.scaler.mean_[self.target_idx]
            scale = self.scaler.scale_[self.target_idx]
            return data * scale + mean
        return data


class Dataset_StockPacked(Dataset):
    """Pack all stocks into one huge tensor (channels = stocks) to learn cross-stock signals.

    Supports:
    - features='S': target-only, channels = stocks
    - features!='S': multi-feature, channels = feature_groups * stocks, where groups follow base feature order + target
      and the last block is always the target (size = stocks), making it easy to slice for IC/CCC losses.
    """

    def __init__(
        self,
        args,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='stock_data.parquet',
        target='lag_return',
        scale=True,
        timeenc=0,
        freq='b',
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 64
            self.label_len = 1
            self.pred_len = 2
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.train_years = parse_years_spec(getattr(args, 'train_years', '2014-2023'))
        self.test_years = parse_years_spec(getattr(args, 'test_years', '2024'))
        self.val_years = parse_years_spec(getattr(args, 'val_years', '2025'))
        self.train_date_range = parse_date_range(
            getattr(args, 'train_start', None),
            getattr(args, 'train_end', None)
        )
        self.test_date_range = parse_date_range(
            getattr(args, 'test_start', None),
            getattr(args, 'test_end', None)
        )
        self.val_date_range = parse_date_range(
            getattr(args, 'val_start', None),
            getattr(args, 'val_end', None)
        )
        self.pack_date_range = parse_date_range(
            getattr(args, 'stock_pack_start', None),
            getattr(args, 'stock_pack_end', None)
        )
        self.strict_pred_end = _parse_bool(getattr(args, 'stock_strict_pred_end', None), default=True)
        self.sample_index = []
        self.sample_meta = []
        self.feature_cols = []
        self.pack_feature_cols = []
        self.universe_codes = []
        self.c_in = 0
        self.n_codes = 0
        self.n_groups = 0
        self.target_slice = None
        self.packed = True
        self.__read_data__()

    def _build_datetime(self, df, time_col):
        time_series = df[time_col]
        return pd.to_datetime(time_series, unit='ms', errors='coerce')

    def _select_split_years(self):
        if self.flag == 'train':
            return self.train_years
        if self.flag == 'test':
            return self.test_years
        return self.val_years

    def _select_split_date_range(self):
        if self.flag == 'train':
            return self.train_date_range
        if self.flag == 'test':
            return self.test_date_range
        return self.val_date_range

    def _base_cache_key(self, data_path):
        key_data = {
            'data_path': os.path.abspath(data_path),
            'mtime': os.path.getmtime(data_path),
            'target': self.target,
            'cache_version': STOCK_BASE_CACHE_VERSION
        }
        raw = str(sorted(key_data.items())).encode('utf-8')
        return hashlib.md5(raw).hexdigest()

    def _packed_cache_key(self, data_path):
        pack_start, pack_end = self.pack_date_range
        key_data = {
            'data_path': os.path.abspath(data_path),
            'mtime': os.path.getmtime(data_path),
            'cache_version': STOCK_PACKED_CACHE_VERSION,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'flag': self.flag,
            'features': self.features,
            'target': self.target,
            'scale': self.scale,
            'timeenc': self.timeenc,
            'freq': self.freq,
            'train_years': sorted(self.train_years),
            'test_years': sorted(self.test_years),
            'val_years': sorted(self.val_years),
            'train_date_range': tuple(str(v) for v in self.train_date_range),
            'test_date_range': tuple(str(v) for v in self.test_date_range),
            'val_date_range': tuple(str(v) for v in self.val_date_range),
            'pack_start': str(pack_start),
            'pack_end': str(pack_end),
            'universe_size': int(getattr(self.args, 'stock_universe_size', 0) or 0),
            'fill_value': float(getattr(self.args, 'stock_pack_fill_value', 0.0)),
            'strict_pred_end': bool(self.strict_pred_end),
        }
        raw = str(sorted(key_data.items())).encode('utf-8')
        return hashlib.md5(raw).hexdigest()

    def _base_cache_path(self, data_path):
        cache_dir = getattr(self.args, 'stock_cache_dir', './cache')
        return os.path.join(cache_dir, f"stock_base_{self._base_cache_key(data_path)}.pkl")

    def _packed_cache_path(self, data_path):
        cache_dir = getattr(self.args, 'stock_cache_dir', './cache')
        return os.path.join(cache_dir, f"stock_packed_{self._packed_cache_key(data_path)}.pkl")

    def _load_pkl(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_pkl(self, path, payload):
        cache_dir = os.path.dirname(path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_base_cache(self, cache_fp):
        payload = self._load_pkl(cache_fp)
        if not isinstance(payload, dict):
            return None
        if payload.get('version') != STOCK_BASE_CACHE_VERSION:
            return None
        return payload

    def _build_base_cache(self, data_path):
        try:
            read_columns = None
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(data_path)
                available = set(pf.schema.names)
                base_cols = {
                    'open', 'high', 'low', 'close', 'volume', 'amount',
                    'settle', 'openInterest', 'preClose', 'suspendFlag'
                }
                wanted = {'code', 'time', self.target} | base_cols
                if self.target in {'lag_return', 'lag_return_rank', 'lag_return_cs_rank'}:
                    wanted |= {'open', 'close', 'preClose'}
                read_columns = [c for c in wanted if c in available]
            except Exception:
                read_columns = None
            df_raw = pd.read_parquet(data_path, columns=read_columns)
        except Exception as exc:
            raise ImportError("pandas.read_parquet requires pyarrow or fastparquet") from exc

        if 'code' not in df_raw.columns:
            raise ValueError("stock data must include 'code' column")
        if 'time' not in df_raw.columns:
            raise ValueError("stock data must include 'time' column with ms timestamp")

        df_raw = df_raw.copy()
        df_raw['datetime'] = self._build_datetime(df_raw, 'time')
        df_raw = df_raw.dropna(subset=['datetime', 'code'])
        df_raw = df_raw.sort_values(['code', 'datetime'])

        if 'open' not in df_raw.columns:
            raise ValueError("stock data must include 'open' column")
        if self.target in {'lag_return', 'lag_return_rank', 'lag_return_cs_rank'}:
            if 'close' not in df_raw.columns:
                raise ValueError("stock data must include 'close' column for lag_return chaining")
            if 'preClose' not in df_raw.columns:
                raise ValueError("stock data must include 'preClose' column for lag_return chaining")
            prev_open = df_raw.groupby('code')['open'].shift(1)
            prev_close = df_raw.groupby('code')['close'].shift(1)
            pre_close = df_raw['preClose']
            open_px = df_raw['open']
            with np.errstate(divide='ignore', invalid='ignore'):
                intraday = prev_close / prev_open
                overnight = open_px / pre_close
                lag_return = intraday * overnight - 1
            valid = (prev_open > 0) & (prev_close > 0) & (pre_close > 0) & (open_px > 0)
            if 'suspendFlag' in df_raw.columns:
                valid &= df_raw['suspendFlag'] == 0
            if STOCK_RETURN_LIMIT is not None:
                valid &= np.abs(lag_return) <= STOCK_RETURN_LIMIT
            df_raw['lag_return'] = lag_return.where(valid)
            if self.target != 'lag_return':
                df_raw[self.target] = df_raw.groupby('datetime')['lag_return'].rank(pct=True)

        if self.target not in df_raw.columns:
            raise ValueError(f"target '{self.target}' not found in stock data")
        if 'suspendFlag' not in df_raw.columns:
            df_raw['suspendFlag'] = 0

        base_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'settle', 'openInterest', 'preClose', 'suspendFlag'
        ]
        base_feature_cols = [col for col in base_cols if col in df_raw.columns]
        columns = base_feature_cols + ([self.target] if self.target not in base_feature_cols else [])
        if not columns:
            raise ValueError("no usable stock columns found for base cache")

        data_by_code = {}
        for code, group in df_raw.groupby('code', sort=True):
            data_by_code[code] = {
                'data': group[columns].values.astype(np.float32, copy=False),
                'dates': group['datetime'].values,
                'suspend': group['suspendFlag'].values.astype(int, copy=False),
            }

        return {
            'version': STOCK_BASE_CACHE_VERSION,
            'columns': columns,
            'data_by_code': data_by_code,
        }

    def _read_base_payload(self, local_fp, use_cache: bool):
        base_key = self._base_cache_key(local_fp)
        base_payload = _STOCK_BASE_MEM_CACHE.get(base_key)
        base_fp = None
        if base_payload is None and use_cache:
            base_fp = self._base_cache_path(local_fp)
            if os.path.exists(base_fp):
                base_payload = self._load_base_cache(base_fp)
        if base_payload is None:
            base_payload = self._build_base_cache(local_fp)
            if use_cache:
                if base_fp is None:
                    base_fp = self._base_cache_path(local_fp)
                self._save_pkl(base_fp, base_payload)
        if use_cache:
            _STOCK_BASE_MEM_CACHE[base_key] = base_payload
        return base_payload

    def __read_data__(self):
        local_fp = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(local_fp):
            raise FileNotFoundError(f"stock data not found: {local_fp}")

        use_cache = getattr(self.args, 'use_stock_cache', True)
        if getattr(self.args, 'disable_stock_cache', False):
            use_cache = False

        cache_fp = self._packed_cache_path(local_fp)
        if use_cache and os.path.exists(cache_fp):
            payload = self._load_pkl(cache_fp)
            if isinstance(payload, dict) and payload.get('version') == STOCK_PACKED_CACHE_VERSION:
                self.feature_cols = payload['feature_cols']
                self.pack_feature_cols = payload.get('pack_feature_cols', payload.get('feature_cols', []))
                self.c_in = payload['c_in']
                self.universe_codes = payload['universe_codes']
                self.dates = payload['dates']
                self.data = payload['data']
                self.open = payload.get('open')
                self.suspend = payload.get('suspend')
                self.stamp = payload['stamp']
                self.sample_index = payload['sample_index']
                self.sample_meta = payload['sample_meta']
                self.scaler = payload.get('scaler', StandardScaler())
                self._scaler_fitted = payload.get('scaler_fitted', False)
                self.n_codes = int(payload.get('n_codes', len(self.universe_codes)))
                self.n_groups = int(payload.get('n_groups', 1))
                self.target_slice = payload.get('target_slice')
                return

        base_payload = self._read_base_payload(local_fp, use_cache=use_cache)
        base_columns = base_payload['columns']
        base_data_by_code = base_payload['data_by_code']

        if self.target not in base_columns:
            raise ValueError(f"target '{self.target}' not in cached base columns")
        if 'open' not in base_columns:
            raise ValueError("stock base cache missing 'open' column")

        base_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'settle', 'openInterest', 'preClose', 'suspendFlag'
        ]
        base_feature_cols = [col for col in base_cols if col in base_columns]

        if self.features == 'S':
            pack_feature_cols = [self.target]
        else:
            pack_feature_cols = [col for col in base_feature_cols if col != self.target]
            if self.target not in pack_feature_cols:
                pack_feature_cols.append(self.target)

        self.pack_feature_cols = pack_feature_cols
        self.feature_cols = pack_feature_cols
        col_idx = [base_columns.index(c) for c in pack_feature_cols]
        open_col_idx = base_columns.index('open')

        pack_start, pack_end = self.pack_date_range
        if pack_start is None:
            pack_start = _parse_date(getattr(self.args, 'train_start', None))
        if pack_end is None:
            pack_end = _parse_date(getattr(self.args, 'val_end', None)) or _parse_date(getattr(self.args, 'test_end', None))
        if pack_start is None or pack_end is None:
            raise ValueError("stock_pack_start/stock_pack_end (or train_start/val_end) must be set for packed stock data")
        if pack_start > pack_end:
            raise ValueError("stock_pack_start must be <= stock_pack_end")
        self.pack_date_range = (pack_start, pack_end)

        universe_size = int(getattr(self.args, 'stock_universe_size', 0) or 0)
        codes = sorted(base_data_by_code.keys())
        selected = []
        for code in codes:
            payload = base_data_by_code.get(code)
            if payload is None:
                continue
            dates = payload.get('dates')
            if dates is None or len(dates) == 0:
                continue
            start_dt = pd.Timestamp(dates[0])
            end_dt = pd.Timestamp(dates[-1])
            if start_dt > pack_start or end_dt < pack_end:
                continue
            selected.append(code)
            if universe_size > 0 and len(selected) >= universe_size:
                break
        if not selected:
            raise ValueError("no stock codes satisfy pack date coverage; try smaller pack range or set stock_universe_size")
        self.universe_codes = selected
        self.n_codes = len(selected)
        self.n_groups = len(pack_feature_cols)
        self.c_in = self.n_codes * self.n_groups
        self.target_slice = ((self.n_groups - 1) * self.n_codes, self.n_groups * self.n_codes)

        self.scaler = StandardScaler()
        self._scaler_fitted = False
        if self.scale:
            for code in selected:
                payload = base_data_by_code[code]
                data_all = payload['data'][:, col_idx]
                dates_all = payload['dates']
                valid = np.isfinite(data_all).all(axis=1)
                if isinstance(dates_all, np.ndarray):
                    valid &= ~pd.isna(dates_all)
                if not valid.any():
                    continue
                data = data_all[valid]
                dates = dates_all[valid]
                if self.train_years:
                    train_mask = pd.DatetimeIndex(dates).year.isin(self.train_years)
                else:
                    train_mask = np.ones(len(dates), dtype=bool)
                train_start, train_end = self.train_date_range
                if train_start is not None:
                    train_mask &= dates >= train_start
                if train_end is not None:
                    train_mask &= dates <= train_end
                if train_mask.any():
                    self.scaler.partial_fit(data[train_mask])
                    self._scaler_fitted = True

        fill_value = float(getattr(self.args, 'stock_pack_fill_value', 0.0))

        # Use a union calendar to avoid empty intersections caused by suspensions/missing targets.
        calendar = None
        for code in selected:
            payload = base_data_by_code[code]
            dates_all = payload.get('dates')
            if dates_all is None or len(dates_all) == 0:
                continue
            range_mask = (~pd.isna(dates_all)) & (dates_all >= pack_start) & (dates_all <= pack_end)
            dates = dates_all[range_mask]
            if len(dates) == 0:
                continue
            calendar = dates if calendar is None else np.union1d(calendar, dates)

        if calendar is None or len(calendar) < (self.seq_len + self.pred_len + 1):
            raise ValueError(
                f"packed calendar too short after union: {0 if calendar is None else len(calendar)}; "
                f"need at least {self.seq_len + self.pred_len + 1}"
            )

        common_dates = np.sort(calendar)
        t_len = len(common_dates)
        n_codes = len(selected)
        n_groups = len(pack_feature_cols)

        data_mat = np.full((t_len, n_groups * n_codes), fill_value, dtype=np.float32)
        open_mat = np.zeros((t_len, n_codes), dtype=np.float32)
        suspend_mat = np.ones((t_len, n_codes), dtype=np.int16)

        mean = getattr(self.scaler, 'mean_', None) if self._scaler_fitted else None
        scale = getattr(self.scaler, 'scale_', None) if self._scaler_fitted else None

        for j, code in enumerate(selected):
            payload = base_data_by_code[code]
            data_all = payload.get('data')
            dates_all = payload.get('dates')
            suspend_all = payload.get('suspend')
            if data_all is None or len(data_all) == 0 or dates_all is None:
                continue

            range_mask = (~pd.isna(dates_all)) & (dates_all >= pack_start) & (dates_all <= pack_end)
            if not range_mask.any():
                continue
            dates = dates_all[range_mask]
            data = data_all[range_mask]
            suspend = suspend_all[range_mask] if suspend_all is not None else np.zeros(len(dates), dtype=int)

            idx = np.searchsorted(common_dates, dates)
            if idx.max() >= len(common_dates):
                raise ValueError("date alignment failed (index out of bounds)")
            if not np.array_equal(common_dates[idx], dates):
                raise ValueError("date alignment failed (mismatched dates)")

            feat = data[:, col_idx].astype(np.float32, copy=False)
            if self.scale and mean is not None and scale is not None:
                for f in range(n_groups):
                    x = feat[:, f]
                    valid = np.isfinite(x)
                    if not valid.any():
                        continue
                    feat[valid, f] = (x[valid] - float(mean[f])) / (float(scale[f]) + 1e-8)

            feat = np.where(np.isfinite(feat), feat, fill_value).astype(np.float32, copy=False)
            for f in range(n_groups):
                data_mat[idx, f * n_codes + j] = feat[:, f]

            open_mat[idx, j] = data[:, open_col_idx].astype(np.float32, copy=False)
            suspend_mat[idx, j] = suspend.astype(np.int16, copy=False)

        if self.timeenc == 0:
            stamp_df = pd.DataFrame({'date': pd.to_datetime(common_dates)})
            stamp_df['month'] = stamp_df.date.dt.month
            stamp_df['day'] = stamp_df.date.dt.day
            stamp_df['weekday'] = stamp_df.date.dt.weekday
            stamp = stamp_df.drop(['date'], axis=1).values.astype(np.float32)
        else:
            stamp = time_features(pd.to_datetime(common_dates), freq=self.freq)
            stamp = stamp.transpose(1, 0).astype(np.float32)

        self.dates = common_dates
        self.data = data_mat
        self.open = open_mat
        self.suspend = suspend_mat
        self.stamp = stamp

        split_years = self._select_split_years()
        split_start, split_end = self._select_split_date_range()
        split_years_list = sorted(split_years)

        data_len = len(self.data)
        sample_index = []
        sample_meta = []
        if data_len >= self.seq_len + self.pred_len:
            end_idx = np.arange(self.seq_len - 1, data_len - self.pred_len)
            trade_idx = end_idx + 1
            trade_years = pd.DatetimeIndex(self.dates).year
            if split_years_list:
                mask = np.isin(trade_years[trade_idx], split_years_list)
                end_idx = end_idx[mask]
                trade_idx = trade_idx[mask]
            if split_start is not None or split_end is not None:
                trade_dates = pd.to_datetime(self.dates[trade_idx])
                date_mask = np.ones(len(trade_idx), dtype=bool)
                if split_start is not None:
                    date_mask &= trade_dates >= split_start
                if split_end is not None:
                    date_mask &= trade_dates <= split_end
                end_idx = end_idx[date_mask]
                trade_idx = trade_idx[date_mask]
            if end_idx.size > 0:
                pred_idx = end_idx + self.pred_len
                if self.strict_pred_end:
                    pred_mask = np.ones(len(pred_idx), dtype=bool)
                    if split_years_list:
                        pred_mask &= np.isin(trade_years[pred_idx], split_years_list)
                    if split_start is not None or split_end is not None:
                        pred_dates = pd.to_datetime(self.dates[pred_idx])
                        if split_start is not None:
                            pred_mask &= pred_dates >= split_start
                        if split_end is not None:
                            pred_mask &= pred_dates <= split_end
                    end_idx = end_idx[pred_mask]
                    trade_idx = trade_idx[pred_mask]
                    pred_idx = pred_idx[pred_mask]
                sample_index = [int(i) for i in end_idx]
                end_dates = self.dates[end_idx]
                trade_dates = self.dates[trade_idx]
                pred_dates = self.dates[pred_idx]
                sample_meta = [
                    {
                        'end_date': pd.Timestamp(end_dates[i]),
                        'trade_date': pd.Timestamp(trade_dates[i]),
                        'pred_date': pd.Timestamp(pred_dates[i]),
                    }
                    for i in range(len(end_idx))
                ]
        self.sample_index = sample_index
        self.sample_meta = sample_meta

        if use_cache:
            payload = {
                'version': STOCK_PACKED_CACHE_VERSION,
                'feature_cols': self.feature_cols,
                'pack_feature_cols': self.pack_feature_cols,
                'c_in': self.c_in,
                'universe_codes': self.universe_codes,
                'n_codes': self.n_codes,
                'n_groups': self.n_groups,
                'target_slice': self.target_slice,
                'dates': self.dates,
                'data': self.data,
                'open': self.open,
                'suspend': self.suspend,
                'stamp': self.stamp,
                'sample_index': self.sample_index,
                'sample_meta': self.sample_meta,
                'scaler': self.scaler,
                'scaler_fitted': self._scaler_fitted,
            }
            self._save_pkl(cache_fp, payload)

    def __getitem__(self, index):
        end_idx = self.sample_index[index]
        s_end = end_idx + 1
        s_begin = s_end - self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.stamp[s_begin:s_end]
        seq_y_mark = self.stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_index)

    def inverse_transform(self, data):
        if not self.scale or not getattr(self, '_scaler_fitted', False):
            return data
        mean = getattr(self.scaler, 'mean_', None)
        scale = getattr(self.scaler, 'scale_', None)
        if mean is None or scale is None:
            return data

        n_codes = int(getattr(self, 'n_codes', 0) or len(getattr(self, 'universe_codes', []) or []))
        n_groups = int(getattr(self, 'n_groups', 0) or 1)
        if n_codes <= 0 or n_groups <= 0:
            return data

        if data.shape[-1] == n_groups * n_codes:
            out = data.copy()
            for f in range(n_groups):
                mu = float(mean[f])
                sd = float(scale[f]) + 1e-8
                start = f * n_codes
                end = (f + 1) * n_codes
                out[..., start:end] = out[..., start:end] * sd + mu
            return out

        # Backwards compat: target-only
        if data.shape[-1] == n_codes and len(mean) >= 1:
            mu = float(mean[-1 if len(mean) > 1 else 0])
            sd = float(scale[-1 if len(scale) > 1 else 0]) + 1e-8
            return data * sd + mu
        return data
