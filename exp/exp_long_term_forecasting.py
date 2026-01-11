from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.wandb_utils import log_wandb
from utils.losses import CCCLoss, ICLoss, WeightedICLoss, HybridICCCLoss, RiskAverseListNetLoss

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, None
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask
        raise ValueError(f"unexpected batch format (len={len(batch) if isinstance(batch, (list, tuple)) else 'n/a'})")

    def _slice_for_stock_pack(self, outputs, batch_y, data_set):
        if getattr(self.args, 'data', '') != 'stock' or not getattr(self.args, 'stock_pack', False):
            return None
        if not getattr(data_set, 'packed', False):
            return None
        target_slice = getattr(data_set, 'target_slice', None)
        if target_slice and isinstance(target_slice, (list, tuple)) and len(target_slice) == 2:
            start, end = target_slice
        else:
            n_codes = int(getattr(data_set, 'n_codes', 0) or len(getattr(data_set, 'universe_codes', []) or []))
            n_groups = int(getattr(data_set, 'n_groups', 0) or 1)
            start, end = (n_groups - 1) * n_codes, n_groups * n_codes
        start = int(start)
        end = int(end)
        if end <= start:
            raise ValueError("invalid target_slice for packed stock dataset")
        outputs = outputs[:, -self.args.pred_len:, start:end]
        batch_y = batch_y[:, -self.args.pred_len:, start:end]
        return outputs, batch_y

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'MSE') or 'MSE').strip().upper()
        if loss_name in {'MSE', 'L2'}:
            return nn.MSELoss()
        if loss_name in {'MAE', 'L1'}:
            return nn.L1Loss()
        if loss_name in {'CCC', 'CCCL'}:
            return CCCLoss()
        if loss_name in {'IC', 'ICL', 'ICLOSS'}:
            ic_dim = -1 if (getattr(self.args, 'data', '') == 'stock' and getattr(self.args, 'stock_pack', False)) else 0
            return ICLoss(dim=ic_dim)
        if loss_name in {'WIC', 'WICL', 'WEIGHTEDIC', 'WEIGHTED_IC', 'WEIGHTEDICLOSS'}:
            ic_dim = -1 if (getattr(self.args, 'data', '') == 'stock' and getattr(self.args, 'stock_pack', False)) else 0
            beta = float(getattr(self.args, 'ic_weight_beta', 5.0))
            return WeightedICLoss(dim=ic_dim, beta=beta)
        if loss_name in {'HYBRID', 'HYBRID_IC_CCC', 'IC_CCC'}:
            ic_dim = -1 if (getattr(self.args, 'data', '') == 'stock' and getattr(self.args, 'stock_pack', False)) else 0
            ic_weight = float(getattr(self.args, 'hybrid_ic_weight', 0.7))
            ic_weight = max(0.0, min(1.0, ic_weight))
            return HybridICCCLoss(ic_weight=ic_weight, ic_loss=ICLoss(dim=ic_dim))
        if loss_name in {'HYBRID_WIC', 'HYBRID_WIC_CCC', 'WIC_CCC'}:
            ic_dim = -1 if (getattr(self.args, 'data', '') == 'stock' and getattr(self.args, 'stock_pack', False)) else 0
            ic_weight = float(getattr(self.args, 'hybrid_ic_weight', 0.7))
            ic_weight = max(0.0, min(1.0, ic_weight))
            beta = float(getattr(self.args, 'ic_weight_beta', 5.0))
            return HybridICCCLoss(ic_weight=ic_weight, ic_loss=WeightedICLoss(dim=ic_dim, beta=beta))
        if loss_name in {'LISTNET', 'RA_LISTNET', 'RISK_LISTNET', 'RALN'}:
            dim = -1 if (getattr(self.args, 'data', '') == 'stock' and getattr(self.args, 'stock_pack', False)) else 0
            temp = float(getattr(self.args, 'ra_temperature', 10.0))
            down_w = float(getattr(self.args, 'ra_downside_weight', 0.1))
            down_g = float(getattr(self.args, 'ra_downside_gamma', 2.0))
            horizon = None
            if getattr(self.args, 'data', '') == 'stock':
                trade_horizon = getattr(self.args, 'trade_horizon', None)
                if trade_horizon is not None:
                    try:
                        horizon = int(trade_horizon) - 1
                    except Exception:
                        horizon = None
            return RiskAverseListNetLoss(dim=dim, temperature=temp, downside_weight=down_w, downside_gamma=down_g, horizon_idx=horizon)
        print(f"[warn] unknown loss '{loss_name}', fallback to MSE")
        return nn.MSELoss()
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if batch_y_mask is not None:
                    batch_y_mask = batch_y_mask.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                sliced = self._slice_for_stock_pack(outputs, batch_y, vali_data)
                if sliced is None:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if batch_y_mask is not None:
                        batch_y_mask = batch_y_mask[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    outputs, batch_y = sliced
                    batch_y = batch_y.to(self.device)
                    if batch_y_mask is not None:
                        mask_sliced = self._slice_for_stock_pack(batch_y_mask, batch_y_mask, vali_data)
                        batch_y_mask = (mask_sliced[0] if mask_sliced is not None else batch_y_mask).to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                if batch_y_mask is not None and getattr(criterion, 'supports_mask', False):
                    loss = criterion(pred, true, mask=batch_y_mask.detach())
                else:
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_y_mask is not None:
                    batch_y_mask = batch_y_mask.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        sliced = self._slice_for_stock_pack(outputs, batch_y, train_data)
                        if sliced is None:
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            if batch_y_mask is not None:
                                batch_y_mask = batch_y_mask[:, -self.args.pred_len:, f_dim:]
                        else:
                            outputs, batch_y = sliced
                            batch_y = batch_y.to(self.device)
                            if batch_y_mask is not None:
                                mask_sliced = self._slice_for_stock_pack(batch_y_mask, batch_y_mask, train_data)
                                batch_y_mask = mask_sliced[0] if mask_sliced is not None else batch_y_mask
                        if batch_y_mask is not None and getattr(criterion, 'supports_mask', False):
                            loss = criterion(outputs, batch_y, mask=batch_y_mask)
                        else:
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    sliced = self._slice_for_stock_pack(outputs, batch_y, train_data)
                    if sliced is None:
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if batch_y_mask is not None:
                            batch_y_mask = batch_y_mask[:, -self.args.pred_len:, f_dim:]
                    else:
                        outputs, batch_y = sliced
                        batch_y = batch_y.to(self.device)
                        if batch_y_mask is not None:
                            mask_sliced = self._slice_for_stock_pack(batch_y_mask, batch_y_mask, train_data)
                            batch_y_mask = mask_sliced[0] if mask_sliced is not None else batch_y_mask
                    if batch_y_mask is not None and getattr(criterion, 'supports_mask', False):
                        loss = criterion(outputs, batch_y, mask=batch_y_mask)
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if getattr(self.args, 'use_wandb', False):
                log_wandb({
                    'train/loss': train_loss,
                    'val/loss': vali_loss,
                    'test/loss': test_loss,
                    'epoch': epoch + 1
                }, step=epoch + 1)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
