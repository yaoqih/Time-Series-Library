# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class CCCLoss(nn.Module):
    supports_mask = True

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        pred = pred.float()
        target = target.float()

        if mask is not None:
            mask = mask.float()
            if mask.shape != pred.shape:
                raise ValueError(f"CCCLoss expects mask same shape as pred/target, got {mask.shape} vs {pred.shape}")
            keep = mask.reshape(-1) > 0
            if not t.any(keep):
                return t.zeros((), device=pred.device, dtype=pred.dtype)
            pred = pred.reshape(-1)[keep]
            target = target.reshape(-1)[keep]
        else:
            pred = pred.reshape(-1)
            target = target.reshape(-1)

        mean_pred = t.mean(pred)
        mean_target = t.mean(target)

        pred_centered = pred - mean_pred
        target_centered = target - mean_target

        var_pred = t.mean(pred_centered ** 2)
        var_target = t.mean(target_centered ** 2)
        cov = t.mean(pred_centered * target_centered)

        mean_diff_sq = (mean_pred - mean_target) ** 2
        denom = var_pred + var_target + mean_diff_sq
        ccc = (2.0 * cov) / (denom + self.eps)
        ccc = t.where(denom < self.eps, t.ones_like(ccc), ccc)
        return 1.0 - ccc


class ICLoss(nn.Module):
    """Information Coefficient (Pearson corr) loss.

    Computes correlation along `dim` (e.g., cross-sectional dim) and returns `1 - mean(ic)`.
    """

    supports_mask = True

    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"ICLoss expects pred/target same shape, got {pred.shape} vs {target.shape}")

        if mask is None:
            mean_pred = pred.mean(dim=self.dim, keepdim=True)
            mean_target = target.mean(dim=self.dim, keepdim=True)
            pred_centered = pred - mean_pred
            target_centered = target - mean_target

            cov = (pred_centered * target_centered).mean(dim=self.dim)
            var_pred = (pred_centered ** 2).mean(dim=self.dim)
            var_target = (target_centered ** 2).mean(dim=self.dim)
            denom = (var_pred.sqrt() * var_target.sqrt()) + self.eps
            corr = cov / denom
            corr = t.where((var_pred < self.eps) | (var_target < self.eps), t.zeros_like(corr), corr)
            ic = corr.mean()
            return 1.0 - ic

        mask = mask.float()
        if mask.shape != pred.shape:
            raise ValueError(f"ICLoss expects mask same shape as pred/target, got {mask.shape} vs {pred.shape}")

        w = mask
        w_sum = w.sum(dim=self.dim, keepdim=True)
        w_norm = w / (w_sum + self.eps)

        mean_pred = (w_norm * pred).sum(dim=self.dim, keepdim=True)
        mean_target = (w_norm * target).sum(dim=self.dim, keepdim=True)
        pred_centered = pred - mean_pred
        target_centered = target - mean_target

        cov = (w_norm * pred_centered * target_centered).sum(dim=self.dim)
        var_pred = (w_norm * (pred_centered ** 2)).sum(dim=self.dim)
        var_target = (w_norm * (target_centered ** 2)).sum(dim=self.dim)
        denom = (var_pred.sqrt() * var_target.sqrt()) + self.eps
        corr = cov / denom
        corr = t.where((w_sum.squeeze(self.dim) < self.eps) | (var_pred < self.eps) | (var_target < self.eps), t.zeros_like(corr), corr)
        ic = corr.mean()
        return 1.0 - ic


class WeightedICLoss(nn.Module):
    """Weighted Information Coefficient loss.

    Weights are computed from `target` to emphasize high-return samples (e.g., winners).
    Default uses softmax(beta * target) along `dim`.
    """

    supports_mask = True

    def __init__(self, dim: int = -1, beta: float = 5.0, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.beta = float(beta)
        self.eps = eps

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"WeightedICLoss expects pred/target same shape, got {pred.shape} vs {target.shape}")

        weights = t.softmax(self.beta * target.detach(), dim=self.dim)
        if mask is not None:
            mask = mask.float()
            if mask.shape != pred.shape:
                raise ValueError(f"WeightedICLoss expects mask same shape as pred/target, got {mask.shape} vs {pred.shape}")
            weights = weights * mask
        weights = weights / (weights.sum(dim=self.dim, keepdim=True) + self.eps)

        mean_pred = (weights * pred).sum(dim=self.dim, keepdim=True)
        mean_target = (weights * target).sum(dim=self.dim, keepdim=True)
        pred_centered = pred - mean_pred
        target_centered = target - mean_target

        cov = (weights * pred_centered * target_centered).sum(dim=self.dim)
        var_pred = (weights * (pred_centered ** 2)).sum(dim=self.dim)
        var_target = (weights * (target_centered ** 2)).sum(dim=self.dim)
        denom = (var_pred.sqrt() * var_target.sqrt()) + self.eps
        corr = cov / denom
        corr = t.where((var_pred < self.eps) | (var_target < self.eps), t.zeros_like(corr), corr)

        ic = corr.mean()
        return 1.0 - ic


class HybridICCCLoss(nn.Module):
    """Hybrid loss: ic_weight * IC + (1-ic_weight) * CCC, both in loss form."""

    supports_mask = True

    def __init__(
        self,
        ic_weight: float = 0.7,
        ic_loss: nn.Module | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.ic_weight = float(ic_weight)
        self.ic_loss = ic_loss if ic_loss is not None else ICLoss(dim=-1, eps=eps)
        self.ccc_loss = CCCLoss(eps=eps)

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        if mask is not None and getattr(self.ic_loss, 'supports_mask', False):
            ic = self.ic_loss(pred, target, mask=mask)
        else:
            ic = self.ic_loss(pred, target)
        if mask is not None and getattr(self.ccc_loss, 'supports_mask', False):
            ccc = self.ccc_loss(pred, target, mask=mask)
        else:
            ccc = self.ccc_loss(pred, target)
        return self.ic_weight * ic + (1.0 - self.ic_weight) * ccc


class RiskAverseListNetLoss(nn.Module):
    """Risk-averse listwise ranking loss (ListNet + downside penalty).

    Designed for stock_pack cross-sectional training:
    - pred/target shape typically: [B, pred_len, n_codes]
    - computes listwise cross-entropy between softmax(target) and softmax(pred)
    - adds a downside penalty to discourage allocating score mass to below-mean targets
    """

    supports_mask = True

    def __init__(
        self,
        dim: int = -1,
        temperature: float = 10.0,
        downside_weight: float = 0.1,
        downside_gamma: float = 2.0,
        horizon_idx: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.downside_weight = float(downside_weight)
        self.downside_gamma = float(downside_gamma)
        self.horizon_idx = None if horizon_idx is None else int(horizon_idx)
        self.eps = float(eps)

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"RiskAverseListNetLoss expects pred/target same shape, got {pred.shape} vs {target.shape}")

        # Optionally focus on one trade horizon (e.g., trade_horizon=2 -> horizon_idx=1).
        if self.horizon_idx is not None and pred.ndim >= 3:
            idx = self.horizon_idx
            if 0 <= idx < pred.shape[1]:
                pred = pred[:, idx, ...]
                target = target[:, idx, ...]
                if mask is not None:
                    mask = mask[:, idx, ...]

        dim = self.dim if self.dim >= 0 else pred.ndim + self.dim
        if dim < 0 or dim >= pred.ndim:
            raise ValueError(f"invalid dim={self.dim} for pred.ndim={pred.ndim}")

        if mask is None:
            pred_centered = pred - pred.mean(dim=dim, keepdim=True)
            target_centered = target - target.mean(dim=dim, keepdim=True)
            pred_std = pred_centered.std(dim=dim, keepdim=True, unbiased=False)
            target_std = target_centered.std(dim=dim, keepdim=True, unbiased=False)

            pred_norm = pred_centered / (pred_std + self.eps)
            target_norm = target_centered / (target_std + self.eps)
            pred_norm = t.where(pred_std < self.eps, t.zeros_like(pred_norm), pred_norm)
            target_norm = t.where(target_std < self.eps, t.zeros_like(target_norm), target_norm)
            valid_any = None
        else:
            mask = mask.float()
            if mask.shape != pred.shape:
                raise ValueError(f"RiskAverseListNetLoss expects mask same shape as pred/target, got {mask.shape} vs {pred.shape}")
            w = mask
            w_sum = w.sum(dim=dim, keepdim=True)
            w_norm = w / (w_sum + self.eps)

            pred_mean = (w_norm * pred).sum(dim=dim, keepdim=True)
            target_mean = (w_norm * target).sum(dim=dim, keepdim=True)
            pred_centered = pred - pred_mean
            target_centered = target - target_mean

            pred_var = (w_norm * (pred_centered ** 2)).sum(dim=dim, keepdim=True)
            target_var = (w_norm * (target_centered ** 2)).sum(dim=dim, keepdim=True)
            pred_std = t.sqrt(pred_var + self.eps)
            target_std = t.sqrt(target_var + self.eps)

            pred_norm = pred_centered / pred_std
            target_norm = target_centered / target_std
            pred_norm = t.where(w_sum < self.eps, t.zeros_like(pred_norm), pred_norm)
            target_norm = t.where(w_sum < self.eps, t.zeros_like(target_norm), target_norm)
            valid_any = (w_sum.squeeze(dim) > self.eps)

        logits_pred = self.temperature * pred_norm
        logits_true = self.temperature * target_norm.detach()

        if mask is None:
            log_p_pred = t.log_softmax(logits_pred, dim=dim)
            p_true = t.softmax(logits_true, dim=dim)
            ce = -(p_true * log_p_pred).sum(dim=dim).mean()
            p_pred = t.softmax(logits_pred, dim=dim)
        else:
            # masked softmax: set invalid logits to a large negative and renormalize
            neg = t.finfo(logits_pred.dtype).min / 4
            logits_pred_m = t.where(mask > 0, logits_pred, neg)
            logits_true_m = t.where(mask > 0, logits_true, neg)

            log_p_pred = t.log_softmax(logits_pred_m, dim=dim)
            p_pred = t.softmax(logits_pred_m, dim=dim)
            p_true = t.softmax(logits_true_m, dim=dim)

            ce_vec = -(p_true * log_p_pred).sum(dim=dim)
            if valid_any is None:
                ce = ce_vec.mean()
            else:
                ce = t.where(valid_any, ce_vec, t.zeros_like(ce_vec))
                denom = valid_any.float().mean().clamp_min(self.eps)
                ce = ce.mean() / denom

        if self.downside_weight <= 0:
            return ce

        # Downside penalty: discourage giving high weight to below-mean targets.
        downside = t.relu(-target_norm.detach())
        if self.downside_gamma != 1.0:
            downside = downside ** self.downside_gamma
        if mask is not None:
            downside = downside * mask
        downside_pen_vec = (p_pred * downside).sum(dim=dim)
        if valid_any is None:
            downside_pen = downside_pen_vec.mean()
        else:
            downside_pen = t.where(valid_any, downside_pen_vec, t.zeros_like(downside_pen_vec))
            denom = valid_any.float().mean().clamp_min(self.eps)
            downside_pen = downside_pen.mean() / denom
        return ce + self.downside_weight * downside_pen


class Top1UtilityLoss(nn.Module):
    """Approximate top-1 selection utility loss (softmax-weighted return).

    Converts model outputs to cross-sectional weights via softmax(temperature * pred_norm),
    then maximizes expected return under those weights. Optional downside/variance penalties
    help reduce drawdowns and tail risk while keeping "top1" alignment.
    """

    supports_mask = True

    def __init__(
        self,
        dim: int = -1,
        temperature: float = 10.0,
        downside_weight: float = 0.0,
        downside_gamma: float = 2.0,
        var_weight: float = 0.0,
        horizon_idx: int | None = None,
        normalize_pred: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.downside_weight = float(downside_weight)
        self.downside_gamma = float(downside_gamma)
        self.var_weight = float(var_weight)
        self.horizon_idx = None if horizon_idx is None else int(horizon_idx)
        self.normalize_pred = bool(normalize_pred)
        self.eps = float(eps)

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"Top1UtilityLoss expects pred/target same shape, got {pred.shape} vs {target.shape}")

        # Optionally focus on one trade horizon (e.g., trade_horizon=2 -> horizon_idx=1).
        if self.horizon_idx is not None and pred.ndim >= 3:
            idx = self.horizon_idx
            if 0 <= idx < pred.shape[1]:
                pred = pred[:, idx, ...]
                target = target[:, idx, ...]
                if mask is not None:
                    mask = mask[:, idx, ...]

        dim = self.dim if self.dim >= 0 else pred.ndim + self.dim
        if dim < 0 or dim >= pred.ndim:
            raise ValueError(f"invalid dim={self.dim} for pred.ndim={pred.ndim}")

        valid_any = None
        if mask is None:
            if self.normalize_pred:
                pred_centered = pred - pred.mean(dim=dim, keepdim=True)
                pred_std = pred_centered.std(dim=dim, keepdim=True, unbiased=False)
                pred_norm = pred_centered / (pred_std + self.eps)
                pred_norm = t.where(pred_std < self.eps, t.zeros_like(pred_norm), pred_norm)
            else:
                pred_norm = pred
            logits = self.temperature * pred_norm
            w = t.softmax(logits, dim=dim)
        else:
            mask = mask.float()
            if mask.shape != pred.shape:
                raise ValueError(f"Top1UtilityLoss expects mask same shape as pred/target, got {mask.shape} vs {pred.shape}")
            w_sum = mask.sum(dim=dim, keepdim=True)
            valid_any = (w_sum.squeeze(dim) > self.eps)
            w_norm = mask / (w_sum + self.eps)

            if self.normalize_pred:
                pred_mean = (w_norm * pred).sum(dim=dim, keepdim=True)
                pred_centered = pred - pred_mean
                pred_var = (w_norm * (pred_centered ** 2)).sum(dim=dim, keepdim=True)
                pred_std = t.sqrt(pred_var + self.eps)
                pred_norm = pred_centered / pred_std
                pred_norm = t.where(w_sum < self.eps, t.zeros_like(pred_norm), pred_norm)
            else:
                pred_norm = t.where(w_sum < self.eps, t.zeros_like(pred), pred)

            logits = self.temperature * pred_norm
            neg = t.finfo(logits.dtype).min / 4
            logits_m = t.where(mask > 0, logits, neg)
            w = t.softmax(logits_m, dim=dim)

        exp_ret = (w * target).sum(dim=dim)
        utility = exp_ret

        if self.downside_weight > 0:
            downside = t.relu(-target)
            if self.downside_gamma != 1.0:
                downside = downside ** self.downside_gamma
            downside_pen = (w * downside).sum(dim=dim)
            utility = utility - self.downside_weight * downside_pen

        if self.var_weight > 0:
            diff = target - exp_ret.unsqueeze(dim)
            var_pen = (w * (diff ** 2)).sum(dim=dim)
            utility = utility - self.var_weight * var_pen

        loss_vec = -utility
        if valid_any is None:
            return loss_vec.mean()

        loss_vec = t.where(valid_any, loss_vec, t.zeros_like(loss_vec))
        denom = valid_any.float().mean().clamp_min(self.eps)
        return loss_vec.mean() / denom


class HybridRankUtilityLoss(nn.Module):
    """Hybrid loss for stock selection: dense listwise ranking + top1 utility alignment.

    Motivation:
    - Pure top1-style objectives are sparse/noisy (especially with large universes).
    - A listwise rank loss provides denser supervision to learn cross-sectional signal.
    - Utility term aligns training with the final top1 PnL objective (with optional downside control).
    """

    supports_mask = True

    def __init__(
        self,
        *,
        rank_loss: nn.Module,
        utility_loss: nn.Module,
        rank_weight: float = 0.7,
        utility_weight: float = 0.3,
        utility_warmup_epochs: int = 0,
        utility_ramp_epochs: int = 0,
        normalize_weights: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.rank_loss = rank_loss
        self.utility_loss = utility_loss
        self.rank_weight = float(rank_weight)
        self.utility_weight = float(utility_weight)
        self.utility_warmup_epochs = int(utility_warmup_epochs)
        self.utility_ramp_epochs = int(utility_ramp_epochs)
        self.normalize_weights = bool(normalize_weights)
        self.eps = float(eps)

        self._epoch = 0
        self._total_epochs = None

    def set_epoch(self, epoch: int, total_epochs: int | None = None):
        self._epoch = int(epoch)
        self._total_epochs = None if total_epochs is None else int(total_epochs)

    def _utility_scale(self) -> float:
        warmup = max(0, int(self.utility_warmup_epochs))
        ramp = max(0, int(self.utility_ramp_epochs))
        if self._epoch < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        # Start ramp immediately after warmup. At epoch==warmup => 1/ramp.
        progress = (self._epoch - warmup + 1) / float(ramp)
        if progress <= 0:
            return 0.0
        if progress >= 1:
            return 1.0
        return float(progress)

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        if mask is not None and getattr(self.rank_loss, 'supports_mask', False):
            rank_term = self.rank_loss(pred, target, mask=mask)
        else:
            rank_term = self.rank_loss(pred, target)

        if mask is not None and getattr(self.utility_loss, 'supports_mask', False):
            utility_term = self.utility_loss(pred, target, mask=mask)
        else:
            utility_term = self.utility_loss(pred, target)

        util_scale = self._utility_scale()
        rw = max(0.0, self.rank_weight)
        uw = max(0.0, self.utility_weight) * util_scale

        if not self.normalize_weights:
            return rw * rank_term + uw * utility_term

        denom = rw + uw
        if denom <= self.eps:
            return rank_term
        return (rw / denom) * rank_term + (uw / denom) * utility_term
