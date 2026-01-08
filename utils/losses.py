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
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        pred = pred.reshape(-1).float()
        target = target.reshape(-1).float()

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

    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"ICLoss expects pred/target same shape, got {pred.shape} vs {target.shape}")

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


class WeightedICLoss(nn.Module):
    """Weighted Information Coefficient loss.

    Weights are computed from `target` to emphasize high-return samples (e.g., winners).
    Default uses softmax(beta * target) along `dim`.
    """

    def __init__(self, dim: int = -1, beta: float = 5.0, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.beta = float(beta)
        self.eps = eps

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise ValueError(f"WeightedICLoss expects pred/target same shape, got {pred.shape} vs {target.shape}")

        weights = t.softmax(self.beta * target.detach(), dim=self.dim)
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

    def forward(self, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        ic = self.ic_loss(pred, target)
        ccc = self.ccc_loss(pred, target)
        return self.ic_weight * ic + (1.0 - self.ic_weight) * ccc
