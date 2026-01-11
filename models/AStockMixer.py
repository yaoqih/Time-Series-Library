import torch
import torch.nn as nn


class _TemporalMixerBlock(nn.Module):
    """TSMixer-style block, but operating on per-stock embeddings.

    Input: [B, N, L, D]
      - N: number of stocks (codes)
      - L: seq_len
      - D: d_model
    """

    def __init__(self, seq_len, d_model, d_ff, dropout):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, L, D]
        x = x + self.temporal(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.norm1(x)
        x = x + self.channel(x)
        x = self.norm2(x)
        return x


class _ISABlock(nn.Module):
    """Induced Set Attention Block (Set Transformer style) for cross-sectional mixing.

    Input: [B, N, D] where N is number of stocks.
    """

    def __init__(self, d_model, n_heads, n_induce, d_ff, dropout):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(n_induce, d_model) * 0.02)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, D]
        bsz = x.shape[0]
        induce = self.inducing.unsqueeze(0).expand(bsz, -1, -1)  # [B, M, D]
        h, _ = self.attn1(induce, x, x, need_weights=False)
        induce = self.norm1(induce + h)
        y, _ = self.attn2(x, induce, induce, need_weights=False)
        x = self.norm2(x + y)
        x = self.norm3(x + self.ff(x))
        return x


class Model(nn.Module):
    """A-share oriented model for stock_pack:
    - encode per-stock OHLCV (+target history) as embeddings
    - temporal mixing within each stock (captures reversal/vol clustering)
    - cross-sectional induced attention (captures style rotation/herding)
    - output only the target slice (others padded as 0 for compatibility)
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = int(configs.pred_len)
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)

        self.d_model = int(getattr(configs, 'd_model', 256))
        self.dropout = float(getattr(configs, 'dropout', 0.1))
        self.temporal_layers = int(getattr(configs, 'e_layers', 2))
        self.n_heads = int(getattr(configs, 'n_heads', 8))

        # Stock-pack meta (set by stock scripts; see scripts/stock/stock_rolling_backtest.py).
        self.stock_pack = bool(getattr(configs, 'stock_pack', False)) and str(getattr(configs, 'data', '')).lower() == 'stock'
        self.n_codes = int(getattr(configs, 'stock_n_codes', 0) or 0)
        self.n_groups = int(getattr(configs, 'stock_n_groups', 0) or 0)

        d_ff = int(getattr(configs, 'd_ff', self.d_model * 4))

        self.use_cs_mixer = bool(getattr(configs, 'stock_use_cs_mixer', True))
        self.cs_layers = int(getattr(configs, 'stock_cs_layers', 1))
        self.cs_induce = int(getattr(configs, 'stock_cs_induce', 16))
        self.use_time_embed = bool(getattr(configs, 'stock_use_time_embed', True))
        self.time_proj = nn.LazyLinear(self.d_model) if self.use_time_embed else None

        # Packed path (preferred): [B,L,G*N] -> [B,N,L,G] -> embed(G)->D
        packed_ready = self.stock_pack and self.n_codes > 0 and self.n_groups > 0 and self.enc_in == self.n_codes * self.n_groups
        self.packed_ready = bool(packed_ready)
        if self.packed_ready:
            self.feature_proj = nn.Linear(self.n_groups, self.d_model)
            self.temporal_blocks = nn.ModuleList([
                _TemporalMixerBlock(self.seq_len, self.d_model, d_ff=d_ff, dropout=self.dropout)
                for _ in range(self.temporal_layers)
            ])
            self.cs_blocks = nn.ModuleList([
                _ISABlock(self.d_model, self.n_heads, self.cs_induce, d_ff=d_ff, dropout=self.dropout)
                for _ in range(max(0, self.cs_layers))
            ])
            self.head = nn.Linear(self.d_model, self.pred_len)
        else:
            # Fallback: treat input as a single multivariate series (works for non-packed runs).
            self.in_proj = nn.Linear(self.enc_in, self.d_model)
            self.temporal_1d = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.seq_len),
                nn.Dropout(self.dropout),
            )
            self.ff_1d = nn.Sequential(
                nn.Linear(self.d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, self.d_model),
                nn.Dropout(self.dropout),
            )
            self.norm1 = nn.LayerNorm(self.d_model)
            self.norm2 = nn.LayerNorm(self.d_model)
            self.head = nn.Linear(self.d_model, self.pred_len)

    def _forecast_packed(self, x_enc, x_mark_enc=None):
        # x_enc: [B, L, G*N]
        bsz, _, dim = x_enc.shape
        if dim != self.n_groups * self.n_codes:
            raise ValueError(f"packed forecast expects enc_in == n_groups*n_codes, got {dim} vs {self.n_groups*self.n_codes}")

        x = x_enc.view(bsz, self.seq_len, self.n_groups, self.n_codes)  # [B,L,G,N]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,N,L,G]
        x = self.feature_proj(x)  # [B,N,L,D]
        if self.time_proj is not None and x_mark_enc is not None:
            t_emb = self.time_proj(x_mark_enc.float())  # [B,L,D]
            x = x + t_emb.unsqueeze(1)

        for block in self.temporal_blocks:
            x = block(x)

        h = x[:, :, -1, :]  # [B,N,D]
        if self.time_proj is not None and x_mark_enc is not None:
            h = h + t_emb[:, -1:, :].expand(-1, h.shape[1], -1)
        if self.use_cs_mixer:
            for block in self.cs_blocks:
                h = block(h)

        pred = self.head(h)  # [B,N,pred_len]
        pred = pred.permute(0, 2, 1).contiguous()  # [B,pred_len,N]

        # Pad to full channel size for compatibility with stock_pack slicing logic.
        out = x_enc.new_zeros(bsz, self.pred_len, self.n_groups * self.n_codes)
        target_start = (self.n_groups - 1) * self.n_codes
        out[:, :, target_start:target_start + self.n_codes] = pred
        return out

    def _forecast_fallback(self, x_enc, x_mark_enc=None):
        # x_enc: [B, L, D_in]
        bsz, seq_len, _ = x_enc.shape
        if seq_len != self.seq_len:
            raise ValueError(f"AStockMixer expects fixed seq_len={self.seq_len}, got {seq_len}")

        x = self.in_proj(x_enc)  # [B,L,D]
        if self.time_proj is not None and x_mark_enc is not None:
            x = x + self.time_proj(x_mark_enc.float())
        x = x + self.temporal_1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x)
        x = x + self.ff_1d(x)
        x = self.norm2(x)

        h = x[:, -1, :]  # [B,D]
        pred = self.head(h).unsqueeze(-1)  # [B,pred_len,1]

        # Only write to target channel (-1) to keep compatibility with MS slicing.
        out = x_enc.new_zeros(bsz, self.pred_len, self.enc_in)
        out[:, :, -1:] = pred
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in {'long_term_forecast', 'short_term_forecast', 'zero_shot_forecast'}:
            raise ValueError('Only forecast tasks implemented for AStockMixer')

        if self.packed_ready:
            return self._forecast_packed(x_enc, x_mark_enc=x_mark_enc)
        return self._forecast_fallback(x_enc, x_mark_enc=x_mark_enc)
