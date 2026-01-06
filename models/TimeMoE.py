import types
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self._patch_dynamic_cache()
        self.model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', trust_remote_code=True)
        self._patch_generation_compat()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    @staticmethod
    def _patch_dynamic_cache():
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            def _get_max_length(self, layer_idx=0):
                return self.get_max_cache_shape(layer_idx)
            DynamicCache.get_max_length = _get_max_length
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, seq_length, layer_idx=0):
                cache_length = self.get_seq_length(layer_idx)
                max_length = self.get_max_cache_shape(layer_idx)
                if max_length is None or max_length < 0:
                    return cache_length
                usable = max_length - seq_length
                if usable <= 0:
                    return 0
                return min(cache_length, usable)
            DynamicCache.get_usable_length = _get_usable_length

    def _patch_generation_compat(self):
        if hasattr(self.model, "_extract_past_from_model_output"):
            return
        def _extract_past_from_model_output(model_self, outputs, standardize_cache_format=False):
            if hasattr(outputs, "past_key_values"):
                return outputs.past_key_values
            if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
                return outputs[1]
            return None
        self.model._extract_past_from_model_output = types.MethodType(
            _extract_past_from_model_output,
            self.model
        )
        if hasattr(self.model, "generate"):
            original_generate = self.model.generate

            def _generate_no_cache(model_self, *args, **kwargs):
                kwargs["use_cache"] = False
                kwargs["do_sample"] = False
                return original_generate(*args, **kwargs)

            self.model.generate = types.MethodType(_generate_no_cache, self.model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        input_size = getattr(self.model.config, "input_size", 1)
        horizons = sorted(getattr(self.model.config, "horizon_lengths", [1]))
        max_horizon = horizons[-1] if horizons else 1
        seq_dim = 1

        if input_size == 1:
            x_seq = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, L)
            batch_size = B * C
        else:
            if input_size != C:
                raise ValueError(
                    f"TimeMoE expects input_size={input_size} but got C={C} features."
                )
            x_seq = x_enc
            batch_size = B

        def pick_horizon(need_len):
            for h in horizons:
                if h >= need_len:
                    return h
            return max_horizon

        if self.pred_len <= max_horizon:
            horizon_len = pick_horizon(self.pred_len)
            outputs = self.model(
                input_ids=x_seq,
                use_cache=False,
                return_dict=True,
                max_horizon_length=horizon_len,
            )
            logits = outputs.logits[:, -1, :]
            if input_size == 1:
                logits = logits.reshape(batch_size, horizon_len)
                preds = logits[:, :self.pred_len]
                dec_out = preds.reshape(B, C, self.pred_len).permute(0, 2, 1).contiguous()
            else:
                logits = logits.reshape(batch_size, horizon_len, input_size)
                dec_out = logits[:, :self.pred_len, :]
        else:
            pred_chunks = []
            remaining = self.pred_len
            while remaining > 0:
                chunk_len = min(remaining, max_horizon)
                horizon_len = pick_horizon(chunk_len)
                outputs = self.model(
                    input_ids=x_seq,
                    use_cache=False,
                    return_dict=True,
                    max_horizon_length=horizon_len,
                )
                logits = outputs.logits[:, -1, :]
                if input_size == 1:
                    logits = logits.reshape(batch_size, horizon_len)
                    next_steps = logits[:, :chunk_len]
                else:
                    logits = logits.reshape(batch_size, horizon_len, input_size)
                    next_steps = logits[:, :chunk_len, :]
                pred_chunks.append(next_steps)
                x_seq = torch.cat([x_seq, next_steps], dim=seq_dim)
                remaining -= chunk_len

            if input_size == 1:
                preds = torch.cat(pred_chunks, dim=seq_dim)
                dec_out = preds.reshape(B, C, self.pred_len).permute(0, 2, 1).contiguous()
            else:
                dec_out = torch.cat(pred_chunks, dim=seq_dim)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
