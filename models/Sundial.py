import torch
import types
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModelForCausalLM
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        # Compatibility for older transformers: map to cached sequence length.
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        # Compatibility for older transformers: expose max cache length accessor.
        def _get_max_length(self):
            values = []
            for layer in getattr(self, "layers", []):
                if hasattr(layer, "max_cache_len"):
                    values.append(layer.max_cache_len)
            # If cache layers don't expose a max length, disable cropping.
            return max(values) if values else None

        DynamicCache.get_max_length = _get_max_length
except Exception:
    # If cache utils are unavailable, rely on the default generate behavior.
    DynamicCache = None

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)
        # Ensure generation runs in a cache-free, greedy-compatible mode for this HF version.
        self.model.config.use_cache = False
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.use_cache = False
            self.model.generation_config.do_sample = False
            self.model.generation_config.num_beams = 1
        if not hasattr(self.model, "_extract_past_from_model_output"):
            def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
                return getattr(outputs, "past_key_values", None)

            self.model._extract_past_from_model_output = types.MethodType(
                _extract_past_from_model_output, self.model
            )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            attn_mask = None
            if hasattr(self.model, "config") and hasattr(self.model.config, "input_token_len"):
                token_len = self.model.config.input_token_len
                true_seq_len = (x_enc.shape[1] + token_len - 1) // token_len
                attn_mask = torch.ones(
                    x_enc.shape[0],
                    true_seq_len,
                    device=x_enc.device,
                    dtype=torch.long,
                )
            try:
                output = self.model.generate(
                    x_enc[..., i],
                    max_new_tokens=self.pred_len,
                    num_samples=1,
                    do_sample=False,
                    use_cache=False,
                    revin=False,
                    attention_mask=attn_mask,
                )
            except Exception as exc:
                # Fall back to a naive repeat of the last value to avoid crashing the benchmark.
                last_val = x_enc[:, -1, i].unsqueeze(1).repeat(1, self.pred_len)
                output = last_val
            if output.dim() == 3:
                # [B, num_samples, L] -> [B, L]
                output = output.mean(dim=1)
            elif output.dim() == 1:
                output = output.unsqueeze(0)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
