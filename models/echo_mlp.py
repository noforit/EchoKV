import torch
from torch import nn


class EchoPartialMLP(nn.Module):
    def __init__(self, config, total_save_dim=1024, block_size=2, input_mode="joint"):
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.feature_dim = self.num_kv_heads * self.head_dim
        self.total_save_dim = total_save_dim
        if not 0 <= total_save_dim <= self.feature_dim:
            raise ValueError("total_save_dim must lie between 0 and the flattened KV dimension")
        self.input_mode = "target_only" if block_size == 1 else input_mode
        dims = {"joint": self.feature_dim + total_save_dim,
                "anchor_only": self.feature_dim, "target_only": total_save_dim}
        if self.input_mode not in dims:
            raise ValueError(f"Unknown input mode: {self.input_mode}")
        self.input_dim = dims[self.input_mode]
        self.output_dim = self.feature_dim - total_save_dim
        self.target_remain_dim = self.output_dim
        self.project_k = nn.Linear(self.input_dim, self.output_dim)
        self.project_v = nn.Linear(self.input_dim, self.output_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):

        for kind in ('k', 'v'):
            old = prefix + f'linear_{kind}.'
            for key in list(state_dict):
                if key.startswith(old):
                    state_dict[prefix + f'project_{kind}.' + key[len(old):]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def select_local_dims(self, flat):
        return flat[..., :self.total_save_dim]

    def _input(self, anchor, local):
        if self.input_mode == 'target_only':
            return local
        if anchor is None:
            return None
        anchor = anchor.transpose(1, 2).flatten(2)
        if self.input_mode == 'anchor_only' or self.total_save_dim == 0:
            return anchor
        return torch.cat((anchor, local), dim=-1)

    def forward(self, anchor_k, anchor_v, target_k_partial_flat, target_v_partial_flat):
        k = self._input(anchor_k, target_k_partial_flat)
        v = self._input(anchor_v, target_v_partial_flat)
        return (self.project_k(k) if k is not None else None,
                self.project_v(v) if v is not None else None)

    def _assemble(self, local, remaining):
        if remaining is None:
            return None
        if self.total_save_dim == 0:
            full = remaining
        else:
            full = torch.cat((local, remaining), dim=-1)
        return full.view(*full.shape[:-1], self.num_kv_heads, self.head_dim).transpose(1, 2)

    def get_projected_kv(self, anchor_k, anchor_v, stored_k_partial, stored_v_partial,
                         self_attn_module=None):
        k, v = self.forward(anchor_k, anchor_v, stored_k_partial, stored_v_partial)
        return self._assemble(stored_k_partial, k), self._assemble(stored_v_partial, v)
