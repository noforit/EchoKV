import torch
import torch.nn as nn

class EchoPartialMLP(nn.Module):
    def __init__(self, config, total_save_dim: int = 512, input_mode: str = 'joint'):
        super().__init__()


        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        self.feature_dim = self.num_kv_heads * self.head_dim
        self.total_save_dim = total_save_dim
        self.input_mode = input_mode

        if self.total_save_dim > self.feature_dim:
            raise ValueError(f"total_save_dim ({self.total_save_dim}) must be <= feature_dim ({self.feature_dim})")

        self.target_remain_dim = self.feature_dim - self.total_save_dim
        self.output_dim = self.target_remain_dim


        if self.input_mode == 'joint':
            self.input_dim = self.feature_dim + self.total_save_dim
        elif self.input_mode == 'target_only':
            self.input_dim = self.total_save_dim
        elif self.input_mode == 'anchor_only':
            self.input_dim = self.feature_dim
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        print(f"EchoMLP Init | Mode: {self.input_mode} | Input Dim: {self.input_dim} -> Output Dim: {self.output_dim}")

        self.linear_k = None
        self.linear_v = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, anchor_k, anchor_v, target_k_partial_flat, target_v_partial_flat):

        k_anchor_flat, v_anchor_flat = None, None

        if self.input_mode in ['joint', 'anchor_only']:
            if anchor_k is not None:
                bsz, _, seq_len, _ = anchor_k.shape
                k_anchor_flat = anchor_k.transpose(1, 2).reshape(bsz, seq_len, -1)
            if anchor_v is not None:
                bsz, _, seq_len, _ = anchor_v.shape
                v_anchor_flat = anchor_v.transpose(1, 2).reshape(bsz, seq_len, -1)


        k_in, v_in = None, None


        if self.input_mode == 'joint':
            if k_anchor_flat is not None and target_k_partial_flat is not None:
                k_in = torch.cat([k_anchor_flat, target_k_partial_flat], dim=-1)
        elif self.input_mode == 'target_only':
            k_in = target_k_partial_flat
        elif self.input_mode == 'anchor_only':
            k_in = k_anchor_flat


        if self.input_mode == 'joint':
            if v_anchor_flat is not None and target_v_partial_flat is not None:
                v_in = torch.cat([v_anchor_flat, target_v_partial_flat], dim=-1)
        elif self.input_mode == 'target_only':
            v_in = target_v_partial_flat
        elif self.input_mode == 'anchor_only':
            v_in = v_anchor_flat


        rec_k_remain_flat = None
        if k_in is not None:
            rec_k_remain_flat = self.linear_k(k_in)

        rec_v_remain_flat = None
        if v_in is not None:
            rec_v_remain_flat = self.linear_v(v_in)

        return rec_k_remain_flat, rec_v_remain_flat

    def get_projected_kv(self, anchor_k, anchor_v, stored_k_partial, stored_v_partial, self_attn_module=None):
        bsz, seq_len = 0, 0
        if stored_v_partial is not None:
            bsz, seq_len, _ = stored_v_partial.shape
        elif anchor_v is not None:
            bsz, _, seq_len, _ = anchor_v.shape

        k_remain_flat, v_remain_flat = self.forward(anchor_k, anchor_v, stored_k_partial, stored_v_partial)

        rec_k_full, rec_v_full = None, None
        if k_remain_flat is not None and stored_k_partial is not None:
            k_recon_flat = torch.cat([stored_k_partial, k_remain_flat], dim=-1)
            rec_k_full = k_recon_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if v_remain_flat is not None and stored_v_partial is not None:
            v_recon_flat = torch.cat([stored_v_partial, v_remain_flat], dim=-1)
            rec_v_full = v_recon_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        return rec_k_full, rec_v_full
